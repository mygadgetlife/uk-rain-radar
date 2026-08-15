#!/usr/bin/env python3
"""
fetch_radar.py
--------------
Runs on a server / Raspberry Pi / any always-on machine (NOT the Pico).

Pipeline:
  1. Ask RainViewer for the latest available radar frame.
  2. Work out which XYZ tiles cover a UK bounding box at a chosen zoom.
  3. Download and stitch those tiles into one image.
  4. Optionally composite a plain basemap (coastline/borders) underneath so
     the radar isn't floating on a blank background.
  5. Crop to the exact bounding box, resize to the e-paper resolution
     (400x300 for the Waveshare 4.2"), and quantize to 4 grayscale levels
     using ordered (Bayer) dithering.
  6. Pack the result to 2 bits-per-pixel and write it to a .bin file that
     the Pico W will download over HTTP.

Data source: RainViewer Weather Maps API (https://www.rainviewer.com/api.html)
  - Free for personal / educational / small-scale use.
  - No API key required for the tile/timeline endpoints used here.
  - Please keep polling infrequent (radar updates every ~10 min) and
    mention RainViewer as the data source if you show this publicly.

Requires: pip install pillow requests numpy
"""

import io
import math
import time
import requests
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Roughly the British Isles. Tweak to taste (e.g. tighten to just England).
UK_BBOX = {
    "lat_min": 49.8,
    "lat_max": 61.0,
    "lon_min": -8.5,
    "lon_max": 2.0,
}

ZOOM = 6              # RainViewer tiles go up to zoom 7 (512px tiles) on the free tier
TILE_SIZE = 256        # 256 or 512
COLOR_SCHEME = 2       # RainViewer palette id (2 = "Universal Blue"), see rainviewer.com/api/color-schemes.html
SMOOTH = 1             # 1 = smoothed tiles, 0 = raw pixels
SNOW = 1               # 1 = show snow separately

# Waveshare 4.2" e-paper (Pico version) native resolution
EPD_WIDTH = 400
EPD_HEIGHT = 300

OUTPUT_BIN = "radar_400x300_2bpp.bin"

# Basemap (optional). OpenStreetMap's standard tile server is free but has a
# strict usage policy for automated/bulk fetching (see
# https://operations.osmfoundation.org/policies/tiles/). Because this script
# only runs every 10-15 minutes and fetches a handful of tiles, it's within
# reasonable personal use, but set your own User-Agent and don't lower the
# refresh interval further. Set USE_BASEMAP = False to skip it entirely and
# just show radar-on-white, which is simpler and avoids the policy question.
USE_BASEMAP = True
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "uk-radar-eink-display/1.0 (personal project; contact: you@example.com)"

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"


# ---------------------------------------------------------------------------
# Slippy-map tile math (standard Web Mercator XYZ scheme)
# ---------------------------------------------------------------------------

def deg2tile(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_pixel_bounds(bbox, zoom, tile_size):
    """Return (x0_tile, y0_tile, x1_tile, y1_tile, px_bbox) where px_bbox is
    the crop rectangle in pixel space of the stitched image."""
    x0, y0 = deg2tile(bbox["lat_max"], bbox["lon_min"], zoom)  # top-left
    x1, y1 = deg2tile(bbox["lat_min"], bbox["lon_max"], zoom)  # bottom-right

    tx0, ty0 = math.floor(x0), math.floor(y0)
    tx1, ty1 = math.floor(x1), math.floor(y1)

    px0 = int((x0 - tx0) * tile_size)
    py0 = int((y0 - ty0) * tile_size)
    px1 = int((x1 - tx0) * tile_size) + (tx1 - tx0) * tile_size
    py1 = int((y1 - ty0) * tile_size) + (ty1 - ty0) * tile_size

    return tx0, ty0, tx1, ty1, (px0, py0, px1, py1)


def fetch_tile(url, session):
    r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def stitch_tiles(url_template, bbox, zoom, tile_size, session):
    tx0, ty0, tx1, ty1, crop_box = tile_pixel_bounds(bbox, zoom, tile_size)
    cols = tx1 - tx0 + 1
    rows = ty1 - ty0 + 1

    canvas = Image.new("RGBA", (cols * tile_size, rows * tile_size), (0, 0, 0, 0))
    for row, ty in enumerate(range(ty0, ty1 + 1)):
        for col, tx in enumerate(range(tx0, tx1 + 1)):
            url = url_template.format(z=zoom, x=tx, y=ty)
            try:
                tile = fetch_tile(url, session)
            except requests.RequestException as e:
                print(f"  tile fetch failed ({tx},{ty}): {e}")
                tile = Image.new("RGBA", (tile_size, tile_size), (255, 255, 255, 0))
            canvas.paste(tile, (col * tile_size, row * tile_size))

    return canvas.crop(crop_box)


# ---------------------------------------------------------------------------
# Radar frame lookup
# ---------------------------------------------------------------------------

def get_latest_radar_path(session):
    r = session.get(RAINVIEWER_API, headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    data = r.json()
    host = data["host"]
    latest_frame = data["radar"]["past"][-1]  # most recent observed frame
    return host, latest_frame["path"], latest_frame["time"]


# ---------------------------------------------------------------------------
# Image processing: composite -> 4-level grayscale -> 2bpp pack
# ---------------------------------------------------------------------------

BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]) / 16.0


def ordered_dither_to_4level(gray_img):
    """gray_img: PIL 'L' image. Returns an array of values in {0,1,2,3}
    (0 = white, 3 = black) using 4x4 Bayer ordered dithering, which looks
    much better on e-ink than naive rounding."""
    arr = np.asarray(gray_img, dtype=np.float64) / 255.0  # 0=black .. 1=white in PIL's L
    h, w = arr.shape
    bayer_tiled = np.tile(BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]

    # invert so higher "ink" value = darker, then quantize to 4 levels with dither
    ink = 1.0 - arr
    levels = 3
    dithered = ink + (bayer_tiled - 0.5) / levels
    quant = np.clip(np.round(dithered * levels), 0, levels).astype(np.uint8)
    return quant  # 0 = white ... 3 = black


def pack_2bpp(levels):
    """Pack a (H, W) array of 2-bit values (0-3) into bytes, 4 pixels/byte,
    MSB-first, matching the format Waveshare's 4Gray demo code expects."""
    h, w = levels.shape
    assert w % 4 == 0, "width must be a multiple of 4 for clean 2bpp packing"
    flat = levels.reshape(h, w // 4, 4)
    packed = (flat[:, :, 0] << 6) | (flat[:, :, 1] << 4) | (flat[:, :, 2] << 2) | flat[:, :, 3]
    return packed.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_frame():
    session = requests.Session()

    host, path, frame_time = get_latest_radar_path(session)
    print(f"Latest radar frame: {time.strftime('%H:%M:%S', time.gmtime(frame_time))} UTC")

    radar_tile_url = (
        host + path + f"/{TILE_SIZE}/{{z}}/{{x}}/{{y}}/{COLOR_SCHEME}/{SMOOTH}_{SNOW}.png"
    )
    radar_img = stitch_tiles(radar_tile_url, UK_BBOX, ZOOM, TILE_SIZE, session)

    if USE_BASEMAP:
        base_img = stitch_tiles(OSM_TILE_URL, UK_BBOX, ZOOM, TILE_SIZE, session).convert("RGBA")
        # Lighten the basemap so radar reads clearly on top once quantized
        base_img = Image.eval(base_img.convert("L"), lambda p: int(200 + p * 55 / 255)).convert("RGBA")
        composite = Image.alpha_composite(base_img, radar_img)
    else:
        white_bg = Image.new("RGBA", radar_img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, radar_img)

    composite = composite.convert("RGB").resize((EPD_WIDTH, EPD_HEIGHT), Image.LANCZOS)
    gray = composite.convert("L")

    levels = ordered_dither_to_4level(gray)
    packed = pack_2bpp(levels)

    with open(OUTPUT_BIN, "wb") as f:
        f.write(packed)

    print(f"Wrote {OUTPUT_BIN}: {len(packed)} bytes for {EPD_WIDTH}x{EPD_HEIGHT} @ 2bpp")
    return OUTPUT_BIN


if __name__ == "__main__":
    build_frame()
