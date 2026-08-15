#!/usr/bin/env python3
"""
fetch_radar.py
--------------
Runs on a server / Raspberry Pi / GitHub Actions (NOT the Pico).

Pipeline:
  1. Ask RainViewer for the latest available radar frame.
  2. Work out which XYZ tiles cover the UK bounding box (from geo_utils,
     already aspect-fitted to a 300x400 portrait canvas) at a chosen zoom.
  3. Download and stitch those tiles into one image, crop to the bbox,
     resize to 300x400.
  4. Composite the static coastline_overlay.png on top (see
     build_coastline_overlay.py -- run once, not part of this hot path).
  5. Quantize to 4 grayscale levels using ordered (Bayer) dithering.
  6. IMPORTANT: the Waveshare 4.2" controller's native buffer is always
     400 (x) x 300 (y), regardless of how you mount the panel. Since we've
     rendered everything in *logical* portrait space (300x400) because
     that's the shape the UK actually fits, the final step rotates the
     image 90 degrees into the controller's native landscape buffer order.
     You then physically mount the panel rotated 90 degrees so it reads
     as portrait. If it comes out upside-down after mounting, flip
     ROTATE_90 to ROTATE_270 below -- that's a one-line fix, not a bug in
     the pipeline.
  7. Pack to 2 bits-per-pixel and write a .bin file for the Pico to fetch.

Data source: RainViewer Weather Maps API (https://www.rainviewer.com/api.html)
  - Free for personal / educational / small-scale use.
  - No API key required for the tile/timeline endpoints used here.
  - Please keep polling infrequent (radar updates every ~10 min) and
    mention RainViewer as the data source if you show this publicly.

Requires: pip install pillow requests numpy
"""

import io
import time
import requests
import numpy as np
from PIL import Image

import geo_utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZOOM = geo_utils.ZOOM
TILE_SIZE = geo_utils.TILE_SIZE
UK_BBOX = geo_utils.UK_BBOX

COLOR_SCHEME = 2       # RainViewer palette id (2 = "Universal Blue"), see rainviewer.com/api/color-schemes.html
SMOOTH = 1             # 1 = smoothed tiles, 0 = raw pixels
SNOW = 1               # 1 = show snow separately

# Logical (portrait) render size -- see geo_utils. The panel's *native*
# buffer is the transpose of this (see rotation step below).
LOGICAL_WIDTH = geo_utils.EPD_WIDTH    # 300
LOGICAL_HEIGHT = geo_utils.EPD_HEIGHT  # 400

COASTLINE_OVERLAY_PATH = "coastline_overlay.png"  # built once by build_coastline_overlay.py

OUTPUT_BIN = "radar_300x400_2bpp.bin"

USER_AGENT = "uk-radar-eink-display/1.0 (personal project; contact: you@example.com)"
RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"


# ---------------------------------------------------------------------------
# Slippy-map tiling: stitch tiles covering UK_BBOX at ZOOM, crop precisely
# ---------------------------------------------------------------------------

def tile_pixel_bounds(bbox, zoom, tile_size):
    x0, y0 = geo_utils.deg2tile(bbox["lat_max"], bbox["lon_min"], zoom)  # top-left
    x1, y1 = geo_utils.deg2tile(bbox["lat_min"], bbox["lon_max"], zoom)  # bottom-right

    tx0, ty0 = int(x0), int(y0)
    tx1, ty1 = int(x1), int(y1)

    px0 = int((x0 - tx0) * tile_size)
    py0 = int((y0 - ty0) * tile_size)
    px1 = int((x1 - tx0) * tile_size) + (tx1 - tx0) * tile_size
    py1 = int((y1 - ty0) * tile_size) + (ty1 - ty0) * tile_size

    return tx0, ty0, tx1, ty1, (px0, py0, px1, py1)


def fetch_tile(url, session):
    r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def stitch_radar_tiles(url_template, bbox, zoom, tile_size, session):
    tx0, ty0, tx1, ty1, crop_box = tile_pixel_bounds(bbox, zoom, tile_size)
    cols, rows = tx1 - tx0 + 1, ty1 - ty0 + 1

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
# Image processing: composite -> 4-level grayscale -> rotate -> 2bpp pack
# ---------------------------------------------------------------------------

BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]) / 16.0


def ordered_dither_to_4level(gray_img):
    """gray_img: PIL 'L' image. Returns an array of values in {0,1,2,3}
    (0 = white, 3 = black) using 4x4 Bayer ordered dithering."""
    arr = np.asarray(gray_img, dtype=np.float64) / 255.0
    h, w = arr.shape
    bayer_tiled = np.tile(BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]

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
    radar_img = stitch_radar_tiles(radar_tile_url, UK_BBOX, ZOOM, TILE_SIZE, session)
    radar_img = radar_img.resize((LOGICAL_WIDTH, LOGICAL_HEIGHT), Image.LANCZOS).convert("RGBA")

    white_bg = Image.new("RGBA", radar_img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, radar_img)

    coastline = Image.open(COASTLINE_OVERLAY_PATH).convert("RGBA")
    if coastline.size != composite.size:
        raise RuntimeError(
            f"coastline_overlay.png is {coastline.size}, expected {composite.size}. "
            "Re-run build_coastline_overlay.py after changing geo_utils dimensions."
        )
    composite = Image.alpha_composite(composite, coastline)

    gray = composite.convert("L")

    # Rotate logical portrait (300x400) into the controller's native buffer
    # order (400x300). Flip to ROTATE_270 if the mounted panel reads upside
    # down or mirrored.
    native = gray.transpose(Image.ROTATE_90)

    levels = ordered_dither_to_4level(native)
    packed = pack_2bpp(levels)

    with open(OUTPUT_BIN, "wb") as f:
        f.write(packed)

    print(f"Wrote {OUTPUT_BIN}: {len(packed)} bytes (native buffer {native.size[0]}x{native.size[1]})")
    return OUTPUT_BIN


if __name__ == "__main__":
    build_frame()
