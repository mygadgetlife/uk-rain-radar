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

# Debug PNGs, saved alongside the .bin so you can visually compare pipeline
# output against a photo of the physical panel. Set False once you trust
# the pipeline -- committing PNGs every 15 min bloats repo history faster
# than the tiny .bin does.
DEBUG_SAVE_PNG = True
DEBUG_RAW_TILES_PATH = "debug_raw_tiles.png"  # straight from RainViewer, before ANY processing
DEBUG_COMPOSITE_PATH = "debug_composite.png"  # full-color, logical portrait, human-readable
DEBUG_PREVIEW_PATH = "debug_preview.png"      # quantized 4-gray, native rotation -- matches panel exactly

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
    # --- FIX START: Force perfect 300x400 bounding box alignment ---
    # Crop to the web-mercator rounded bounding box first
    raw_cropped = canvas.crop(crop_box)
    
    # Calculate exactly what height matches the cropped width to maintain a strict 300:400 aspect ratio
    target_ratio = LOGICAL_WIDTH / LOGICAL_HEIGHT # 0.75
    w, h = raw_cropped.size
    
    if (w / h) > target_ratio:
        # Image is too wide; trim the sides evenly
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        final_crop = (left, 0, left + new_w, h)
    else:
        # Image is too tall; trim the top/bottom evenly
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        final_crop = (0, top, w, top + new_h)
        
    return raw_cropped.crop(final_crop)
    # --- FIX END ---
    # return canvas.crop(crop_box)


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

def quantize_to_4level(gray_img):
    """gray_img: PIL 'L' image. Returns an array of values in {0,1,2,3}
    (0 = white, 3 = black) using simple nearest-level thresholding --
    no dithering. RainViewer's radar tiles are already flat colour bands
    rather than smooth gradients, so dithering (designed to fake extra
    shades by mixing black/white pixels at a fine spatial scale) mostly
    adds visual noise here rather than useful detail."""
    arr = np.asarray(gray_img, dtype=np.float64) / 255.0
    ink = 1.0 - arr  # 0 = white ... 1 = black
    levels = np.clip(np.round(ink * 3), 0, 3).astype(np.uint8)
    return levels  # 0 = white ... 3 = black


def levels_to_preview_image(levels):
    """Turn the quantized 0-3 level array back into a viewable 8-bit
    grayscale PNG (0->white, 3->black), so you can look at exactly what
    the panel is being sent, pixel-for-pixel, without decoding the .bin
    by hand."""
    shade_map = np.array([255, 170, 85, 0], dtype=np.uint8)  # level 0..3 -> gray value
    preview_arr = shade_map[levels]
    return Image.fromarray(preview_arr, mode="L")


def pack_2bpp(levels):
    """Pack a (H, W) array of 2-bit values (0-3) into bytes, 4 pixels/byte.

    Bit order matches Waveshare's EPD_4IN2_4GrayDisplay() unpacking loop
    specifically (confirmed by tracing their driver source) -- it reads
    each byte's four pixels starting from the LOWEST bits, so the first
    (leftmost) pixel of each group of 4 needs to go in the lowest 2 bits,
    not the highest. This is the opposite of the "obvious" MSB-first
    layout, which is exactly what caused a scrambled image originally."""
    h, w = levels.shape
    assert w % 4 == 0, "width must be a multiple of 4 for clean 2bpp packing"
    flat = levels.reshape(h, w // 4, 4)
    packed = (flat[:, :, 3] << 6) | (flat[:, :, 2] << 4) | (flat[:, :, 1] << 2) | flat[:, :, 0]
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
    print(f"Tile URL template: {radar_tile_url}")
    radar_img = stitch_radar_tiles(radar_tile_url, UK_BBOX, ZOOM, TILE_SIZE, session)

    if DEBUG_SAVE_PNG:
        # Straight from RainViewer, before resize/composite/coastline/quantize
        # -- if this alone doesn't show a believable precipitation pattern,
        # the bug is in the fetch itself, not anything downstream.
        radar_img.convert("RGB").save(DEBUG_RAW_TILES_PATH)
        print(f"Wrote {DEBUG_RAW_TILES_PATH} (raw stitched tiles, {radar_img.size[0]}x{radar_img.size[1]})")

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

    if DEBUG_SAVE_PNG:
        composite.convert("RGB").save(DEBUG_COMPOSITE_PATH)
        print(f"Wrote {DEBUG_COMPOSITE_PATH} (full-color, {composite.size[0]}x{composite.size[1]})")

    gray = composite.convert("L")

    # Rotate logical portrait (300x400) into the controller's native buffer
    # order (400x300). Flip to ROTATE_270 if the mounted panel reads upside
    # down or mirrored.
    native = gray.transpose(Image.ROTATE_90)

    levels = quantize_to_4level(native)  # our convention: 0=white ... 3=black

    if DEBUG_SAVE_PNG:
        preview = levels_to_preview_image(levels)
        preview.save(DEBUG_PREVIEW_PATH)
        print(f"Wrote {DEBUG_PREVIEW_PATH} (quantized, native orientation, {preview.size[0]}x{preview.size[1]})")

    # Waveshare's EPD_4IN2_V2_4GrayDisplay() expects raw codes 0=black and
    # 3=white as you'd assume, but its actual grayscale LUT swaps the two
    # middle codes relative to a naive ascending-brightness assumption:
    # raw 1 renders as LIGHT gray and raw 2 as DARK gray (confirmed via
    # test_4gray_bands.py against the physical panel -- not something
    # derivable from the driver source alone). Map explicitly rather than
    # with a linear formula, so `levels` elsewhere stays in the intuitive
    # white-to-black convention regardless of this panel-specific quirk.
    LEVEL_TO_DRIVER_CODE = np.array([3, 1, 2, 0], dtype=np.uint8)  # index = our level (0=white..3=black)
    driver_levels = LEVEL_TO_DRIVER_CODE[levels]
    packed = pack_2bpp(driver_levels)

    with open(OUTPUT_BIN, "wb") as f:
        f.write(packed)

    print(f"Wrote {OUTPUT_BIN}: {len(packed)} bytes (native buffer {native.size[0]}x{native.size[1]})")
    return OUTPUT_BIN


if __name__ == "__main__":
    build_frame()
