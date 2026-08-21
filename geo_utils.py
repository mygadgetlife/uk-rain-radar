"""
geo_utils.py
------------
Shared by fetch_radar.py and build_coastline_overlay.py so that the radar
imagery and the static coastline overlay always land on exactly the same
pixels. Import this rather than duplicating the projection math.

All rendering happens in "logical portrait" space: EPD_WIDTH x EPD_HEIGHT
= 300 x 400. This is NOT the panel's native buffer order (see the rotation
step in fetch_radar.py) -- it's just the orientation that's convenient to
reason about while plotting the UK, which is itself taller than it is wide.
"""

import math

TILE_SIZE = 512
ZOOM = 6

# Logical (portrait) output size -- what you actually see once the panel
# is physically mounted rotated 90 degrees.
EPD_WIDTH = 300
EPD_HEIGHT = 400

# Roughly the British Isles, before aspect correction.
_RAW_BBOX = {
    "lat_min": 49.8,
    "lat_max": 61.0,
    "lon_min": -8.5,
    "lon_max": 2.0,
}


def deg2tile(lat_deg, lon_deg, zoom=ZOOM):
    """Standard Web Mercator slippy-map projection: lat/lon -> fractional tile coords."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _tile2deg(x, y, zoom=ZOOM):
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def _pixel_bbox(bbox, zoom=ZOOM, tile_size=TILE_SIZE):
    x0, y0 = deg2tile(bbox["lat_max"], bbox["lon_min"], zoom)
    x1, y1 = deg2tile(bbox["lat_min"], bbox["lon_max"], zoom)
    return x0 * tile_size, y0 * tile_size, x1 * tile_size, y1 * tile_size


def fit_bbox_to_aspect(bbox, target_w, target_h, zoom=ZOOM, tile_size=TILE_SIZE):
    """Grow the bbox on whichever axis is too narrow so its pixel aspect
    ratio matches target_w/target_h exactly, keeping the same center. This
    avoids stretching/squashing the UK shape when resizing to the panel
    resolution."""
    px0, py0, px1, py1 = _pixel_bbox(bbox, zoom, tile_size)
    cur_w, cur_h = px1 - px0, py1 - py0
    target_ratio = target_w / target_h
    cur_ratio = cur_w / cur_h

    if cur_ratio > target_ratio:
        new_h = cur_w / target_ratio
        extra = (new_h - cur_h) / 2
        py0, py1 = py0 - extra, py1 + extra
    else:
        new_w = cur_h * target_ratio
        extra = (new_w - cur_w) / 2
        px0, px1 = px0 - extra, px1 + extra

    lat_max, lon_min = _tile2deg(px0 / tile_size, py0 / tile_size, zoom)
    lat_min, lon_max = _tile2deg(px1 / tile_size, py1 / tile_size, zoom)
    return {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}


# The bbox actually used everywhere -- computed once at import time so
# fetch_radar.py and build_coastline_overlay.py are guaranteed to agree.
UK_BBOX = fit_bbox_to_aspect(_RAW_BBOX, EPD_WIDTH, EPD_HEIGHT)


def latlon_to_output_px(lat, lon, bbox=UK_BBOX, zoom=ZOOM, tile_size=TILE_SIZE,
                         out_w=EPD_WIDTH, out_h=EPD_HEIGHT):
    """Project a lat/lon straight to a pixel coordinate in the final
    EPD_WIDTH x EPD_HEIGHT logical image -- the same crop+resize that
    fetch_radar.py applies to the radar tiles."""
    px0, py0 = deg2tile(bbox["lat_max"], bbox["lon_min"], zoom)
    px0, py0 = px0 * tile_size, py0 * tile_size
    px1, py1 = deg2tile(bbox["lat_min"], bbox["lon_max"], zoom)
    px1, py1 = px1 * tile_size, py1 * tile_size

    x, y = deg2tile(lat, lon, zoom)
    x, y = x * tile_size, y * tile_size

    out_x = (x - px0) / (px1 - px0) * out_w
    out_y = (y - py0) / (py1 - py0) * out_h
    return out_x, out_y
