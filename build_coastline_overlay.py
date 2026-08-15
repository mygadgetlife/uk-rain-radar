#!/usr/bin/env python3
"""
build_coastline_overlay.py
---------------------------
Run this ONCE (or again only if you change EPD_WIDTH/EPD_HEIGHT/UK_BBOX in
geo_utils.py). It fetches a UK boundary polygon, projects every vertex
into the same pixel space fetch_radar.py uses, and rasterizes a crisp
coastline as a transparent PNG. fetch_radar.py then just loads and
composites this static file on every 15-minute run -- no vector geometry
processing needed in the hot path.

Data source: Natural Earth boundary data via the johan/world.geo.json
GitHub mirror (public domain, derived from Natural Earth). If this
specific URL has moved, search "GBR geojson" or grab
ne_50m_admin_0_countries.geojson from Natural Earth directly and filter
for the United Kingdom feature -- the parsing logic below only assumes
standard GeoJSON Polygon/MultiPolygon structure, so it isn't tied to one
mirror.

Requires: pip install pillow requests
"""

import json
import requests
from PIL import Image, ImageDraw

import geo_utils

GEOJSON_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/GBR.geo.json"

OUTPUT_PATH = "coastline_overlay.png"

# Rings (individual islands/landmasses) smaller than this, in output pixels
# squared, are dropped as clutter -- tiny specks don't read on a 4-gray
# 300x400 panel. Lower this if you want more small islands to survive.
MIN_RING_PIXEL_AREA = 8

SUPERSAMPLE = 4          # render at 4x then downscale for smoother lines
LINE_WIDTH_PX = 1        # at final (non-supersampled) resolution
LINE_RGBA = (60, 60, 60, 255)  # medium-gray line, distinguishable from radar tones


def fetch_uk_geometry():
    r = requests.get(GEOJSON_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Handle either a bare Feature or a FeatureCollection with one feature.
    if data.get("type") == "FeatureCollection":
        geometry = data["features"][0]["geometry"]
    elif data.get("type") == "Feature":
        geometry = data["geometry"]
    else:
        geometry = data  # already a raw geometry object

    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    else:
        raise ValueError(f"Unexpected geometry type: {geometry['type']}")


def project_ring(ring):
    """ring: list of [lon, lat] pairs (GeoJSON order) -> list of (x, y) in
    supersampled output pixel space."""
    pts = []
    for lon, lat in ring:
        x, y = geo_utils.latlon_to_output_px(lat, lon)
        pts.append((x * SUPERSAMPLE, y * SUPERSAMPLE))
    return pts


def ring_pixel_area(pts):
    """Shoelace formula, for filtering out tiny islands."""
    area = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0 / (SUPERSAMPLE ** 2)


def build_overlay():
    polygons = fetch_uk_geometry()

    canvas_size = (geo_utils.EPD_WIDTH * SUPERSAMPLE, geo_utils.EPD_HEIGHT * SUPERSAMPLE)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    ring_count = 0
    for polygon in polygons:
        # polygon[0] is the outer ring; polygon[1:] are holes (lakes etc.) --
        # we only care about the outline, so draw every ring the same way.
        for ring in polygon:
            pts = project_ring(ring)
            if ring_pixel_area(pts) < MIN_RING_PIXEL_AREA:
                continue
            draw.line(pts + [pts[0]], fill=LINE_RGBA, width=LINE_WIDTH_PX * SUPERSAMPLE)
            ring_count += 1

    print(f"Drew {ring_count} coastline ring(s)")

    final = canvas.resize(
        (geo_utils.EPD_WIDTH, geo_utils.EPD_HEIGHT), Image.LANCZOS
    )
    final.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} ({geo_utils.EPD_WIDTH}x{geo_utils.EPD_HEIGHT})")


if __name__ == "__main__":
    build_overlay()
