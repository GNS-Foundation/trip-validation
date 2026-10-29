#!/usr/bin/env python3
"""
make_serhii_set.py  —  the 64-cell NL synthetic trajectory for Serhii,
verified the way his script will verify it: every WHOLE H3 cell inside the
Netherlands polygon, not just the centroid.

Placed in Utrecht (interior NL, far from any border) so all 64 cells clear the
whole-cell containment test with margin. 15-minute cadence, res-10, labelled
synthetic in the file. Serhii drops this in place of his placeholder evidence.
"""

import json
import math
import os
import h3
from shapely.geometry import shape, Point, Polygon

HERE = os.path.dirname(__file__)
NL = os.path.join(HERE, "geo", "netherlands.json")
OUT = os.path.join(HERE, "results", "trip_synthetic_example.json")

RES = 10
STEP_S = 15 * 60
N_CELLS = 64
UTRECHT = (52.0907, 5.1214)          # interior NL, far from borders
T0 = 1767225600                      # 2026-01-01T00:00:00Z, arbitrary
rng = __import__("random").Random(20260910)


def nl_polygon():
    d = json.load(open(NL))
    g = d["features"][0]["geometry"] if d["type"] == "FeatureCollection" else \
        (d["geometry"] if d["type"] == "Feature" else d)
    return shape(g)


def cell_polygon(cell):
    # h3 v4: cell_to_boundary -> list of (lat, lng); shapely wants (lng, lat)
    ring = [(lng, lat) for (lat, lng) in h3.cell_to_boundary(cell)]
    return Polygon(ring)


def whole_cell_inside(cell, country):
    return country.contains(cell_polygon(cell))


def build_set(country):
    """Local random walk from Utrecht -> 64 distinct res-10 cells, all whole-cell
    inside NL. Small steps keep the path compact and interior."""
    lat, lon = UTRECHT
    cells, times = [], []
    prev = None
    t = T0
    guard = 0
    bearing = rng.uniform(0, 2 * math.pi)      # persistent heading -> a walk, not milling
    while len(cells) < N_CELLS and guard < 100000:
        guard += 1
        # step ~120-360 m with mild bearing persistence (a plausible walk across town)
        r_km = rng.uniform(0.12, 0.36)
        bearing += rng.uniform(-0.6, 0.6)
        b = bearing
        nlat = lat + (r_km / 111.32) * math.cos(b)
        nlon = lon + (r_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(b)
        cell = h3.latlng_to_cell(nlat, nlon, RES)
        if cell == prev:
            continue                       # 4.1 dedup
        if not whole_cell_inside(cell, country):
            continue                       # keep the path interior; skip edge cells
        lat, lon = nlat, nlon
        t += STEP_S
        cells.append(cell); times.append(t); prev = cell
    return cells, times


def main():
    country = nl_polygon()
    cells, times = build_set(country)
    assert len(cells) == N_CELLS, f"only got {len(cells)} cells"

    # verify 64/64 whole-cell containment (what Serhii's script requires)
    inside = [whole_cell_inside(c, country) for c in cells]
    n_inside = sum(inside)
    span_h = (times[-1] - times[0]) / 3600.0

    out = {
        "_label": "SYNTHETIC — illustrative only, not real movement. Generated for "
                  "the geographic-results basis=evidence worked example (draft-ayerbe-"
                  "trip-protocol producer side).",
        "parameters": {
            "h3_resolution": RES,
            "collection_interval_s": STEP_S,
            "n_breadcrumbs": len(cells),
            "span_hours": round(span_h, 2),
        },
        "jurisdiction_country": "NL",
        "containment_check": {
            "rule": "each whole H3 cell fully inside the Netherlands polygon",
            "cells_inside": n_inside,
            "cells_total": len(cells),
            "all_inside": n_inside == len(cells),
            "polygon_source": "georgique/world-geojson netherlands.json",
        },
        "basis": "evidence",
        "basis_note": "device-signed H3 cells appraised directly by the Verifier "
                      "(Criticality Engine); no coordinate is signed or seen.",
        "claim_uuid": "b6f1e0c2-0000-4a1a-9f00-000000000001",
        "h3_cells": cells,
        "collection_times_unix_s": times,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"cells: {len(cells)}   whole-cell inside NL: {n_inside}/{len(cells)}   "
          f"span: {span_h:.2f} h")
    print(f"first cell: {cells[0]}   last cell: {cells[-1]}")
    la0, lo0 = h3.cell_to_latlng(cells[0]); la1, lo1 = h3.cell_to_latlng(cells[-1])
    print(f"centroids: start ({la0:.4f},{lo0:.4f})  end ({la1:.4f},{lo1:.4f})")
    print(f"distinct cells: {len(set(cells))}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
