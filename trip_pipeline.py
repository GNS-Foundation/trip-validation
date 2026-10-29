#!/usr/bin/env python3
"""
trip_pipeline.py  —  TRIP -04 breadcrumb pipeline over real GPS tracks.

This applies the *actual* draft-ayerbe-trip-protocol-04 chain-construction rules
to raw GPS data, so that any statistics computed downstream (PSD alpha, Levy fit,
Hamiltonian, ROC) are measured against the pipeline the spec really defines --
not an idealized version of it.

Faithful to the draft:
  - Section 3.1 : quantize each position to an H3 cell (default resolution 10)
                  BEFORE anything else. Raw coordinates never enter a breadcrumb.
  - Section 4.2 : breadcrumbs collected at >= 15 min intervals (SHOULD);
                  never < 5 min (MUST NOT). We sample the dense GPS track on a
                  15-minute collection schedule.
  - Section 4.1 : reject a breadcrumb whose H3 cell is identical to the
                  immediately preceding breadcrumb (MUST). Optional cap of 10
                  breadcrumbs at any single cell (SHOULD) is tracked, not enforced,
                  in this increment.
  - Section 7.1 : displacement series d(i) = haversine(cell(i), cell(i-1)),
                  i.e. distance between consecutive H3 *cell centres*, not raw GPS.

MODELLING CHOICE (documented honestly, per reviewer concern on preprocessing):
  GeoLife samples every 1-5 s; a TRIP device collects a breadcrumb no more often
  than every ~15 min. So we subsample each user's time-sorted track onto a
  15-minute collection schedule, then quantize, then apply consecutive-cell dedup.
  A user's full trajectory is the concatenation of all their .plt sessions in time
  order -- the append-only chain a single TRIP identity would accumulate.
"""

import math
import os
import glob
from dataclasses import dataclass, field
from typing import Iterator

import h3

# ---- draft parameters (Section references in comments) ----
H3_RES = 10                 # 3.1 default resolution
MIN_INTERVAL_S = 15 * 60    # 4.2 SHOULD: >= 15 minutes between breadcrumbs
HARD_MIN_S = 5 * 60         # 4.2 MUST NOT: < 5 minutes
PSD_FLOOR = 64              # 7.4.4 minimum breadcrumbs for PSD computation
CLASS_RECOMMEND = 200       # 7.4.4 recommended for reliable human classification

# GeoLife .plt: 6 header lines, then rows:
#   lat, lon, 0, altitude_ft, days_since_1899_12_30, date, time
_PLT_HEADER_LINES = 6
_DAYS_EPOCH_TO_SECONDS = 86400.0


@dataclass
class RawPoint:
    t: float       # seconds since 1899-12-30 (monotonic, from .plt field 4)
    lat: float
    lon: float


@dataclass
class Trajectory:
    """A user's full time-ordered raw track, concatenated across sessions."""
    user: str
    points: list  # list[RawPoint], sorted by t


def parse_plt(path: str) -> list:
    """Parse one .plt session file into RawPoints."""
    pts = []
    with open(path, "r", errors="ignore") as fh:
        for _ in range(_PLT_HEADER_LINES):
            fh.readline()
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                lat = float(parts[0]); lon = float(parts[1])
                t = float(parts[4]) * _DAYS_EPOCH_TO_SECONDS
            except ValueError:
                continue
            # GeoLife has a few corrupt (0,0) or out-of-range rows; drop them.
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            if lat == 0.0 and lon == 0.0:
                continue
            pts.append(RawPoint(t, lat, lon))
    return pts


def load_user(user_dir: str, user: str) -> Trajectory:
    """Concatenate all of a user's .plt sessions into one time-sorted trajectory."""
    pts = []
    for plt in glob.glob(os.path.join(user_dir, "Trajectory", "*.plt")):
        pts.extend(parse_plt(plt))
    pts.sort(key=lambda p: p.t)
    return Trajectory(user=user, points=pts)


def iter_users(data_root: str) -> Iterator[Trajectory]:
    """Yield one Trajectory per GeoLife user directory."""
    for user in sorted(os.listdir(data_root)):
        udir = os.path.join(data_root, user)
        if os.path.isdir(os.path.join(udir, "Trajectory")):
            yield load_user(udir, user)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class Breadcrumbs:
    """Output of the TRIP -04 pipeline for one trajectory."""
    user: str
    cells: list = field(default_factory=list)       # H3 cell ids, post-dedup
    times: list = field(default_factory=list)        # collection time (s) per cell
    # diagnostics
    n_raw: int = 0                                    # raw GPS points in
    n_collected: int = 0                              # after 15-min schedule
    n_dedup_dropped: int = 0                          # consecutive-cell rejects (4.1)
    span_days: float = 0.0
    n_gaps_over_1h: int = 0                           # collection gaps > 1 h

    @property
    def n_breadcrumbs(self) -> int:
        return len(self.cells)

    @property
    def n_unique_cells(self) -> int:
        return len(set(self.cells))

    def displacements_km(self) -> list:
        """d(i) = haversine between consecutive H3 cell centres (Section 7.1)."""
        out = []
        for a, b in zip(self.cells, self.cells[1:]):
            la, lna = h3.cell_to_latlng(a)
            lb, lnb = h3.cell_to_latlng(b)
            out.append(haversine_km(la, lna, lb, lnb))
        return out


def build_breadcrumbs(traj: Trajectory,
                      res: int = H3_RES,
                      interval_s: int = MIN_INTERVAL_S) -> Breadcrumbs:
    """Apply Sections 3.1, 4.2, 4.1 to a raw trajectory."""
    bc = Breadcrumbs(user=traj.user)
    bc.n_raw = len(traj.points)
    if not traj.points:
        return bc

    bc.span_days = (traj.points[-1].t - traj.points[0].t) / _DAYS_EPOCH_TO_SECONDS

    last_emit_t = None      # last collection tick time
    prev_cell = None        # immediately preceding emitted cell (for 4.1 dedup)
    for p in traj.points:
        # 4.2 : collection schedule -- only sample every >= interval_s
        if last_emit_t is not None and (p.t - last_emit_t) < interval_s:
            continue
        if last_emit_t is not None and (p.t - last_emit_t) > 3600:
            bc.n_gaps_over_1h += 1
        last_emit_t = p.t
        bc.n_collected += 1
        # 3.1 : quantize to H3 cell (raw coords discarded here)
        cell = h3.latlng_to_cell(p.lat, p.lon, res)
        # 4.1 : reject if identical to immediately preceding breadcrumb
        if cell == prev_cell:
            bc.n_dedup_dropped += 1
            continue
        bc.cells.append(cell)
        bc.times.append(p.t)
        prev_cell = cell
    return bc
