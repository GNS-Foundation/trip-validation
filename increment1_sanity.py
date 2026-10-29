#!/usr/bin/env python3
"""
increment1_sanity.py  —  Feed real GeoLife through the real TRIP -04 pipeline.

No statistics yet. This proves the pipeline runs against real data and shows what
comes out: how many breadcrumbs survive Sections 4.2 (interval) and 4.1 (dedup),
how many users reach the PSD floor (64) and the classification recommendation
(200), and what the collection gaps look like. Those gaps matter because the
reviewer's stationarity concern (Section 7.1) lives exactly there.
"""

import os
import statistics as st
from trip_pipeline import (
    iter_users, build_breadcrumbs,
    PSD_FLOOR, CLASS_RECOMMEND, H3_RES, MIN_INTERVAL_S,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data",
                         "Geolife Trajectories 1.3", "Data")


def main():
    print(f"TRIP -04 pipeline over GeoLife  (H3 res {H3_RES}, "
          f"collection interval {MIN_INTERVAL_S//60} min)\n")
    rows = []
    for traj in iter_users(DATA_ROOT):
        bc = build_breadcrumbs(traj)
        rows.append(bc)

    rows.sort(key=lambda b: b.n_breadcrumbs, reverse=True)

    print(f"{'user':>5} {'raw':>9} {'collect':>8} {'dropped':>8} "
          f"{'crumbs':>7} {'uniq':>6} {'days':>7} {'gaps>1h':>7}")
    print("-" * 66)
    for b in rows[:15]:
        print(f"{b.user:>5} {b.n_raw:>9} {b.n_collected:>8} {b.n_dedup_dropped:>8} "
              f"{b.n_breadcrumbs:>7} {b.n_unique_cells:>6} {b.span_days:>7.1f} "
              f"{b.n_gaps_over_1h:>7}")
    print(f"   ... ({len(rows)} users total; top 15 by breadcrumb count shown)\n")

    crumbs = [b.n_breadcrumbs for b in rows]
    uniq = [b.n_unique_cells for b in rows]
    ge64 = sum(1 for c in crumbs if c >= PSD_FLOOR)
    ge200 = sum(1 for c in crumbs if c >= CLASS_RECOMMEND)
    total_raw = sum(b.n_raw for b in rows)
    total_collect = sum(b.n_collected for b in rows)
    total_drop = sum(b.n_dedup_dropped for b in rows)

    print("SUMMARY")
    print(f"  users                         : {len(rows)}")
    print(f"  raw GPS points (all users)    : {total_raw:,}")
    print(f"  collected on 15-min schedule  : {total_collect:,} "
          f"({100*total_collect/max(total_raw,1):.2f}% of raw)")
    print(f"  dropped by 4.1 consec-dedup   : {total_drop:,} "
          f"({100*total_drop/max(total_collect,1):.1f}% of collected)")
    print(f"  breadcrumbs (median / max)    : {int(st.median(crumbs))} / {max(crumbs)}")
    print(f"  unique cells (median / max)   : {int(st.median(uniq))} / {max(uniq)}")
    print(f"  users >= {PSD_FLOOR:<3} (PSD floor)     : {ge64}  "
          f"({100*ge64/len(rows):.0f}%)")
    print(f"  users >= {CLASS_RECOMMEND:<3} (class. rec.)   : {ge200}  "
          f"({100*ge200/len(rows):.0f}%)")
    print()
    print("Reading: the usable cohort is the users clearing the PSD floor. Those")
    print("are the trajectories we can compute alpha on in the next increment.")
    print("Collection gaps > 1 h are where the stationarity question (7.1) bites;")
    print("we surface them now so the alpha method is chosen with eyes open.")


if __name__ == "__main__":
    main()
