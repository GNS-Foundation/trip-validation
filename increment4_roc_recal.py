#!/usr/bin/env python3
"""
increment4_roc_recal.py  —  ROC curves, an honest recalibrated band, and the
synthetic worked example promised to Serhii.

Increment 3 gave AUCs (~0.75-0.80) but only the draft's two-sided in-band rule at
its fixed [0.30, 0.80] boundary. This increment does three things:

  1. Adds a "brownian_drift" adversary whose displacement SERIES is itself
     integrated (brown noise, high alpha). The draft's UPPER bound (0.80) exists
     to catch exactly this "replay with drift" case; none of the Increment-3
     adversaries went high, so we add one that does and test whether an upper
     bound is actually needed.

  2. Builds real ROC curves and searches for the alpha band that best separates
     real humans from pooled adversaries (Youden's J), and reports that
     recalibrated band against the draft's [0.30, 0.80] operating point.

  3. Emits the synthetic worked-example trajectory for Serhii: 64 res-10 H3 cells
     at 15-min cadence, all within one jurisdiction, labelled synthetic, for him
     to encode and byte-count as the basis=evidence third example.

Plots: alpha-by-class distribution (the recalibration made visual) and ROC curves.
"""

import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h3

from trip_pipeline import (Trajectory, RawPoint, build_breadcrumbs, PSD_FLOOR, H3_RES)
from increment2_alpha import psd_alpha, dfa_exponent, load_displacements, WINDOW, BAND
from increment3_roc import (GENERATORS, _track_from_steps, alpha_set, humanness,
                            auc, in_band, N_TRACKS, N_STEPS, STEP_S)

rng = np.random.default_rng(4)
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


# ---- the missing "high alpha" adversary: drift / replay-with-drift ----
def gen_brownian_drift(scale_km):
    # step magnitude is itself a random walk -> integrated -> brown-noise
    # displacement series -> high alpha (the case the draft's 0.80 upper bound targets)
    r = scale_km
    steps = []
    for _ in range(N_STEPS):
        r = max(0.05, r + rng.normal(0, scale_km * 0.4))
        steps.append(r)
    bearings = rng.uniform(0, 2 * math.pi, N_STEPS)
    return _track_from_steps(np.array(steps), bearings)


def roc_curve(pos, neg):
    """One-sided score ROC: predict 'human' if score >= t. Returns fpr, tpr, auc."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    thr = np.unique(np.concatenate([pos, neg]))
    thr = np.concatenate([[thr.min() - 1], thr, [thr.max() + 1]])[::-1]
    tpr = [(pos >= t).mean() for t in thr]
    fpr = [(neg >= t).mean() for t in thr]
    return np.array(fpr), np.array(tpr), auc(pos, neg)


def best_two_sided_band(human_a, synth_a):
    """Grid-search the [lo,hi] band maximizing Youden's J = TPR_human - FPR_synth."""
    los = np.round(np.arange(-0.20, 0.61, 0.02), 2)
    his = np.round(np.arange(0.40, 2.21, 0.05), 2)
    best = (-1, None, None, None)
    for lo in los:
        for hi in his:
            if hi <= lo:
                continue
            tpr = float(np.mean((human_a >= lo) & (human_a <= hi)))
            fpr = float(np.mean((synth_a >= lo) & (synth_a <= hi)))
            J = tpr - fpr
            if J > best[0]:
                best = (J, (lo, hi), tpr, fpr)
    return best


def main():
    human_disp = load_displacements()
    human_d = [human_disp[u] for u in sorted(human_disp)]
    scale = float(np.median([np.median(d) for d in human_d]))

    gens = dict(GENERATORS)
    gens["brownian_drift"] = gen_brownian_drift

    hA = {"psd": alpha_set(human_d, "psd"), "dfa": alpha_set(human_d, "dfa")}
    synth = {}
    for name, gen in gens.items():
        dl = [build_breadcrumbs(gen(scale)).displacements_km() for _ in range(N_TRACKS)]
        synth[name] = {"psd": alpha_set(dl, "psd"), "dfa": alpha_set(dl, "dfa")}

    # ---- text report ----
    for method, label in (("psd", "Section 7.1 raw-DFT"), ("dfa", "DFA (beta=2H-1)")):
        print("=" * 68)
        print(f"METHOD: {label}")
        print("=" * 68)
        h = hA[method]
        print(f"  {'class':<18}{'n':>5}{'median a':>10}{'in draft band':>15}")
        print(f"  {'HUMAN (real)':<18}{len(h):>5}{np.median(h):>10.3f}"
              f"{100*in_band(h):>13.0f}%")
        pooled = []
        for name in gens:
            a = synth[name][method]; a = a[np.isfinite(a)]
            pooled.append(a)
            print(f"  {name:<18}{len(a):>5}{np.median(a):>10.3f}{100*in_band(a):>13.0f}%")
        pooled = np.concatenate(pooled)
        # recalibrate
        J, band, tpr, fpr = best_two_sided_band(h, pooled)
        dtpr = float(np.mean((h >= BAND[0]) & (h <= BAND[1])))
        dfpr = float(np.mean((pooled >= BAND[0]) & (pooled <= BAND[1])))
        print(f"\n  DRAFT band   [0.30, 0.80] -> human TPR {100*dtpr:.0f}%, "
              f"bot FPR {100*dfpr:.0f}%   (Youden J={dtpr-dfpr:+.2f})")
        print(f"  RECAL band   [{band[0]:.2f}, {band[1]:.2f}] -> human TPR {100*tpr:.0f}%, "
              f"bot FPR {100*fpr:.0f}%   (Youden J={J:+.2f})")
        # AUC using centered human-ness score
        aucv = auc(humanness(h), humanness(pooled))
        print(f"  AUC (centered score, human vs pooled bots): {aucv:.3f}\n")

    # ---- plots (psd method) ----
    hpsd = hA["psd"]
    pooled_psd = np.concatenate([synth[n]["psd"][np.isfinite(synth[n]["psd"])]
                                 for n in gens])
    levy_psd = synth["truncated_levy"]["psd"]

    # plot 1: alpha distribution by class
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1.2, 2.2, 45)
    ax.hist(hpsd, bins=bins, alpha=0.65, label=f"real humans (n={len(hpsd)})",
            color="#1f77b4", density=True)
    ax.hist(pooled_psd, bins=bins, alpha=0.5, label=f"synthetic bots (n={len(pooled_psd)})",
            color="#d62728", density=True)
    ax.axvspan(BAND[0], BAND[1], color="green", alpha=0.12,
               label="draft band [0.30, 0.80]")
    J, band, *_ = best_two_sided_band(hpsd, pooled_psd)
    ax.axvspan(band[0], band[1], color="orange", alpha=0.15,
               label=f"recalibrated [{band[0]:.2f}, {band[1]:.2f}]")
    ax.set_xlabel("PSD scaling exponent alpha (Section 7.1 method)")
    ax.set_ylabel("density")
    ax.set_title("alpha by class: real humans vs synthetic adversaries (GeoLife)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/alpha_by_class.png", dpi=130); plt.close(fig)

    # plot 2: ROC curves (centered human-ness score)
    fig, ax = plt.subplots(figsize=(6, 6))
    for neg, name, col in ((pooled_psd, "vs pooled bots", "#d62728"),
                           (levy_psd, "vs truncated-Levy (informed)", "#9467bd")):
        f, t, a = roc_curve(humanness(hpsd), humanness(neg))
        ax.plot(f, t, color=col, label=f"{name}  AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("false positive rate (bots accepted)")
    ax.set_ylabel("true positive rate (humans accepted)")
    ax.set_title("ROC: alpha as a human/bot score (Section 7.1 method)")
    ax.legend(fontsize=9); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{OUT}/alpha_roc.png", dpi=130); plt.close(fig)
    print(f"Wrote plots: {OUT}/alpha_by_class.png , {OUT}/alpha_roc.png")

    # ---- Serhii worked example: synthetic NL trajectory, 64 res-10 cells ----
    ams = (52.3702, 4.8952)
    lat, lon = ams
    t = 1.767e9  # arbitrary unix-ish seconds
    cells, times = [], []
    prev = None
    steps = _truncated_local(scale, 200)
    i = 0
    while len(cells) < 64 and i < len(steps):
        r, b = steps[i]; i += 1
        lat, lon = (lat + (r/111.32)*math.cos(b),
                    lon + (r/(111.32*math.cos(math.radians(lat))))*math.sin(b))
        c = h3.latlng_to_cell(lat, lon, H3_RES)
        t += STEP_S
        if c == prev:
            continue
        cells.append(c); times.append(int(t)); prev = c
    example = {
        "_comment": "SYNTHETIC, illustrative only. Not real movement. Generated for "
                    "the geographic-results basis=evidence worked example.",
        "parameters": {"h3_resolution": H3_RES, "collection_interval_s": STEP_S,
                       "window_cells": len(cells)},
        "jurisdiction": "NL",
        "jurisdiction_note": "all cells placed within Amsterdam, Netherlands",
        "basis": "evidence",
        "basis_note": "the Verifier appraised these device-signed H3 cells directly",
        "claim_uuid": "b6f1e0c2-0000-4a1a-9f00-000000000001",
        "h3_cells": cells,
        "collection_times_unix_s": times,
    }
    with open(f"{OUT}/serhii_synthetic_example.json", "w") as fh:
        json.dump(example, fh, indent=2)
    print(f"Wrote Serhii worked example: {OUT}/serhii_synthetic_example.json "
          f"({len(cells)} cells, jurisdiction NL, basis=evidence)")


def _truncated_local(scale, n):
    # short local steps so 64 cells stay within one city (one jurisdiction)
    out = []
    for _ in range(n):
        r = min(abs(rng.normal(scale * 0.5, scale * 0.3)) + 0.05, 0.6)
        out.append((r, rng.uniform(0, 2 * math.pi)))
    return out


if __name__ == "__main__":
    main()
