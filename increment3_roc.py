#!/usr/bin/env python3
"""
increment3_roc.py  —  Can alpha separate real humans from bots at all?

Increment 2 showed only ~33% of real humans land in the draft's [0.30, 0.80]
band. That is one half of the picture. The half that decides whether alpha is
salvageable is separation: do SYNTHETIC trajectories land somewhere different?

  - If bots land where humans land, alpha has no discriminative power and the
    [0.30, 0.80] gate is not fixable by re-tuning -- it has to be demoted.
  - If bots cluster somewhere humans don't, a boundary exists; it just isn't
    [0.30, 0.80], and -05 can recalibrate.

We generate five adversary classes as raw lat/lon tracks at the 15-min collection
cadence, push them through the SAME build_breadcrumbs pipeline (H3 quantize +
dedup) and the SAME two alpha estimators as the humans, then measure separability
with ROC AUC. Human-ness score = -(|alpha - 0.55|), i.e. the draft's own notion
that 0.55 is the centre of "human" (Section 7.2 alpha_score).

The decisive class is truncated_levy: an adversary sampling steps from the exact
truncated-Levy law the draft itself uses to MODEL humans (Section 8.1). If alpha
cannot separate real humans from that, the SOC thesis does not survive contact
with an informed adversary.
"""

import math
import numpy as np

from trip_pipeline import Trajectory, RawPoint, build_breadcrumbs, PSD_FLOOR
from increment2_alpha import psd_alpha, dfa_exponent, load_displacements, WINDOW, BAND

rng = np.random.default_rng(20260909)

ORIGIN = (39.9, 116.4)          # Beijing-ish, matching GeoLife's region
STEP_S = 15 * 60                # 15-min cadence
N_TRACKS = 128                  # match the human cohort size
N_STEPS = 420                   # enough to survive dedup to a >=256 window


def _move(lat, lon, r_km, bearing):
    dlat = (r_km / 111.32) * math.cos(bearing)
    dlon = (r_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(bearing)
    return lat + dlat, lon + dlon


def _track_from_steps(step_km_seq, bearings):
    lat, lon = ORIGIN
    t0 = 3.9e9 / 86400 * 86400   # arbitrary epoch-seconds base
    pts = [RawPoint(t0, lat, lon)]
    t = t0
    for r, b in zip(step_km_seq, bearings):
        lat, lon = _move(lat, lon, r, b)
        t += STEP_S
        pts.append(RawPoint(t, lat, lon))
    return Trajectory(user="syn", points=pts)


# ---------- adversary generators (all calibrated to human step scale) ----------
def gen_random_walk(scale_km):
    steps = np.full(N_STEPS, scale_km)
    bearings = rng.uniform(0, 2 * math.pi, N_STEPS)
    return _track_from_steps(steps, bearings)


def gen_gaussian_walk(scale_km):
    steps = np.abs(rng.normal(scale_km, scale_km * 0.6, N_STEPS)) + 0.02
    bearings = rng.uniform(0, 2 * math.pi, N_STEPS)
    return _track_from_steps(steps, bearings)


def gen_correlated_walk(scale_km):
    # bearing persistence: a smoother, more "human-looking" wander
    steps = np.abs(rng.normal(scale_km, scale_km * 0.5, N_STEPS)) + 0.02
    bearings = np.cumsum(rng.normal(0, 0.5, N_STEPS))
    return _track_from_steps(steps, bearings)


def _truncated_levy_steps(n, beta, kappa_km, r_min):
    # P(r) ~ r^-beta * exp(-r/kappa), sampled by rejection from a power law
    out = []
    while len(out) < n:
        u = rng.uniform()
        r = r_min * (1 - u) ** (-1.0 / (beta - 1.0))     # power-law inverse CDF
        if r > 50 * kappa_km:
            continue
        if rng.uniform() < math.exp(-r / kappa_km):       # truncation
            out.append(r)
    return np.array(out)


def gen_truncated_levy(scale_km):
    # the informed adversary: uses the draft's OWN Section 8.1 mobility law
    steps = _truncated_levy_steps(N_STEPS, beta=1.75, kappa_km=max(scale_km * 6, 1.0),
                                  r_min=max(scale_km * 0.3, 0.1))
    bearings = rng.uniform(0, 2 * math.pi, N_STEPS)
    return _track_from_steps(steps, bearings)


def gen_scripted_commute(scale_km):
    # deterministic home<->work<->errand cycle: mimics human routine
    legs = [scale_km * 4, scale_km * 4, scale_km * 1.5, scale_km * 1.5]
    dirs = [0.0, math.pi, math.pi / 2, 3 * math.pi / 2]
    steps, bearings = [], []
    for i in range(N_STEPS):
        steps.append(legs[i % len(legs)])
        bearings.append(dirs[i % len(dirs)])
    return _track_from_steps(np.array(steps), np.array(bearings))


GENERATORS = {
    "random_walk":     gen_random_walk,
    "gaussian_walk":   gen_gaussian_walk,
    "correlated_walk": gen_correlated_walk,
    "truncated_levy":  gen_truncated_levy,     # the decisive adversary
    "scripted_commute": gen_scripted_commute,
}


def alpha_set(disp_list, method):
    A = []
    for d in disp_list:
        d = d[-WINDOW:]
        if len(d) < PSD_FLOOR:
            continue
        if method == "psd":
            a, _ = psd_alpha(d)
        else:
            a = 2 * dfa_exponent(d) - 1
        A.append(a)
    return np.array(A)


def humanness(alpha_arr):
    return -np.abs(alpha_arr - 0.55)     # draft's own "closeness to human centre"


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    wins = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return wins / (len(pos) * len(neg))


def in_band(alpha_arr):
    return float(np.mean((alpha_arr >= BAND[0]) & (alpha_arr <= BAND[1])))


def main():
    human_disp = load_displacements()
    human_d = [human_disp[u] for u in sorted(human_disp)]
    # calibrate synthetic step scale to the human median displacement
    scale = float(np.median([np.median(d) for d in human_d]))
    print(f"Human cohort: {len(human_d)}   calibrated synthetic step scale: "
          f"{scale:.3f} km (human median displacement)\n")

    # human alpha, both methods
    hA = {"psd": alpha_set(human_d, "psd"), "dfa": alpha_set(human_d, "dfa")}

    # synthetic alpha, both methods
    synth = {}
    for name, gen in GENERATORS.items():
        dlist = []
        for _ in range(N_TRACKS):
            bc = build_breadcrumbs(gen(scale))
            dlist.append(bc.displacements_km())
        synth[name] = {"psd": alpha_set(dlist, "psd"),
                       "dfa": alpha_set(dlist, "dfa")}

    for method, label in (("psd", "Section 7.1 raw-DFT method"),
                          ("dfa", "DFA cross-check (beta=2H-1)")):
        print("=" * 70)
        print(f"METHOD: {label}")
        print("=" * 70)
        print(f"  {'class':<18} {'n':>4} {'median a':>9} {'in-band%':>9} "
              f"{'AUC vs human':>13}")
        print(f"  {'HUMAN (real)':<18} {len(hA[method]):>4} "
              f"{np.median(hA[method]):>9.3f} {100*in_band(hA[method]):>8.0f}% "
              f"{'--':>13}")
        print("  " + "-" * 58)
        hscore = humanness(hA[method])
        for name in GENERATORS:
            a = synth[name][method]
            a = a[np.isfinite(a)]
            sscore = humanness(a)
            au = auc(hscore, sscore)          # P(human scored more human than bot)
            print(f"  {name:<18} {len(a):>4} {np.median(a):>9.3f} "
                  f"{100*in_band(a):>8.0f}% {au:>13.3f}")
        print()

    print("=" * 70)
    print("READING")
    print("=" * 70)
    print("  AUC ~ 0.5  => alpha cannot tell that class of bot from a real human.")
    print("  AUC ~ 1.0  => alpha separates cleanly (a boundary exists, even if it")
    print("               is not [0.30, 0.80]).")
    print("  The truncated_levy row is the one that matters: it is an adversary")
    print("  using the draft's own Section 8.1 model. Its AUC is the honest ceiling")
    print("  on what alpha-based classification can achieve against someone who has")
    print("  read the spec. Compare in-band% for HUMAN vs each bot: if a bot's")
    print("  in-band% is >= the human's, the [0.30,0.80] gate lets that bot through")
    print("  while rejecting real people.")


if __name__ == "__main__":
    main()
