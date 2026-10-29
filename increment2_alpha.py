#!/usr/bin/env python3
"""
increment2_alpha.py  —  Does real human movement actually land in [0.30, 0.80]?

Section 7.1 makes the load-bearing claim of the whole protocol: the PSD scaling
exponent alpha of a human displacement series falls in [0.30, 0.80] (1/f "pink
noise"), and a conforming Verifier MUST classify a trajectory as biological only
if alpha is in that band. Everything else (trust score cap, PoH acceptance) hangs
off it.

This increment tests that claim against real human trajectories (GeoLife), the
honest way: compute alpha TWO ways and compare.

  (A) The draft's method, LITERALLY as Section 7.1 specifies it:
        S(f) = |DFT(d)|^2 ,  fit  S(f) ~ 1/f^alpha  in log-log space.
      No detrending, no windowing, no taper -- because that is what the spec says,
      and the point is to test the spec, not an improved version of it.

  (B) Detrended Fluctuation Analysis (DFA), the standard tool for scaling
      exponents on NON-STATIONARY series. This is the reviewer's concern made
      concrete: human mobility has strong day/night periodicity and long gaps
      (Increment 1 counted thousands of >1 h gaps), which violate the stationarity
      that a raw DFT assumes. DFA detrends each window, so it is robust to exactly
      that. We convert the DFA exponent H to an equivalent PSD slope beta = 2H - 1
      (fractional Gaussian noise relation) so the two are on the same axis.

If (A) and (B) agree AND real humans cluster inside [0.30, 0.80], Section 7.1 is
vindicated. If they diverge, or if real humans scatter outside the band under the
draft's own method, that is the finding -05 has to act on -- and far better we
find it than a reviewer does.
"""

import os
import pickle
import numpy as np

from trip_pipeline import iter_users, build_breadcrumbs, PSD_FLOOR

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data",
                         "Geolife Trajectories 1.3", "Data")
CACHE = os.path.join(os.path.dirname(__file__), "displacements.pkl")
WINDOW = 256          # 7.1 "recommended"; we use most-recent min(256, len)
BAND = (0.30, 0.80)   # 7.1 biological band


# ---------- (A) the draft's Section 7.1 method, literal ----------
def psd_alpha(d):
    d = np.asarray(d, float)
    n = len(d)
    F = np.fft.rfft(d)
    power = np.abs(F) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    m = freqs > 0                        # drop DC
    lf = np.log(freqs[m])
    lp = np.log(power[m] + 1e-30)
    slope, c = np.polyfit(lf, lp, 1)
    alpha = -slope
    pred = slope * lf + c
    ss_res = float(np.sum((lp - pred) ** 2))
    ss_tot = float(np.sum((lp - lp.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, r2


# ---------- (B) DFA, robust to non-stationarity ----------
def dfa_exponent(x):
    x = np.asarray(x, float)
    N = len(x)
    y = np.cumsum(x - x.mean())          # integrated profile
    scales = np.unique(np.floor(np.logspace(np.log10(4),
                                            np.log10(max(8, N // 4)), 20)).astype(int))
    F = []
    used = []
    for n in scales:
        nb = N // n
        if nb < 1:
            continue
        rms = []
        t = np.arange(n)
        for i in range(nb):
            seg = y[i * n:(i + 1) * n]
            coef = np.polyfit(t, seg, 1)
            fit = np.polyval(coef, t)
            rms.append(np.mean((seg - fit) ** 2))
        f = np.sqrt(np.mean(rms))
        if f > 0:
            F.append(f)
            used.append(n)
    if len(used) < 3:
        return float("nan")
    H = np.polyfit(np.log(used), np.log(F), 1)[0]
    return H


def load_displacements():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    disp = {}
    for traj in iter_users(DATA_ROOT):
        bc = build_breadcrumbs(traj)
        d = bc.displacements_km()
        if len(d) >= PSD_FLOOR:
            disp[bc.user] = d
    with open(CACHE, "wb") as fh:
        pickle.dump(disp, fh)
    return disp


def hist(vals, lo, hi, nbins=12, width=40):
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    peak = max(counts) if len(counts) and max(counts) else 1
    lines = []
    for i in range(nbins):
        bar = "#" * int(round(width * counts[i] / peak))
        lines.append(f"  [{edges[i]:5.2f},{edges[i+1]:5.2f})  {counts[i]:4d} |{bar}")
    return "\n".join(lines)


def main():
    disp = load_displacements()
    users = sorted(disp)
    print(f"Cohort: {len(users)} users clearing the {PSD_FLOOR}-displacement floor.\n")

    A, R2, Bdfa = [], [], []
    for u in users:
        d = disp[u][-WINDOW:]            # most-recent window (7.1)
        a, r2 = psd_alpha(d)
        H = dfa_exponent(d)
        A.append(a); R2.append(r2); Bdfa.append(2 * H - 1)   # fGn: beta = 2H-1

    A = np.array(A); R2 = np.array(R2); Bdfa = np.array(Bdfa)
    in_band_A = (A >= BAND[0]) & (A <= BAND[1])
    in_band_B = (Bdfa >= BAND[0]) & (Bdfa <= BAND[1])

    print("=" * 66)
    print("(A) Draft Section 7.1 method (raw DFT power-law fit), literal")
    print("=" * 66)
    print(f"  alpha  median {np.median(A):.3f}   mean {A.mean():.3f}   "
          f"IQR [{np.percentile(A,25):.3f}, {np.percentile(A,75):.3f}]")
    print(f"  fit quality R^2  median {np.median(R2):.3f}   "
          f"(low R^2 = the power-law fit is poor; reviewer's concern)")
    print(f"\n  REAL humans landing inside the biological band [0.30, 0.80]:")
    print(f"    {in_band_A.sum()} / {len(A)}  = {100*in_band_A.mean():.0f}%")
    print(f"  => draft-method FALSE-NEGATIVE rate on real humans: "
          f"{100*(1-in_band_A.mean()):.0f}%")
    print(f"\n  alpha distribution (draft method):")
    print(hist(A, min(A.min(), 0.0), max(A.max(), 1.2)))

    print("\n" + "=" * 66)
    print("(B) DFA cross-check, converted to equivalent PSD slope beta = 2H-1")
    print("=" * 66)
    print(f"  beta_DFA  median {np.median(Bdfa):.3f}   mean {Bdfa.mean():.3f}   "
          f"IQR [{np.percentile(Bdfa,25):.3f}, {np.percentile(Bdfa,75):.3f}]")
    print(f"  real humans inside [0.30, 0.80] by DFA: "
          f"{in_band_B.sum()} / {len(Bdfa)} = {100*in_band_B.mean():.0f}%")
    print(f"\n  beta_DFA distribution:")
    print(hist(Bdfa, min(Bdfa.min(), 0.0), max(Bdfa.max(), 1.2)))

    print("\n" + "=" * 66)
    print("AGREEMENT between the two methods")
    print("=" * 66)
    diff = np.abs(A - Bdfa)
    corr = np.corrcoef(A, Bdfa)[0, 1]
    print(f"  Pearson corr(alpha_7.1, beta_DFA) : {corr:+.3f}")
    print(f"  mean |alpha_7.1 - beta_DFA|       : {diff.mean():.3f}")
    print(f"  (near 0 diff + high corr => 7.1 raw DFT is trustworthy despite")
    print(f"   non-stationarity; large diff => 7.1 alpha is contaminated.)")

    print("\n" + "=" * 66)
    print("READING")
    print("=" * 66)
    print(f"  These are all real human trajectories. Section 7.1 predicts ~100%")
    print(f"  land in [0.30, 0.80]. The draft method puts {100*in_band_A.mean():.0f}% in band.")
    print(f"  Whatever that number is, it is the honest false-negative rate the")
    print(f"  [0.30, 0.80] boundary carries on real data -- the first hard input")
    print(f"  to the -05 recalibration, and the ROC work in the next increment.")


if __name__ == "__main__":
    main()
