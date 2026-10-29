# trip-validation

Empirical validation of the statistical core of
**draft-ayerbe-trip-protocol** against real human mobility data, feeding the
`-05` revision. These are the **public** validation artifacts for the
Section 7 ("Criticality Engine") statistical claims of
draft-ayerbe-trip-protocol. Code components are Contributions to the IETF
Standards Process and are provided under the Revised BSD License; see
`LICENSE`.

The lab runs the *actual* draft pipeline (H3 quantization, dedup, interval rules)
over the GeoLife dataset and a set of synthetic/adversarial trajectories, and
measures whether the Section 7.1 criticality claim holds. It is built to try to
*break* the claim, not to confirm it.

## Finding (short version)

- **The `[0.30, 0.80]` band in Section 7.1 does not hold as written.** Run
  literally on real human trajectories, only ~33% (raw-DFT method) / ~46% (DFA)
  of genuine humans land in the band — a false-negative rate that makes the MUST
  gate unusable as specified.
- **The power-law fit is weak.** Median R² of the log-log fit is ~0.04; the
  displacement spectra are not fittable power laws, so the extracted α is close to
  noise (reviewer concern, confirmed).
- **But α still carries real signal.** As a *score* (not a hard band), α separates
  humans from bots with AUC ≈ 0.85 (pooled adversaries) and ≈ 0.78 against an
  informed truncated-Lévy adversary using the draft's own Section 8.1 model.
- **The two-sided shape is right; the numbers are wrong.** A drift adversary lands
  at α ≈ 1.8, so an upper bound is genuinely needed. A recalibrated band of about
  `[0.16, 1.25]` roughly doubles Youden's J (human TPR 59–68%, bot FPR 5–8%).
- **The hard-to-forge signal is temporal correlation, not the step-size
  distribution.** An adversary copying the draft's Section 8.1 magnitude law still
  lands far from humans — a narrower, stronger, defensible version of the claim.

The `-05` implication: retire the fixed `[0.30, 0.80]` MUST gate, reframe α as a
calibrated *scored feature* (per the independent review), recalibrate the band
from the ROC, and state the temporal-correlation claim rather than the
SOC-as-physical-law claim. See `results/alpha_by_class.png` and
`results/alpha_roc.png`.

## Layout

```
trip_pipeline.py          loader + faithful -04 pipeline (3.1 quantize, 4.1 dedup,
                          4.2 interval, 7.1 displacement series)
increment1_sanity.py      real GeoLife through the pipeline; cohort/gap sanity, no stats
increment2_alpha.py       alpha two ways: Section 7.1 literal + DFA robust cross-check
increment3_roc.py         synthetic adversaries through the same pipeline; AUC
increment4_roc_recal.py   ROC curves, recalibrated band, drift adversary, plots
make_serhii_set.py        the 64-cell NL worked example (polygon-verified)
scripts/fetch_data.sh     downloads GeoLife + NL polygon (data is gitignored)
results/                  committed artifacts: the two plots + the worked example
```

Data (`data/`, `geo/`) and the `displacements.pkl` cache are **not** committed.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_data.sh            # GeoLife (~313 MB) + NL polygon
python increment1_sanity.py           # cohort sanity (builds the cache, ~1 min)
python increment2_alpha.py            # alpha two ways
python increment3_roc.py              # adversaries + AUC
python increment4_roc_recal.py        # ROC curves + recalibration -> results/*.png
python make_serhii_set.py             # -> results/trip_synthetic_example.json
```

## Honesty rails

- **Symbolic of nothing but this pipeline.** All statistics are computed on the
  breadcrumb sequence the draft actually produces (quantized + deduped + interval),
  not on raw GPS, so the numbers reflect the spec, not an idealized version.
- **Cohort is the ≥64-breadcrumb set** (129 of 182 GeoLife users clear the PSD
  floor; 80 reach the 200 the draft recommends). Report the denominator.
- **This is a bounded empirical study, not a theorem.** AUCs are on GeoLife +
  these specific adversaries; a different dataset or a cleverer adversary could
  move them. The synthetic adversaries are illustrative, not exhaustive.
- The `results/trip_synthetic_example.json` trajectory is **synthetic**, generated
  for the geographic-results worked example; it is not real movement.

## Credits

- GeoLife lab, pipeline, and ROC/recalibration (`trip_pipeline.py`,
  `increment1`–`increment4`): Camilo Ayerbe Posada.
- Structural results (white step-magnitude spectrum, the log-periodogram
  R² ceiling, the standard-error window bound) and the worked example
  (`make_serhii_set.py`): Serhii Nikolaichuk. These are reported from
  numerical simulation and are stated as such in the draft.
- `test-vectors/`: Field 15 (calibration-provenance) signing vectors.

## Data provenance

- GeoLife GPS Trajectories 1.3 — Microsoft Research.
- Netherlands boundary — `georgique/world-geojson`.

Both are fetched by `scripts/fetch_data.sh` and excluded from the repository.
