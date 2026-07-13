# Under-ascertainment of claims-based acute care utilization in Medicaid

Code and aggregate outputs for the study of claims-based outcome under-ascertainment and a
positive-unlabeled correction, using a real-time admission-discharge-transfer (ADT) feed as an
independent record of acute care events.

## What this repository contains

- `src/` — the analysis pipeline (four scripts). These read from an internal data connector and
  are provided for methods transparency; they are not runnable outside the secure data environment.
  - `build_dataset.py` — cohort, features, outcomes for a given index date.
  - `eval_full.py` — naive and corrected models; full metric suite with bootstrap confidence intervals.
  - `eval_robust.py` — observation-excluded and second-index-date analyses.
  - `make_figs_canon.py` — figures and canonical-number export.
  - `model.py` — exploratory modeling used during development.
- `results/` — aggregate outputs only (no member-level data): `aggregate_summary.json`,
  `results_full.json`, `results_robust.json`. Plan identifiers are anonymized (Plan A–G).
- `audit/` — reproducibility audit: `data_provenance.csv` (number-provenance registry) and
  `verify_numbers.py` (re-derives every headline number from the aggregate outputs).
- `figs/` — the four manuscript figures.
- `docs/` — the analysis specification and the canonical-numbers file.

## Reproducing the audit (no data required)

```
pip install -r requirements.txt
python3 audit/verify_numbers.py
```

This re-derives all fourteen registered headline numbers from `results/` and asserts they match
the reported values. It runs fully offline and requires no member-level data.

## Data

Member-level data are protected under the care organization's data-governance policy and are not
shared. The pipeline in `src/` documents exactly how the aggregate outputs were produced.

## License

MIT (code). See `LICENSE`.
