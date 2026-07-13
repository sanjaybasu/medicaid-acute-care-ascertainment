# Analysis spec: continuity-trajectory features + SAR-PU for acute-care-utilization prediction

Status: active. Data = the care organization data warehouse (`dbt_tuva_core`, `dbt`), prod, read-only. All work local; no PHI leaves infra; repo gets code + aggregates only.

## Objective
Test two additions to a Signal-like baseline for predicting acute-care utilization (ED + hospitalization) in Medicaid members, and the targeting implications for Compass/care teams:
1. Care-continuity/fragmentation trajectory features (does the *shape* of prior care add discrimination beyond volume?).
2. SAR-PU correction for outcome under-ascertainment (do incomplete/uneven claims+ADT+EHR feeds bias risk, and does correction fix ranking + measurement?).

## Data
- `dbt_tuva_core.eligibility`: enrollment spans, payer, payer_type, birth_date, death_date. 283k persons, 7 Medicaid payers across OH/VA/WA.
- `dbt_tuva_core.encounter`: 7.2M rows, 260k persons. `data_source` ∈ {Claims (7.05M), HL7 ADT Feed (173k)}. Claims acute = `ed_flag=1 OR encounter_group='inpatient'`. ADT acute = `encounter_group='clinical' AND encounter_type IN ('emergency department','acute inpatient','observation')`. Continuity fields: `attending_provider_id`, `facility_id`, `encounter_type/group`, `primary_diagnosis_code`.
- `dbt.pqi_claims_utilization`: UNUSABLE (2,176 rows). Compute ACSC from inpatient `primary_diagnosis_code` vs AHRQ PQI ICD-10 lists instead.
- `dbt.non_emergent_ed_claims_utilization`: preventable-ED (NYU) weights, 76k persons — tertiary outcome.
- eligibility↔encounter overlap = 236k persons → analysis cohort.

## Cohort & windows
- Index date t0 = 2025-07-01. Baseline = [t0−365d, t0). Follow-up = [t0, t0+90d] (primary) and +180d (secondary).
- Include members with ≥1 eligibility day in baseline and enrolled at t0; require ≥ N days baseline enrollment (≥180) to observe care pattern.
- Exclude dirty dates (> 2026-07-12 or < 2005). Dedup acute events to person-date.
- Claims maturity: follow-up ends 2025-09-30; extraction 2026-07 → claims settled.

## Outcome
- Primary Y_obs: any acute-care event (ED or hospitalization) in follow-up, from Claims provenance (the label a claims-only model sees).
- Y_full: acute event from Claims OR ADT provenance (adds ADT-only events) — used to quantify under-ascertainment and as the SAR-PU validation target on the anchor.
- Secondary: ACSC (AHRQ PQI ICD-10 on inpatient dx); preventable ED (NYU).

## Features
- Baseline (Signal-like): age; payer/state; prior-utilization counts over baseline (ED, inpatient, office visits, urgent care, outpatient, total); distinct primary-diagnosis count; total paid_amount; enrollment months.
- Continuity/fragmentation (from baseline encounters):
  - UPC (usual provider of care) = max visits to one provider / total ambulatory visits.
  - Bice–Boxerman COC = (Σ_j n_j² − N) / (N(N−1)) over providers j.
  - Days since last primary-care (office-visit) encounter at t0.
  - Peripheral-substitution share = (urgent care + treat-and-release ED + outpatient) / ambulatory visits.
  - Distinct facilities; distinct providers.
  - Drift/slope: Δ in monthly PC-visit rate and peripheral share, first half vs second half of baseline.
- Ascertainment covariates x_a (for e(x)): payer, state, payer-level ADT coverage, claims-lag proxy (gap between last claim date and t0), enrollment continuity (gaps, fraction of baseline covered).

## Models
- M0 baseline GBM (LightGBM/XGBoost) on baseline features.
- M1 = M0 + continuity features.
- M2 = SAR-PU wrapper: estimate e(x_a) (calibrated on ADT-complete anchor where e→1); reweight (observed positives ×1/e; unlabeled split); refit base learner; recover P(Y=1|x). Same learner as M1.
- Split: member-disjoint, temporal-honest; 70/30; 5-fold CV for tuning. Report on held-out.

## Tests / go–no-go
Addition 1 positive if, on held-out:
- ΔAUROC(M1−M0) > 0 and DeLong p < 0.05, AND meaningful AUPRC gain, AND top-decile capture (recall@10%) improves.
Addition 2 positive if:
- (a) under-ascertainment is real: ADT reveals acute events absent from claims (crux query), and e(x_a) varies materially by coverage covariates;
- (b) anchor recovery: on held-out ADT-complete anchor, corrected risk matches observed Y_full while naive under-predicts;
- (c) reclassification: SAR-PU moves low-coverage members up the ranking (top-decile composition shifts toward low-coverage strata).
Write-up only if ≥1 addition clears its bar. Report honest null otherwise.

## Known limitations
- e(x) identification rests on anchor near-completeness + coverage/clinical covariate separation → sensitivity analysis.
- Continuity features associational → modifiable claim tested vs intervention response, not asserted.
- ACSC via self-computed AHRQ lists, not the (unusable) PQI table.
- Single index date v1; multi-index robustness deferred.
