# CANONICAL NUMBERS (source of truth for manuscript/appendix/repo)

- Primary cohort N = 111,660 (index 2025-07-01, 90-day follow-up)
- Plans = 7 Medicaid MCOs across OH, VA, WA
- Member 90-day acute rate: claims 1.77%, ADT 6.22%, union 7.28%
- PRIMARY capture (ADT-covered members, identifiable stratum) = 14.8% [13.5, 16.1] (n=2,770)
- Blended all-plan capture (diluted by non-ADT plans; NOT identifiable) = 24.3% [23.4, 25.2] (n=8,227)
- Non-ADT stratum capture (mechanical, union≈claims) = 29.3% (n=5,356)
  - ABHVA: 12.6% [11.4, 13.9] (n=2895)
  - UHCWA: 12.8% [10.7, 15.1] (n=894)
  - UHCOH: 13.8% [11.4, 16.8] (n=621)
  - CHPW: 15.1% [12.8, 17.7] (n=841)
  - PROVWA: 24.3% [20.8, 28.1] (n=523)
  - SHPVA: 24.5% [22.5, 26.6] (n=1682)
  - CSOH: 99.7% [99.1, 99.9] (n=771)
- Capture by claims lag (days): ≤30=29.0%, 31-90=17.5%, 91-180=15.2%, >180=13.9%
- Ambulatory-claims among ADT-only-acute members = 93.9% (39,642/42,232) [source: characterization query crux4]
- Capture range across the six ADT-covered plans = 12.6% to 24.5% (four plans 12.6-15.1%; PROVWA 24.3%, SHPVA 24.5%)
- UNIFIED primary correction (held-out ADT anchor, n=3,099, truth 27.9%; all 95% CI by 1000-boot):
  - mean predicted: naive 3.5% [3.2,3.7] -> corrected 20.2% [19.4,21.1]
  - AUROC: naive 0.622 [0.601,0.645], corrected 0.608 [0.586,0.630]
  - AUPRC: naive 0.400 [0.370,0.433], corrected 0.381 [0.353,0.413]
  - @top-10%: naive sens 0.169 [0.149,0.190] spec 0.927 [0.919,0.934] PPV 0.471 [0.410,0.529] NPV 0.743 [0.725,0.758] F1 0.249 [0.220,0.279]
  - @top-10%: corrected sens 0.168 [0.151,0.188] spec 0.926 [0.920,0.934] PPV 0.468 [0.416,0.529] NPV 0.742 [0.725,0.758] F1 0.247 [0.222,0.277]
  - Brier: naive 0.254 [0.241,0.270], corrected 0.227 [0.215,0.239]
  - Calibration slope: naive 0.357 [0.296,0.429], corrected 0.099 [0.079,0.122]; intercept naive 0.49 [0.24,0.76], corrected -0.839 [-0.921,-0.756]
- Capture-model AUROC 95% CI: event 0.788 [0.780,0.797]; member old 0.569 [0.517,0.620]; member new 0.640
- MECHANISM: of ADT acute events lacking an acute-coded claim within 30d, 81.8% [81.5,82.1] have some claim within +/-7d (18.2% entirely absent); acute-claim capture rises 14.8%(30d) -> 20.0%(90d) -> 24.1%(180d)
- Recovery ratio naive->truth ~ eightfold (27.9/3.5); population claims vs union ~ fourfold (7.3/1.8)
- Robustness [primary_union]: capture 24.3% [23.4,25.3]; anchor truth 27.9%, naive 3.5% -> corrected 20.2%; AUROC 0.622->0.608; Brier 0.2544->0.2269
- Robustness [exclude_observation]: capture 24.5% [23.6,25.5]; anchor truth 27.4%, naive 3.2% -> corrected 19.2%; AUROC 0.628->0.619; Brier 0.2516->0.2221
- Robustness [altdate_2025-04-01]: capture 21.6% [20.6,22.7]; anchor truth 25.1%, naive 3.5% -> corrected 20.2%; AUROC 0.633->0.625; Brier 0.227->0.2061
- e(x) floor sensitivity (corrected mean vs truth 28.2%): 0.02=20.5%, 0.05=20.4%, 0.1=20.0%, 0.15=18.6%, 0.2=16.0%
- Ablation SCAR(c=0.147) vs SAR: corrected mean 0.192 vs 0.202 (obs 0.279); high-lag stratum (>90d) obs 0.150, naive 0.014, SCAR 0.079, SAR 0.140 (n=420); low-lag obs 0.299 (n=2,679)
- Capture-model AUROC 0.571 (drop claims-lag→0.525; drop plan→0.612; drop spans→0.573; drop ADT→0.568)
- STRENGTHENED capture model (event level, n=98,539 ADT acute events, 954 facilities, overall event capture 15.3%): AUROC 0.79 (GBM 0.788, logistic 0.778); facility-reporting-rate only 0.654; plan(mean-encoded)-only 0.732
- Member-level prospective capture AUROC: old (plan one-hot+lag) 0.569 -> new (+baseline facility-mix + plan mean-encode) 0.640; members with baseline facility-mix = 10,208/111,660
- Facility-informed correction (v2) recovery unchanged: naive mean 3.5% -> corrected 19.8% (obs 27.9%), Brier 0.254->0.222
- RUNOUT sensitivity (events with >=12mo runout, n=64,868): acute-coded capture 10.5%(7d) 14.4%(30d) 19.4%(90d) 24.0%(180d) 29.6%(365d) 31.9%(540d) -> plateau ~32%, ~2/3 permanent gap (not lag)
- Event->first acute-coded claim lag (captured events): median 13d, IQR 0-106d, p90 248d

## Targeting improvement (label + representation), eval vs true (union) events on held-out ADT anchor (n=3,099; 864 true pos; top-10% outreach)
- M0 status quo (claims label + tabular): AUROC 0.622, recall@10% 0.169 (146/864), PPV 0.471, F1 0.249
- +ADT-completed label (tabular): AUROC 0.655, recall@10% 0.191, PPV 0.532, F1 0.281
- +event-sequence model (GRU, 3-seed avg, ADT-completed label): AUROC 0.681, recall@10% 0.225 (194/864), PPV 0.626, F1 0.330
- delta sequence vs tabular(ADT-label): dAUROC +0.026 [0.011,0.043]; drecall +0.034 [0.009,0.053] (both significant)
- FULL STACK vs status quo: AUROC 0.622->0.681; recall@10% 16.9%->22.5% = 146->194 true events per 10% outreach (+33% relative)
- Enriched tabular features vs ADT-label baseline: dAUROC +0.004 [-0.006,0.014] NS; drecall +0.013 NS (tabular enrichment does NOT significantly help)
- LABEL-SIDE NULLS (no targeting gain): nnPU AUROC 0.616-0.621; outcome-imputation 0.610; PU rescale 0.608 (all <= status quo 0.622)
- Latent-class (Dawid-Skene, ADT-covered): true acute prevalence 30.6%, claims sensitivity 13.0%, ADT sensitivity 87.3%

## Targeting ladder — definitive CI run (results_ladder_ci.json; held-out ADT anchor n=3,099; 864 true events; top-10%)
- Status quo (claims label, tabular): AUROC vs TRUE 0.620 [0.599,0.644]; AUROC vs CLAIMS outcome 0.758; recall 0.163 [0.147,0.186]; PPV 0.455; F1 0.240
- ADT-completed label, tabular: AUROC vs TRUE 0.655 [0.634,0.677]; AUROC vs CLAIMS 0.716; recall 0.191 [0.174,0.213]; PPV 0.532; F1 0.281
- ADT-completed label, event-sequence (3-seed): AUROC vs TRUE 0.681 [0.660,0.701]; AUROC vs CLAIMS 0.691; recall 0.225 [0.206,0.243]; PPV 0.626; F1 0.330
- HEADLINE: recall 16.3%->22.5% at fixed 10% outreach (141->194 of 864 true events, +38% relative); AUROC vs true 0.620->0.681
- CLAIMS-OPTIMISM: status-quo model AUROC 0.758 vs claims outcome but 0.620 vs true (gap 0.138); claims metric MISRANKS (status-quo 0.758 > sequence 0.691 on claims, but sequence 0.681 > status-quo 0.620 on true events)

## Targeting ladder — single consistent split (ladder_ci.py + ladder_labelside.py), status_quo reproduces 0.620/0.163 exactly
- status_quo (claims,tabular): AUROC-true 0.620, recall@10 0.163, PPV 0.455, F1 0.240; AUROC-claims 0.758
- constant Elkan-Noto rescale (claims): 0.620 / 0.163 / 0.455 / 0.240 (RANK-PRESERVING → identical to status quo)
- covariate PU reweight (claims): 0.614 / 0.176 / 0.490 / 0.259
- nnPU (claims): 0.633 / 0.170 / 0.474 / 0.250
- outcome imputation (claims): 0.614 / 0.176 / 0.490 / 0.259
- adt-completed tabular: 0.655 / 0.191 / 0.532 / 0.281
- adt-completed + enriched: 0.658 / 0.204 / 0.568 / 0.300
- event-sequence (ADT-completed): 0.681 / 0.225 / 0.626 / 0.330; AUROC-claims 0.691
- Label-side range recall 0.170–0.176 within status-quo 95% CI [0.147,0.186]; none reach 0.191/0.225

## Duration / churn / missing sensitivity (sensitivity_duration_churn.py, coredb live 2026-07-13; results_sensitivity.json)
- Capture by follow-up window: 30d 14.3% | 60d 15.0% | 90d 14.8% (reproduces headline) | 180d 16.1% -> window-invariant
- Targeting recall@10 (tabular sq -> ADT-completed): 30d 0.218->0.197 (gain absent, few events) | 60d 0.179->0.206 | 90d 0.163->0.191 | 180d 0.158->0.165 (attenuates as claims run out)
- Churn: 4.7% disenroll by 90d; capture 14.8% all vs 15.1% continuously-enrolled (n=106,472) -> not churn-driven; 2.8% of ADT acute events post-disenrollment (HIE not claims-gated)
- Missing baseline: claims_lag null 12.6% (no prior claim); facility/provider 17.3%; zero-filled + indicators
