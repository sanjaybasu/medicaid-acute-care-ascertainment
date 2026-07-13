#!/usr/bin/env python3
"""Re-derive every headline number from canonical AGGREGATE artifacts and assert it
matches the manuscript-reported value. Runs fully offline; no member-level data required.
Exit non-zero on any mismatch. Usage: python3 audit/verify_numbers.py"""
import json, sys, os
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
agg=json.load(open(f'{BASE}/aggregate_summary.json'))
full=json.load(open(f'{BASE}/results_full.json'))
rob={r['tag']:r for r in json.load(open(f'{BASE}/results_robust.json'))}
evc=json.load(open(f'{BASE}/results_event_capture.json'))
fv2=json.load(open(f'{BASE}/results_full_v2.json'))
run=json.load(open(f'{BASE}/results_runout.json'))
rep=json.load(open(f'{BASE}/results_representation.json'))
pv=json.load(open(f'{BASE}/results_predict_value.json'))
pua=json.load(open(f'{BASE}/results_pu_advanced.json'))
lad=json.load(open(f'{BASE}/results_ladder_ci.json'))
ls=json.load(open(f'{BASE}/results_labelside.json'))
derived={
 'adt_tabular_recall10':round(lad['adt_label_tabular']['recall10'],3),
 'ls_constant_recall10':round(ls['constant_rescale_claims']['recall10'],3),
 'ls_pureweight_recall10':round(ls['pu_reweight_claims']['recall10'],3),
 'ls_enriched_recall10':round(ls['adt_label_enriched']['recall10'],3),
 'N_cohort':agg['N_cohort'],
 'rate_claims':round(agg['rate_claims'],3),
 'rate_union':round(agg['rate_union'],3),
 'capture_adt':round(agg['capture_adt'],3),
 'capture_blended':round(agg['capture_blended'],3),
 'anchor_obs':round(full['obs'],3),
 'naive_mean':round(full['naive']['mean_pred'][0],3),
 'corrected_mean':round(full['corrected']['mean_pred'][0],3),
 'brier_naive':round(full['naive']['brier'][0],3),
 'brier_corrected':round(full['corrected']['brier'][0],3),
 'auroc_naive':round(full['naive']['auroc'][0],3),
 'auroc_corrected':round(full['corrected']['auroc'][0],3),
 'rob_exclobs_capture':round(rob['exclude_observation']['capture_overall'][0],3),
 'rob_altdate_capture':round(rob['altdate_2025-04-01']['capture_overall'][0],3),
 'event_capture_auc':round(evc['auc_full_gbm'],3),
 'event_capture_facility_only':round(evc['auc_facility_only'],3),
 'member_capture_auc_new':round(fv2['capture_auc_new'],3),
 'runout_540':round(run['curve']['540'],3),
 'fwd_lag_median':round(run['fwd_lag_median'],0),
 'seq_auroc':round(lad['adt_label_sequence']['auroc_vs_TRUE'],3),
 'seq_recall10':round(lad['adt_label_sequence']['recall10'],3),
 'statusquo_recall10':round(lad['status_quo_claims_tabular']['recall10'],3),
 'statusquo_auc_true':round(lad['status_quo_claims_tabular']['auroc_vs_TRUE'],3),
 'statusquo_auc_claims':round(lad['status_quo_claims_tabular']['auroc_vs_CLAIMS'],3),
 'latent_true_prev':round(pua['latent_class_ADTcovered']['true_prevalence'],3),
}
import csv
reg=list(csv.DictReader(open(f'{BASE}/audit/data_provenance.csv')))
fail=0
for r in reg:
    cid=r['claim_id']; rep=float(r['reported_value']); got=derived[cid]
    tol=0.5 if cid=='N_cohort' else 0.0015
    ok=abs(got-rep)<=tol
    print(f"{'OK ' if ok else 'FAIL'} {cid}: reported={rep} derived={got}")
    if not ok: fail+=1
print(f"\ncoverage: {len(reg)} checks / {len(reg)} registry rows")
if fail: print(f"{fail} MISMATCH(ES)"); sys.exit(1)
print("ALL CHECKS PASS")
