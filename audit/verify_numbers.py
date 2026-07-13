#!/usr/bin/env python3
"""Re-derive every headline number from canonical AGGREGATE artifacts and assert it
matches the manuscript-reported value. Runs fully offline; no member-level data required.
Exit non-zero on any mismatch. Usage: python3 audit/verify_numbers.py"""
import json, sys, os
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES=os.path.join(BASE,'results')
agg=json.load(open(f'{RES}/aggregate_summary.json'))
full=json.load(open(f'{RES}/results_full.json'))
rob={r['tag']:r for r in json.load(open(f'{RES}/results_robust.json'))}
evc=json.load(open(f'{RES}/results_event_capture.json'))
fv2=json.load(open(f'{RES}/results_full_v2.json'))
derived={
 'N_cohort':agg['N_cohort'],
 'rate_claims':round(agg['rate_claims'],3),
 'rate_union':round(agg['rate_union'],3),
 'capture_adt':round(agg['capture_adt'],3),
 'capture_blended':round(agg['capture_blended'],3),
 'anchor_obs':round(full['naive']['obs'],3),
 'naive_mean':round(full['naive']['mean_pred'],3),
 'corrected_mean':round(full['corrected']['mean_pred'],3),
 'brier_naive':round(full['naive']['brier'],3),
 'brier_corrected':round(full['corrected']['brier'],3),
 'auroc_naive':round(full['naive']['auroc'][0],3),
 'auroc_corrected':round(full['corrected']['auroc'][0],3),
 'rob_exclobs_capture':round(rob['exclude_observation']['capture_overall'][0],3),
 'rob_altdate_capture':round(rob['altdate_2025-04-01']['capture_overall'][0],3),
 'event_capture_auc':round(evc['auc_full_gbm'],3),
 'event_capture_facility_only':round(evc['auc_facility_only'],3),
 'member_capture_auc_new':round(fv2['capture_auc_new'],3),
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
