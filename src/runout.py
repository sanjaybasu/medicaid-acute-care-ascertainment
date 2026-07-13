import sys, pathlib, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
# Internal data connector (wm_conn) loaded from secure environment; not included here.
from wm_conn import coredb, query
cd=coredb("prod"); BASE='.'
# ADT acute events with >=12 mo potential runout (event before 2025-07; extraction ~2026-07)
adt=query(cd,"""SELECT DISTINCT person_id, encounter_start_date d FROM dbt_tuva_core.encounter
 WHERE data_source='HL7 ADT Feed' AND encounter_start_date BETWEEN '2024-01-01' AND '2025-06-30'
   AND encounter_type IN ('emergency department','acute inpatient','observation')""")
pids=list(adt.person_id.unique())
clA=query(cd,"""SELECT DISTINCT person_id, encounter_start_date d FROM dbt_tuva_core.encounter
 WHERE data_source='Claims' AND (ed_flag=1 OR encounter_group='inpatient')
   AND encounter_start_date BETWEEN '2023-06-01' AND '2026-07-13' AND person_id=ANY(%(p)s)""",p=pids)
adt['d']=pd.to_datetime(adt.d); clA['d']=pd.to_datetime(clA.d)
g=clA.groupby('person_id')['d'].apply(lambda s:np.sort(s.values)).to_dict()
def mindiff_signed(pid,d):
    a=g.get(pid)
    if a is None or len(a)==0: return None
    diff=(a-np.datetime64(d)).astype('timedelta64[D]').astype(int)
    j=np.argmin(np.abs(diff)); return int(diff[j])
adt['nearest']=[mindiff_signed(p,d) for p,d in zip(adt.person_id,adt.d)]
n=len(adt)
print(f"ADT acute events (>=12mo runout): {n:,}")
print("=== capture as a function of allowed runout window (|nearest acute claim| <= W days) ===")
for W in [7,14,30,60,90,120,180,270,365,540]:
    cap=np.mean([(x is not None and abs(x)<=W) for x in adt.nearest])
    print(f"  +/-{W:>3}d: {cap*100:5.1f}%")
ever=np.mean([x is not None for x in adt.nearest])  # any acute claim ever (within pull)
print(f"  EVER (any acute claim 2023-06..2026-07): {ever*100:.1f}%")
# event-to-claim lag among captured-forward (first acute claim AFTER event)
def fwd(pid,d):
    a=g.get(pid)
    if a is None: return None
    diff=(a-np.datetime64(d)).astype('timedelta64[D]').astype(int); diff=diff[diff>=0]
    return int(diff.min()) if len(diff) else None
lags=[fwd(p,d) for p,d in zip(adt.person_id,adt.d)]; lags=[x for x in lags if x is not None]
if lags:
    import numpy as np
    q=np.percentile(lags,[25,50,75,90])
    print(f"=== forward event->first acute claim lag (captured events, n={len(lags):,}) days: median {q[1]:.0f} [IQR {q[0]:.0f}-{q[2]:.0f}], p90 {q[3]:.0f}")
res={'n':n,'ever_capture':round(float(ever),3),
     'curve':{str(W):round(float(np.mean([(x is not None and abs(x)<=W) for x in adt.nearest])),3) for W in [30,90,180,365,540]},
     'fwd_lag_median':float(np.percentile(lags,50)),'fwd_lag_iqr':[float(np.percentile(lags,25)),float(np.percentile(lags,75))]}
json.dump(res,open(f'{BASE}/results_runout.json','w'),indent=2)
print("saved results_runout.json")
