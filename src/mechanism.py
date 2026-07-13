import sys, pathlib, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
# Internal data connector (wm_conn) loaded from secure environment; not included here.
from wm_conn import coredb, query
from statsmodels.stats.proportion import proportion_confint
cd=coredb("prod"); BASE='/Users/sanjaybasu/notebooks/pu-underascertainment'
W0,W1='2024-01-01','2025-12-31'
# ADT acute events
adt=query(cd,f"""SELECT DISTINCT person_id, encounter_start_date d FROM dbt_tuva_core.encounter
 WHERE data_source='HL7 ADT Feed' AND encounter_start_date BETWEEN '{W0}' AND '{W1}'
   AND encounter_type IN ('emergency department','acute inpatient','observation')""")
pids=list(adt.person_id.unique())
# claims ACUTE person-dates
clA=query(cd,f"""SELECT DISTINCT person_id, encounter_start_date d FROM dbt_tuva_core.encounter
 WHERE data_source='Claims' AND (ed_flag=1 OR encounter_group='inpatient')
   AND encounter_start_date BETWEEN '2023-12-01' AND '2026-02-01' AND person_id=ANY(%(p)s)""",p=pids)
# claims ANY-type person-dates
clAny=query(cd,f"""SELECT DISTINCT person_id, encounter_start_date d FROM dbt_tuva_core.encounter
 WHERE data_source='Claims' AND encounter_start_date BETWEEN '2023-12-01' AND '2026-02-01' AND person_id=ANY(%(p)s)""",p=pids)
adt['d']=pd.to_datetime(adt.d); clA['d']=pd.to_datetime(clA.d); clAny['d']=pd.to_datetime(clAny.d)
gA=clA.groupby('person_id')['d'].apply(lambda s:np.sort(s.values)).to_dict()
gAny=clAny.groupby('person_id')['d'].apply(lambda s:np.sort(s.values)).to_dict()
def near(g,pid,d,win):
    a=g.get(pid)
    if a is None or len(a)==0: return False
    return bool(np.min(np.abs((a-np.datetime64(d)).astype('timedelta64[D]').astype(int)))<=win)
adt['acute30']=[near(gA,p,d,30) for p,d in zip(adt.person_id,adt.d)]
miss=adt[~adt.acute30].copy()  # ADT acute w/ no acute-claim within 30d
miss['any7']=[near(gAny,p,d,7) for p,d in zip(miss.person_id,miss.d)]
n=len(miss); k=int(miss.any7.sum()); lo,hi=proportion_confint(k,n,method='wilson')
print(f"ADT acute events with NO acute-coded claim within 30d: {n:,}")
print(f"  of these, ANY claim within +/-7d: {k:,} = {100*k/n:.1f}% [{lo*100:.1f},{hi*100:.1f}]")
print(f"  -> entirely absent from claims feed (no claim +/-7d): {100*(1-k/n):.1f}%")
# lag test: capture at 30d vs 90d vs 180d windows
for win in (30,90,180):
    cap=np.mean([near(gA,p,d,win) for p,d in zip(adt.person_id,adt.d)])
    print(f"  acute-claim capture within +/-{win}d: {cap*100:.1f}%")
json.dump({'n_missing':n,'any7_pct':round(100*k/n,1),'any7_ci':[round(lo*100,1),round(hi*100,1)],
           'entirely_absent_pct':round(100*(1-k/n),1)}, open(f'{BASE}/results_mechanism.json','w'),indent=2)
print("saved results_mechanism.json")
