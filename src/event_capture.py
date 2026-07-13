import sys, pathlib, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
# Internal data connector (wm_conn) is loaded from the secure environment; not included here.
from wm_conn import coredb, query
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
cd = coredb("prod"); BASE='/Users/sanjaybasu/notebooks/pu-underascertainment'
W0,W1='2024-01-01','2025-12-31'
# ADT acute events with facility + type + admit source
adt = query(cd, f"""
 SELECT person_id, encounter_start_date d, facility_id, encounter_type, admit_source_description asrc
 FROM dbt_tuva_core.encounter
 WHERE data_source='HL7 ADT Feed' AND encounter_start_date BETWEEN '{W0}' AND '{W1}'
   AND encounter_type IN ('emergency department','acute inpatient','observation')""")
adt=adt.drop_duplicates(['person_id','d','facility_id'])
pids=list(adt.person_id.unique())
clm = query(cd, f"""SELECT DISTINCT person_id, encounter_start_date d FROM dbt_tuva_core.encounter
  WHERE data_source='Claims' AND (ed_flag=1 OR encounter_group='inpatient')
    AND encounter_start_date BETWEEN '2023-12-01' AND '2026-02-01' AND person_id = ANY(%(p)s)""", p=pids)
pay = query(cd, "SELECT DISTINCT ON(person_id) person_id, payer FROM dbt_tuva_core.eligibility WHERE person_id=ANY(%(p)s) ORDER BY person_id, enrollment_end_date DESC", p=pids)
adt['d']=pd.to_datetime(adt.d); clm['d']=pd.to_datetime(clm.d)
clmg=clm.groupby('person_id')['d'].apply(lambda s: np.sort(s.values)).to_dict()
def captured(pid,d,win=30):
    arr=clmg.get(pid)
    if arr is None or len(arr)==0: return 0
    return int(np.min(np.abs((arr-np.datetime64(d)).astype('timedelta64[D]').astype(int)))<=win)
adt['captured']=[captured(p,d) for p,d in zip(adt.person_id,adt.d)]
adt=adt.merge(pay,on='person_id',how='left')
adt['month']=adt.d.dt.month + 12*(adt.d.dt.year-2024)
adt['is_ed']=(adt.encounter_type=='emergency department').astype(int)
adt['is_inpat']=(adt.encounter_type=='acute inpatient').astype(int)
adt['is_obs']=(adt.encounter_type=='observation').astype(int)
print(f"event-level ADT acute: n={len(adt):,}, overall captured={adt.captured.mean():.3f}, facilities={adt.facility_id.nunique()}")

# member-disjoint split; reporting rates from TRAIN only (no leakage)
gss=GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
tri,tei=next(gss.split(adt,groups=adt.person_id))
tr,te=adt.iloc[tri].copy(),adt.iloc[tei].copy()
m=tr.captured.mean(); A=20.0
def te_encode(col):
    g=tr.groupby(col).captured.agg(['sum','count'])
    rate=((g['sum']+A*m)/(g['count']+A)).to_dict()
    return rate
frate=te_encode('facility_id'); prate=te_encode('payer'); arate=te_encode('asrc')
for df_ in (tr,te):
    df_['fac_rate']=df_.facility_id.map(frate).fillna(m)
    df_['plan_rate']=df_.payer.map(prate).fillna(m)
    df_['asrc_rate']=df_.asrc.map(arate).fillna(m)
FEATS=['fac_rate','plan_rate','asrc_rate','is_ed','is_inpat','is_obs','month']
def auc(model,feats):
    model.fit(tr[feats],tr.captured); p=model.predict_proba(te[feats])[:,1]; return roc_auc_score(te.captured,p),p
res={}
# baseline: OLD member-level style (plan + month only, no facility)
a_old,_=auc(LogisticRegression(max_iter=1000),['plan_rate','month','is_ed','is_inpat','is_obs'])
# facility reporting rate alone
a_fac,_=auc(LogisticRegression(max_iter=1000),['fac_rate'])
# full logistic
a_lr,_=auc(LogisticRegression(max_iter=1000),FEATS)
# full gbm
a_gbm,pg=auc(LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1),FEATS)
res={'n_events':int(len(adt)),'overall_capture':round(float(adt.captured.mean()),3),
     'n_facilities':int(adt.facility_id.nunique()),
     'auc_old_planonly':round(float(a_old),3),'auc_facility_only':round(float(a_fac),3),
     'auc_full_logistic':round(float(a_lr),3),'auc_full_gbm':round(float(a_gbm),3)}
print(json.dumps(res,indent=2))
json.dump(res,open(f'{BASE}/results_event_capture.json','w'),indent=2)
adt['fac_rate']=adt.facility_id.map(frate).fillna(m)
adt['plan_rate']=adt.payer.map(prate).fillna(m)
adt['asrc_rate']=adt.asrc.map(arate).fillna(m)
adt[['person_id','captured','fac_rate','plan_rate','asrc_rate','is_ed','is_inpat','is_obs','month']].to_parquet(f'{BASE}/_event_capture.parquet')
json.dump({'global_mean':float(m),'pseudo':A,'facility_rate':{str(k):float(v) for k,v in frate.items()},
           'plan_rate':{str(k):float(v) for k,v in prate.items()}}, open(f'{BASE}/facility_rate_map.json','w'))
print("saved results_event_capture.json + _event_capture.parquet + facility_rate_map.json")
