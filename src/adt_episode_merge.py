import sys, pathlib, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
cd=coredb("prod"); W0,W1='2024-01-01','2025-12-31'
adt=query(cd,f"""SELECT person_id, encounter_start_date d, facility_id, encounter_type, admit_source_description asrc
 FROM dbt_tuva_core.encounter WHERE data_source='HL7 ADT Feed' AND encounter_start_date BETWEEN '{W0}' AND '{W1}'
   AND encounter_type IN ('emergency department','acute inpatient','observation')""")
adt=adt.drop_duplicates(['person_id','d','facility_id'])
pids=list(adt.person_id.unique())
clm=query(cd,f"""SELECT DISTINCT person_id, encounter_start_date d FROM dbt_tuva_core.encounter
  WHERE data_source='Claims' AND (ed_flag=1 OR encounter_group='inpatient')
    AND encounter_start_date BETWEEN '2023-12-01' AND '2026-02-01' AND person_id = ANY(%(p)s)""", p=pids)
pay=query(cd,"SELECT DISTINCT ON(person_id) person_id, payer FROM dbt_tuva_core.eligibility WHERE person_id=ANY(%(p)s) ORDER BY person_id, enrollment_end_date DESC", p=pids)
adt['d']=pd.to_datetime(adt.d); clm['d']=pd.to_datetime(clm.d)
clmg=clm.groupby('person_id')['d'].apply(lambda s: np.sort(s.values)).to_dict()

def episode_merge(df, gap=1):
    d=df.sort_values(['person_id','d']).copy()
    d['prec']=d.encounter_type.map({'acute inpatient':3,'observation':2,'emergency department':1})
    d['gd']=d.groupby('person_id')['d'].diff().dt.days
    d['newep']=((d.gd.isna())|(d.gd>gap)).astype(int)
    d['epid']=d.groupby('person_id')['newep'].cumsum()
    d['key']=d.person_id.astype(str)+'_'+d.epid.astype(str)
    # anchor = highest-precedence row (IP>obs>ED), earliest on ties
    anchor=d.sort_values(['key','prec','d'],ascending=[True,False,True]).groupby('key',as_index=False).head(1)
    return anchor.drop(columns=['prec','gd','newep','epid','key'])

def metrics(df,label):
    x=df.copy()
    def cap(pid,dt,win=30):
        arr=clmg.get(pid)
        if arr is None or len(arr)==0: return 0
        return int(np.min(np.abs((arr-np.datetime64(dt)).astype('timedelta64[D]').astype(int)))<=win)
    x['captured']=[cap(p,dt) for p,dt in zip(x.person_id,x.d)]
    x=x.merge(pay,on='person_id',how='left')
    x['month']=x.d.dt.month+12*(x.d.dt.year-2024)
    x['is_ed']=(x.encounter_type=='emergency department').astype(int)
    x['is_inpat']=(x.encounter_type=='acute inpatient').astype(int)
    x['is_obs']=(x.encounter_type=='observation').astype(int)
    # runout curve (|nearest acute claim| <= W)
    def curve(win):
        return round(float(np.mean([cap(p,dt,win) for p,dt in zip(x.person_id,x.d)])),3)
    cur={str(w):curve(w) for w in [30,90,180,365,540]}
    # capture-model AUROC (member-disjoint, train-only target encoding)
    gss=GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
    tri,tei=next(gss.split(x,groups=x.person_id)); tr,te=x.iloc[tri].copy(),x.iloc[tei].copy()
    m=tr.captured.mean(); A=20.0
    def enc(col):
        g=tr.groupby(col).captured.agg(['sum','count']); return ((g['sum']+A*m)/(g['count']+A)).to_dict()
    fr,pr,ar=enc('facility_id'),enc('payer'),enc('asrc')
    for df_ in (tr,te):
        df_['fac_rate']=df_.facility_id.map(fr).fillna(m); df_['plan_rate']=df_.payer.map(pr).fillna(m); df_['asrc_rate']=df_.asrc.map(ar).fillna(m)
    F=['fac_rate','plan_rate','asrc_rate','is_ed','is_inpat','is_obs','month']
    gb=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[F],tr.captured)
    auc=roc_auc_score(te.captured,gb.predict_proba(te[F])[:,1])
    r={'label':label,'n_events':int(len(x)),'overall_capture_30d':round(float(x.captured.mean()),3),
       'runout_curve':cur,'capture_model_auc_gbm':round(float(auc),3),
       'pct_ed':round(float(x.is_ed.mean()),3),'pct_inpat':round(float(x.is_inpat.mean()),3)}
    print(json.dumps(r,indent=2)); return r

before=metrics(adt,'before_merge (dedup person/date/facility)')
after=metrics(episode_merge(adt,gap=1),'after_merge (ED->IP within 1 day, IP precedence)')
out={'before':before,'after':after,
     'n_collapsed':before['n_events']-after['n_events'],
     'pct_events_collapsed':round((before['n_events']-after['n_events'])/before['n_events'],3)}
json.dump(out,open('results_adt_merge.json','w'),indent=2)
print("\nSUMMARY: events %d -> %d (%.1f%% collapsed); runout@540 %.3f -> %.3f; capture-AUC %.3f -> %.3f"%(
 before['n_events'],after['n_events'],100*out['pct_events_collapsed'],
 before['runout_curve']['540'],after['runout_curve']['540'],before['capture_model_auc_gbm'],after['capture_model_auc_gbm']))
