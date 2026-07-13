import sys, pathlib, json, warnings, numpy as np, pandas as pd
from datetime import date, timedelta
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path.home() / ".claude/skills/waymark-data-access/scripts"))
from wm_conn import coredb, query
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
cd=coredb("prod"); T0='2025-07-01'; t0=date.fromisoformat(T0)
BL=(t0-timedelta(days=365)).isoformat(); W180=(t0+timedelta(days=180)).isoformat()
np.random.seed(1)

df=pd.read_parquet('dataset_2025-07-01.parquet')                    # cohort + features + adt_active

# acute events (both sources) with dates, T0..T0+180, replicating build_dataset acute defs
ev=query(cd,f"""
SELECT person_id, encounter_start_date d,
  (data_source='Claims' AND (ed_flag=1 OR encounter_group='inpatient'))::int claims_acute,
  (data_source='HL7 ADT Feed' AND encounter_type IN ('emergency department','acute inpatient','observation'))::int adt_acute
FROM dbt_tuva_core.encounter
WHERE encounter_start_date>=DATE '{T0}' AND encounter_start_date<=DATE '{W180}'
  AND ((data_source='Claims' AND (ed_flag=1 OR encounter_group='inpatient'))
    OR (data_source='HL7 ADT Feed' AND encounter_type IN ('emergency department','acute inpatient','observation')))""")
ev=ev[ev.person_id.isin(df.person_id)]
ev['dd']=(pd.to_datetime(ev.d)-pd.Timestamp(t0)).dt.days

# enrollment end of the span covering T0 (for right-censoring / churn)
enr=query(cd,f"""SELECT person_id, MAX(enrollment_end_date) enr_end FROM dbt_tuva_core.eligibility
  WHERE enrollment_start_date<=DATE '{T0}' AND enrollment_end_date>=DATE '{T0}' GROUP BY person_id""")
enr['enr_days']=(pd.to_datetime(enr.enr_end)-pd.Timestamp(t0)).dt.days
df=df.merge(enr[['person_id','enr_days']],on='person_id',how='left')

def labels_at(W):
    c=ev[(ev.claims_acute==1)&(ev.dd<=W)].person_id.unique()
    a=ev[(ev.adt_acute==1)&(ev.dd<=W)].person_id.unique()
    yc=df.person_id.isin(c).astype(int).values; ya=df.person_id.isin(a).astype(int).values
    return yc, np.maximum(yc,ya)

# feature matrix (same as ladder_ci)
d=df.copy()
for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: d[c]=d[c].fillna(0)
d['no_claims']=d['claims_lag_days'].isna().astype(int); d['claims_lag_days']=d['claims_lag_days'].fillna(9999)
d['age']=d['age'].fillna(d['age'].median()); d['n_spans']=d['n_spans'].fillna(1); d['adt_active']=d['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(d['payer'].fillna('UNK'),prefix='pay'); d=pd.concat([d,pay],axis=1)
F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+list(pay.columns)
def LG(): return LGBMClassifier(n_estimators=500,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1)
def recall10(ytrue,p):
    thr=np.quantile(p,0.9); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(ytrue,yh).ravel(); return tp/(tp+fn)

res={'windows':{}}
for W in [30,60,90,180]:
    yc,yu=labels_at(W); d['_yc']=yc; d['_yu']=yu
    m=d.adt_active.values==1                                        # ADT-covered
    cap= (yc[m & (yu==1)].sum())/max((yu[m]==1).sum(),1)            # capture among ADT-covered union events
    tr,te=train_test_split(d,test_size=0.3,random_state=42,stratify=d._yu)
    anc=te.adt_active.values==1; yU=te.loc[anc,'_yu'].values
    p_sq=LG().fit(tr[F],tr._yc).predict_proba(te[F])[:,1][anc]      # status quo (claims label)
    p_adt=LG().fit(tr[F],tr._yu).predict_proba(te[F])[:,1][anc]     # ADT-completed label
    res['windows'][W]={'union_rate':round(float(yu.mean()),4),'obs_adt_rate':round(float(yu[m].mean()),4),
        'capture':round(float(cap),4),'n_union_events_adt':int((yu[m]==1).sum()),
        'recall10_statusquo':round(float(recall10(yU,p_sq)),4),'recall10_adt':round(float(recall10(yU,p_adt)),4),
        'auroc_statusquo_vs_true':round(float(roc_auc_score(yU,p_sq)),4),'auroc_adt_vs_true':round(float(roc_auc_score(yU,p_adt)),4)}
    print(f"W={W:>3}d union={yu.mean():.3f} capture={cap:.3f} recall_sq={res['windows'][W]['recall10_statusquo']:.3f} recall_adt={res['windows'][W]['recall10_adt']:.3f}")

# churn / right-censoring (at W=90)
yc90,yu90=labels_at(90)
disenroll_90=float((df.enr_days<90).mean())
cont=df.enr_days>=90
m=(df.adt_active.values==1)
cap_all=yc90[m&(yu90==1)].sum()/max((yu90[m]==1).sum(),1)
mc=m & cont.values
cap_cont=yc90[mc&(yu90==1)].sum()/max((yu90[mc]==1).sum(),1)
# ADT acute events occurring AFTER disenrollment (feed not plan-claims-gated)
evm=ev[ev.adt_acute==1].merge(df[['person_id','enr_days']],on='person_id',how='left')
post=float((evm.dd>evm.enr_days).mean())
res['churn']={'disenroll_by_90d':round(disenroll_90,4),'capture_all_adt':round(float(cap_all),4),
  'capture_continuous_enrolled':round(float(cap_cont),4),'n_continuous':int(cont.sum()),
  'adt_acute_events_post_disenrollment_frac':round(post,4)}
print("churn:",json.dumps(res['churn']))

# missingness of key features
miss={c:round(float(df[c].isna().mean()),4) for c in ['age','payer','claims_lag_days','n_fac','n_prov','enr_days']}
miss['no_prior_claim (claims_lag null)']=round(float(df['claims_lag_days'].isna().mean()),4)
res['missingness']=miss
print("missing:",json.dumps(miss))
json.dump(res,open('results_sensitivity.json','w'),indent=2,default=float)
print("SAVED results_sensitivity.json")
