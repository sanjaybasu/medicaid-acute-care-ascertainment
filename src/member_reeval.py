import sys, pathlib, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
# Internal data connector (wm_conn) is loaded from the secure environment; not included here.
from wm_conn import coredb, query
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score
rng=np.random.default_rng(20260713)
BASE='.'
cd=coredb("prod")
df=pd.read_parquet(f'{BASE}/dataset_2025-07-01.parquet')
fmap=json.load(open(f'{BASE}/facility_rate_map.json')); GM=fmap['global_mean']
frate=fmap['facility_rate']; prate=fmap['plan_rate']
BL,T0='2024-07-01','2025-07-01'
pids=list(df.person_id.unique())
# member baseline facility-mix reporting rate from prior-year ADT acute exposure (prospectively available)
fac=query(cd, f"""SELECT person_id, facility_id, COUNT(*) n FROM dbt_tuva_core.encounter
 WHERE data_source='HL7 ADT Feed' AND encounter_type IN ('emergency department','acute inpatient','observation')
   AND encounter_start_date>=DATE '{BL}' AND encounter_start_date<DATE '{T0}' AND person_id=ANY(%(p)s)
 GROUP BY 1,2""", p=pids)
fac['rate']=fac.facility_id.map(frate)
fac=fac.dropna(subset=['rate'])
fmix=fac.groupby('person_id').apply(lambda g: np.average(g.rate, weights=g.n)).rename('fac_mix_rate').reset_index()
df=df.merge(fmix,on='person_id',how='left')
df['has_fac_mix']=df.fac_mix_rate.notna().astype(int)
df['fac_mix_rate']=df.fac_mix_rate.fillna(GM)
df['plan_rate']=df.payer.map(prate).fillna(GM)
print(f"members with baseline facility-mix rate: {df.has_fac_mix.sum():,} / {len(df):,}")

# prep
for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1); df['adt_active']=df['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1); paycols=list(pay.columns)
BASE_F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
ASC_OLD=['claims_lag_days','no_claims','n_spans','adt_active']+paycols
ASC_NEW=['claims_lag_days','no_claims','n_spans','adt_active','plan_rate','fac_mix_rate','has_fac_mix']
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
mS=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[BASE_F],tr.y_claims)
pS=mS.predict_proba(te[BASE_F])[:,1]
ap=tr[(tr.adt_active==1)&(tr.y_union==1)]; apte=te[(te.adt_active==1)&(te.y_union==1)]
def emodel(feats):
    e=LogisticRegression(max_iter=2000).fit(ap[feats],ap.y_claims)
    auc=roc_auc_score(apte.y_claims,e.predict_proba(apte[feats])[:,1]) if apte.y_claims.nunique()>1 else None
    return e,round(float(auc),3)
eO,aucO=emodel(ASC_OLD); eN,aucN=emodel(ASC_NEW)
print(f"member-level capture AUROC: OLD(plan one-hot+lag)={aucO}  NEW(+facility-mix+plan-rate)={aucN}")
def ehat(e,feats,X): return np.clip(e.predict_proba(X[feats])[:,1],0.05,1.0)
pY=np.clip(pS/ehat(eN,ASC_NEW,te),0,1)
anc=te.adt_active==1; y=te.loc[anc,'y_union'].values; ps=pS[anc.values]; py=pY[anc.values]
def m_at_k(y,p,k=0.10):
    thr=np.quantile(p,1-k); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return dict(sens=tp/(tp+fn),spec=tn/(tn+fp),ppv=tp/(tp+fp) if tp+fp else np.nan,npv=tn/(tn+fn) if tn+fn else np.nan,f1=f1_score(y,yh))
def cal(y,p):
    pc=np.clip(p,1e-6,1-1e-6); lr=LogisticRegression().fit(np.log(pc/(1-pc)).reshape(-1,1),y)
    return dict(slope=float(lr.coef_[0,0]),intercept=float(lr.intercept_[0]),brier=float(brier_score_loss(y,np.clip(p,0,1))),mean=float(p.mean()))
def bootauc(y,p,n=1000):
    idx=np.arange(len(y)); v=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),True)
        if y[b].sum()<2: continue
        v.append(roc_auc_score(y[b],p[b]))
    return round(np.percentile(v,2.5),3),round(np.percentile(v,97.5),3)
R={'members_with_fac_mix':int(df.has_fac_mix.sum()),'n_cohort':int(len(df)),
   'capture_auc_old':aucO,'capture_auc_new':aucN,'n_test_anchor':int(anc.sum()),'obs':round(float(y.mean()),3)}
for nm,p in [('naive',ps),('corrected',py)]:
    R[nm]={'auroc':round(float(roc_auc_score(y,p)),3),'auroc_ci':bootauc(y,p),
           'auprc':round(float(average_precision_score(y,p)),3),**{k:round(float(v),3) for k,v in m_at_k(y,p).items()},
           **{k:round(float(v),3) for k,v in cal(y,p).items()}}
# SCAR vs SAR with new e
c_const=(ap.y_claims==1).mean(); pY_scar=np.clip(ps/max(c_const,0.05),0,1)
teA=te.loc[anc].copy(); teA['y']=y; teA['naive']=ps; teA['sar']=py; teA['scar']=pY_scar; teA['hilag']=(teA.claims_lag_days>90).astype(int)
R['ablation']={'scar_mean':round(float(pY_scar.mean()),3),'sar_mean':round(float(py.mean()),3),
   'strata':{('lag>90d' if g==1 else 'lag<=90d'):{'n':int(len(s)),'obs':round(float(s.y.mean()),3),'naive':round(float(s.naive.mean()),3),'scar':round(float(s.scar.mean()),3),'sar':round(float(s.sar.mean()),3)} for g,s in teA.groupby('hilag')}}
json.dump(R,open(f'{BASE}/results_full_v2.json','w'),indent=2,default=float)
print(json.dumps(R,indent=2,default=float))
