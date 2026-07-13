import sys, pathlib, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
# Internal data connector (wm_conn) loaded from secure environment; not included here.
from wm_conn import coredb, query
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score
rng=np.random.default_rng(20260713); BASE='.'
def prep(df):
    for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
    df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
    df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1); df['adt_active']=df['adt_active'].fillna(0).astype(int)
    pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1)
    return df,list(pay.columns)
def m_at_k(y,p,k=0.10):
    thr=np.quantile(p,1-k); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return dict(sens=tp/(tp+fn),spec=tn/(tn+fp),ppv=tp/(tp+fp) if tp+fp else np.nan,npv=tn/(tn+fn) if tn+fn else np.nan,f1=f1_score(y,yh))
def cal(y,p):
    pc=np.clip(p,1e-6,1-1e-6); lr=LogisticRegression().fit(np.log(pc/(1-pc)).reshape(-1,1),y)
    return dict(cal_slope=float(lr.coef_[0,0]),cal_intercept=float(lr.intercept_[0]),brier=float(brier_score_loss(y,np.clip(p,0,1))),mean_pred=float(p.mean()))
def allm(y,p):
    d=dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p))); d.update(m_at_k(y,p)); d.update(cal(y,p)); return d
def ci(y,p,n=1000):
    base=allm(y,p); keys=list(base); acc={k:[] for k in keys}; idx=np.arange(len(y))
    for _ in range(n):
        b=rng.choice(idx,len(idx),True)
        if y[b].sum()<2: continue
        mm=allm(y[b],p[b])
        for k in keys: acc[k].append(mm[k])
    return {k:[round(base[k],4),round(float(np.percentile(acc[k],2.5)),4),round(float(np.percentile(acc[k],97.5)),4)] for k in keys}

df=pd.read_parquet(f'{BASE}/dataset_2025-07-01.parquet'); df,paycols=prep(df)
BASE_F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
ASC=['claims_lag_days','no_claims','n_spans','adt_active']+paycols
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
mS=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[BASE_F],tr.y_claims)
pS=mS.predict_proba(te[BASE_F])[:,1]
ap=tr[(tr.adt_active==1)&(tr.y_union==1)]
e=LogisticRegression(max_iter=2000).fit(ap[ASC],ap.y_claims)
pY=np.clip(pS/np.clip(e.predict_proba(te[ASC])[:,1],0.05,1.0),0,1)
anc=te.adt_active==1; y=te.loc[anc,'y_union'].values
R={'n_test_anchor':int(anc.sum()),'obs':round(float(y.mean()),4),'naive':ci(y,pS[anc.values]),'corrected':ci(y,pY[anc.values])}
json.dump(R,open(f'{BASE}/results_full.json','w'),indent=2,default=float)
print("Table2 with CIs: naive mean",R['naive']['mean_pred'],"corrected mean",R['corrected']['mean_pred'],"brier",R['naive']['brier'],R['corrected']['brier'])

# event-level capture AUROC CI
ev=pd.read_parquet(f'{BASE}/_event_capture.parquet')
FEV=['fac_rate','plan_rate','asrc_rate','is_ed','is_inpat','is_obs','month']
gss=GroupShuffleSplit(n_splits=1,test_size=0.3,random_state=42); tri,tei=next(gss.split(ev,groups=ev.person_id))
etr,ete=ev.iloc[tri],ev.iloc[tei]
gm=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(etr[FEV],etr.captured)
pe=gm.predict_proba(ete[FEV])[:,1]; ye=ete.captured.values
def aucci(y,p,n=1000):
    b0=roc_auc_score(y,p); v=[]; idx=np.arange(len(y))
    for _ in range(n):
        bb=rng.choice(idx,len(idx),True)
        if y[bb].sum()<2: continue
        v.append(roc_auc_score(y[bb],p[bb]))
    return [round(float(b0),3),round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]
CAP={'event_auc':aucci(ye,pe)}
# member capture old
apte=te[(te.adt_active==1)&(te.y_union==1)]
eo=LogisticRegression(max_iter=2000).fit(ap[ASC],ap.y_claims); po=eo.predict_proba(apte[ASC])[:,1]
CAP['member_auc_old']=aucci(apte.y_claims.values,po)
json.dump(CAP,open(f'{BASE}/results_capture_ci.json','w'),indent=2)
print("capture CIs:",CAP)
print("saved results_full.json + results_capture_ci.json")
