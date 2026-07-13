import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
BASE='/Users/sanjaybasu/notebooks/pu-underascertainment'
df=pd.read_parquet(f'{BASE}/dataset_2025-07-01.parquet')
for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1); df['adt_active']=df['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1); paycols=list(pay.columns)
BASE_F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
ASC_full={'plan':paycols,'claims_lag':['claims_lag_days','no_claims'],'enrollment_spans':['n_spans'],'adt_coverage':['adt_active']}
ALL_ASC=sum(ASC_full.values(),[])
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
mS=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[BASE_F],tr.y_claims)
pS=mS.predict_proba(te[BASE_F])[:,1]
anc=te.adt_active==1; yv=te.loc[anc,'y_union'].values; pSa=pS[anc.values]
ap=tr[(tr.adt_active==1)&(tr.y_union==1)]
R={}
# 1) SCAR (constant c) vs SAR e(x)
c_const=(ap.y_claims==1).mean()
e_sar=LogisticRegression(max_iter=2000).fit(ap[ALL_ASC],ap.y_claims)
def ehat(X): return np.clip(e_sar.predict_proba(X[ALL_ASC])[:,1],0.05,1.0)
pY_scar=np.clip(pSa/max(c_const,0.05),0,1); pY_sar=np.clip(pSa/ehat(te[anc.values.astype(bool)] if False else te.loc[anc]),0,1)
R['obs']=round(float(yv.mean()),3); R['naive_mean']=round(float(pSa.mean()),3)
R['scar_c']=round(float(c_const),3); R['scar_mean']=round(float(pY_scar.mean()),3); R['sar_mean']=round(float(pY_sar.mean()),3)
# stratified by claims lag (low vs high coverage proxy) on anchor
teA=te.loc[anc].copy(); teA['pS']=pSa; teA['pY_scar']=pY_scar; teA['pY_sar']=pY_sar; teA['y']=yv
teA['hilag']=(teA.claims_lag_days>90).astype(int)
strat={}
for g,sub in teA.groupby('hilag'):
    strat['lag>90d' if g==1 else 'lag<=90d']={'n':int(len(sub)),'obs':round(float(sub.y.mean()),3),
        'naive':round(float(sub.pS.mean()),3),'scar':round(float(sub.pY_scar.mean()),3),'sar':round(float(sub.pY_sar.mean()),3)}
R['strata']=strat
# 2) e(x) covariate drop-one: capture-model AUROC on held-out anchor union-positives
apte=te[(te.adt_active==1)&(te.y_union==1)]
def cap_auc(feats):
    m=LogisticRegression(max_iter=2000).fit(ap[feats],ap.y_claims)
    p=m.predict_proba(apte[feats])[:,1]
    return round(float(roc_auc_score(apte.y_claims,p)),3) if apte.y_claims.nunique()>1 else None
R['capture_auc_full']=cap_auc(ALL_ASC)
R['capture_auc_dropone']={k:cap_auc([f for f in ALL_ASC if f not in v]) for k,v in ASC_full.items()}
# 3) e(x) odds ratios (interpretability, non-plan covariates)
coef=dict(zip(ALL_ASC,e_sar.coef_[0]))
R['e_odds_ratios']={k:round(float(np.exp(coef[k])),3) for k in ['claims_lag_days','no_claims','n_spans','adt_active']}
json.dump(R,open(f'{BASE}/results_ablation.json','w'),indent=2,default=float)
print(json.dumps(R,indent=2,default=float))
