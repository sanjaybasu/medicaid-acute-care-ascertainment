import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
rng=np.random.default_rng(20260713)
df=pd.read_parquet('dataset_2025-07-01.parquet')
for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1); df['adt_active']=df['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1); paycols=list(pay.columns)
F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
ASC=['claims_lag_days','no_claims','n_spans','adt_active']+paycols
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
def lg(y,w=None): 
    m=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1)
    m.fit(tr[F],y,sample_weight=w); return m
# e(x) capture model (train anchor union-positives)
ap=tr[(tr.adt_active==1)&(tr.y_union==1)]; eclf=LogisticRegression(max_iter=2000).fit(ap[ASC],ap.y_claims)
etr=np.clip(eclf.predict_proba(tr[ASC])[:,1],0.05,1.0); ete=np.clip(eclf.predict_proba(te[ASC])[:,1],0.05,1.0)
m0=lg(tr.y_claims); m1=lg(tr.y_union)
# M3: PU inverse-capture-weighted training on claims label (up-weight low-capture members)
w=np.where(tr.y_claims==1, 1.0/etr, 1.0); m3=lg(tr.y_claims,w=w)
p0=m0.predict_proba(te[F])[:,1]; p1=m1.predict_proba(te[F])[:,1]; p3=m3.predict_proba(te[F])[:,1]
p2=np.clip(p0/ete,0,1)
anc=te.adt_active.values==1; y=te.loc[anc,'y_union'].values
P={'M0_claims':p0[anc],'M1_ADTlabel':p1[anc],'M2_PUrescale':p2[anc],'M3_PUweighted':p3[anc]}
def rec_at(y,p,k=0.10):
    thr=np.quantile(p,1-k); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return tp/(tp+fn),(tp/(tp+fp) if tp+fp else np.nan),f1_score(y,yh),int(tp),int(tp+fn)
def aucci(y,p):
    idx=np.arange(len(y)); v=[roc_auc_score(y[b],p[b]) for b in (rng.choice(idx,len(idx),True) for _ in range(1000)) if y[b].sum()>2]
    return [round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]
def paired(y,pa,pb,metric,n=1000):
    idx=np.arange(len(y)); d=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),True)
        if y[b].sum()<3: continue
        d.append(metric(y[b],pb[b])-metric(y[b],pa[b]))
    return [round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)]
recm=lambda yy,pp: rec_at(yy,pp)[0]
R={'n_anchor':int(anc.sum()),'obs':round(float(y.mean()),3),'true_pos_total':int(y.sum())}
for nm,p in P.items():
    rc,ppv,f1,tp,npos=rec_at(y,p)
    R[nm]={'auroc':round(roc_auc_score(y,p),3),'auroc_ci':aucci(y,p),'auprc':round(float(average_precision_score(y,p)),3),
           'recall_top10':round(rc,3),'ppv_top10':round(ppv,3),'f1_top10':round(f1,3),'tp_top10':tp}
for nm in ['M1_ADTlabel','M3_PUweighted','M2_PUrescale']:
    R[f'delta_{nm}_vs_M0']={'d_auroc':round(R[nm]['auroc']-R['M0_claims']['auroc'],3),
        'd_auroc_ci':paired(y,P['M0_claims'],P[nm],lambda a,b:roc_auc_score(a,b)),
        'd_recall':round(R[nm]['recall_top10']-R['M0_claims']['recall_top10'],3),
        'd_recall_ci':paired(y,P['M0_claims'],P[nm],recm)}
json.dump(R,open('results_predict_value.json','w'),indent=2,default=float)
print(json.dumps(R,indent=2,default=float))
