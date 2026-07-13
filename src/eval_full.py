import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
   confusion_matrix, f1_score)
rng=np.random.default_rng(20260713)
OUT='/Users/sanjaybasu/notebooks/pu-underascertainment'
df=pd.read_parquet(f'{OUT}/dataset.parquet')

# ---- clean ----
cnt=['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']
for c in cnt: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int)
df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1)
df['adt_active']=df['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1)
paycols=list(pay.columns)
BASE=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
ASC=['claims_lag_days','no_claims','n_spans','adt_active']+paycols

tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)

# ---- naive claims model + SAR-PU correction ----
def lgbm(y,X): 
    m=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,
        min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1); m.fit(X[BASE],y); return m
mS=lgbm(tr.y_claims.values,tr); pS=mS.predict_proba(te[BASE])[:,1]
# e(x): anchor(adt_active) train, among union-positives P(claims=1|x)
ap=tr[(tr.adt_active==1)&(tr.y_union==1)]
eclf=LogisticRegression(max_iter=2000).fit(ap[ASC],ap.y_claims)
def ehat(X,floor=0.05): return np.clip(eclf.predict_proba(X[ASC])[:,1],floor,1.0)
pY=np.clip(pS/ehat(te),0,1)

# ---- metric suite (evaluated on held-out ADT-anchor where y_union~truth) ----
anc=te.adt_active==1
y=te.loc[anc,'y_union'].values; ps=pS[anc.values]; py=pY[anc.values]
def metrics_at_k(y,p,k=0.10):
    thr=np.quantile(p,1-k); yhat=(p>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yhat).ravel()
    sens=tp/(tp+fn); spec=tn/(tn+fp); ppv=tp/(tp+fp) if (tp+fp) else np.nan
    npv=tn/(tn+fn) if (tn+fn) else np.nan; f1=f1_score(y,yhat)
    return dict(sens=sens,spec=spec,ppv=ppv,npv=npv,f1=f1)
def calib(y,p):
    pc=np.clip(p,1e-6,1-1e-6); lr=LogisticRegression().fit(np.log(pc/(1-pc)).reshape(-1,1),y)
    return dict(cal_slope=float(lr.coef_[0,0]),cal_intercept=float(lr.intercept_[0]),
                brier=float(brier_score_loss(y,np.clip(p,0,1))),mean_pred=float(p.mean()),obs=float(y.mean()))
def full(y,p):
    d=dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)))
    d.update(metrics_at_k(y,p)); d.update(calib(y,p)); return d
def boot(y,p,fn,n=1000):
    idx=np.arange(len(y)); vals=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if y[b].sum()<2: continue
        vals.append(fn(y[b],p[b]))
    return np.percentile(vals,[2.5,97.5])
def ci(y,p):
    base=full(y,p); out={}
    for k,f in [('auroc',roc_auc_score),('auprc',average_precision_score)]:
        lo,hi=boot(y,p,f); out[k]=[round(base[k],4),round(lo,4),round(hi,4)]
    for k in ['sens','spec','ppv','npv','f1']:
        f=lambda yy,pp,kk=k: metrics_at_k(yy,pp)[kk]; lo,hi=boot(y,p,f)
        out[k]=[round(base[k],4),round(lo,4),round(hi,4)]
    for k in ['cal_slope','cal_intercept','brier','mean_pred','obs']: out[k]=round(base[k],4)
    return out

R={'n_test_anchor':int(anc.sum()),'obs_union_rate':float(y.mean()),
   'naive':ci(y,ps),'corrected':ci(y,py)}
print("=== Metric suite on held-out ADT-anchor (n=%d, obs rate=%.3f) ==="%(anc.sum(),y.mean()))
for name in ['naive','corrected']:
    m=R[name]; print(f"\n[{name}] AUROC {m['auroc']}  AUPRC {m['auprc']}")
    print(f"  @top10%: sens {m['sens']} spec {m['spec']} PPV {m['ppv']} NPV {m['npv']} F1 {m['f1']}")
    print(f"  calib: slope {m['cal_slope']} intercept {m['cal_intercept']} Brier {m['brier']} mean_pred {m['mean_pred']} obs {m['obs']}")

# ---- descriptive capture with Wilson CI ----
from statsmodels.stats.proportion import proportion_confint
pos=df[df.y_union==1]
def wil(s): 
    k=int(s.sum()); n=len(s); lo,hi=proportion_confint(k,n,method='wilson'); return round(k/n,3),round(lo,3),round(hi,3),n
cap={'overall':wil(pos.y_claims)}
for pv,g in pos.groupby('payer'): cap[pv]=wil(g.y_claims)
R['capture']=cap
print("\n=== Capture P(claims|union) with 95% Wilson CI ===")
for k,v in cap.items(): print(f"  {k}: {v[0]} [{v[1]},{v[2]}] (n={v[3]})")

# ---- e(x) floor sensitivity ----
sens={}
for fl in [0.02,0.05,0.10,0.15,0.20]:
    pyf=np.clip(pS/ehat(te,floor=fl),0,1)[anc.values]
    sens[fl]={'mean_corrected':round(float(pyf.mean()),3)}
R['efloor_sensitivity']=sens; R['obs_for_sens']=float(y.mean())
print("\n=== e(x) floor sensitivity (mean corrected vs obs %.3f) ==="%y.mean())
for fl,v in sens.items(): print(f"  floor={fl}: mean_corrected={v['mean_corrected']}")

json.dump(R,open(f'{OUT}/results_full.json','w'),indent=2,default=float)
print("\nsaved results_full.json")
