import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
rng = np.random.default_rng(20260712)
OUT='.'
df = pd.read_parquet(f'{OUT}/dataset.parquet')

# ---- clean / feature engineering ----
cnt=['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','office_h1','office_h2','amb_visits','sumsq','maxn','nprov_amb']
for c in cnt: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int)
df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['no_office']=df['days_since_office'].isna().astype(int)
df['days_since_office']=df['days_since_office'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median())
df['n_spans']=df['n_spans'].fillna(1)
df['adt_active']=df['adt_active'].fillna(0).astype(int)
# continuity
df['UPC']=np.where(df.amb_visits>0, df.maxn/df.amb_visits.replace(0,np.nan),0).astype(float); df['UPC']=df['UPC'].fillna(0)
den=df.amb_visits*(df.amb_visits-1)
df['COC']=np.where(df.amb_visits>1,(df.sumsq-df.amb_visits)/den.replace(0,np.nan),0).astype(float); df['COC']=df['COC'].fillna(0)
pden=df.n_office+df.n_urgent+df.n_ed+df.n_outp
df['peripheral_share']=np.where(pden>0,(df.n_urgent+df.n_ed)/pden.replace(0,np.nan),0).astype(float); df['peripheral_share']=df['peripheral_share'].fillna(0)
df['office_drift']=df.office_h2-df.office_h1
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay')
df=pd.concat([df,pay],axis=1)
payer_cols=list(pay.columns)

BASE=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','claims_lag_days','no_claims','n_spans']+payer_cols
CONT=['UPC','COC','peripheral_share','days_since_office','no_office','office_drift','n_fac','n_prov']

tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
def fit_pred(feats,ytr,Xtr,Xte):
    m=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,
                     colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1)
    m.fit(Xtr[feats],ytr); return m,m.predict_proba(Xte[feats])[:,1]

def paired_boot_auc(y,p0,p1,n=1000):
    d=[]; idx=np.arange(len(y))
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if y[b].sum()==0 or y[b].sum()==len(b): continue
        d.append(roc_auc_score(y[b],p1[b])-roc_auc_score(y[b],p0[b]))
    d=np.array(d); return d.mean(), np.percentile(d,2.5),np.percentile(d,97.5),(d<=0).mean()
def recall_at_k(y,p,k=0.10):
    thr=np.quantile(p,1-k); sel=p>=thr; return y[sel].sum()/y.sum()

res={}
# ===== ADDITION 1: continuity incremental value (outcome=y_union) =====
y=te.y_union.values
m0,p0=fit_pred(BASE,tr.y_union.values,tr,te)
m1,p1=fit_pred(BASE+CONT,tr.y_union.values,tr,te)
auc0,auc1=roc_auc_score(y,p0),roc_auc_score(y,p1)
ap0,ap1=average_precision_score(y,p0),average_precision_score(y,p1)
dmean,dlo,dhi,pval=paired_boot_auc(y,p0,p1)
res['A1']={'auroc_base':auc0,'auroc_cont':auc1,'dAUROC':dmean,'ci':[dlo,dhi],'p_onesided':pval,
           'auprc_base':ap0,'auprc_cont':ap1,'recall10_base':recall_at_k(y,p0),'recall10_cont':recall_at_k(y,p1)}
print("=== ADDITION 1: continuity features (predict y_union) ===")
print(f"AUROC base={auc0:.4f}  +cont={auc1:.4f}  dAUROC={dmean:.4f} [{dlo:.4f},{dhi:.4f}] p(1-sided)={pval:.3f}")
print(f"AUPRC base={ap0:.4f}  +cont={ap1:.4f}   Recall@10%: base={recall_at_k(y,p0):.3f} +cont={recall_at_k(y,p1):.3f}")
imp=pd.Series(m1.feature_importances_,index=BASE+CONT).sort_values(ascending=False)
print("top features:", list(imp.head(12).index))

# ===== ADDITION 2: under-ascertainment + SAR-PU =====
print("\n=== ADDITION 2: capture rate e = P(claims=1 | union=1) ===")
pos=df[df.y_union==1]
print("overall capture:", round((pos.y_claims==1).mean(),3), " n_union_pos=",len(pos))
print("by payer:\n", pos.groupby('payer').y_claims.mean().round(3).to_string())
print("by adt_active:\n", pos.groupby('adt_active').y_claims.mean().round(3).to_string())
pos['lag_bucket']=pd.cut(pos.claims_lag_days,[0,30,90,180,10000],labels=['<=30','31-90','91-180','>180'])
print("by claims-lag bucket:\n", pos.groupby('lag_bucket').y_claims.mean().round(3).to_string())

# SAR-PU: anchor = adt_active (union ~ complete). estimate e(x) on anchor; naive claims model; corrected=p_S/e
ASC=['claims_lag_days','no_claims','n_spans','adt_active']+payer_cols
anchor=tr[tr.adt_active==1]
# e(x): among anchor union-positives, P(claims=1| x_asc)
ap_=anchor[anchor.y_union==1]
eclf=LogisticRegression(max_iter=1000,C=1.0)
if ap_.y_claims.nunique()>1:
    eclf.fit(ap_[ASC],ap_.y_claims)
    def ehat(X): 
        e=eclf.predict_proba(X[ASC])[:,1]; return np.clip(e,0.05,1.0)
else:
    def ehat(X): return np.full(len(X),(ap_.y_claims==1).mean())
# naive claims risk model (train on y_claims, all train)
mS,pS_te=fit_pred(BASE+CONT,tr.y_claims.values,tr,te)
e_te=ehat(te)
pY_te=np.clip(pS_te/e_te,0,1)
# validate on held-out ADT anchor (union ~ truth)
anc_te=te[te.adt_active==1].index
mask=te.index.isin(anc_te)
yv=te.loc[mask,'y_union'].values
print("\n--- Anchor recovery (held-out ADT-active, union~truth) ---")
print(f"observed union rate={yv.mean():.3f} | mean naive p_S={pS_te[mask].mean():.3f} | mean corrected p_Y={pY_te[mask].mean():.3f}")
print(f"AUROC naive={roc_auc_score(yv,pS_te[mask]):.3f} corrected={roc_auc_score(yv,pY_te[mask]):.3f}")
# reclassification: who enters top decile under corrected vs naive (full test)
k=0.10
top_S=pS_te>=np.quantile(pS_te,1-k); top_Y=pY_te>=np.quantile(pY_te,1-k)
gained=(~top_S)&top_Y
print("\n--- Reclassification into top-decile (corrected vs naive) ---")
print(f"n gained into top decile: {gained.sum()}  ({100*gained.mean():.1f}% of test)")
print("median claims_lag: kept-out-then-in vs stayed:", round(te.claims_lag_days.values[gained].mean(),1),"vs",round(te.claims_lag_days.values[top_S].mean(),1))
lowcov=(te.adt_active.values==0)
print("gained members that are non-ADT (low-coverage):", round(100*lowcov[gained].mean(),1),"%")
res['A2']={'capture_overall':float((pos.y_claims==1).mean()),
           'capture_by_payer':pos.groupby('payer').y_claims.mean().round(3).to_dict(),
           'anchor_obs_union':float(yv.mean()),'anchor_naive_mean':float(pS_te[mask].mean()),
           'anchor_corrected_mean':float(pY_te[mask].mean()),
           'reclassified_pct':float(100*gained.mean())}
json.dump(res,open(f'{OUT}/results.json','w'),indent=2,default=float)
print("\nsaved results.json")
