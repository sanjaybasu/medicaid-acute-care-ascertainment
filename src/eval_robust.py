import warnings, json, sys, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from statsmodels.stats.proportion import proportion_confint
OUT='/Users/sanjaybasu/notebooks/pu-underascertainment'

def prep(df):
    for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
    df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
    df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1)
    df['adt_active']=df['adt_active'].fillna(0).astype(int)
    pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1)
    return df,list(pay.columns)
def wil(s): k=int(s.sum()); n=len(s); lo,hi=proportion_confint(k,n,method='wilson'); return [round(k/n,3),round(lo,3),round(hi,3),n]

def run(tag, path, ycol):
    df=pd.read_parquet(path); df,paycols=prep(df)
    BASE=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
    ASC=['claims_lag_days','no_claims','n_spans','adt_active']+paycols
    tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df[ycol])
    m=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1)
    m.fit(tr[BASE],tr.y_claims); pS=m.predict_proba(te[BASE])[:,1]
    ap=tr[(tr.adt_active==1)&(tr[ycol]==1)]
    e=LogisticRegression(max_iter=2000).fit(ap[ASC],ap.y_claims)
    pY=np.clip(pS/np.clip(e.predict_proba(te[ASC])[:,1],0.05,1.0),0,1)
    anc=te.adt_active==1; y=te.loc[anc,ycol].values
    pos=df[df[ycol]==1]
    return {'tag':tag,'ycol':ycol,'n':len(df),'y_claims':round(float(df.y_claims.mean()),4),
            'y_outcome':round(float(df[ycol].mean()),4),
            'capture_overall':wil(pos.y_claims),
            'capture_by_payer':{pv:wil(g.y_claims) for pv,g in pos.groupby('payer')},
            'anchor_n':int(anc.sum()),'obs':round(float(y.mean()),3),
            'naive_mean':round(float(pS[anc.values].mean()),3),'corrected_mean':round(float(pY[anc.values].mean()),3),
            'naive_auroc':round(float(roc_auc_score(y,pS[anc.values])),3),'corrected_auroc':round(float(roc_auc_score(y,pY[anc.values])),3),
            'naive_brier':round(float(brier_score_loss(y,np.clip(pS[anc.values],0,1))),4),'corrected_brier':round(float(brier_score_loss(y,pY[anc.values])),4)}

R=[]
R.append(run('primary_union','%s/dataset_2025-07-01.parquet'%OUT,'y_union'))
R.append(run('exclude_observation','%s/dataset_2025-07-01.parquet'%OUT,'y_union_noobs'))
R.append(run('altdate_2025-04-01','%s/dataset_2025-04-01.parquet'%OUT,'y_union'))
for r in R:
    print(f"\n=== {r['tag']} (outcome={r['ycol']}, N={r['n']:,}) ===")
    print(f"  y_claims={r['y_claims']}  y_outcome={r['y_outcome']}  capture_overall={r['capture_overall'][0]} [{r['capture_overall'][1]},{r['capture_overall'][2]}]")
    print(f"  anchor n={r['anchor_n']} obs={r['obs']} | naive_mean={r['naive_mean']} corrected_mean={r['corrected_mean']} | AUROC {r['naive_auroc']}->{r['corrected_auroc']} | Brier {r['naive_brier']}->{r['corrected_brier']}")
json.dump(R,open('%s/results_robust.json'%OUT,'w'),indent=2,default=float)
print("\nsaved results_robust.json")
