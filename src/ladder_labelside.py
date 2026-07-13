import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
# identical setup to ladder_ci.py
rng=np.random.default_rng(20260713); torch.manual_seed(1); np.random.seed(1)
df=pd.read_parquet('dataset_2025-07-01.parquet')
for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1); df['adt_active']=df['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1); paycols=list(pay.columns)
F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
anc=te.adt_active.values==1; yU=te.loc[anc,'y_union'].values; yC=te.loc[anc,'y_claims'].values

def LG(): return LGBMClassifier(n_estimators=500,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1)
def M(y,p):
    au=roc_auc_score(y,p); thr=np.quantile(p,0.9); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return round(au,3),round(tp/(tp+fn),3),round(tp/(tp+fp) if tp+fp else np.nan,3),round(f1_score(y,yh),3)

# status quo (must reproduce ladder_ci: 0.620 / 0.163)
m0=LG().fit(tr[F],tr.y_claims); p0=m0.predict_proba(te[F])[:,1][anc]
sq=M(yU,p0); print("status_quo reproduce:",sq)

# enriched-feature tabular on ADT-completed label
enr=pd.read_parquet('_enriched_features.parquet'); ecols=[c for c in enr.columns if c!='person_id']
trE=tr.merge(enr,on='person_id',how='left'); teE=te.merge(enr,on='person_id',how='left')
for c in ecols: trE[c]=trE[c].fillna(0); teE[c]=teE[c].fillna(0)
F2=F+ecols
p_enr=LG().fit(trE[F2],trE.y_union).predict_proba(teE[F2])[:,1][anc]

# capture propensity e(x) by logistic regression on ascertainment covariates
xa=['claims_lag_days','no_claims','n_spans','adt_active']+paycols
pos=tr[(tr.adt_active==1)&(tr.y_union==1)]
ex_model=LogisticRegression(max_iter=1000,C=1.0).fit(pos[xa],pos.y_claims)
def ecap(d):
    e=ex_model.predict_proba(d[xa])[:,1]; return np.clip(e,0.05,1.0)
e_tr=ecap(tr); ps_tr=np.clip(m0.predict_proba(tr[F])[:,1],1e-4,1-1e-4)  # in-sample P(S=1|x)

# (1) constant Elkan-Noto rescale: rank-preserving -> recall identical to status quo
c_const=float(ps_tr[tr.y_claims.values==1].mean())
p_const=np.clip(p0/c_const,0,1)
# (2) covariate PU reweighting (Elkan-Noto, SAR): two-copy weighted retrain on claims label
w=np.clip((1-e_tr)/e_tr * ps_tr/(1-ps_tr),0,1)  # P(y=1 | s=0, x)
s=tr.y_claims.values.astype(int)
Xw=pd.concat([tr[F],tr[F]],ignore_index=True)
yw=np.concatenate([np.ones(len(tr)),np.zeros(len(tr))]).astype(int)
sw=np.concatenate([np.where(s==1,1.0,w), np.where(s==1,0.0,1-w)])
keep=sw>1e-6
p_pu=LG().fit(Xw[keep],yw[keep],sample_weight=sw[keep]).predict_proba(te[F])[:,1][anc]
# (3) outcome imputation: soft label z = 1 if s=1 else P(y=1|s=0,x); LGBM regressor
z=np.where(s==1,1.0,w)
p_imp=LGBMRegressor(n_estimators=500,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[F],z).predict(te[F])[anc]
p_imp=np.clip(p_imp,0,1)
# (4) nnPU (Kiryo non-negative risk) MLP on standardized F, claims label as PU
Xtr=torch.tensor(tr[F].astype('float32').values); Xte=torch.tensor(te[F].astype('float32').values)
mu,sdv=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sdv; Xte=(Xte-mu)/sdv
st=torch.tensor(s,dtype=torch.float32); pi=float(yU.mean())  # class prior for true positives
class MLP(nn.Module):
    def __init__(s2,d): super().__init__(); s2.n=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(s2,x): return s2.n(x).squeeze(1)
torch.manual_seed(1); net=MLP(len(F)); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
sig=nn.Sigmoid(); N=len(tr)
for ep in range(30):
    perm=torch.randperm(N)
    for i in range(0,N,4096):
        b=perm[i:i+4096]; opt.zero_grad(); g=net(Xtr[b]); pr=sig(g); sb=st[b]
        lp=sb*(-torch.log(pr+1e-6)); lu=(-torch.log(1-pr+1e-6))
        Rp=pi*(sb*(-torch.log(pr+1e-6))).sum()/(sb.sum()+1e-6)
        Ru=(lu.sum()-pi*(sb*(-torch.log(1-pr+1e-6))).sum()/(sb.sum()+1e-6))/len(b)
        loss=Rp+torch.clamp(Ru,min=0.0); loss.backward(); opt.step()
net.eval()
with torch.no_grad(): p_nn=sig(net(Xte)).numpy()[anc]

rows={
 'status_quo_claims_tabular':p0,
 'pu_reweight_claims':p_pu,
 'nnpu_claims':p_nn,
 'imputation_claims':p_imp,
 'constant_rescale_claims':p_const,
 'adt_label_enriched':p_enr,
}
out={k:{'auroc_vs_TRUE':M(yU,v)[0],'recall10':M(yU,v)[1],'ppv10':M(yU,v)[2],'f1':M(yU,v)[3]} for k,v in rows.items()}
json.dump(out,open('results_labelside.json','w'),indent=2,default=float)
print(json.dumps(out,indent=2,default=float))
