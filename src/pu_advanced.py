import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
rng=np.random.default_rng(20260713); torch.manual_seed(1); np.random.seed(1)
df=pd.read_parquet('dataset_2025-07-01.parquet')
for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1); df['adt_active']=df['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1); paycols=list(pay.columns)
F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
ASC=['claims_lag_days','no_claims','n_spans','adt_active']+paycols
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
anc=te.adt_active.values==1; y=te.loc[anc,'y_union'].values
def metrics(p):
    au=roc_auc_score(y,p); thr=np.quantile(p,0.9); yh=(p>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return round(au,3),round(tp/(tp+fn),3),round(tp/(tp+fp),3),round(f1_score(y,yh),3),int(tp)
R={'M0_claims':None,'M1_union_ceiling':None}
def lg(yt): 
    m=LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1); m.fit(tr[F],yt); return m.predict_proba(te[F])[:,1]
R['M0_claims']=metrics(lg(tr.y_claims)[anc])
R['M1_union_ceiling']=metrics(lg(tr.y_union)[anc])

# ---- (a) nnPU (Kiryo 2017) ----
sc=StandardScaler().fit(tr[F]); Xtr=torch.tensor(sc.transform(tr[F]),dtype=torch.float32); Xte=torch.tensor(sc.transform(te[F]),dtype=torch.float32)
s=torch.tensor(tr.y_claims.values,dtype=torch.float32)  # labeled positives = claims events
def nnpu(prior, epochs=60):
    torch.manual_seed(1)
    net=nn.Sequential(nn.Linear(Xtr.shape[1],128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    P=(s==1); U=(s==0); nP=int(P.sum()); nU=int(U.sum())
    sig=lambda z: torch.sigmoid(-z)  # loss ell(g,+1)=sigmoid(-g); ell(g,-1)=sigmoid(g)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(Xtr.shape[0])
        for i in range(0,len(perm),4096):
            idx=perm[i:i+4096]; g=net(Xtr[idx]).squeeze(1); sb=s[idx]
            gp=g[sb==1]; gu=g[sb==0]
            if len(gp)<2 or len(gu)<2: continue
            Rp_pos=prior*torch.sigmoid(-gp).mean()
            Rp_neg=prior*torch.sigmoid(gp).mean()
            Ru_neg=torch.sigmoid(gu).mean()
            neg=Ru_neg-Rp_neg
            loss = Rp_pos + (neg if neg>=0 else -neg)  # nnPU non-negative correction
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(Xte).squeeze(1).numpy()
for pi in [0.08,0.10,0.12]:
    R[f'nnPU_pi{pi}']=metrics(nnpu(pi)[anc])

# ---- (b) outcome imputation (Elkan-Noto soft labels), whole panel ----
ps_clf=LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[F],tr.y_claims)
ps_tr=np.clip(ps_clf.predict_proba(tr[F])[:,1],1e-4,1-1e-4)
eclf=LogisticRegression(max_iter=2000).fit(tr[(tr.adt_active==1)&(tr.y_union==1)][ASC],tr[(tr.adt_active==1)&(tr.y_union==1)].y_claims)
e_tr=np.clip(eclf.predict_proba(tr[ASC])[:,1],0.05,1.0)
w=(1-e_tr)/e_tr * ps_tr/(1-ps_tr)  # Elkan-Noto P(y=1|s=0,x)
soft=np.where(tr.y_claims.values==1,1.0,np.clip(w,0,1))
reg=LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[F],soft)
R['imputed_softlabel']=metrics(np.clip(reg.predict(te[F])[anc],0,1))

# ---- (c) latent-class (Dawid-Skene, 2 indicators) prevalence among ADT-covered ----
cov=df[df.adt_active==1]
a=cov.y_adt.values if 'y_adt' in cov else None
# claims & ADT indicators
cl=cov.y_claims.values; ad=(cov.y_union.values & ~cov.y_claims.values.astype(bool)).astype(int) | 0
# simple EM 2-test latent class
adt_ind=cov.y_adt.values if 'y_adt' in cov.columns else (cov.y_union.values)
import numpy as np
cl=cov.y_claims.values.astype(float); at=cov.y_adt.values.astype(float) if 'y_adt' in cov.columns else cov.y_union.values.astype(float)
pi=0.3; se_c=0.3; se_a=0.9; sp_c=0.999; sp_a=0.999
for _ in range(200):
    lik1=pi*(se_c**cl*(1-se_c)**(1-cl))*(se_a**at*(1-se_a)**(1-at))
    lik0=(1-pi)*((1-sp_c)**cl*sp_c**(1-cl))*((1-sp_a)**at*sp_a**(1-at))
    post=lik1/(lik1+lik0+1e-12)
    pi=post.mean(); se_c=(post*cl).sum()/post.sum(); se_a=(post*at).sum()/post.sum()
R['latent_class_ADTcovered']={'true_prevalence':round(float(pi),3),'claims_sensitivity':round(float(se_c),3),'adt_sensitivity':round(float(se_a),3)}
print(json.dumps(R,indent=2,default=float))
json.dump(R,open('results_pu_advanced.json','w'),indent=2,default=float)
