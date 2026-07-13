import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
rng=np.random.default_rng(20260713); torch.manual_seed(1); np.random.seed(1)
df=pd.read_parquet('dataset_2025-07-01.parquet')
for c in ['n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov']: df[c]=df[c].fillna(0)
df['no_claims']=df['claims_lag_days'].isna().astype(int); df['claims_lag_days']=df['claims_lag_days'].fillna(9999)
df['age']=df['age'].fillna(df['age'].median()); df['n_spans']=df['n_spans'].fillna(1); df['adt_active']=df['adt_active'].fillna(0).astype(int)
pay=pd.get_dummies(df['payer'].fillna('UNK'),prefix='pay'); df=pd.concat([df,pay],axis=1); paycols=list(pay.columns)
F=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
anc=te.adt_active.values==1; yU=te.loc[anc,'y_union'].values; yC=te.loc[anc,'y_claims'].values
def lgp(label): 
    m=LGBMClassifier(n_estimators=500,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[F],label); return m.predict_proba(te[F])[:,1][anc]
p0=lgp(tr.y_claims); p1=lgp(tr.y_union)
# sequence model (3-seed) on ADT-completed label
seq=pd.read_parquet('_event_sequences.parquet'); vocab={t:i+1 for i,t in enumerate(sorted(seq.etype.unique()))}
seq['tok']=seq.etype.map(vocab); grp=seq.groupby('person_id').agg(toks=('tok',list),wks=('wk',list)); ML=48
padt=lambda l,f=0:[f]*(ML-len(l[-ML:]))+list(l[-ML:])
tm=grp.toks.apply(lambda l:padt(l)).to_dict(); wm=grp.wks.apply(lambda l:padt(l,55)).to_dict()
def arr(d):
    T=np.array([tm.get(p,[0]*ML) for p in d.person_id]); W=np.array([wm.get(p,[55]*ML) for p in d.person_id]); return torch.tensor(T),torch.tensor(W)
Ttr,Wtr=arr(tr); Tte,Wte=arr(te); ytr=torch.tensor(tr.y_union.values,dtype=torch.float32)
Xb=torch.tensor(tr[F].astype('float32').values); Xbte=torch.tensor(te[F].astype('float32').values); mu,sd=Xb.mean(0),Xb.std(0)+1e-6; Xb=(Xb-mu)/sd; Xbte=(Xbte-mu)/sd
class Net(nn.Module):
    def __init__(s,V,nb): super().__init__(); s.e=nn.Embedding(V+1,32,padding_idx=0); s.p=nn.Embedding(56,32); s.g=nn.GRU(32,64,batch_first=True); s.h=nn.Sequential(nn.Linear(64+nb,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(s,T,W,X): o,_=s.g(s.e(T)+s.p(W)); return s.h(torch.cat([o[:,-1,:],X],1)).squeeze(1)
pw=torch.tensor([(ytr==0).sum()/(ytr==1).sum()]); lf=nn.BCEWithLogitsLoss(pos_weight=pw); N=Ttr.shape[0]
def trainseq(sd):
    torch.manual_seed(sd); net=Net(len(vocab),len(F)); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
    for ep in range(20):
        pm=torch.randperm(N)
        for i in range(0,N,2048): b=pm[i:i+2048]; opt.zero_grad(); lf(net(Ttr[b],Wtr[b],Xb[b]),ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(Tte,Wte,Xbte)).numpy()[anc]
p2=np.mean([trainseq(s) for s in [1,2,3]],axis=0)
def M(y,p):
    au=roc_auc_score(y,p); thr=np.quantile(p,0.9); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return au,tp/(tp+fn),(tp/(tp+fp) if tp+fp else np.nan),f1_score(y,yh)
def ci(y,p,fn,n=1000):
    idx=np.arange(len(y)); v=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),True)
        if y[b].sum()<3: continue
        v.append(fn(y[b],p[b]))
    return [round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]
R={'n_anchor':int(anc.sum())}
for nm,p in [('status_quo_claims_tabular',p0),('adt_label_tabular',p1),('adt_label_sequence',p2)]:
    au,rc,pp,f1=M(yU,p)
    R[nm]={'auroc_vs_TRUE':round(au,3),'auroc_vs_TRUE_ci':ci(yU,p,lambda a,b:roc_auc_score(a,b)),
           'auroc_vs_CLAIMS':round(roc_auc_score(yC,p),3),
           'recall10':round(rc,3),'recall10_ci':ci(yU,p,lambda a,b:M(a,b)[1]),
           'ppv10':round(pp,3),'f1':round(f1,3)}
json.dump(R,open('results_ladder_ci.json','w'),indent=2,default=float)
print(json.dumps(R,indent=2,default=float))
