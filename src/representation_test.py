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
BASE=['age','n_office','n_urgent','n_ed','n_inpat','n_outp','n_total','n_fac','n_prov','claims_lag_days','no_claims','n_spans']+paycols
# enriched
enr=pd.read_parquet('_enriched_features.parquet')
df=df.merge(enr,on='person_id',how='left')
ENRcols=['n_dx_cat','tot_paid','dx_diabetes','dx_chf','dx_resp','dx_ckd','dx_bh','n_rx','n_rx_distinct']
for c in ENRcols: df[c]=df[c].fillna(0)
tr,te=train_test_split(df,test_size=0.3,random_state=42,stratify=df.y_union)
anc=te.adt_active.values==1; y=te.loc[anc,'y_union'].values
def M(p):
    au=roc_auc_score(y,p); thr=np.quantile(p,0.9); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return [round(au,3),round(tp/(tp+fn),3),round(tp/(tp+fp),3),round(f1_score(y,yh),3)]
def lg(feats):
    m=LGBMClassifier(n_estimators=500,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[feats],tr.y_union)
    return m.predict_proba(te[feats])[:,1][anc]
R={}
R['R0_basic']=M(lg(BASE))
R['R1_enriched']=M(lg(BASE+ENRcols))

# ---- R2 sequence model (GRU over event tokens) ----
seq=pd.read_parquet('_event_sequences.parquet')
vocab={t:i+1 for i,t in enumerate(sorted(seq.etype.unique()))}  # 0=pad
seq['tok']=seq.etype.map(vocab)
grp=seq.groupby('person_id').agg(toks=('tok',list),wks=('wk',list))
MAXLEN=48
def pad(l,fill=0):
    l=l[-MAXLEN:]; return [fill]*(MAXLEN-len(l))+list(l)
tokmap=grp.toks.apply(lambda l: pad(l)).to_dict(); wkmap=grp.wks.apply(lambda l: pad(l,55)).to_dict()
def seqarr(dfx):
    T=np.array([tokmap.get(p,[0]*MAXLEN) for p in dfx.person_id],dtype=np.int64)
    W=np.array([wkmap.get(p,[55]*MAXLEN) for p in dfx.person_id],dtype=np.int64)
    return torch.tensor(T),torch.tensor(W)
Ttr,Wtr=seqarr(tr); Tte,Wte=seqarr(te)
ytr=torch.tensor(tr.y_union.values,dtype=torch.float32)
Xbtr=torch.tensor(tr[BASE].astype('float32').values,dtype=torch.float32); Xbte=torch.tensor(te[BASE].astype('float32').values,dtype=torch.float32)
mu,sd=Xbtr.mean(0),Xbtr.std(0)+1e-6; Xbtr=(Xbtr-mu)/sd; Xbte=(Xbte-mu)/sd
class SeqNet(nn.Module):
    def __init__(s,V,nb):
        super().__init__(); s.emb=nn.Embedding(V+1,32,padding_idx=0); s.pos=nn.Embedding(56,32)
        s.gru=nn.GRU(32,64,batch_first=True); s.head=nn.Sequential(nn.Linear(64+nb,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(s,T,W,Xb):
        e=s.emb(T)+s.pos(W); o,_=s.gru(e); h=o[:,-1,:]; return s.head(torch.cat([h,Xb],1)).squeeze(1)
pos_w=torch.tensor([(ytr==0).sum()/(ytr==1).sum()])
lossf=nn.BCEWithLogitsLoss(pos_weight=pos_w); N=Ttr.shape[0]
def train_seq(seed):
    torch.manual_seed(seed); net=SeqNet(len(vocab),len(BASE)); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
    for ep in range(20):
        net.train(); perm=torch.randperm(N)
        for i in range(0,N,2048):
            b=perm[i:i+2048]; opt.zero_grad(); out=net(Ttr[b],Wtr[b],Xbtr[b]); lossf(out,ytr[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(Tte,Wte,Xbte)).numpy()
seqpreds=np.mean([train_seq(sd) for sd in [1,2,3]],axis=0)
pseq=seqpreds[anc]
# store R0/R1 preds too for CIs
p0=None
import lightgbm as lgbm
def lgpred(feats):
    m=LGBMClassifier(n_estimators=500,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,random_state=1,n_jobs=-1,verbose=-1).fit(tr[feats],tr.y_union)
    return m.predict_proba(te[feats])[:,1][anc]
pr0=lgpred(BASE); pr1=lgpred(BASE+ENRcols)
R['R2_sequence_GRU_3seed']=M(pseq)
def rec(y,p): thr=np.quantile(p,0.9); yh=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel(); return tp/(tp+fn)
def paired(pa,pb,fn,n=1000):
    idx=np.arange(len(y)); d=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),True)
        if y[b].sum()<3: continue
        d.append(fn(y[b],pb[b])-fn(y[b],pa[b]))
    return [round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)]
R['delta_R2_vs_R0']={'d_auroc':round(roc_auc_score(y,pseq)-roc_auc_score(y,pr0),3),
  'd_auroc_ci':paired(pr0,pseq,lambda a,b:roc_auc_score(a,b)),
  'd_recall':round(rec(y,pseq)-rec(y,pr0),3),'d_recall_ci':paired(pr0,pseq,rec)}
R['delta_R1_vs_R0']={'d_auroc':round(roc_auc_score(y,pr1)-roc_auc_score(y,pr0),3),
  'd_auroc_ci':paired(pr0,pr1,lambda a,b:roc_auc_score(a,b)),
  'd_recall':round(rec(y,pr1)-rec(y,pr0),3),'d_recall_ci':paired(pr0,pr1,rec)}
R['_legend']='[AUROC, recall@10%, PPV@10%, F1@10%]; label=ADT-completed union; eval vs true events on ADT anchor; R2=3-seed avg'
print(json.dumps(R,indent=2))
json.dump(R,open('results_representation.json','w'),indent=2)
