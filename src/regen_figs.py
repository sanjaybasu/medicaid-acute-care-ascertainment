import warnings, json, numpy as np, pandas as pd, pathlib
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
OUT='/Users/sanjaybasu/notebooks/pu-underascertainment'; pathlib.Path(f'{OUT}/figs').mkdir(exist_ok=True)
df=pd.read_parquet(f'{OUT}/dataset_2025-07-01.parquet'); df['adt_active']=df['adt_active'].fillna(0).astype(int)
full=json.load(open(f'{OUT}/results_full.json'))
CB=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#999999']
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':0.3,'axes.axisbelow':True})

# Fig1 prevalence
fig,ax=plt.subplots(figsize=(4.4,3.6),constrained_layout=True)
v=[df.y_claims.mean()*100,df.y_adt.mean()*100,df.y_union.mean()*100]
b=ax.bar(['Claims\nonly','ADT\nonly','Claims or ADT\n(union)'],v,color=[CB[0],CB[1],CB[2]],width=0.6)
ax.set_ylim(0,max(v)*1.20)
for i,x in enumerate(v): ax.text(i,x+max(v)*0.02,f'{x:.1f}%',ha='center',va='bottom',fontsize=10)
ax.set_ylabel('90-day acute care rate (%)'); ax.set_title('Figure 1. Acute care rate by data source')
plt.savefig(f'{OUT}/figs/fig1_prevalence.png',dpi=200); plt.close()

# Fig2 capture by plan (anonymized, canonical Table 1 values for exact consistency)
pos=df[df.y_union==1]
CANON=[("A",12.6,11.4,13.9),("B",12.8,10.7,15.1),("C",13.8,11.4,16.8),("D",15.1,12.8,17.7),
       ("E",24.3,20.8,28.1),("F",24.5,22.5,26.6),("G",99.7,99.1,99.9)]
labels=[]; est=[]; lerr=[]; herr=[]
for L,e,lo,hi in CANON:
    lab=f'Plan {L}\n(no ADT feed)' if e>90 else f'Plan {L}'
    labels.append(lab); est.append(e); lerr.append(e-lo); herr.append(hi-e)
overall=14.8
fig,ax=plt.subplots(figsize=(5.6,3.8),constrained_layout=True)
colors=[CB[6] if e>90 else CB[0] for e in est]
y=np.arange(len(labels))
ax.barh(y,est,xerr=[lerr,herr],color=colors,capsize=3,height=0.62)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(0,112)
for i,e in enumerate(est): ax.text(min(e+max(herr[i],2)+3,104),i,f'{e:.0f}%',va='center',ha='left',fontsize=9)
ax.axvline(overall,ls='--',color=CB[3],lw=1.3,label=f'ADT-covered overall {overall:.0f}%')
ax.set_xlabel('Claims capture of acute care events (%, 95% CI)')
ax.set_title('Figure 2. Claims capture by plan'); ax.legend(loc='lower right',framealpha=0.9)
plt.savefig(f'{OUT}/figs/fig2_capture_payer.png',dpi=200); plt.close()

# Fig3 capture by lag
pos['lag']=pd.cut(pos.claims_lag_days.fillna(9999),[0,30,90,180,1e9],labels=['≤30','31-90','91-180','>180'])
g=pos.groupby('lag').y_claims.mean()*100
fig,ax=plt.subplots(figsize=(4.4,3.6),constrained_layout=True)
ax.bar(g.index.astype(str),g.values,color=CB[4],width=0.6); ax.set_ylim(0,max(g.values)*1.20)
for i,x in enumerate(g.values): ax.text(i,x+max(g.values)*0.02,f'{x:.0f}%',ha='center',va='bottom',fontsize=10)
ax.set_xlabel('Claims lag (days)'); ax.set_ylabel('Claims capture (%)'); ax.set_title('Figure 3. Capture by claims lag')
plt.savefig(f'{OUT}/figs/fig3_capture_lag.png',dpi=200); plt.close()

# Fig4 recovery
obs=full['obs']*100; nm=full['naive']['mean_pred'][0]*100; cm=full['corrected']['mean_pred'][0]*100
fig,ax=plt.subplots(figsize=(4.4,3.6),constrained_layout=True)
v=[obs,nm,cm]; ax.bar(['Observed\n(union)','Naive\nclaims','Corrected\n(PU)'],v,color=[CB[2],CB[3],CB[0]],width=0.6)
ax.set_ylim(0,max(v)*1.20)
for i,x in enumerate(v): ax.text(i,x+max(v)*0.02,f'{x:.1f}%',ha='center',va='bottom',fontsize=10)
ax.set_ylabel('Mean predicted / observed rate (%)'); ax.set_title('Figure 4. Calibration on held-out ADT anchor')
plt.savefig(f'{OUT}/figs/fig4_recovery.png',dpi=200); plt.close()
print("regenerated 4 figures (constrained layout, headroom, no overlapping annotation)")
print("recovery obs/naive/corrected:",round(obs,1),round(nm,1),round(cm,1))
