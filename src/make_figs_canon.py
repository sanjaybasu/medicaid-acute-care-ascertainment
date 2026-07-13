import warnings, json, numpy as np, pandas as pd, pathlib
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='.'; pathlib.Path(f'{OUT}/figs').mkdir(exist_ok=True)
df=pd.read_parquet(f'{OUT}/dataset_2025-07-01.parquet')
full=json.load(open(f'{OUT}/results_full.json')); rob=json.load(open(f'{OUT}/results_robust.json'))
CB=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9']
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':0.3})

# Fig1 prevalence
fig,ax=plt.subplots(figsize=(4.2,3.4))
v=[df.y_claims.mean(),df.y_adt.mean(),df.y_union.mean()]
ax.bar(['Claims only','ADT only','Claims ∪ ADT'],v,color=[CB[0],CB[1],CB[2]])
for i,x in enumerate(v): ax.text(i,x+0.001,f'{x*100:.1f}%',ha='center')
ax.set_ylabel('90-day acute care rate'); ax.set_title('Figure 1. Outcome ascertainment by source (N=111,660)')
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig1_prevalence.png',dpi=200); plt.close()

# Fig2 capture by payer with Wilson CI
cap=full['capture']; items=[(k,v) for k,v in cap.items() if k!='overall']
items=sorted(items,key=lambda x:x[1][0]); labels=[k for k,_ in items]
est=[v[0]*100 for _,v in items]; lo=[(v[0]-v[1])*100 for _,v in items]; hi=[(v[2]-v[0])*100 for _,v in items]
fig,ax=plt.subplots(figsize=(5.2,3.6))
ax.barh(labels,est,xerr=[lo,hi],color=CB[0],capsize=3)
ax.axvline(cap['overall'][0]*100,ls='--',color=CB[3],label=f"overall {cap['overall'][0]*100:.0f}%")
ax.set_xlabel('Claims capture of acute events, % (95% CI)'); ax.set_title('Figure 2. Capture varies by payer'); ax.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig2_capture_payer.png',dpi=200); plt.close()

# Fig3 capture by lag
pos=df[df.y_union==1].copy()
pos['lag']=pd.cut(pos.claims_lag_days.fillna(9999),[0,30,90,180,1e9],labels=['≤30','31-90','91-180','>180'])
g=pos.groupby('lag').y_claims.mean()*100
fig,ax=plt.subplots(figsize=(4.2,3.4)); ax.bar(g.index.astype(str),g.values,color=CB[4])
for i,x in enumerate(g.values): ax.text(i,x+1,f'{x:.0f}%',ha='center')
ax.set_xlabel('Claims lag (days)'); ax.set_ylabel('Claims capture, %'); ax.set_title('Figure 3. Capture declines with claims lag')
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig3_capture_lag.png',dpi=200); plt.close()

# Fig4 recovery
a=full; obs=a['naive']['obs']; nm=a['naive']['mean_pred']; cm=a['corrected']['mean_pred']
fig,ax=plt.subplots(figsize=(4.2,3.4))
v=[obs,nm,cm]; ax.bar(['Observed\n(truth)','Naive\nclaims','SAR-PU\ncorrected'],v,color=[CB[2],CB[3],CB[0]])
for i,x in enumerate(v): ax.text(i,x+0.005,f'{x*100:.1f}%',ha='center')
ax.set_ylabel('Mean predicted / observed rate'); ax.set_title('Figure 4. Calibration-in-the-large (held-out ADT anchor)')
plt.tight_layout(); plt.savefig(f'{OUT}/figs/fig4_recovery.png',dpi=200); plt.close()
print("figures regenerated")

# ---- CANONICAL NUMBERS ----
lines=[]
A=lambda s:lines.append(s)
A("# CANONICAL NUMBERS (source of truth for manuscript/appendix/repo)\n")
A(f"- Primary cohort N = {len(df):,} (index 2025-07-01, 90-day follow-up)")
A(f"- Plans = 7 Medicaid MCOs across OH, VA, WA")
A(f"- Member 90-day acute rate: claims {df.y_claims.mean()*100:.2f}%, ADT {df.y_adt.mean()*100:.2f}%, union {df.y_union.mean()*100:.2f}%")
A(f"- Capture P(claims|union) overall = {cap['overall'][0]*100:.1f}% [{cap['overall'][1]*100:.1f}, {cap['overall'][2]*100:.1f}] (n={cap['overall'][3]:,})")
for k,v in sorted(items,key=lambda x:x[1][0]):
    A(f"  - {k}: {v[0]*100:.1f}% [{v[1]*100:.1f}, {v[2]*100:.1f}] (n={v[3]})")
A(f"- Capture by claims lag (days): " + ", ".join(f"{i}={x:.1f}%" for i,x in g.items()))
n=full; c=full['corrected']
A(f"- Anchor recovery (held-out ADT, n={full['n_test_anchor']:,}, truth={n['naive']['obs']*100:.1f}%): naive mean {n['naive']['mean_pred']*100:.1f}% -> corrected {c['mean_pred']*100:.1f}%")
A(f"- Discrimination (anchor): AUROC naive {n['naive']['auroc'][0]} [{n['naive']['auroc'][1]},{n['naive']['auroc'][2]}], corrected {c['auroc'][0]} [{c['auroc'][1]},{c['auroc'][2]}]")
A(f"- AUPRC: naive {n['naive']['auprc'][0]}, corrected {c['auprc'][0]}")
A(f"- @top-10%: naive sens {n['naive']['sens'][0]} spec {n['naive']['spec'][0]} PPV {n['naive']['ppv'][0]} NPV {n['naive']['npv'][0]} F1 {n['naive']['f1'][0]}")
A(f"- @top-10%: corrected sens {c['sens'][0]} spec {c['spec'][0]} PPV {c['ppv'][0]} NPV {c['npv'][0]} F1 {c['f1'][0]}")
A(f"- Brier: naive {n['naive']['brier']}, corrected {c['brier']}")
A(f"- Calibration slope/intercept: naive {n['naive']['cal_slope']}/{n['naive']['cal_intercept']}, corrected {c['cal_slope']}/{c['cal_intercept']}")
for r in rob:
    A(f"- Robustness [{r['tag']}]: capture {r['capture_overall'][0]*100:.1f}% [{r['capture_overall'][1]*100:.1f},{r['capture_overall'][2]*100:.1f}]; anchor truth {r['obs']*100:.1f}%, naive {r['naive_mean']*100:.1f}% -> corrected {r['corrected_mean']*100:.1f}%; AUROC {r['naive_auroc']}->{r['corrected_auroc']}; Brier {r['naive_brier']}->{r['corrected_brier']}")
A(f"- e(x) floor sensitivity (corrected mean vs truth {full['obs_for_sens']*100:.1f}%): " + ", ".join(f"{k}={v['mean_corrected']*100:.1f}%" for k,v in full['efloor_sensitivity'].items()))
open(f'{OUT}/CANONICAL_NUMBERS.md','w').write("\n".join(lines)+"\n")
print("wrote CANONICAL_NUMBERS.md"); print("\n".join(lines))
