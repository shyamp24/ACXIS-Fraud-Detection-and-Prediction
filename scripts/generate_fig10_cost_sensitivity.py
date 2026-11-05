"""
Fig. 10 generator (PaySim cost sensitivity).
Usage (from the repository root; arguments are optional):
  python scripts/generate_fig10_cost_sensitivity.py
  python scripts/generate_fig10_cost_sensitivity.py <paysim cache .npz> <exp6 results .json> <output prefix>
Defaults: cache/paysim_stage1_and_shap.npz, results_json/exp6_cost_sensitivity_results.json,
          figures/Fig10_Cost_Sensitivity_Synchronized
Inputs : PaySim validation cache (p_val, y_val, amt_val) and Table 7 results JSON.
Panel (a): validation cost curves for FP-cost multipliers 0.5x/1.0x/1.5x/2.0x with the
           fixed Stage 2 threshold tau* = 0.2179 marked (same threshold as Table 7).
Panel (b): Table 7 tiered-multiplier savings (read from the JSON).
Outputs: <out>.png and <out>.pdf
"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_a = sys.argv[1:4] + [None] * (3 - len(sys.argv[1:4]))
CACHE = _a[0] or os.path.join(ROOT, 'cache', 'paysim_stage1_and_shap.npz')
JSONF = _a[1] or os.path.join(ROOT, 'results_json', 'exp6_cost_sensitivity_results.json')
OUT = _a[2] or os.path.join(ROOT, 'figures', 'Fig10_Cost_Sensitivity_Synchronized')
d = np.load(CACHE, allow_pickle=True)
p, y, a = d['p_val'].astype(float), d['y_val'].astype(int), d['amt_val'].astype(float)
br = [(50_000,500),(100_000,1000),(150_000,1500),(200_000,2000),(500_000,3000),(1_000_000,5000),(float('inf'),10000)]
lim = np.array([b[0] for b in br]); cst = np.array([b[1] for b in br], float)
cfp = cst[np.searchsorted(lim, a, side='left')]
thr = np.linspace(0.01, 0.99, 100)
def curve(m):
    return np.array([a[(y==1)&~(p>=t)].sum() + m*cfp[(y==0)&(p>=t)].sum() for t in thr])
sc = [(0.5,'Low (-50%)','#15803d'),(1.0,'Baseline','#1d4ed8'),(1.5,'High (+50%)','#ea580c'),(2.0,'Very High (+100%)','#b91c1c')]
plt.rcParams['font.family']='serif'
fig,(ax,bx)=plt.subplots(1,2,figsize=(12.9,4.9),dpi=300,gridspec_kw={'width_ratios':[1,1.05]})
TAU=0.2179
j=int(np.argmin(np.abs(thr-TAU)))
ax.axvline(thr[j],color='black',ls='--',lw=1.3,zorder=1)
for m,n,c in sc:
    cu=curve(m)
    ax.plot(thr,cu,color=c,lw=2.2,label=n)
    ax.scatter([thr[j]],[cu[j]],s=60,color=c,edgecolor='black',zorder=6)
ax.text(thr[j]+0.012,15.5e6,'fixed \u03c4* = 0.2179\n(Stage 2, baseline schedule)',fontsize=8.5,va='top',ha='left')
ax.set_xlim(0,0.92); ax.set_ylim(0,25e6)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:'₹%dM'%(v/1e6) if v else '₹0M'))
ax.set_xlabel('Classification Decision Threshold (τ)',fontweight='bold')
ax.set_ylabel('Validation Misclassification Cost (₹ Millions)',fontweight='bold')
ax.set_title('(a) Cost Curves Across FP Investigation Multipliers',fontweight='bold',fontsize=11)
ax.grid(True,ls='--',alpha=.5); ax.legend(loc='upper left',bbox_to_anchor=(0.30,0.99),fontsize=9)
# panel (b) from results json (Table 7 tiered rows)
sc7=[s for s in json.load(open(JSONF))['paysim']['scenarios'] if s['category']=='Tiered Multiplier']
pct=[s['pct'] for s in sc7]; sav=[s['savings']/1e3 for s in sc7]
lab=['Low (-50%)\n[Mult = 0.5×]','Baseline\n[Mult = 1.0×]','High (+50%)\n[Mult = 1.5×]','Very High (+100%)\n[Mult = 2.0×]']
x=np.arange(4); w=0.36
b1=bx.bar(x-w/2,pct,w,color='#16a34a',edgecolor='black'); bx.set_ylim(0,4.5)
bx.set_ylabel('Net Financial Cost Savings (%)',color='#15803d',fontweight='bold'); bx.set_xticks(x); bx.set_xticklabels(lab,fontsize=8.5,fontweight='bold')
for r,v in zip(b1,pct): bx.text(r.get_x()+r.get_width()/2,v+0.06,f'+{v:.2f}%',ha='center',color='#15803d',fontweight='bold',fontsize=9)
bx2=bx.twinx(); b2=bx2.bar(x+w/2,sav,w,color='#2563eb',edgecolor='black'); bx2.set_ylim(0,140)
bx2.set_ylabel('Absolute Currency Savings (₹ Thousands)',color='#1d4ed8',fontweight='bold')
for r,v in zip(b2,sav): bx2.text(r.get_x()+r.get_width()/2,v+2,f'₹{v:.1f}k',ha='center',color='#1d4ed8',fontweight='bold',fontsize=9)
bx.set_title('(b) Net Savings Across Operational Cost Schedules (Table 7)',fontweight='bold',fontsize=11); bx.grid(True,axis='y',ls='--',alpha=.5)
fig.tight_layout(); fig.savefig(OUT+'.png'); fig.savefig(OUT+'.pdf')
