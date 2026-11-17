import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],"font.size":7.5})
BLUE="#2a78d6"; ORANGE="#eb6834"; INK="#2c2c2a"; MUTED="#6b6a66"; RULE="#e2e1dc"; BAND="#eef4fc"
# ---------- (1) MASE offline vs deployment
mase={"HASHIRA-CF":(0.859,0.744),"T-learner":(0.863,0.801),"Drift":(1.127,0.842),"Random forest":(0.883,0.864),
      "Gradient boosting":(0.866,1.073),"Linear regression":(0.891,2.696),"Ridge regression":(0.891,5.444),
      "Linear DML":(0.861,17.797),"DR-learner":(0.936,27.724)}
names=list(mase)[::-1]
fig,ax=plt.subplots(figsize=(3.46,2.6),dpi=300)
for i,n in enumerate(names):
    o,d=mase[n]
    if n=="HASHIRA-CF": ax.axhspan(i-0.45,i+0.45,color=BAND,zorder=0)
    ax.plot([o,d],[i,i],color="#b9b8b3",lw=1.0,zorder=1)
    ax.scatter([o],[i],s=22,facecolor="white",edgecolor=BLUE,linewidth=1.3,zorder=3)
    ax.scatter([d],[i],s=22,color=ORANGE,edgecolor="white",linewidth=0.6,zorder=4)
ax.set_xscale("log"); ax.set_xlim(0.6,40)
ax.xaxis.set_major_locator(FixedLocator([0.7,1,2,5,10,30])); ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FixedFormatter(["0.7","1","2","5","10","30"]))
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for t in ax.get_yticklabels():
    if t.get_text()=="HASHIRA-CF": t.set_fontweight("bold")
ax.axvline(1,color="#9a9994",lw=0.6,ls=(0,(2,2)),zorder=0)
ax.set_xlabel("MASE (log scale)")
ax.tick_params(length=2,pad=2)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#9a9994"); ax.tick_params(axis="y",length=0)
ax.grid(axis="x",color=RULE,lw=0.5); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h=[Line2D([],[],marker="o",ls="",markerfacecolor="white",markeredgecolor=BLUE,markeredgewidth=1.3,markersize=4.5,label="Offline"),
   Line2D([],[],marker="o",ls="",color=ORANGE,markersize=4.5,label="Deployment")]
ax.legend(handles=h,frameon=False,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2,handletextpad=0.3,columnspacing=1.0)
fig.tight_layout(pad=0.3); fig.savefig("fig_mase_settings.pdf"); fig.savefig("fig_mase_settings.png",dpi=300); plt.close(fig)
# ---------- (2) scarcity box plot
d=pd.read_csv('../results/scarcity_results.csv')
nm={'HASHIRA-CF':'HASHIRA-CF','tlearner':'T-learner','lineardml':'Linear DML','drlearner':'DR-learner','Drift':'Drift','linear':'Linear regression','ridge':'Ridge regression','rf':'Random forest','gb':'Gradient boosting'}
d['model']=d.model.map(nm); w=d.pivot(index='origin',columns='model',values='mae')
order=w.median().sort_values().index.tolist()[::-1]
fig,ax=plt.subplots(figsize=(3.46,2.45),dpi=300)
rng=np.random.default_rng(1)
for i,n in enumerate(order):
    v=w[n].values; hl=n=="HASHIRA-CF"
    if hl: ax.axhspan(i-0.45,i+0.45,color=BAND,zorder=0)
    q1,med,q3=np.percentile(v,[25,50,75])
    ax.plot([v.min(),v.max()],[i,i],color="#9a9994",lw=0.7,zorder=1)
    ax.add_patch(plt.Rectangle((q1,i-0.28),q3-q1,0.56,facecolor=(BLUE if hl else "#d6d5cf"),edgecolor="none",zorder=2))
    ax.plot([med,med],[i-0.28,i+0.28],color=(INK if not hl else "white"),lw=1.2,zorder=3)
    ax.scatter(v,i+rng.uniform(-0.18,0.18,len(v)),s=3,color=INK,alpha=0.45,linewidths=0,zorder=4)
ax.set_xscale("log"); ax.set_xlim(1.5,150)
ax.xaxis.set_major_locator(FixedLocator([2,3,5,10,20,50,100])); ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FixedFormatter(["2","3","5","10","20","50","100"]))
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
for t in ax.get_yticklabels():
    if t.get_text()=="HASHIRA-CF": t.set_fontweight("bold")
ax.set_xlabel("MAE per window (pp, log scale)")
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#9a9994"); ax.tick_params(axis="y",length=0); ax.tick_params(axis="x",length=2,pad=2)
ax.grid(axis="x",color=RULE,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3); fig.savefig("fig_scarcity.pdf"); fig.savefig("fig_scarcity.png",dpi=300); plt.close(fig)
print('ok', order[::-1][:3])
