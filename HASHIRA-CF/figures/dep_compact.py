import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],"font.size":7})
INK="#2c2c2a"; BLUE="#2a78d6"
ev=pd.read_csv('../data/live_eval_10sep.csv',parse_dates=['timestamp'])
ev=ev[(ev.timestamp>='2026-09-04')&(ev.timestamp<'2026-09-11')].dropna(subset=['aktual','sebelum','hum_pred_1h'])
dep={'HASHIRA-CF':'hum_pred_1h','T-learner':'hum_pred_tlearner','Drift':'hum_pred_drift','Random forest':'hum_pred_rf','Gradient boosting':'hum_pred_gb',
     'Linear regression':'hum_pred_lr','Ridge regression':'hum_pred_ridge','Linear DML':'hum_pred_lineardml','DR-learner':'hum_pred_drlearner'}
y=ev.aktual.values
fig,axs=plt.subplots(2,5,figsize=(7.0,3.2),dpi=300,sharex=True,sharey=True)
for ax,(n,c) in zip(axs.flat,dep.items()):
    p=ev[c].values; mae=np.abs(p-y).mean()
    ax.plot([0,100],[0,100],color=INK,lw=0.6,ls=(0,(3,2)),zorder=1)
    ax.scatter(y,p,s=5,color=(BLUE if n=="HASHIRA-CF" else "#6b6a66"),alpha=0.75,linewidths=0,zorder=2,clip_on=False)
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect("equal")
    ax.set_xticks([0,50,100]); ax.set_yticks([0,50,100])
    ax.set_title(n,fontsize=7.5,fontweight=("bold" if n=="HASHIRA-CF" else "normal"),pad=2)
    ax.text(0.05,0.95,f"MAE {mae:.2f}",transform=ax.transAxes,va="top",fontsize=6.5)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2,pad=1.5,labelsize=6.5)
axs.flat[-1].axis("off")
for ax in axs[1,:4]: ax.set_xlabel("Observed RH (%)",fontsize=7)
axs[0,4].tick_params(labelbottom=True); axs[0,4].set_xlabel("Observed RH (%)",fontsize=7)
for ax in axs[:,0]: ax.set_ylabel("Forecast RH (%)",fontsize=7)
fig.tight_layout(pad=0.3,h_pad=0.6,w_pad=0.4,rect=(0,0,1,0.97))
fig.savefig("fig_deployment_scatter.pdf"); fig.savefig("fig_deployment_scatter.png",dpi=300)
