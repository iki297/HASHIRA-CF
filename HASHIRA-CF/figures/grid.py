import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],"font.size":7})
INK="#2c2c2a"; BLUE="#2a78d6"
R=pd.read_csv('../results/fair_neg40C.csv')
off={'HASHIRA-CF':'HASHIRA-CF','Linear DML':'lineardml','T-learner':'tlearner','Gradient boosting':'gb[X+T]','Random forest':'rf[X+T]',
     'Ridge regression':'ridge[X+T]','Linear regression':'linear[X+T]','DR-learner':'drlearner','Drift':'Drift'}
ev=pd.read_csv('../data/live_eval_10sep.csv',parse_dates=['timestamp'])
ev=ev[(ev.timestamp>='2026-09-04')&(ev.timestamp<'2026-09-11')].dropna(subset=['aktual','sebelum','hum_pred_1h'])
dep={'HASHIRA-CF':'hum_pred_1h','T-learner':'hum_pred_tlearner','Drift':'hum_pred_drift','Random forest':'hum_pred_rf','Gradient boosting':'hum_pred_gb',
     'Linear regression':'hum_pred_lr','Ridge regression':'hum_pred_ridge','Linear DML':'hum_pred_lineardml','DR-learner':'hum_pred_drlearner'}
def grid(y,P,fname,title,hexb,lim):
    fig,axs=plt.subplots(3,3,figsize=(7.0,7.2),dpi=300,sharex=True,sharey=True)
    for ax,(n,p) in zip(axs.flat,P.items()):
        e=p-y; mae=np.abs(e).mean(); r2=1-(e**2).sum()/((y-y.mean())**2).sum()
        cmap="Blues" if n=="HASHIRA-CF" else "Greys"
        if hexb: ax.hexbin(y,p,gridsize=40,cmap=cmap,mincnt=1,bins="log",linewidths=0,extent=(lim[0],lim[1],lim[0],lim[1]))
        else: ax.scatter(y,p,s=6,color=(BLUE if n=="HASHIRA-CF" else "#6b6a66"),alpha=0.7,linewidths=0)
        ax.plot(lim,lim,color=INK,lw=0.7,ls=(0,(3,2)))
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_title(n,fontsize=8,fontweight=("bold" if n=="HASHIRA-CF" else "normal"),pad=3)
        ax.text(0.04,0.96,f"MAE {mae:.3f}\nR$^2$ {r2:.3f}",transform=ax.transAxes,va="top",fontsize=6.5)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        ax.tick_params(length=2,pad=2,labelsize=6.5)
    for ax in axs[-1]: ax.set_xlabel("Observed RH (%)")
    for ax in axs[:,0]: ax.set_ylabel("Forecast RH (%)")
    fig.suptitle(title,fontsize=8.5,x=0.02,ha="left",y=0.995)
    fig.tight_layout(pad=0.4,h_pad=0.8,w_pad=0.6)
    fig.savefig(fname+".png",dpi=300); fig.savefig(fname+".pdf"); plt.close(fig)
y=R.actual.values; grid(y,{k:R[v].values for k,v in off.items()},"fig_scatter_offline","Offline test partition (n = 7,083)",True,(10,100))
y=ev.aktual.values; grid(y,{k:ev[v].values for k,v in dep.items()},"fig_scatter_deployment","Real-time deployment (n = 138)",False,(0,100))
print('ok')
