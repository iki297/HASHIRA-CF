import pandas as pd, numpy as np
from scipy import stats
def dm(e1,e2,L=0):
    d=np.abs(e1)-np.abs(e2); n=len(d); m=d.mean(); dc=d-m
    v=(dc**2).mean()
    for k in range(1,L+1): v+=2*(1-k/(L+1))*(dc[k:]*dc[:-k]).mean()
    t=m/np.sqrt(v/n); return 2*stats.t.sf(abs(t),n-1)
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); a=np.empty(m); r=0
    for i,j in enumerate(o): r=max(r,(m-i)*p[j]); a[j]=min(1,r)
    return a
def metr(y,p,s):
    e=p-y; mae=np.abs(e).mean()
    return mae, np.sqrt((e**2).mean()), mae/s, 1-(e**2).sum()/((y-y.mean())**2).sum()
NAMES=['HASHIRA-CF','T-learner','Linear DML','DR-learner','Gradient boosting','Random forest','Linear regression','Ridge regression','Drift']
# ---- scales: mean |dH| over FITTING observations
def scale(path, nfit):
    df=pd.read_csv(path); df['timestamp']=pd.to_datetime(df.timestamp,format='mixed')
    for c in ['UV','Temperature','PM2_5','PM10','Humidity','BarPressure']: df[c]=pd.to_numeric(df[c],errors='coerce')
    r=(df.UV>=0)&df.Temperature.between(0,60)&(df.PM2_5>=0)&(df.PM10>=0)&df.Humidity.between(0,100)&(df.BarPressure>0)
    o=df[r].sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')[['UV','Temperature','PM2_5','PM10','Humidity','BarPressure']].resample('1h').mean().interpolate(method='linear',limit=3,limit_direction='both')
    for c in ['Temperature','UV','Humidity']:
        for L in (1,2,3): o[f'{c}_l{L}']=o[c].shift(L)
    Y=o.Humidity.shift(-1)-o.Humidity; m=o.notna().all(axis=1)&Y.notna(); y=Y[m]
    return len(y), y.iloc[:nfit].abs().mean()
n_off,s_off=scale('../data/kanazawa_outdoor_training.csv',int(0.8*int(0.8*34874)))
n_dep,s_dep=scale('../data/20260902_telemetry_training.csv',268)
print(f"scales: offline model-ready {n_off}, fit {int(0.8*int(0.8*34874))}, s_off={s_off:.4f}; deploy model-ready {n_dep}, fit 268, s_dep={s_dep:.4f}")
# ---- offline
O=pd.read_csv('offline_preds.csv'); yo=O.actual.values
om={'HASHIRA-CF':'HASHIRA-CF','T-learner':'tlearner','Linear DML':'lineardml','DR-learner':'drlearner','Gradient boosting':'gb','Random forest':'rf','Linear regression':'linear','Ridge regression':'ridge','Drift':'Drift'}
# ---- deployment
d=pd.read_csv('../data/live_eval_10sep.csv',parse_dates=['timestamp'])
d=d[(d.timestamp>='2026-09-04')&(d.timestamp<'2026-09-11')].dropna(subset=['aktual','sebelum','hum_pred_1h']).reset_index(drop=True)
yd=d.aktual.values; Hd=d.sebelum.values
dmap={'HASHIRA-CF':'hum_pred_1h','T-learner':'hum_pred_tlearner','Linear DML':'hum_pred_lineardml','DR-learner':'hum_pred_drlearner','Gradient boosting':'hum_pred_gb','Random forest':'hum_pred_rf','Linear regression':'hum_pred_lr','Ridge regression':'hum_pred_ridge','Drift':'hum_pred_drift'}
for label,y,P,s in [('OFFLINE',yo,{k:O[v].values for k,v in om.items()},s_off),('DEPLOYMENT',yd,{k:d[v].values for k,v in dmap.items()},s_dep)]:
    print(f"\n==== {label} (n={len(y)})")
    for k in sorted(NAMES,key=lambda k:np.abs(P[k]-y).mean()):
        print(f"  {k:18s} MAE {metr(y,P[k],s)[0]:7.3f} RMSE {metr(y,P[k],s)[1]:7.3f} MASE {metr(y,P[k],s)[2]:7.3f} R2 {metr(y,P[k],s)[3]:8.3f}")
    eH=P['HASHIRA-CF']-y; ks=[k for k in NAMES if k!='HASHIRA-CF']
    nom=[dm(eH,P[k]-y,0) for k in ks]; nw=[dm(eH,P[k]-y,4) for k in ks]
    hn=holm(nom); hw=holm(nw)
    for k,a,b,c,e in zip(ks,nom,hn,nw,hw):
        w=stats.wilcoxon(np.abs(eH),np.abs(P[k]-y)).pvalue
        print(f"  vs {k:18s} DM nom {a:.3g} Holm {b:.3g} | NW4 {c:.3g} Holm {e:.3g} | Wilc {w:.3g}  medAE diff {np.median(np.abs(eH))-np.median(np.abs(P[k]-y)):+.3f}")
    dc=np.abs(eH)-np.abs(P['T-learner']-y); print('  r1 loss diff vs T:',round(pd.Series(dc).autocorr(1),3))
# ---- regime realised
rc_o=np.abs(yo-O._naive.values); rc_d=np.abs(yd-Hd)
print('\nOffline band counts',[(rc_o<2).sum(),((rc_o>=2)&(rc_o<5)).sum(),(rc_o>=5).sum()],' deploy',[(rc_d<2).sum(),(rc_d>=2).sum(),(rc_d>=5).sum()])
for k in NAMES:
    eo=np.abs(O[om[k]].values-yo); ed=np.abs(d[dmap[k]].values-yd)
    print(f"  {k:18s} off {eo[rc_o<2].mean():5.2f} {eo[(rc_o>=2)&(rc_o<5)].mean():5.2f} {eo[rc_o>=5].mean():5.2f} | dep {ed[rc_d<2].mean():5.2f} {ed[rc_d>=2].mean():5.2f}")
# ---- ex-ante
up=np.abs(d.hum_pred_1h.values-Hd)>=2
print('\nEx-ante routine/upper n',(~up).sum(),up.sum())
eH=d.hum_pred_1h.values-yd
for nm,mask in [('routine',~up),('upper',up)]:
    print(' ',nm, ' | '.join(f"{k} {np.abs(d[v].values-yd)[mask].mean():.2f}" for k,v in sorted(dmap.items(),key=lambda kv:np.abs(d[kv[1]].values-yd)[mask].mean())))
    for k,v in dmap.items():
        if k=='HASHIRA-CF':continue
        e=d[v].values-yd; p=dm(eH[mask],e[mask],0); w=stats.wilcoxon(np.abs(eH[mask]),np.abs(e[mask])).pvalue
        print(f"     vs {k:18s} DM {p:.3g}  W {w:.3g}")
print('\nSignals >=2pp:',{k:int((np.abs(d[v].values-Hd)>=2).sum()) for k,v in dmap.items()})
