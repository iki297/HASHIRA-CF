import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
R=pickle.load(open('replay.pkl','rb')); B=pickle.load(open('replay_bundles.pkl','rb'))
ev=pd.read_csv('../data/live_eval_10sep.csv',parse_dates=['timestamp'])
ev=ev[(ev.timestamp>='2026-09-04')&(ev.timestamp<'2026-09-11')].dropna(subset=['aktual','sebelum','hum_pred_1h']).reset_index(drop=True)
y=ev.aktual.values; Hn=ev.sebelum.values; s=2.1968
def dm(e1,e2,L=4):
    d=np.abs(e1)-np.abs(e2); n=len(d); m=d.mean(); dc=d-m; v=(dc**2).mean()
    for k in range(1,L+1): v+=2*(1-k/(L+1))*(dc[k:]*dc[:-k]).mean()
    return 2*stats.t.sf(abs(m/np.sqrt(v/n)),n-1)
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); a=np.empty(m); r=0
    for i,j in enumerate(o): r=max(r,(m-i)*p[j]); a[j]=min(1,r)
    return a
def table(P,title):
    print('\n== '+title)
    eH=P['HASHIRA-CF']-y; ks=[k for k in P if k!='HASHIRA-CF']; ph=dict(zip(ks,holm([dm(eH,P[k]-y) for k in ks])))
    for k in sorted(P,key=lambda k:np.abs(P[k]-y).mean()):
        e=P[k]-y; mae=np.abs(e).mean(); r2=1-(e**2).sum()/((y-y.mean())**2).sum()
        w='' if k=='HASHIRA-CF' else f" W {stats.wilcoxon(np.abs(eH),np.abs(e)).pvalue:.3g}"
        print(f"  {k:18s} MAE {mae:7.3f} RMSE {np.sqrt((e**2).mean()):7.3f} MASE {mae/s:6.3f} R2 {r2:8.3f}  p {ph.get(k,np.nan):.3g}{w}")
def fit_base(mod,var):
    d=R[mod]; X,Tc,Ha,Ya,sv=d['X'],d['Tc'],d['Ha'],d['Ya'],d['sv']; thr=B[mod]['thr']
    Tb=(Tc>=thr).astype(float); Ylvl=Ha[:sv]+Ya[:sv]
    xr=np.array([r['x'] for r in d['rows']]); ct=np.array([r['cur_temp'] for r in d['rows']])
    if var=='X': Xa,xa=X,xr
    elif var=='X+T': Xa,xa=np.column_stack([X,Tb]),np.column_stack([xr,(ct>=thr).astype(float)])
    else: Xa,xa=np.column_stack([X,Tc]),np.column_stack([xr,ct])
    out={}
    for nm,mdl in [('Linear regression',LinearRegression()),('Ridge regression',RidgeCV(alphas=np.logspace(-3,3,13))),
                   ('Random forest',RandomForestRegressor(n_estimators=500,min_samples_leaf=3,random_state=42,n_jobs=-1)),
                   ('Gradient boosting',HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,random_state=42))]:
        mdl.fit(Xa[:sv],Ylvl); out[nm]=np.clip(mdl.predict(xa),0,100)
    return out
def causal(mod):
    rr=R[mod]['rows']; g=lambda k: np.array([r['bp_'+k] for r in rr])
    return {'HASHIRA-CF':np.array([r['hum'] for r in rr]),'T-learner':g('tlearner'),'Linear DML':g('lineardml'),'DR-learner':g('drlearner'),'Drift':g('drift')}
live={'HASHIRA-CF':ev.hum_pred_1h.values,'T-learner':ev.hum_pred_tlearner.values,'Linear DML':ev.hum_pred_lineardml.values,'DR-learner':ev.hum_pred_drlearner.values,'Drift':ev.hum_pred_drift.values,
      'Linear regression':ev.hum_pred_lr.values,'Ridge regression':ev.hum_pred_ridge.values,'Random forest':ev.hum_pred_rf.values,'Gradient boosting':ev.hum_pred_gb.values}
table(live,'LIVE (as in manuscript)')
P=causal('crfmod'); P.update(fit_base('crfmod','X')); table(P,'REPLAY, baselines on X (fidelity reference)')
P=causal('crfmod'); P.update(fit_base('crfmod','X+T')); table(P,'REPLAY, baselines on X+T  [A1 parity]'); PT=P
P=causal('crfmod'); P.update(fit_base('crfmod','X+Temp')); table(P,'REPLAY, baselines on X+Temp  [A1 sensitivity]')
P=causal('crfmod_nocal'); P.update(fit_base('crfmod_nocal','X+T')); table(P,'REPLAY NO CALENDAR, baselines on X+T  [A4]'); PN=P
pickle.dump({'PT':PT,'PN':PN,'live':live},open('dep_preds.pkl','wb'))
