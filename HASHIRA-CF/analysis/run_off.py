import sys, numpy as np, pandas as pd, logging
sys.argv=['x']
import crfmod as M
M.DEPLOY_MODE=False
logging.basicConfig(level=logging.WARNING)
df_raw=pd.read_csv(sys.argv_path if False else '../data/kanazawa_outdoor_training.csv')
out,mets,meta,df,b=M.train_iot_crf(df_raw,'timestamp',freq='1h',horizon_steps=1)
print('n_model',meta['n_model'],'conf',meta['confounder_count'],'dropped',meta['dropped_collinear_features'])
print('test n',len(out))
ac=meta['actual_col']
F=df.set_index('timestamp')
ts=pd.to_datetime(out['timestamp_t'])
X=F.loc[ts,b['conf_cols']].apply(pd.to_numeric).to_numpy(float)
H=out['Humidity_t_naive'].to_numpy(); T=out['Temperature_t'].to_numpy()
tb=(T>=b['thr']).astype(int); p=b['p_ref']
P={'HASHIRA-CF':out['Humidity_pred_CRFonly'].to_numpy()}
prev=F.loc[ts,'Humidity_lag1'].to_numpy() if 'Humidity_lag1' in F else pd.to_numeric(F['Humidity'],errors='coerce').shift(1).loc[ts].to_numpy()
P['Drift']=np.clip(H+(H-prev),0,100)
for nm,mdl in b['baselines'].items(): P[nm]=np.clip(mdl.predict(X),0,100)
for nm,(est,g) in b['causal_models'].items():
    P[nm]=np.clip(H+g.predict(X)+est.effect(X)*(tb-p),0,100)
P['_naive']=H
R=pd.DataFrame(P); R.insert(0,'timestamp',ts.values); R.insert(1,'actual',out[ac].to_numpy())
R.to_csv('offline_preds.csv',index=False)
for k in P: print(f"{k:12s} MAE {np.abs(R[k]-R.actual).mean():.3f}")
