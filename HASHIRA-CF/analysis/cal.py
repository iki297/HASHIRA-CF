import sys, numpy as np, pandas as pd, warnings, importlib
warnings.filterwarnings('ignore'); sys.argv=['x']; sys.path.insert(0,'.')
def run(modname, path):
    M=importlib.import_module(modname); M.DEPLOY_MODE=False
    out,mets,meta,df,b=M.train_iot_crf(pd.read_csv(path),'timestamp',freq='1h',horizon_steps=1)
    F=df.set_index('timestamp'); ts=pd.to_datetime(out['timestamp_t'])
    X=F.loc[ts,b['conf_cols']].apply(pd.to_numeric).to_numpy(float)
    H=out['Humidity_t_naive'].to_numpy(); tb=(out['Temperature_t'].to_numpy()>=b['thr']).astype(int); p=b['p_ref']
    y=out[meta['actual_col']].to_numpy()
    P={'HASHIRA-CF':out['Humidity_pred_CRFonly'].to_numpy()}
    for nm,mdl in b['baselines'].items(): P[nm]=np.clip(mdl.predict(X),0,100)
    for nm,(est,g) in b['causal_models'].items(): P[nm]=np.clip(H+g.predict(X)+est.effect(X)*(tb-p),0,100)
    return {k:np.abs(v-y).mean() for k,v in P.items()}, meta['confounder_count'], len(y), ts.min(), ts.max()
path='../data/20260902_telemetry_training.csv'
a=run('crfmod',path); b=run('crfmod_nocal',path)
print('with cal: conf',a[1],'test n',a[2],a[3],'->',a[4]); print('no cal:   conf',b[1])
for k in a[0]:
    w,n=a[0][k],b[0][k]; print(f"{k:11s} with {w:7.3f}  without {n:7.3f}  change {n-w:+7.3f} ({(n-w)/w*100:+.0f}%)")
