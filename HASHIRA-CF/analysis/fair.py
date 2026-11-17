import sys, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.argv=['x']; sys.path.insert(0,'.')
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
modname=sys.argv_mod if False else None
def run(modname, tag):
    M=importlib.import_module(modname); M.DEPLOY_MODE=False
    out,mets,meta,df,b=M.train_iot_crf(pd.read_csv('../data/kanazawa_outdoor_training.csv'),'timestamp',freq='1h',horizon_steps=1)
    cc=b['conf_cols']; X_df=df[cc].apply(pd.to_numeric,errors='coerce')
    T=pd.to_numeric(df['Temperature'],errors='coerce'); H=pd.to_numeric(df['Humidity'],errors='coerce'); Y=H.shift(-1)-H
    m=(X_df.notna().all(axis=1)&T.notna()&Y.notna()&H.notna()).to_numpy()
    X=X_df[m].to_numpy(float); Tc=T[m].to_numpy(float); Ha=H[m].to_numpy(float); Ya=Y[m].to_numpy(float); ts=df.loc[m,'timestamp'].to_numpy()
    n=len(X); sp=int(0.8*n); sv=int(0.8*sp)
    assert (pd.to_datetime(ts[sp:])==pd.to_datetime(out['timestamp_t']).values).all()
    Tb=(Tc>=b['thr']).astype(float); p=b['p_ref']
    y=out[meta['actual_col']].to_numpy(); Hte=Ha[sp:]
    P={'HASHIRA-CF':out['Humidity_pred_CRFonly'].to_numpy()}
    Xte=X[sp:]
    for nm,(est,g) in b['causal_models'].items(): P[nm]=np.clip(Hte+g.predict(Xte)+est.effect(Xte)*(Tb[sp:]-p),0,100)
    prev=pd.to_numeric(df['Humidity'],errors='coerce').shift(1)[m].to_numpy()[sp:]
    P['Drift']=np.clip(Hte+(Hte-prev),0,100)
    Ylvl=Ha[:sv]+Ya[:sv]
    def mk(): return {'linear':LinearRegression(),'ridge':RidgeCV(alphas=np.logspace(-3,3,13)),
        'rf':RandomForestRegressor(n_estimators=500,min_samples_leaf=3,random_state=42,n_jobs=-1),
        'gb':HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,random_state=42)}
    for var,add in [('X',None),('X+T',Tb),('X+Temp',Tc)]:
        Xa=X if add is None else np.column_stack([X,add])
        for nm,mdl in mk().items():
            mdl.fit(Xa[:sv],Ylvl); P[f'{nm}[{var}]']=np.clip(mdl.predict(Xa[sp:]),0,100)
    R=pd.DataFrame(P); R.insert(0,'actual',y); R.insert(0,'H_t',Hte); R.insert(0,'timestamp',ts[sp:])
    R.to_csv(f'fair_{tag}.csv',index=False)
    s=np.abs(Ya[:sv]).mean()
    print(f"[{tag}] model-ready {n}, train {sp}, fit {sv}, test {n-sp}, conf {len(cc)}, dropped {meta['dropped_collinear_features']}, s={s:.4f}, thr={b['thr']:.2f}")
    return s
for mod,tag in [('crfmod','orig0C'),('crfmod_neg','neg40C')]:
    s=run(mod,tag); open(f's_{tag}.txt','w').write(str(s))
