import sys, warnings, numpy as np, pandas as pd, time
warnings.filterwarnings('ignore'); sys.argv=['x']; sys.path.insert(0,'.')
import crfmod_neg as M
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
M.DEPLOY_MODE=False
raw=pd.read_csv('../data/kanazawa_outdoor_training.csv'); raw['timestamp']=pd.to_datetime(raw.timestamp)
raw=raw.sort_values('timestamp').reset_index(drop=True)
W=424; K=int(sys.argv_k) if False else 24
starts=np.linspace(0,len(raw)-W-1,K).astype(int)
rows=[]
for o,st in enumerate(starts):
    t0=time.time()
    df_raw=raw.iloc[st:st+W].copy()
    try:
        out,mets,meta,df,b=M.train_iot_crf(df_raw,'timestamp',freq='1h',horizon_steps=1)
    except Exception as e:
        print('origin',o,'failed',str(e)[:80]); continue
    cc=b['conf_cols']; X_df=df[cc].apply(pd.to_numeric,errors='coerce')
    T=pd.to_numeric(df['Temperature'],errors='coerce'); H=pd.to_numeric(df['Humidity'],errors='coerce'); Y=H.shift(-1)-H
    m=(X_df.notna().all(axis=1)&T.notna()&Y.notna()&H.notna()).to_numpy()
    X=X_df[m].to_numpy(float); Tc=T[m].to_numpy(float); Ha=H[m].to_numpy(float); Ya=Y[m].to_numpy(float)
    n=len(X); sp=int(0.8*n); sv=int(0.8*sp)
    Tb=(Tc>=b['thr']).astype(float); p=b['p_ref']; Hte=Ha[sp:]; y=out[meta['actual_col']].to_numpy()
    P={'HASHIRA-CF':out['Humidity_pred_CRFonly'].to_numpy()}
    for nm,(est,g) in b['causal_models'].items(): P[nm]=np.clip(Hte+g.predict(X[sp:])+est.effect(X[sp:])*(Tb[sp:]-p),0,100)
    prev=H.shift(1)[m].to_numpy()[sp:]; P['Drift']=np.clip(Hte+(Hte-prev),0,100)
    Xa=np.column_stack([X,Tb]); Ylvl=Ha[:sv]+Ya[:sv]
    for nm,mdl in [('linear',LinearRegression()),('ridge',RidgeCV(alphas=np.logspace(-3,3,13))),
                   ('rf',RandomForestRegressor(n_estimators=500,min_samples_leaf=3,random_state=42,n_jobs=-1)),
                   ('gb',HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,random_state=42))]:
        mdl.fit(Xa[:sv],Ylvl); P[nm]=np.clip(mdl.predict(Xa[sp:]),0,100)
    s=np.abs(Ya[:sv]).mean()
    for k,v in P.items(): rows.append(dict(origin=o,start=str(df_raw.timestamp.iloc[0]),n=n,fit=sv,test=n-sp,model=k,mae=np.abs(v-y).mean(),mase=np.abs(v-y).mean()/s))
    pd.DataFrame(rows).to_csv('scarcity_results.csv',index=False)
    print(f'origin {o} start {df_raw.timestamp.iloc[0]} n={n} fit={sv} test={n-sp} HASHIRA {np.abs(P["HASHIRA-CF"]-y).mean():.3f} ({time.time()-t0:.0f}s)',flush=True)
