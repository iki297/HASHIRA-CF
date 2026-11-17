import sys, importlib, warnings, numpy as np, pandas as pd, pickle
warnings.filterwarnings('ignore'); sys.argv=['x']; sys.path.insert(0,'.')
TEL='../data/20260913_telemetry_with_all_predictions.csv'
tel=pd.read_csv(TEL); tel['ts']=pd.to_datetime(tel.timestamp,format='mixed')
raw=tel[tel.Temperature.notna()].sort_values('ts').reset_index(drop=True)
ev=pd.read_csv('../data/live_eval_10sep.csv',parse_dates=['timestamp'])
ev=ev[(ev.timestamp>='2026-09-04')&(ev.timestamp<'2026-09-11')].dropna(subset=['aktual','sebelum','hum_pred_1h']).reset_index(drop=True)
class Buf:
    def __init__(s,recs): s.r=recs
    def snapshot(s): return s.r
def replay(modname):
    M=importlib.import_module(modname); M.DEPLOY_MODE=True
    out,mets,meta,df,b=M.train_iot_crf(pd.read_csv('../data/20260902_telemetry_training.csv'),'timestamp',freq='1h',horizon_steps=1)
    # rebuild fitting matrix exactly as in train_iot_crf
    cc=b['conf_cols']; X_df=df[cc].apply(pd.to_numeric,errors='coerce')
    T=pd.to_numeric(df['Temperature'],errors='coerce'); H=pd.to_numeric(df['Humidity'],errors='coerce'); Y=H.shift(-1)-H
    m=(X_df.notna().all(axis=1)&T.notna()&Y.notna()&H.notna()).to_numpy()
    X=X_df[m].to_numpy(float); Tc=T[m].to_numpy(float); Ha=H[m].to_numpy(float); Ya=Y[m].to_numpy(float)
    n=len(X); sv=int(0.8*n)
    rows=[]
    cols=['timestamp','UV','Temperature','PM2_5','PM10','Humidity','BarPressure']
    for T0 in ev.timestamp:
        win=raw[(raw.ts<T0)&(raw.ts>=T0-pd.Timedelta(hours=1))]
        if win.empty: rows.append(None); continue
        r=win.ts.iloc[-1]
        bufdf=raw[(raw.ts<=r)&(raw.ts>=r-pd.Timedelta(minutes=1440))].copy()
        bufdf['timestamp']=bufdf.ts
        recs=bufdf[cols].to_dict('records')
        f=M.build_live_feature_row(Buf(recs),b)
        if f is None: rows.append(None); continue
        x,ch,ct,ft,ph=f
        res=M.predict_live_humidity(b,x,ch,ct,ph)
        rows.append(dict(x=x[0],cur_hum=ch,cur_temp=ct,ft=ft,prev=ph,hum=res['hum_pred'],**{f'bp_{k}':v for k,v in res['baseline_preds'].items()}))
    return dict(b=b,X=X,Tc=Tc,Ha=Ha,Ya=Ya,sv=sv,rows=rows,n=n)
R={}
for mod in ['crfmod','crfmod_nocal']:
    R[mod]=replay(mod)
    ok=[r for r in R[mod]['rows'] if r is not None]
    print(mod,'model-ready',R[mod]['n'],'fit',R[mod]['sv'],'conf',len(R[mod]['b']['conf_cols']),'replayed',len(ok),'of',len(ev))
pickle.dump({k:{kk:vv for kk,vv in v.items() if kk!='b'} for k,v in R.items()},open('replay.pkl','wb'))
# keep bundles for later fitting
pickle.dump({k:v['b'] for k,v in R.items()},open('replay_bundles.pkl','wb'))
# fidelity check on original model
rr=R['crfmod']['rows']
mk=np.array([r is not None for r in rr])
rep=np.array([r['hum'] if r else np.nan for r in rr]); live=ev.hum_pred_1h.values
ft_ok=np.mean([ (r['ft']==t-pd.Timedelta(hours=1)) for r,t in zip(rr,ev.timestamp) if r])
cur_ok=np.mean([abs(r['cur_hum']-s)<0.051 for r,s in zip(rr,ev.sebelum) if r])
d=rep[mk]-live[mk]
print('feature bin == T-1h:',ft_ok,' current_hum == sebelum:',cur_ok)
print('HASHIRA replay vs live: mean|diff|',np.abs(d).mean().round(4),' max|diff|',np.abs(d).max().round(3),' corr',np.corrcoef(rep[mk],live[mk])[0,1].round(5))
for k,c in [('tlearner','hum_pred_tlearner'),('rf','hum_pred_rf'),('gb','hum_pred_gb'),('linear','hum_pred_lr'),('ridge','hum_pred_ridge'),('lineardml','hum_pred_lineardml'),('drift','hum_pred_drift')]:
    v=np.array([r.get('bp_'+k,np.nan) if r else np.nan for r in rr])
    dd=v[mk]-ev[c].values[mk]; print(f'  {k:10s} mean|diff| {np.nanmean(np.abs(dd)):.4f} max {np.nanmax(np.abs(dd)):.3f}')
