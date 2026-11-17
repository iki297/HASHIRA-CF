import sys, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.argv=['x']; sys.path.insert(0,'.')
TEL='../data/20260913_telemetry_with_all_predictions.csv'
tel=pd.read_csv(TEL); tel['ts']=pd.to_datetime(tel.timestamp,format='mixed'); raw=tel[tel.Temperature.notna()].sort_values('ts').reset_index(drop=True)
ev=pd.read_csv('../data/live_eval_10sep.csv',parse_dates=['timestamp'])
ev=ev[(ev.timestamp>='2026-09-04')&(ev.timestamp<'2026-09-11')].dropna(subset=['aktual','sebelum','hum_pred_1h']).reset_index(drop=True); y=ev.aktual.values
class Buf:
    def __init__(s,r): s.r=r
    def snapshot(s): return s.r
cols=['timestamp','UV','Temperature','PM2_5','PM10','Humidity','BarPressure']; recs=[]
for T0 in ev.timestamp:
    win=raw[(raw.ts<T0)&(raw.ts>=T0-pd.Timedelta(hours=1))]; r=win.ts.iloc[-1]
    b=raw[(raw.ts<=r)&(raw.ts>=r-pd.Timedelta(minutes=1440))].copy(); b['timestamp']=b.ts; recs.append(b[cols].to_dict('records'))
d=pd.read_csv('../data/20260902_telemetry_training.csv')
for mod in ['crfmod','crfmod_nocal']:
    M=importlib.import_module(mod); maes=[]; ates=[]
    for rep in range(5):
        M.DEPLOY_MODE=True; M.PUBLISH_BASELINES=False; M.COLLINEAR_THR=0.93
        out,mets,meta,df,bd=M.train_iot_crf(d,'timestamp',freq='1h',horizon_steps=1)
        pr=np.array([M.predict_live_humidity(bd,*M.build_live_feature_row(Buf(rc),bd)[:3],None if False else M.build_live_feature_row(Buf(rc),bd)[4])['hum_pred'] for rc in recs])
        maes.append(np.abs(pr-y).mean()); ates.append(meta['ATE_mean_test'])
    print(f"{mod}: MAE over 5 refits {np.round(maes,3)} mean {np.mean(maes):.3f} range {min(maes):.3f}-{max(maes):.3f}; ATE range {min(ates):.3f}..{max(ates):.3f}",flush=True)
