import sys, importlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.argv=['x']; sys.path.insert(0,'.')
import crfmod as M
TEL='../data/20260913_telemetry_with_all_predictions.csv'
tel=pd.read_csv(TEL); tel['ts']=pd.to_datetime(tel.timestamp,format='mixed'); raw=tel[tel.Temperature.notna()].sort_values('ts').reset_index(drop=True)
ev=pd.read_csv('../data/live_eval_10sep.csv',parse_dates=['timestamp'])
ev=ev[(ev.timestamp>='2026-09-04')&(ev.timestamp<'2026-09-11')].dropna(subset=['aktual','sebelum','hum_pred_1h']).reset_index(drop=True)
y=ev.aktual.values
class Buf:
    def __init__(s,r): s.r=r
    def snapshot(s): return s.r
cols=['timestamp','UV','Temperature','PM2_5','PM10','Humidity','BarPressure']
recs=[]
for T0 in ev.timestamp:
    win=raw[(raw.ts<T0)&(raw.ts>=T0-pd.Timedelta(hours=1))]; r=win.ts.iloc[-1]
    b=raw[(raw.ts<=r)&(raw.ts>=r-pd.Timedelta(minutes=1440))].copy(); b['timestamp']=b.ts; recs.append(b[cols].to_dict('records'))
M.PUBLISH_BASELINES=False
for thr in [0.90,0.93,0.95,1.01]:
    M.DEPLOY_MODE=True; M.COLLINEAR_THR=thr
    out,mets,meta,df,bd=M.train_iot_crf(pd.read_csv('../data/20260902_telemetry_training.csv'),'timestamp',freq='1h',horizon_steps=1)
    pr=[]
    for rc in recs:
        x,ch,ct,ft,ph=M.build_live_feature_row(Buf(rc),bd); pr.append(M.predict_live_humidity(bd,x,ch,ct,ph)['hum_pred'])
    pr=np.array(pr)
    print(f"DEPLOY thr={thr}: confounders {meta['confounder_count']}, dropped {meta['dropped_collinear_features']}, ATE {meta['ATE_mean_test']:.3f}, MAE {np.abs(pr-y).mean():.3f}",flush=True)
