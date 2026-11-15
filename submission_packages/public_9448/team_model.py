
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

SIGNALS=['meas_v','meas_t1','meas_t2','meas_p','meas_co2','meas_co','meas_ch4','meas_hf','atmp_t','thermal_t']
LAGS=(5,10,15,20,30)
IMAGE_COLS=['img_mean','img_std','img_max','img_p95','img_p99','img_hot_ratio',
            'r_mean','g_mean','b_mean','r_std','g_std','b_std']


def image_features(frame,root):
    root=Path(root)
    def extract(path):
        with Image.open(root/path) as im:
            rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255
            gray=np.asarray(im.convert('L'),dtype=np.float32)/255
        return [gray.mean(),gray.std(),gray.max(),*np.percentile(gray,[95,99]),
                (gray>=.8).mean(),*rgb.mean(axis=(0,1)),*rgb.std(axis=(0,1))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(extract,frame.image_path))
    return pd.DataFrame(rows,columns=IMAGE_COLS,index=frame.index,dtype=np.float32)


def sensor_features(frame,variant='combined'):
    if frame[['experiment_id','relative_time_s']].isna().any().any():raise ValueError('Missing time/experiment')
    if frame.duplicated(['experiment_id','relative_time_s']).any():raise ValueError('Duplicate time')
    t=frame.relative_time_s.to_numpy(float)
    if not np.isfinite(t).all():raise ValueError('Nonfinite time')
    raw=frame[SIGNALS].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    values=raw.to_numpy(float)
    parts=[raw.reset_index(drop=True)]
    for lag in LAGS:
        delta=np.full_like(values,np.nan); rate=delta.copy(); age=np.full(len(t),np.nan)
        for idx in frame.groupby('experiment_id',sort=False).indices.values():
            idx=idx[np.argsort(t[idx],kind='stable')]
            old=np.searchsorted(t[idx],t[idx]-lag,side='right')-1
            good=np.flatnonzero(old>=0)
            dt=t[idx[good]]-t[idx[old[good]]]
            delta[idx[good]]=values[idx[good]]-values[idx[old[good]]]
            rate[idx[good]]=delta[idx[good]]/dt[:,None]
            age[idx[good]]=dt
        parts.extend([pd.DataFrame(delta,columns=[f'{s}_delta{lag}s' for s in SIGNALS]),
                      pd.DataFrame(rate,columns=[f'{s}_rate{lag}s' for s in SIGNALS]),
                      pd.DataFrame({f'actual_lag{lag}s':age})])
    extras=pd.DataFrame({'t1_minus_t2':values[:,1]-values[:,2],
                        'temp_minus_ambient':(values[:,1]+values[:,2])/2-values[:,8]})
    parts.append(extras)
    if variant in ('metadata','combined','full_available','full_available_time'):
        meta={}
        for col,values_allowed in [('bat_class',(1,2,3)),('soc_status',(1,2)),('tr_method',(1,2))]:
            observed=pd.to_numeric(frame[col],errors='coerce').to_numpy()
            for category in values_allowed:meta[f'{col}_is{category}']=(observed==category).astype(float)
            meta[f'{col}_unknown']=(~np.isin(observed,values_allowed)).astype(float)
        meta['bat_cc']=pd.to_numeric(frame.bat_cc,errors='coerce').to_numpy()
        meta['bat_cc_log1p']=np.sign(meta['bat_cc'])*np.log1p(np.abs(meta['bat_cc']))
        parts.append(pd.DataFrame(meta))
    if variant in ('prefix','combined','full_available','full_available_time'):
        relative=np.full_like(values,np.nan)
        peak=np.full((len(frame),3),np.nan)
        event_age=np.full((len(frame),7),np.nan)
        voltage_ratio=np.full(len(frame),np.nan)
        for idx in frame.groupby('experiment_id',sort=False).indices.values():
            idx=idx[np.argsort(t[idx],kind='stable')]
            relative[idx]=values[idx]-values[idx[0]]
            voltage_ratio[idx]=(values[idx,0]-values[idx[0],0])/max(abs(values[idx[0],0]),1e-3)
            temperatures=values[idx][:,[1,2,9]]
            peak[idx]=np.fmax.accumulate(temperatures,axis=0)
            current=np.fmax(values[idx,1],values[idx,2])
            for j,threshold in enumerate((45,60,80,100,120,150,200)):
                first=np.minimum.accumulate(np.where(current>=threshold,t[idx],np.inf))
                event_age[idx,j]=np.where(np.isfinite(first),t[idx]-first,np.nan)
        parts.append(pd.DataFrame(relative,columns=[f'{s}_from_first' for s in SIGNALS]))
        parts.append(pd.DataFrame(peak,columns=['t1_prefix_peak','t2_prefix_peak','thermal_prefix_peak']))
        parts.append(pd.DataFrame(values[:,[1,2,9]]-peak,columns=['t1_from_peak','t2_from_peak','thermal_from_peak']))
        parts.append(pd.DataFrame(event_age,columns=[f'seconds_since_temp_ge_{v}' for v in (45,60,80,100,120,150,200)]))
        parts.append(pd.DataFrame({'voltage_relative_drop':voltage_ratio}))
    if variant in ('full_available','full_available_time'):
        other=['meas_hcn','meas_hcl','meas_n2o','meas_no','meas_no2','meas_so2','indu_t','indu_t.1','meas_t1_plus_t2']
        more=frame[other].apply(pd.to_numeric,errors='coerce').reset_index(drop=True)
        parts.append(more)
    if variant=='full_available_time':
        parts.append(pd.DataFrame({'relative_time_s':t}))
    return pd.concat(parts,axis=1).replace([np.inf,-np.inf],np.nan).astype(np.float32)


def thin(frame,interval):

    origin=frame.groupby('experiment_id').relative_time_s.transform('min')
    return frame[(frame.relative_time_s-origin).mod(interval)==0].copy()


def augmented(frame,variant='combined'):
    versions=[thin(frame,s) for s in (5,10,15)]
    x=pd.concat([sensor_features(f,variant) for f in versions],ignore_index=True)
    y=np.concatenate([f.label.to_numpy(int) for f in versions])

    weights=np.concatenate([np.full(len(f),len(x)/(3*len(f))) for f in versions])
    return x,y,weights
