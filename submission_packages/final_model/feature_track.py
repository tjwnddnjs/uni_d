"""Causal physical/history feature ablations on fixed experiment folds."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report,confusion_matrix
import compare as C

RAW_EXCLUDE=['sample_id','experiment_id','image_path','label']
GASES=['meas_ch4','meas_co','meas_co2','meas_hcn','meas_hcl','meas_hf','meas_n2o','meas_no','meas_no2','meas_so2']
SIGNALS=['meas_v','meas_t1','meas_t2','meas_p','thermal_t']+GASES


def build_features(df):
    """No fit, labels or future statistics. Uses only supplied current/past rows."""
    df=df.reset_index(drop=True)
    raw=df[[c for c in df if c not in RAW_EXCLUDE]].apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    physical={}
    for c in ['meas_t1','meas_t2','thermal_t']:
        physical[f'{c}_above_ambient']=raw[c]-raw.atmp_t
    physical['t1_minus_t2']=raw.meas_t1-raw.meas_t2
    physical['t1_minus_thermal']=raw.meas_t1-raw.thermal_t
    physical['t2_minus_thermal']=raw.meas_t2-raw.thermal_t
    physical['surface_max']=raw[['meas_t1','meas_t2']].max(axis=1)
    physical['surface_min']=raw[['meas_t1','meas_t2']].min(axis=1)
    physical['surface_mean']=raw[['meas_t1','meas_t2']].mean(axis=1)
    physical['gas_total']=raw[GASES].sum(axis=1)
    physical['co_to_co2']=raw.meas_co/(raw.meas_co2+.1)
    physical['co2_fraction']=raw.meas_co2/(raw[GASES].sum(axis=1)+.1)
    for c in GASES+['meas_p','meas_v']:
        physical[f'{c}_log1p']=np.log1p(raw[c].clip(lower=0))
    physical=pd.DataFrame(physical)
    signals=raw[SIGNALS].copy()
    for c in ['t1_minus_t2','gas_total','surface_max']:
        signals[c]=physical[c]
    names=[]
    for s in signals:
        names += [f'{s}_{k}' for k in ['initial_delta','initial_ratio','past_max','past_min','below_peak','above_min','past_integral','last_slope']]
        for w in [15,60,180]:
            names += [f'{s}_{k}_{w}s' for k in ['delta_slope','mean','std','max','min']]
    names += ['voltage_retained','peak_temp_since_start','time_above_45','time_above_80','time_above_150','time_since_45','time_since_80','time_since_150','time_since_peak_t1','time_since_peak_thermal']
    history=pd.DataFrame(np.nan,index=df.index,columns=names,dtype=float)
    for _,ii in df.groupby('experiment_id',sort=False).indices.items():
        ii=ii[np.argsort(raw.relative_time_s.to_numpy()[ii],kind='stable')]
        t=raw.relative_time_s.to_numpy(dtype=float)[ii]
        dt=np.diff(t,prepend=t[0])
        for s in signals:
            x=signals[s].to_numpy(dtype=float)[ii]
            finite=np.isfinite(x)
            # Current signal known; no future filling is performed.
            mx=np.maximum.accumulate(np.where(finite,x,-np.inf));mn=np.minimum.accumulate(np.where(finite,x,np.inf))
            vals={f'{s}_initial_delta':x-x[0],f'{s}_initial_ratio':x/(abs(x[0])+.1),f'{s}_past_max':mx,f'{s}_past_min':mn,f'{s}_below_peak':mx-x,f'{s}_above_min':x-mn,f'{s}_past_integral':np.cumsum(np.nan_to_num(x)*dt),f'{s}_last_slope':np.divide(np.diff(x,prepend=x[0]),dt,out=np.zeros_like(x),where=dt>0)}
            for w in [15,60,180]:
                start=np.searchsorted(t,t-w,side='right');lag=start-1;valid=lag>=0
                delta=np.full(len(t),np.nan);delta[valid]=(x[valid]-x[lag[valid]])/(t[valid]-t[lag[valid]])
                vals[f'{s}_delta_slope_{w}s']=delta
                for kind,fn in [('mean',np.mean),('std',np.std),('max',np.max),('min',np.min)]:
                    vals[f'{s}_{kind}_{w}s']=np.array([fn(x[a:i+1]) for i,a in enumerate(start)])
            for c,v in vals.items():history.loc[ii,c]=v
        v=raw.meas_v.to_numpy(dtype=float)[ii];t1=raw.meas_t1.to_numpy(dtype=float)[ii]
        history.loc[ii,'voltage_retained']=v/(abs(v[0])+.1)
        history.loc[ii,'peak_temp_since_start']=np.maximum.accumulate(t1)-t1[0]
        for threshold in [45,80,150]:
            above=t1>=threshold
            history.loc[ii,f'time_above_{threshold}']=np.cumsum(above*dt)
            # The first crossing time is used only from that time onward.
            crossing=np.flatnonzero(above)
            elapsed=np.full(len(t),np.nan)
            if len(crossing):elapsed[crossing[0]:]=t[crossing[0]:]-t[crossing[0]]
            history.loc[ii,f'time_since_{threshold}']=elapsed
        for c,name in [('meas_t1','t1'),('thermal_t','thermal')]:
            x=raw[c].to_numpy(dtype=float)[ii];mx=np.maximum.accumulate(x)
            peak_time=np.maximum.accumulate(np.where(x>=mx,t,-np.inf))
            history.loc[ii,f'time_since_peak_{name}']=t-peak_time
    return {'raw':raw,'physical':pd.concat([raw,physical],axis=1),'history':pd.concat([raw,history],axis=1),'full':pd.concat([raw,physical,history],axis=1)}


def load(root,split,out):
    path=root/split/f'{split}.csv';df=pd.read_csv(path)
    key=hashlib.sha256(path.read_bytes()+b'featuretrack-v1').hexdigest()[:12]
    cache=out/f'features_{split}_{key}.joblib'
    if cache.exists():sets=joblib.load(cache)
    else:sets=build_features(df);joblib.dump(sets,cache)
    return df,sets


def analyse(train,sets,out):
    train.groupby('label')[[c for c in train if c not in RAW_EXCLUDE]].agg(['median','min','max']).to_csv(out/'label_raw_summary.csv')
    transitions=[]
    for exp,g in train.groupby('experiment_id'):
        g=g.sort_values('relative_time_s');change=g.label!=g.label.shift()
        for _,r in g[change].iterrows():transitions.append(r.to_dict())
    pd.DataFrame(transitions).to_csv(out/'label_onsets.csv',index=False)
    sets['full'].assign(label=train.label).groupby('label').median().to_csv(out/'label_engineered_medians.csv')


def predict_feature_bundle(bundle, df, image_features):
    """Predict saved selected engineered model from supplied rows/image vectors only."""
    name=bundle['selection']['feature_set']
    if name=='original':
        raise ValueError('Use compare feature loader for the original, non-engineered model')
    sets=build_features(df)
    if name not in sets:
        refined=refined_sets(df.reset_index(drop=True),sets)
        sets=refined if name in refined else kinetic_sets(df.reset_index(drop=True),sets)
    spec=bundle['feature_models'][name]
    assert list(sets[name].columns)==spec['columns']
    p=spec['model'].predict_proba(sets[name].to_numpy(dtype=np.float32))
    image=bundle['image_model'].predict_proba(image_features)
    return (1-spec['alpha'])*p+spec['alpha']*image


def refined_sets(df, sets):
    """Reduce noisy gas histories; add persistent voltage-collapse state."""
    history=sets['history'];raw=list(sets['raw'].columns)
    key=['meas_v','meas_t1','meas_t2','meas_p','thermal_t']
    state_suffix=['initial_delta','initial_ratio','past_max','past_min','below_peak','above_min','last_slope']
    keep=[c for c in history if c in raw or any(c.startswith(s+'_') for s in key) or c.startswith('time_') or c in ['voltage_retained','peak_temp_since_start']]
    concise=[c for c in keep if c in raw or any(c.endswith(s) for s in state_suffix) or c.startswith('time_') or c in ['voltage_retained','peak_temp_since_start']]
    core=history[keep].copy()
    state=history[concise].copy()
    voltage={}
    for threshold in [.5,1.,2.]:
        voltage[f'time_since_vdrop_{threshold}']=np.full(len(df),np.nan)
        voltage[f'past_vdrop_{threshold}']=np.zeros(len(df))
    for fraction in [.75,.5,.25]:
        voltage[f'time_since_vfrac_{fraction}']=np.full(len(df),np.nan)
    for _,idx in df.groupby('experiment_id',sort=False).indices.items():
        idx=idx[np.argsort(df.relative_time_s.to_numpy()[idx],kind='stable')]
        t=df.relative_time_s.to_numpy(dtype=float)[idx];v=df.meas_v.to_numpy(dtype=float)[idx]
        peak=np.maximum.accumulate(v)
        for threshold in [.5,1.,2.]:
            crossed=np.flatnonzero(peak-v>=threshold)
            if len(crossed):
                first=crossed[0];voltage[f'time_since_vdrop_{threshold}'][idx[first:]]=t[first:]-t[first];voltage[f'past_vdrop_{threshold}'][idx[first:]]=1
        for fraction in [.75,.5,.25]:
            crossed=np.flatnonzero(v<=fraction*max(v[0],.01))
            if len(crossed):
                first=crossed[0];voltage[f'time_since_vfrac_{fraction}'][idx[first:]]=t[first:]-t[first]
    voltage=pd.DataFrame(voltage)
    return {'core_history':core,'concise_state':state,'core_voltage_state':pd.concat([core,voltage],axis=1),'concise_voltage_state':pd.concat([state,voltage],axis=1)}


def kinetic_sets(df, sets):
    """Third round: heating kinetics and early-background gas deviations."""
    core=refined_sets(df,sets)['core_voltage_state']
    extra={};gas={}
    for s in ['meas_t1','meas_t2','thermal_t','meas_p']:
        for w in [15,30,60,120]:
            for k in ['slope','acceleration','range']:
                extra[f'kinetic_{s}_{k}_{w}']=np.full(len(df),np.nan)
    for th in [45,50,55,60,70,80,90]:
        extra[f'heating_elapsed_{th}']=np.full(len(df),np.nan)
    for c in ['elapsed_50_over_45','rise_from45_per_s','rise_from50_per_s','heating_fraction_t1_t2','thermal_minus_t1_rate15']:
        extra[c]=np.full(len(df),np.nan)
    for s in GASES+['meas_p']:
        for k in ['early_mean_delta','early_zscore','relative_peak','slope30','slope120']:
            gas[f'background_{s}_{k}']=np.full(len(df),np.nan)
    for _,ii in df.groupby('experiment_id',sort=False).indices.items():
        ii=ii[np.argsort(df.relative_time_s.to_numpy()[ii],kind='stable')]
        t=df.relative_time_s.to_numpy(dtype=float)[ii]
        elapsed={}
        t1=df.meas_t1.to_numpy(dtype=float)[ii]
        for th in [45,50,55,60,70,80,90]:
            a=np.flatnonzero(t1>=th);e=np.full(len(t),np.nan)
            if len(a):e[a[0]:]=t[a[0]:]-t[a[0]]
            elapsed[th]=e;extra[f'heating_elapsed_{th}'][ii]=e
        extra['elapsed_50_over_45'][ii]=elapsed[50]/(elapsed[45]+1)
        extra['rise_from45_per_s'][ii]=(t1-45)/(elapsed[45]+1)
        extra['rise_from50_per_s'][ii]=(t1-50)/(elapsed[50]+1)
        t2=df.meas_t2.to_numpy(dtype=float)[ii]
        extra['heating_fraction_t1_t2'][ii]=(t1-t1[0])/(np.abs(t2-t2[0])+1)
        slopes={}
        for s in ['meas_t1','meas_t2','thermal_t','meas_p']:
            x=df[s].to_numpy(dtype=float)[ii]
            for w in [15,30,60,120]:
                lag=np.searchsorted(t,t-w,side='right')-1;valid=lag>=0
                slope=np.full(len(t),np.nan);slope[valid]=(x[valid]-x[lag[valid]])/(t[valid]-t[lag[valid]])
                slopes[(s,w)]=slope
                acc=np.full(len(t),np.nan);acc[valid]=(slope[valid]-slope[lag[valid]])/(t[valid]-t[lag[valid]])
                start=lag+1
                span=np.array([np.ptp(x[a:i+1]) for i,a in enumerate(start)])
                extra[f'kinetic_{s}_slope_{w}'][ii]=slope;extra[f'kinetic_{s}_acceleration_{w}'][ii]=acc;extra[f'kinetic_{s}_range_{w}'][ii]=span
        extra['thermal_minus_t1_rate15'][ii]=slopes[('thermal_t',15)]-slopes[('meas_t1',15)]
        end=np.searchsorted(t,t[0]+60,side='right')
        for s in GASES+['meas_p']:
            x=df[s].to_numpy(dtype=float)[ii]
            # First 60 seconds as background, using only available past rows before 60s.
            n=np.minimum(np.arange(len(t))+1,end);sm=np.cumsum(x);sq=np.cumsum(x*x)
            mean=sm[n-1]/n;std=np.sqrt(np.maximum(sq[n-1]/n-mean**2,0))
            gas[f'background_{s}_early_mean_delta'][ii]=x-mean
            gas[f'background_{s}_early_zscore'][ii]=(x-mean)/(std+.1)
            gas[f'background_{s}_relative_peak'][ii]=np.maximum.accumulate(x)-mean
            for w in [30,120]:
                lag=np.searchsorted(t,t-w,side='right')-1;valid=lag>=0;slope=np.full(len(t),np.nan);slope[valid]=(x[valid]-x[lag[valid]])/(t[valid]-t[lag[valid]])
                gas[f'background_{s}_slope{w}'][ii]=slope
    kinetic=pd.concat([core,pd.DataFrame(extra)],axis=1)
    # Preserve event crossing states and thermal changes, remove raw/rolling absolute temperatures.
    temps=['meas_t1','meas_t2','thermal_t']
    noabs=[c for c in core if c not in temps+['meas_t1_plus_t2'] and not any(c.startswith(s+'_') and ('_mean_' in c or '_max_' in c or '_min_' in c or c.endswith('_past_max') or c.endswith('_past_min') or c.endswith('_past_integral')) for s in temps)]
    return {'core_no_absolute_temp':core[noabs], 'core_kinetic':kinetic,'core_kinetic_gas':pd.concat([kinetic,pd.DataFrame(gas)],axis=1)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'));ap.add_argument('--out',type=Path,default=Path('experiments/feature_results'));ap.add_argument('--refine',action='store_true');ap.add_argument('--kinetic',action='store_true');args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    cache=args.out.parent if args.refine or args.kinetic else args.out
    train,sets=load(args.data_dir,'train',cache)
    if args.kinetic:sets=kinetic_sets(train,sets)
    elif args.refine:sets=refined_sets(train,sets)
    else:analyse(train,sets,args.out)
    folds=pd.read_csv('experiments/results/folds.csv');assert list(folds.sample_id)==list(train.sample_id)
    y=train.label.to_numpy();oof={};scores=[];fold_scores=[]
    base=np.load('experiments/results/probabilities.npz');image_oof=base['oof_image']
    oof['original']=base['oof_tabular']
    for name,features in sets.items():
        features=features.to_numpy(dtype=np.float32);p=np.zeros((len(train),6))
        for f in sorted(folds.fold.unique()):
            a=np.flatnonzero(folds.fold.to_numpy()!=f);b=np.flatnonzero(folds.fold.to_numpy()==f)
            assert set(train.experiment_id.iloc[a]).isdisjoint(train.experiment_id.iloc[b])
            m=C.estimator().fit(features[a],y[a]);p[b]=m.predict_proba(features[b]);s=C.score(y[b],p[b]);fold_scores.append({'feature_set':name,'fold':int(f),'f1':s});print(name,'fold',f,'f1',s,flush=True)
        oof[name]=p
    for name,p in oof.items():
        for alpha in np.linspace(0,.5,6):
            scores.append({'feature_set':name,'alpha':float(alpha),'oof_macro_f1':C.score(y,(1-alpha)*p+alpha*image_oof)})
    ranked=pd.DataFrame(scores).sort_values('oof_macro_f1',ascending=False);ranked.to_csv(args.out/'oof_ablation.csv',index=False);pd.DataFrame(fold_scores).to_csv(args.out/'fold_scores.csv',index=False)
    best=ranked.iloc[0].to_dict();(args.out/'selection.json').write_text(json.dumps(best,indent=2));print('FROZEN SELECTION',best,flush=True)
    # Evaluate every feature ablation at its own OOF-selected image weight, not validation-selected weights.
    val,vs=load(args.data_dir,'validation',cache);test,ts=load(args.data_dir,'test',cache)
    if args.kinetic:vs=kinetic_sets(val,vs);ts=kinetic_sets(test,ts)
    elif args.refine:vs=refined_sets(val,vs);ts=refined_sets(test,ts)
    assert set(train.experiment_id).isdisjoint(val.experiment_id)
    sample=pd.read_csv(args.data_dir/'sample_submission.csv');rows=[];bundles={};probs={}
    original=joblib.load('experiments/results/models.joblib')['models'];image_val=base['validation_image'];image_test=base['test_image']
    for name in oof:
        alpha=float(ranked[ranked.feature_set==name].iloc[0].alpha)
        if name=='original':m=original['tabular'];pv=base['validation_tabular'];pt=base['test_tabular'];cols=None
        else:
            m=C.estimator().fit(sets[name].to_numpy(dtype=np.float32),y);cols=list(sets[name].columns)
            pv=m.predict_proba(vs[name].to_numpy(dtype=np.float32));pt=m.predict_proba(ts[name].to_numpy(dtype=np.float32))
        combined=(1-alpha)*pv+alpha*image_val;pred=C.LABELS[combined.argmax(1)]
        rows.append({'feature_set':name,'alpha':alpha,'oof_macro_f1':C.score(y,(1-alpha)*oof[name]+alpha*image_oof),'validation_macro_f1':C.score(val.label,combined),'validation_tabular_f1':C.score(val.label,pv)})
        bundles[name]={'model':m,'columns':cols,'alpha':alpha};probs[f'validation_{name}']=pv;probs[f'test_{name}']=pt
        pd.DataFrame(confusion_matrix(val.label,pred,labels=C.LABELS),index=C.LABELS,columns=C.LABELS).to_csv(args.out/f'confusion_{name}.csv')
        (args.out/f'classification_{name}.json').write_text(json.dumps(classification_report(val.label,pred,labels=C.LABELS,output_dict=True,zero_division=0),indent=2))
        pd.DataFrame({'sample_id':val.sample_id,'label':val.label,'pred':pred}).to_csv(args.out/f'validation_{name}.csv',index=False)
        pp=(1-alpha)*pt+alpha*image_test;predtest=pd.Series(C.LABELS[pp.argmax(1)],index=test.sample_id)
        pd.DataFrame({'sample_id':sample.sample_id,'label':predtest.loc[sample.sample_id].to_numpy()}).to_csv(args.out/f'submission_{name}.csv',index=False)
    np.savez_compressed(args.out/'probabilities.npz',**{f'oof_{k}':p for k,p in oof.items()},**probs)
    joblib.dump({'feature_models':bundles,'image_model':original['image'],'selection':best},args.out/'models.joblib')
    pd.DataFrame(rows).sort_values('oof_macro_f1',ascending=False).to_csv(args.out/'comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    # Prefix and row order invariance on engineered features (no future access).
    small=train[train.experiment_id.isin(train.experiment_id.unique()[:2])].copy()
    full=build_features(small)['full'];prefix=small.groupby('experiment_id',sort=False).head(50)
    prefixf=build_features(prefix)['full'];lookup=pd.DataFrame(full.to_numpy(),index=small.sample_id,columns=full.columns)
    assert np.allclose(lookup.loc[prefix.sample_id],prefixf,equal_nan=True)
    shuffled=small.sample(frac=1,random_state=4);sf=build_features(shuffled)['full']
    assert np.allclose(lookup.loc[shuffled.sample_id],sf,equal_nan=True)
    print('Feature causality/order checks passed',flush=True)

if __name__=='__main__':main()
