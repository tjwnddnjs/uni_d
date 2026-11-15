import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
import model as M

def select_features(frame):
    # Selection uses only this training partition and grouped subsplits of it.
    # No validation rows or labels participate.
    cols=M.SIGNALS
    def direction(f):
        means=f[f.label.isin([4,5])].groupby('label')[cols].mean()
        if not {4,5}.issubset(means.index):raise ValueError('Training split lacks stage 4 or 5')
        return np.sign(means.loc[5]-means.loc[4]).fillna(0)
    full=direction(frame);votes=[]
    for seed in range(71,81):
        a,_=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(frame,groups=frame.experiment_id))
        votes.append(direction(frame.iloc[a]))
    selected=[c for c in cols if full[c]!=0 and all(v[c]==full[c] for v in votes)]
    if not selected:raise ValueError('No stable training-only features')
    return selected

def expert(frame,balanced):
    pair=frame[frame.label.isin([4,5])]
    selected=select_features(frame)
    # Build causal features BEFORE filtering labels, preserving the full prior history.
    versions=[M.thin(frame,s) for s in (5,10,15)]
    xs=[];ys=[];ws=[]
    for f in versions:
        xf=M.sensor_features(f,'prefix');mask=f.label.isin([4,5]).to_numpy()
        # Retain the chosen signals' causal derivatives and histories, so the
        # comparison tests signal selection rather than removing all dynamics.
        def allowed(c):
            if c=='t1_minus_t2':return {'meas_t1','meas_t2'}.issubset(selected)
            if c in selected or any(c.startswith(s+'_') for s in selected):return True
            if c.startswith('actual_lag'):return True
            if c.startswith('t1_') and 'meas_t1' in selected:return True
            if c.startswith('t2_') and 'meas_t2' in selected:return True
            if c.startswith('thermal_') and 'thermal_t' in selected:return True
            if c.startswith('seconds_since_temp_') and {'meas_t1','meas_t2'}.issubset(selected):return True
            if c=='temp_minus_ambient' and {'meas_t1','meas_t2','atmp_t'}.issubset(selected):return True
            if c=='voltage_relative_drop' and 'meas_v' in selected:return True
            return False
        kept=[c for c in xf if allowed(c)]
        xs.append(xf.iloc[mask][kept]);ys.append(f.label.to_numpy()[mask])
    total=sum(len(v) for v in ys)
    for yv in ys:ws.append(np.full(len(yv),total/(3*len(yv))))
    x=pd.concat(xs,ignore_index=True);y=np.concatenate(ys)
    counts=np.array([(pair.label==v).sum() for v in (4,5)])
    cw=np.sqrt(len(pair)/counts);cw=cw/cw.mean()
    return CatBoostClassifier(iterations=800,depth=5,learning_rate=.05,l2_leaf_reg=8,
        loss_function='Logloss',task_type='GPU',devices='0',gpu_ram_part=.5,thread_count=2,
        random_seed=42,class_weights=cw if balanced else None,allow_writing_files=False,verbose=False).fit(x,y,sample_weight=np.concatenate(ws))
