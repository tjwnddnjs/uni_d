"""Rebuild all three classifiers from the labeled Train CSV and Train images."""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
import argparse,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
import compare as C,feature_track as F,team_model as M

def select_signals(frame):
    def direction(f):
        means=f[f.label.isin([4,5])].groupby('label')[M.SIGNALS].mean()
        if not {4,5}.issubset(means.index):raise ValueError('Train split requires stages 4 and 5')
        return np.sign(means.loc[5]-means.loc[4]).fillna(0)
    full=direction(frame);votes=[]
    for seed in range(71,81):
        a,_=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(frame,groups=frame.experiment_id));votes.append(direction(frame.iloc[a]))
    selected=[c for c in M.SIGNALS if full[c]!=0 and all(v[c]==full[c] for v in votes)]
    if not selected:raise ValueError('No stable expert features')
    return selected

def fit_expert(frame):
    selected=select_signals(frame);xs=[];ys=[]
    def allowed(c):
        if c=='t1_minus_t2':return {'meas_t1','meas_t2'}.issubset(selected)
        if c in selected or any(c.startswith(s+'_') for s in selected):return True
        if c.startswith('actual_lag'):return True
        if c.startswith('t1_') and 'meas_t1' in selected:return True
        if c.startswith('t2_') and 'meas_t2' in selected:return True
        if c.startswith('thermal_') and 'thermal_t' in selected:return True
        if c.startswith('seconds_since_temp_') and {'meas_t1','meas_t2'}.issubset(selected):return True
        if c=='temp_minus_ambient' and {'meas_t1','meas_t2','atmp_t'}.issubset(selected):return True
        return c=='voltage_relative_drop' and 'meas_v' in selected
    for sec in (5,10,15):
        f=M.thin(frame,sec);x=M.sensor_features(f,'prefix');mask=f.label.isin([4,5]).to_numpy();xs.append(x.iloc[mask][[c for c in x if allowed(c)]]);ys.append(f.label.to_numpy()[mask])
    total=sum(len(y) for y in ys);weights=np.concatenate([np.full(len(y),total/(3*len(y))) for y in ys]);pair=frame[frame.label.isin([4,5])];counts=np.array([(pair.label==v).sum() for v in (4,5)]);cw=np.sqrt(len(pair)/counts);cw/=cw.mean()
    model=CatBoostClassifier(iterations=800,depth=5,learning_rate=.05,l2_leaf_reg=8,loss_function='Logloss',task_type='GPU',devices='0',gpu_ram_part=.5,thread_count=2,random_seed=42,class_weights=cw,allow_writing_files=False,verbose=False)
    return model.fit(pd.concat(xs,ignore_index=True),np.concatenate(ys),sample_weight=weights),selected

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train_csv',required=True,type=Path);ap.add_argument('--image_root',required=True,type=Path);ap.add_argument('--output_dir',required=True,type=Path);a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False);frame=pd.read_csv(a.train_csv)
    if not frame.sample_id.is_unique or frame.duplicated(['experiment_id','relative_time_s']).any():raise ValueError('Duplicate sample or experiment/time')
    if set(frame.label.unique())!={1,2,3,4,5,6}:raise ValueError('Train must contain all six stages')
    cols=json.loads((Path(__file__).resolve().parent/'sensor_features.json').read_text());x=F.kinetic_sets(frame,F.build_features(frame))['core_no_absolute_temp'][cols].to_numpy('float32')
    print('Training sensor on',len(frame),'rows',frame.experiment_id.nunique(),'experiments',flush=True);sensor=C.estimator().fit(x,frame.label.to_numpy())
    print('Reading training images',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:images=np.stack(list(pool.map(C.image_vector,[a.image_root/p for p in frame.image_path])))
    print('Training image classifier',flush=True);image=C.estimator().fit(images,frame.label.to_numpy())
    print('Training 4/5 expert on GPU',flush=True);expert,selected=fit_expert(frame);expert.save_model(str(a.output_dir/'expert.cbm'))
    joblib.dump({'sensor':sensor,'image':image,'columns':cols,'sensor_weight':.6,'image_weight':.4,'excluded':[]},a.output_dir/'model.joblib')
    (a.output_dir/'training_metadata.json').write_text(json.dumps({'train_rows':len(frame),'train_experiments':int(frame.experiment_id.nunique()),'excluded_experiments':[],'expert_signals':selected,'test_validation_fit':False},indent=2))
    print('Saved weights in',a.output_dir,flush=True)
if __name__=='__main__':main()
