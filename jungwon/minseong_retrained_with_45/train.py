import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
import compare as C,feature_track as F,model as M,expert_train as E
HERE=Path(__file__).resolve().parent

def base_features(f,columns):
 f=f.reset_index(drop=True);return F.kinetic_sets(f,F.build_features(f))['core_no_absolute_temp'][columns].to_numpy(np.float32)

def image_features(f,root):
 with ThreadPoolExecutor(max_workers=6) as pool:return np.stack(list(pool.map(C.image_vector,[root/p for p in f.image_path])))

def estimator():return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.08,l2_regularization=5,max_bins=63,early_stopping=False,random_state=42)

def score(y,p):return float(f1_score(y,p,labels=range(1,7),average='macro',zero_division=0))

def run_cv(frame,images,columns):
 versions=[]
 for seconds in (5,10,15):
  f=M.thin(frame,seconds);indices=f.index.to_numpy();versions.append((f.reset_index(drop=True),base_features(f,columns),images[indices]));print('PREPARED',seconds,len(f),flush=True)
 results=[]
 for seed in (42,43,44):
  a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(frame,groups=frame.experiment_id));inner=frame.iloc[a];held_ids=set(frame.experiment_id.iloc[b]);train_ids=set(inner.experiment_id);t,xf,im=versions[0];keep=t.experiment_id.isin(train_ids).to_numpy();sensor=estimator().fit(xf[keep],t.label.to_numpy()[keep]);image=estimator().fit(im[keep],t.label.to_numpy()[keep]);expert=E.expert(inner,True)
  for seconds,(f,x,img) in zip((5,10,15),versions):
   mask=f.experiment_id.isin(held_ids).to_numpy();g=f.loc[mask].reset_index(drop=True);probs=.6*sensor.predict_proba(x[mask])+.4*image.predict_proba(img[mask]);pred=probs.argmax(1)+1;gate=np.isin(pred,[4,5]);features=M.sensor_features(g,'prefix');pred[gate]=expert.predict_proba(features.loc[gate,expert.feature_names_],thread_count=2).argmax(1)+4;value=score(g.label,pred);results.append(value);print('CV',seed,seconds,value,flush=True)
 print('TRAIN_GROUP_CV_MEAN',float(np.mean(results)),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--image_root',type=Path,required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--cv',action='store_true');a=p.parse_args();frame=pd.read_csv(a.input);assert frame.sample_id.is_unique and not frame.duplicated(['experiment_id','relative_time_s']).any();assert 'label' in frame;columns=json.loads((HERE/'feature_columns.json').read_text());images=image_features(frame,a.image_root);print('TRAIN_ROWS',len(frame),'TRAIN_EXPERIMENTS',frame.experiment_id.nunique(),flush=True)
 if a.cv:run_cv(frame,images,columns)
 x=base_features(frame,columns);sensor=estimator().fit(x,frame.label);image=estimator().fit(images,frame.label);expert=E.expert(frame,True);a.output_dir.mkdir(parents=True,exist_ok=True);joblib.dump({'sensor':sensor,'image':image,'columns':columns,'sensor_weight':.6,'image_weight':.4},a.output_dir/'model.joblib');expert.save_model(str(a.output_dir/'expert.cbm'));print('MODEL_SAVED',a.output_dir,flush=True)
if __name__=='__main__':main()
