import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier
import compare as C,feature_track as F,model as M
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--image_root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model_dir',type=Path,default=HERE/'weights');p.add_argument('--sample_submission',type=Path);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
f=pd.read_csv(a.input);assert f.sample_id.is_unique and not f.duplicated(['experiment_id','relative_time_s']).any();bundle=joblib.load(a.model_dir/'model.joblib');x=F.kinetic_sets(f.reset_index(drop=True),F.build_features(f))['core_no_absolute_temp'][bundle['columns']].to_numpy(np.float32)
with ThreadPoolExecutor(max_workers=6) as pool:images=np.stack(list(pool.map(C.image_vector,[a.image_root/p for p in f.image_path])))
probs=bundle['sensor_weight']*bundle['sensor'].predict_proba(x)+bundle['image_weight']*bundle['image'].predict_proba(images);pred=probs.argmax(1)+1;gate=np.isin(pred,[4,5]);e=CatBoostClassifier().load_model(str(a.model_dir/'expert.cbm'));z=M.sensor_features(f,'prefix');pred[gate]=e.predict_proba(z.loc[gate,e.feature_names_],thread_count=2).argmax(1)+4;ids=pd.read_csv(a.sample_submission).sample_id if a.sample_submission else f.sample_id;assert ids.is_unique and set(ids)==set(f.sample_id);pd.DataFrame({'sample_id':ids,'label':pd.Series(pred,index=f.sample_id).loc[ids].to_numpy()}).to_csv(a.output,index=False);print('Saved',len(ids),'predictions')
