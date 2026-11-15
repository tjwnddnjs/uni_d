"""Inference for the all-Train sensor/image + frozen 4/5 expert model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import compare as C
import feature_track as F
import team_model as T

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--image_root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample_submission',type=Path)
    parser.add_argument('--weights_dir',type=Path,help='Directory with model.joblib and expert.cbm; defaults to bundled weights')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'Output already exists: {args.output}')
    root=Path(__file__).resolve().parent
    frame=pd.read_csv(args.input)
    if not frame.sample_id.is_unique:raise ValueError('sample_id must be unique')
    if frame.duplicated(['experiment_id','relative_time_s']).any():raise ValueError('Duplicate experiment/time')
    weights=args.weights_dir or root/'weights'
    bundle=joblib.load(weights/'model.joblib')
    features=F.kinetic_sets(frame,F.build_features(frame))['core_no_absolute_temp'][bundle['columns']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        images=np.stack(list(pool.map(C.image_vector,[args.image_root/p for p in frame.image_path])))
    probs=bundle['sensor_weight']*bundle['sensor'].predict_proba(features.to_numpy('float32'))+bundle['image_weight']*bundle['image'].predict_proba(images)
    prediction=probs.argmax(1)+1
    gate=np.isin(prediction,[4,5])
    if gate.any():
        expert=CatBoostClassifier().load_model(str(weights/'expert.cbm'))
        x=T.sensor_features(frame,'prefix')
        prediction[gate]=expert.predict_proba(x.loc[gate,expert.feature_names_],thread_count=2).argmax(1)+4
    ids=pd.read_csv(args.sample_submission).sample_id if args.sample_submission else frame.sample_id
    if not ids.is_unique or set(ids)!=set(frame.sample_id):raise ValueError('sample_submission IDs do not match input')
    pred=pd.Series(prediction,index=frame.sample_id).loc[ids].to_numpy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'sample_id':ids,'label':pred}).to_csv(args.output,index=False)
    print(f'Saved {len(pred)} predictions to {args.output}; routed {int(gate.sum())} rows')

if __name__=='__main__':main()
