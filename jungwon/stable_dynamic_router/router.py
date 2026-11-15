import numpy as np
from catboost import CatBoostClassifier
import model as M

def load(folder):
    models=[]
    for name in ('sensor.cbm','image.cbm','expert.cbm'):
        m=CatBoostClassifier();m.load_model(str(folder/name));models.append(m)
    return models

def infer(frame,images,models):
    sensor,image,expert=models
    x=M.sensor_features(frame,'prefix')
    primary=.975*sensor.predict_proba(x,thread_count=2)+.025*image.predict_proba(images,thread_count=2)
    base=primary.argmax(1)+1
    routed=base.copy();mask=np.isin(base,[4,5])
    if mask.any():routed[mask]=expert.predict_proba(x.iloc[mask][expert.feature_names_],thread_count=2).argmax(1)+4
    return base,routed,mask
