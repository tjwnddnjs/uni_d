import json,shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score,classification_report,confusion_matrix
import model as M
import base_train as B
import router

DATA=Path('/home/jungwon/data/battery_data_light')
ANCHOR=Path('/home/jungwon/uni_d/jungwon/goal90/lag530_late_context/runs/20261114_224921_late_prefix')
CACHE=Path('/home/jungwon/uni_d/jungwon/catboost_multirate/runs')

def select_features(frame):


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

    versions=[M.thin(frame,s) for s in (5,10,15)]
    xs=[];ys=[];ws=[]
    for f in versions:
        xf=M.sensor_features(f,'prefix');mask=f.label.isin([4,5]).to_numpy()


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

def f1(y,p):return float(f1_score(y,p,labels=range(1,7),average='macro',zero_division=0))

def main():
    frames={s:pd.read_csv(DATA/s/f'{s}.csv') for s in ('train','validation','test')}
    images={}
    for s,f in frames.items():
        cached=pd.read_pickle(CACHE/f'{s}_image12.pkl');assert cached['ids'].equals(f.sample_id)
        images[s]=cached['features']
    tr=frames['train'];scores={False:[],True:[]};details=[]
    for seed in (42,43,44):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr,groups=tr.experiment_id))
        inner,held=tr.iloc[a],tr.iloc[b]
        x,y,sw=M.augmented(inner,'prefix')
        primary=B.classifier(5,1000,B.class_weights(inner),seed).fit(x,y,sample_weight=sw)
        context=B.classifier(4,350,B.class_weights(inner),seed).fit(images['train'].iloc[a],inner.label)
        experts={key:expert(inner,key) for key in scores}
        for sec in (5,10,15):
            f=M.thin(held,sec);xi=M.sensor_features(f,'prefix')
            p=.975*primary.predict_proba(xi)+.025*context.predict_proba(images['train'].loc[f.index])
            base=p.argmax(1)+1;mask=np.isin(base,[4,5])
            for key,ex in experts.items():
                pred=base.copy();pred[mask]=ex.predict_proba(xi.iloc[mask][ex.feature_names_]).argmax(1)+4
                val=f1(f.label,pred);scores[key].append(val)
                details.append(dict(seed=seed,interval=sec,balanced=key,base_f1=f1(f.label,base),router_f1=val,routed=int(mask.sum())))
                print(details[-1],flush=True)
    selected=max(scores,key=lambda key:np.mean(scores[key]))
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
    out=Path('runs')/(stamp+'_stable_dynamic_router');out.mkdir(parents=True)
    for name in ('model.py','base_train.py','train.py','router.py','test.py','requirements.txt'):shutil.copy2(name,out/name)
    for name in ('sensor.cbm','image.cbm'):shutil.copy2(ANCHOR/name,out/name)
    config=dict(expert_balanced=selected,selection='Train 3 group holdouts x 5/10/15 seconds',
        primary_feature_lags=[5,10,15,20,30],expert_features=select_features(tr),
        expert_selection='Train-only grouped mean-direction stability; selected signals retain causal derived features',
        training_split='Train only',primary=str(ANCHOR),routing='primary argmax in {4,5}')
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'cv.json').write_text(json.dumps(details,indent=2))
    final_expert=expert(tr,selected)
    config['expert_model_features']=final_expert.feature_names_
    (out/'config.json').write_text(json.dumps(config,indent=2))
    final_expert.save_model(str(out/'expert.cbm'))
    models=router.load(out);metrics=dict(config,train_cv_mean=float(np.mean(scores[selected])))
    for split in ('validation','test'):
        f=frames[split];base,pred,mask=router.infer(f,images[split],models)
        if split=='validation':
            metrics.update(base_macro_f1=f1(f.label,base),router_macro_f1=f1(f.label,pred),
                routed=int(mask.sum()),changed=int((pred!=base).sum()),
                classification_report=classification_report(f.label,pred,labels=range(1,7),output_dict=True,zero_division=0),
                confusion_matrix=confusion_matrix(f.label,pred,labels=range(1,7)).tolist())
            tmp=f[['bat_class','experiment_id','label']].copy();tmp['base_prediction']=base;tmp['prediction']=pred
            groups=[]
            for key,g in tmp.groupby('bat_class',dropna=False):
                groups.append(dict(bat_class=key,rows=len(g),experiments=g.experiment_id.nunique(),
                    base_accuracy=float((g.label==g.base_prediction).mean()),router_accuracy=float((g.label==g.prediction).mean()),
                    base_macro_f1=f1(g.label,g.base_prediction),router_macro_f1=f1(g.label,g.prediction)))
            pd.DataFrame(groups).to_csv(out/'battery_class_metrics.csv',index=False)
            tmp.to_csv(out/'validation_predictions.csv',index=False)
        else:
            ids=pd.read_csv(DATA/'sample_submission.csv').sample_id
            result=pd.Series(pred,index=f.sample_id)
            pd.DataFrame({'sample_id':ids,'label':result.loc[ids].to_numpy()}).to_csv(out/'submission.csv',index=False)
            metrics['test_changed']=int((pred!=base).sum())
            black=pd.DataFrame(np.zeros_like(images[split]),columns=M.IMAGE_COLS)
            metrics['black_image_changed_predictions']=int((router.infer(f,black,models)[1]!=pred).sum())
        assert np.array_equal(pred[~mask],base[~mask])
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print('FINISHED',out,json.dumps(metrics),flush=True)

if __name__=='__main__':main()
