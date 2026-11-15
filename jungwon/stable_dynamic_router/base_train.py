
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import GroupShuffleSplit
import model as M


def score(y,p):return float(f1_score(y,p.argmax(1)+1,labels=range(1,7),average='macro',zero_division=0))


def classifier(depth,trees,weights,seed=42):
    return CatBoostClassifier(iterations=trees,depth=depth,learning_rate=.05,l2_leaf_reg=8,
        loss_function='MultiClass',task_type='GPU',devices='0',thread_count=2,gpu_ram_part=.25,
        random_seed=seed,class_weights=weights,allow_writing_files=False,verbose=200,border_count=64)


def class_weights(frame):
    count=np.bincount(frame.label.to_numpy()-1,minlength=6)
    weight=np.sqrt(len(frame)/np.maximum(count,1))
    return weight/weight.mean()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,default=Path('runs'))
    parser.add_argument('--variants',nargs='+',default=['prefix','full_available_time'])
    args=parser.parse_args()
    assert get_gpu_device_count()>0,'GPU required'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frames={};images={}
    cache_root=Path('/home/jungwon/uni_d/jungwon/catboost_multirate/runs')
    for split in ('train','validation','test'):
        f=pd.read_csv(args.data_dir/split/f'{split}.csv')
        assert f.sample_id.is_unique
        frames[split]=f
        cache=cache_root/f'{split}_image12.pkl'
        if cache.exists():
            saved=pd.read_pickle(cache)
            assert f.sample_id.equals(saved['ids'])
            images[split]=saved['features']
        else:images[split]=M.image_features(f,args.data_dir/split)
    tr=frames['train']
    for other in ('validation','test'):assert set(tr.experiment_id).isdisjoint(frames[other].experiment_id)
    folds=[]
    for seed in (42,43,44):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr,groups=tr.experiment_id))
        inner,hold=tr.iloc[a].copy(),tr.iloc[b].copy()
        held=[M.thin(hold,s) for s in (5,10,15)]
        folds.append(dict(seed=seed,inner=inner,held=held))
    candidates=[]
    for variant in args.variants:
        stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
        folder=args.output_dir/f'{stamp}_late_{variant}'
        folder.mkdir()
        for name in ('model.py','train.py','test.py','verify.py','requirements.txt'):
            shutil.copy2(Path(__file__).parent/name,folder/name)
        probabilities={};fold_scores={}
        for fold in folds:
            seed=fold['seed'];inner=fold['inner'];held=fold['held']
            x,y,sw=M.augmented(inner,variant)
            print('Inner sensor',variant,seed,'features',x.shape[1],flush=True)
            sensor=classifier(5,1000,class_weights(inner),seed).fit(x,y,sample_weight=sw)


            print('Primary frozen; training auxiliary image context',seed,flush=True)
            image_model=classifier(4,350,class_weights(inner),seed).fit(images['train'].loc[inner.index],inner.label)
            hi=[image_model.predict_proba(images['train'].loc[f.index],thread_count=2) for f in held]
            for trees in (300,600,1000):
                hp=[sensor.predict_proba(M.sensor_features(f,variant),ntree_end=trees,thread_count=2) for f in held]
                for weight in (0.,.025,.05,.10,.15):
                    key=(trees,weight)
                    ps=[(1-weight)*p+weight*q for p,q in zip(hp,hi)]
                    scores=[score(f.label,p) for f,p in zip(held,ps)]
                    fold_scores.setdefault(key,[]).extend(scores)
                    if seed==42:probabilities[key]=ps[0]
        options=[dict(variant=variant,depth=5,trees=k[0],image_weight=k[1],fold_interval_f1=v,
                      internal_robust_mean=float(np.mean(v))) for k,v in fold_scores.items()]

        best=max((o for o in options if o['image_weight']>0),key=lambda r:r['internal_robust_mean'])
        config=dict(best,selection='mean 3 Train grouped holdouts (seeds42/43/44) at 5/10/15s',
                    train_intervals=[5,10,15],training_split='Train only',seed=42,gpu_ram_part=.25,
                    image_trees=350,image_depth=4,task_type='GPU',data_dir=str(args.data_dir))
        config.update(feature_lags=list(M.LAGS),fusion='late_probability',
                      training_order=['sensor_primary','freeze_sensor','image_auxiliary','late_fusion'],
                      sensor_only_control=max((o for o in options if o['image_weight']==0),key=lambda r:r['internal_robust_mean']))
        (folder/'config.json').write_text(json.dumps(config,indent=2))
        (folder/'internal_search.json').write_text(json.dumps(options,indent=2))
        selected_probs=probabilities[(best['trees'],best['image_weight'])]
        held=folds[0]['held'][0]
        np.save(folder/'inner_probabilities.npy',selected_probs)
        pd.DataFrame({'sample_id':held.sample_id,'label':held.label,'prediction':selected_probs.argmax(1)+1}).to_csv(folder/'inner_predictions.csv',index=False)
        print('SELECTED BEFORE VALIDATION',json.dumps(best),flush=True)
        x,y,sw=M.augmented(tr,variant)
        final=classifier(5,best['trees'],class_weights(tr)).fit(x,y,sample_weight=sw)
        print('Final primary frozen; fitting auxiliary context',flush=True)
        finalim=classifier(4,350,class_weights(tr)).fit(images['train'],tr.label)
        final.save_model(str(folder/'sensor.cbm'));finalim.save_model(str(folder/'image.cbm'))
        w=best['image_weight']
        metrics=dict(best,training_rows=len(tr),augmented_training_rows=len(y),training_split='Train only',
                     feature_count=x.shape[1],feature_lags=list(M.LAGS),fusion='late_probability',
                     training_order=config['training_order'],sensor_only_control=config['sensor_only_control'],
                     internal_f1_5s=score(held.label,selected_probs))
        for split in ('validation','test'):
            f=frames[split]
            p=(1-w)*final.predict_proba(M.sensor_features(f,variant),thread_count=2)+w*finalim.predict_proba(images[split],thread_count=2)
            np.save(folder/f'{split}_probabilities.npy',p)
            if split=='validation':
                metrics['validation_macro_f1']=score(f.label,p)
                metrics['validation_classification_report']=classification_report(f.label,p.argmax(1)+1,output_dict=True,zero_division=0)
                audit=[]
                for seconds in (5,10,15):
                    sub=M.thin(f,seconds)
                    prob=(1-w)*final.predict_proba(M.sensor_features(sub,variant),thread_count=2)+w*finalim.predict_proba(images[split].loc[sub.index],thread_count=2)
                    audit.append(dict(interval=seconds,macro_f1=score(sub.label,prob),rows=len(sub)))
                metrics['validation_sampling_audit']=audit
            else:
                sample=pd.read_csv(args.data_dir/'sample_submission.csv')
                pred=pd.Series(p.argmax(1)+1,index=f.sample_id)
                pd.DataFrame({'sample_id':sample.sample_id,'label':pred.loc[sample.sample_id].to_numpy()}).to_csv(folder/'submission.csv',index=False)
                black=pd.DataFrame(np.zeros_like(images[split]),columns=M.IMAGE_COLS)
                bp=(1-w)*final.predict_proba(M.sensor_features(f,variant),thread_count=2)+w*finalim.predict_proba(black,thread_count=2)
                metrics['black_image_changed_predictions']=int((p.argmax(1)!=bp.argmax(1)).sum())
        pd.DataFrame({'feature':x.columns,'importance':final.feature_importances_}).sort_values('importance',ascending=False).to_csv(folder/'sensor_importance.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
        print('FINISHED',folder.name,'Train robust',best['internal_robust_mean'],'Validation',metrics['validation_macro_f1'],flush=True)
        candidates.append(dict(folder=str(folder.resolve()),**best,validation_macro_f1=metrics['validation_macro_f1']))
        pd.DataFrame(candidates).to_csv(args.output_dir/'timeline.csv',index=False)
    best=max(candidates,key=lambda c:c['internal_robust_mean'])
    (args.output_dir/'best_by_train.json').write_text(json.dumps(best,indent=2))
    print('BEST BY TRAIN',json.dumps(best),flush=True)


if __name__=='__main__':main()
