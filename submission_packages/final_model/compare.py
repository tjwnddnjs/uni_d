"""CPU multimodal comparison. Selection uses train-group OOF, never validation/test."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

LABELS = np.arange(1, 7)
SENSORS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'thermal_t']
VERSION = 'rgb8-hist16-grid4-causal30-60-v1'


def image_vector(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB').resize((32, 32)), dtype=np.float32) / 255
        small = np.asarray(im.convert('RGB').resize((8, 8)), dtype=np.float32).ravel() / 255
    hist = np.concatenate([np.histogram(rgb[:, :, c], bins=16, range=(0, 1), density=False)[0] / 1024 for c in range(3)])
    tiles = rgb.reshape(4, 8, 4, 8, 3)
    return np.concatenate([small, hist, tiles.mean(axis=(1, 3)).ravel(), tiles.std(axis=(1, 3)).ravel(), np.quantile(rgb, [.1, .5, .9, .99], axis=(0, 1)).ravel()]).astype('float32')


def load_features(root, split, cache, csv_path=None, image_root=None):
    csv = Path(csv_path) if csv_path is not None else root / split / f'{split}.csv'
    image_root = Path(image_root) if image_root is not None else root / split
    df = pd.read_csv(csv)
    assert df.sample_id.is_unique
    assert not df.duplicated(['experiment_id', 'relative_time_s']).any()
    signature = hashlib.sha256(csv.read_bytes() + str(image_root.resolve()).encode() + VERSION.encode()).hexdigest()[:16]
    target = cache / f'{split}_{signature}.joblib'
    if target.exists():
        image = joblib.load(target)
    else:
        paths = [image_root / p for p in df.image_path]
        with ThreadPoolExecutor(max_workers=4) as pool:
            image = np.stack(list(pool.map(image_vector, paths)))
        joblib.dump(image, target)
    # Explicit numeric columns only: IDs and labels cannot enter the model.
    cols = [c for c in df.columns if c not in ['sample_id', 'experiment_id', 'image_path', 'label']]
    tab = df[cols].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan).copy()
    for window in [30, 60]:
        for signal in SENSORS:
            for name in ['mean', 'delta_per_s']:
                tab[f'{signal}_{name}_{window}s'] = np.nan
        for _, idx in df.groupby('experiment_id', sort=False).indices.items():
            idx = idx[np.argsort(df.relative_time_s.to_numpy()[idx], kind='stable')]
            t = df.relative_time_s.to_numpy(dtype=float)[idx]
            start = np.searchsorted(t, t-window, side='right')
            for signal in SENSORS:
                x = df[signal].to_numpy(dtype=float)[idx]
                means = np.array([np.mean(x[a:i+1]) for i, a in enumerate(start)])
                lag = start-1
                delta = np.full(len(t), np.nan)
                valid = lag >= 0
                delta[valid] = (x[valid]-x[lag[valid]])/(t[valid]-t[lag[valid]])
                tab.loc[idx, f'{signal}_mean_{window}s'] = means
                tab.loc[idx, f'{signal}_delta_per_s_{window}s'] = delta
    return df, {'tabular': tab.to_numpy(dtype=np.float32), 'image': image, 'early_fusion': np.column_stack([tab, image]).astype('float32')}, list(tab.columns)


def partitions(items):
    if not items:
        yield []
    else:
        a = items[0]
        for b in items[1:]:
            for rest in partitions([x for x in items[1:] if x != b]):
                yield [[a,b]] + rest


def hierarchy(gate, expert, pairs):
    # A soft group gate from one modality; conditional pair probabilities from the other.
    out = np.zeros_like(gate)
    for pair in pairs:
        cols = np.array(pair)-1
        group_prob = gate[:, cols].sum(axis=1, keepdims=True)
        conditional = expert[:, cols] / np.maximum(expert[:, cols].sum(axis=1, keepdims=True), 1e-12)
        out[:, cols] = group_prob * conditional
    return out / out.sum(axis=1, keepdims=True)


def score(y, p):
    return float(f1_score(y, LABELS[p.argmax(axis=1)], labels=LABELS, average='macro', zero_division=0))


def candidate_predictions(probs):
    yield 'tabular', {}, probs['tabular']
    yield 'image', {}, probs['image']
    yield 'early_fusion', {}, probs['early_fusion']
    for alpha in np.linspace(0, 1, 11):
        yield f'late_fusion_image_{alpha:.1f}', {'alpha': float(alpha)}, (1-alpha)*probs['tabular']+alpha*probs['image']
    for pairs in partitions(list(range(1,7))):
        key = '_'.join(''.join(map(str,p)) for p in pairs)
        for gate,expert in [('tabular','image'),('image','tabular')]:
            yield f'hierarchy_{gate}_to_{expert}_{key}', {'gate':gate, 'expert':expert, 'pairs':pairs}, hierarchy(probs[gate],probs[expert],pairs)


def apply_spec(name, spec, probs):
    if name in probs: return probs[name]
    if 'alpha' in spec: return (1-spec['alpha'])*probs['tabular']+spec['alpha']*probs['image']
    return hierarchy(probs[spec['gate']], probs[spec['expert']], spec['pairs'])


def estimator():
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=.08, l2_regularization=5, max_bins=63, early_stopping=False, random_state=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--out', type=Path, default=Path('experiments/results'))
    ap.add_argument('--folds', type=int, default=3)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out/'cache'; cache.mkdir(exist_ok=True)
    print('Extracting train features', flush=True)
    train, X, columns = load_features(args.data_dir, 'train', cache)
    y = train.label.to_numpy(); groups = train.experiment_id.to_numpy()
    oof = {k: np.zeros((len(y),6)) for k in X}
    folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42).split(X['tabular'], y, groups))
    fold_ids = np.full(len(y), -1)
    for f, (a,b) in enumerate(folds):
        assert set(groups[a]).isdisjoint(groups[b])
        assert set(y[a]) == set(LABELS)
        fold_ids[b] = f
        for name, features in X.items():
            m = estimator().fit(features[a], y[a]); oof[name][b] = m.predict_proba(features[b])
            print(f'Fold {f+1}/{args.folds} {name}: Macro F1={score(y[b],oof[name][b]):.4f}', flush=True)
    train[['sample_id','experiment_id','label']].assign(fold=fold_ids).to_csv(args.out/'folds.csv',index=False)
    candidates = [(name,spec,score(y,p)) for name,spec,p in candidate_predictions(oof)]
    candidates.sort(key=lambda x:x[2],reverse=True)
    pd.DataFrame([{'name':n,'oof_macro_f1':s,'spec':json.dumps(c)} for n,c,s in candidates]).to_csv(args.out/'oof_search.csv',index=False)
    # Freeze all selections before opening validation labels.
    selected = {n:{} for n in ['tabular','image','early_fusion']}
    for prefix in ['late_fusion_', 'hierarchy_tabular_to_image_', 'hierarchy_image_to_tabular_']:
        n,c,_ = next(r for r in candidates if r[0].startswith(prefix)); selected[n]=c
    for gate,expert in [('tabular','image'),('image','tabular')]:
        selected[f'hierarchy_{gate}_to_{expert}_12_34_56']={'gate':gate,'expert':expert,'pairs':[[1,2],[3,4],[5,6]]}
    best_name,best_spec,_=candidates[0]
    # For a multimodal submission require both modalities by construction.
    multi = next(r for r in candidates if r[0]=='early_fusion' or r[0].startswith('hierarchy_') or ('alpha' in r[1] and 0<r[1]['alpha']<1))
    chosen_name,chosen_spec,_=multi
    selected[chosen_name]=chosen_spec
    print('OOF selection:', best_name, 'multimodal:', chosen_name, flush=True)
    print('Extracting validation/test features',flush=True)
    val,V,vcols=load_features(args.data_dir,'validation',cache)
    test,T,tcols=load_features(args.data_dir,'test',cache)
    assert columns==vcols==tcols
    for a,b in [(train,val),(train,test),(val,test)]:
        assert set(a.experiment_id).isdisjoint(b.experiment_id)
        assert set(a.sample_id).isdisjoint(b.sample_id)
    vp={};tp={};models={}
    for name in X:
        m=estimator().fit(X[name],y);models[name]=m
        vp[name]=m.predict_proba(V[name]);tp[name]=m.predict_proba(T[name])
        print('Full train fit:',name,flush=True)
    metrics=[]
    sample=pd.read_csv(args.data_dir/'sample_submission.csv')
    assert set(sample.sample_id)==set(test.sample_id)
    for name,spec in selected.items():
        p=apply_spec(name,spec,vp);pred=LABELS[p.argmax(axis=1)]
        metrics.append({'model':name,'oof_macro_f1':score(y,apply_spec(name,spec,oof)),'validation_macro_f1':score(val.label,p),'validation_accuracy':float(accuracy_score(val.label,pred))})
        pd.DataFrame(confusion_matrix(val.label,pred,labels=LABELS),index=LABELS,columns=LABELS).to_csv(args.out/f'confusion_{name}.csv')
        (args.out/f'classification_{name}.json').write_text(json.dumps(classification_report(val.label,pred,labels=LABELS,output_dict=True,zero_division=0),indent=2))
        pd.DataFrame({'sample_id':val.sample_id,'label':val.label,'prediction':pred,**{f'p{i}':p[:,i-1] for i in LABELS}}).to_csv(args.out/f'validation_{name}.csv',index=False)
        testp=apply_spec(name,spec,tp)
        s=pd.Series(LABELS[testp.argmax(axis=1)],index=test.sample_id)
        pd.DataFrame({'sample_id':sample.sample_id,'label':s.loc[sample.sample_id].to_numpy()}).to_csv(args.out/f'submission_{name}.csv',index=False)
    report=pd.DataFrame(metrics).sort_values('oof_macro_f1',ascending=False);report.to_csv(args.out/'comparison.csv',index=False)
    joblib.dump({'models':models,'selected_name':chosen_name,'selected_spec':chosen_spec,'tabular_columns':columns,'version':VERSION},args.out/'models.joblib')
    np.savez_compressed(args.out/'probabilities.npz',**{f'oof_{k}':v for k,v in oof.items()},**{f'validation_{k}':v for k,v in vp.items()},**{f'test_{k}':v for k,v in tp.items()})
    (args.out/'selection.json').write_text(json.dumps({'overall_oof_winner':best_name,'multimodal_oof_winner':chosen_name,'selected':selected,'folds':args.folds,'seed':42,'feature_version':VERSION},indent=2))
    print(report.to_string(index=False),flush=True)

if __name__=='__main__':main()
