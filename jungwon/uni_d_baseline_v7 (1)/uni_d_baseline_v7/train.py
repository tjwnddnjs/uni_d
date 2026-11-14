"""베이스라인 학습 스크립트.

예) python train.py --data_dir ./data/battery_data_light --model_out ./model.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import model as M


def load_split(data_dir: Path, split: str):
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    img = M.extract_image_features(frame, paths)
    X = M.build_features(frame, img)
    y = pd.to_numeric(frame['label'], errors='raise').astype(int)
    assert y.isin(M.LABELS).all()
    return frame, X, y


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    parser.add_argument('--model_out', type=Path, default=Path('model.joblib'))
    parser.add_argument('--refit_train_val', action='store_true', help='모델 선택 후 Train+Validation으로 다시 학습')
    args = parser.parse_args()

    train, X_tr, y_tr = load_split(args.data_dir, 'train')
    val, X_va, y_va = load_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(val.experiment_id)

    results = {}
    n_splits = min(5, train.experiment_id.nunique())
    for name, cw in [('none', None), ('balanced', 'balanced')]:
        cv = []
        for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X_tr, y_tr, groups=train.experiment_id):
            m = M.build_model(cw).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            cv.append(macro_f1(y_tr.iloc[va_idx], m.predict(X_tr.iloc[va_idx])))
        m = M.build_model(cw).fit(X_tr, y_tr)
        results[name] = (cw, macro_f1(y_va, m.predict(X_va)))
        print(f'class_weight={name}: Train GroupKFold Macro F1 {np.mean(cv):.4f} ± {np.std(cv):.4f} | Validation Macro F1 {results[name][1]:.4f}')

    best = max(results, key=lambda k: results[k][1])
    cw = results[best][0]
    print('선택:', best)
    if args.refit_train_val:
        final = M.build_model(cw).fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
    else:
        final = M.build_model(cw).fit(X_tr, y_tr)
    joblib.dump({'pipeline': final, 'features': list(X_tr.columns), 'class_weight': cw}, args.model_out)
    print('모델 저장:', args.model_out)


if __name__ == '__main__':
    main()
