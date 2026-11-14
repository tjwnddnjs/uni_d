"""베이스라인 추론 스크립트.

예) python test.py --input ./data/battery_data_light/test/test.csv \
                   --image_root ./data/battery_data_light/test \
                   --model ./model.joblib --output ./submission.csv \
                   [--sample_submission ./data/battery_data_light/sample_submission.csv]

입력 CSV에 들어 있는 행만으로 피처를 만들기 때문에, 주최 측이 Test 일부만 넣거나
실험별로 나누어 실행해도 같은 샘플에는 같은 예측이 나와야 합니다.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('model.joblib'))
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
    img = M.extract_image_features(frame, paths)
    X = M.build_features(frame, img)

    bundle = joblib.load(args.model)
    X = X[bundle['features']]
    pred = pd.Series(bundle['pipeline'].predict(X).astype(int), index=frame.sample_id.values)

    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
        assert set(order) == set(pred.index), '입력 CSV와 제출 양식의 ID가 다릅니다.'
        ids = order.tolist()
    else:
        ids = frame.sample_id.tolist()
    submission = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].values})
    assert submission.label.isin(M.LABELS).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission)}행)')


if __name__ == '__main__':
    main()
