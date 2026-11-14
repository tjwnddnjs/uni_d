# Uni-D 배터리 열폭주 6단계 분류 — 베이스라인 v7 (제출 템플릿)

> 참가팀은 이 README를 자기 팀 내용으로 바꿔서 제출하십시오. `<>` 부분을 채우면 됩니다.

## 팀 정보
- 팀명: <팀명>
- 최종 선택 제출 파일: <리더보드에서 선택한 파일명>, Private 복원 대상

## 개발 환경
- OS: <예: Ubuntu 22.04>
- Python: <예: 3.10>
- GPU: <예: 대회 제공 GPU 서버, 모델명>
- 라이브러리: `requirements.txt` 참고

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 피처 생성(이미지 통계·센서·과거 방향 시계열)과 모델 정의 |
| `train.py` | Train으로 학습, Validation으로 평가, 가중치 저장 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성 |
| `model.joblib` | 학습된 가중치. 재학습 없이 `test.py`만으로 결과를 재현할 수 있어야 합니다. |

## 실행 방법
```bash
pip install -r requirements.txt

# 학습 (선택: --refit_train_val 을 주면 Train+Validation으로 최종 학습)
python train.py --data_dir <battery_data_light 경로> --model_out model.joblib

# 추론
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --model model.joblib --output submission.csv \
               --sample_submission <battery_data_light 경로>/sample_submission.csv
```

## 학습·검증 구성
- 학습 데이터: <예: Train만 / Train+Validation>
- 검증 방법: <예: 제공 Validation + Train 실험 단위 GroupKFold(5)>
- 검증 점수: <Validation Macro F1>

## 시계열 피처 구성
- <예: 같은 experiment_id 안에서 relative_time_s 기준 과거 30초 구간의 평균·최댓값, 30초 전 관측과의 차이>
- 미래 시점 관측과 실험 전체 기준 통계는 사용하지 않습니다.

## 사용한 사전학습 모델
- <없음, 또는 모델명 / 출처 / 버전 / 라이선스>
