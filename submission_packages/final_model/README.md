# Uni-DTHON final multimodal model

Train 78개 실험 전체의 센서·열화상 분류기와 기존 4/5 전문가를 결합한 최종 후보입니다. 학습 데이터 및 제출 CSV는 저장소에 포함하지 않습니다. 제공된 저장 가중치는 원본 데이터에서 만든 최종 Test CSV와 3,451행 모두 동일한 예측을 출력합니다. 이 후보의 정확한 Public 점수는 저장된 결과에서 확인되지 않아 임의로 기록하지 않습니다. Validation은 반복 개발에 사용됐고, Private 성능은 미확인입니다.

## 환경

- 검증 환경: Linux, Python 3.10, `requirements.txt`에 고정한 Python 패키지 버전.
- 추론: CPU 가능. 학습: 센서·이미지 HistGradientBoosting은 CPU, 4/5 CatBoost 전문가는 CUDA 지원 NVIDIA GPU와 호환 드라이버 필요.
- GPU 재현 검증: NVIDIA RTX 5080, NVIDIA driver 580.178.04, CatBoost 1.2.10. 다른 드라이버나 장치에서는 GPU 수치 연산 차이로 재학습 결과가 달라질 수 있으므로, 제출 예측을 정확히 재현하려면 포함된 가중치를 사용하세요.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r submission_packages/final_model/requirements.txt
```

`Dockerfile`은 Python 의존성 설치용이며 이미지 빌드 자체는 이 환경에서 검증하지 않았습니다. GPU 학습에는 호스트 NVIDIA 드라이버와 NVIDIA Container Toolkit이 별도로 필요합니다.

## 저장된 최종 가중치로 추론

```bash
python submission_packages/final_model/test.py \
  --input /path/to/test/test.csv \
  --image_root /path/to/test \
  --sample_submission /path/to/sample_submission.csv \
  --output /path/to/new_submission.csv
```

CSV의 `image_path`는 `--image_root`를 기준으로 읽습니다. sample submission을 생략하면 입력 순서대로 출력합니다. 출력 파일이 이미 존재하면 덮어쓰지 않습니다. 추론은 모델을 fit하지 않으며 각 실험의 현재·과거 관측을 이용합니다. 시점별 예측에 필요한 과거 행은 입력에 함께 제공해야 합니다.

## Train부터 재학습

```bash
python submission_packages/final_model/train.py \
  --train_csv /path/to/train/train.csv \
  --image_root /path/to/train \
  --output_dir /path/to/new_weights

python submission_packages/final_model/test.py \
  --input /path/to/test/test.csv \
  --image_root /path/to/test \
  --sample_submission /path/to/sample_submission.csv \
  --weights_dir /path/to/new_weights \
  --output /path/to/retrained_submission.csv
```

`--output_dir`은 아직 존재하지 않는 디렉터리여야 합니다. 학습에서 Validation이나 Test를 읽거나 사용하지 않습니다. Train에 단계 건너뛰기 관측이 있는 실험도 모두 포함합니다.

## 모델 구성

1. 119개 현재·과거 센서 피처와 348개 현재 RGB/공간 이미지 피처로 각각 HistGradientBoostingClassifier를 학습합니다.
2. 6-class 확률을 센서 60% + 이미지 40%로 결합합니다.
3. 기본 예측이 4 또는 5인 경우에만 93개 피처 CatBoost 전문가가 4/5를 다시 결정합니다. 그 밖의 기본 예측은 유지합니다. 전문가 구현과 가중치의 출처는 팀원의 `feature/jungwon` 커밋 `d8754b287404c0ff0f9bf105d824040a36b010c9`입니다.
4. 제공 가중치의 센서·이미지 모델은 Train 33,522행/78개 실험으로 학습됐으며, 팀원의 전문가는 Train에서 학습됐습니다. 외부 사전학습 가중치는 사용하지 않습니다.

## 재현성 검증

- 동봉 가중치로 Test 전체를 원본 이미지에서 추론한 CSV가 기존 최종 후보와 **3,451/3,451행 일치**했습니다.
- 동일 서버에서 Train CSV와 원본 Train 이미지만으로 세 모델을 처음부터 재학습한 뒤 Test에 추론했고, 동봉 가중치 예측과 **3,451/3,451행 일치**했습니다. Test는 예측 동일성 검증에만 사용했고 학습에는 사용하지 않았습니다.
- Test의 한 실험에서 행을 섞거나 후반부를 제거했을 때 기존 시점 예측이 각각 237/237행과 118/118행 일치했습니다. 전 실험에 대한 포괄적인 인과성 증명은 아닙니다.
- 실제 Test 정답은 없고, 제출 성능은 Public의 일부 샘플에 대한 점수입니다.
