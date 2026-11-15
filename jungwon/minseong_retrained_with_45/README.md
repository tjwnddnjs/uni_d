# Retrained sensor/image model + 4/5 specialist

This package reproduces the `0.8847` Train group-CV / `0.8172` full Validation experiment. The 119-column sensor HistGradientBoosting model and 348-feature image HistGradientBoosting model are trained on all Train rows, including experiments with observed skipped stages. Their class probabilities are blended 60:40. When the blended argmax is 4 or 5, a separately trained CatBoost 4/5 specialist determines the final prediction.

The 119 input columns are fixed from the original `public_9448` feature configuration. The 4/5 specialist selects its source sensors from each training partition. The fixed columns were selected in earlier development, so the group-CV estimate is not an untouched, fully nested evaluation.

## Setup

Install the dependencies in `requirements.txt`. Specialist training requires a CUDA GPU supported by CatBoost.

## Train

```bash
python train.py --input /path/to/train/train.csv --image_root /path/to/train --output_dir ./weights_retrained --cv
```

`--cv` runs the original three grouped holdouts (seeds 42, 43, 44) at 5/10/15-second sampling intervals before fitting all Train rows. Omit it to fit all rows directly. All inference image paths are relative to the supplied image root. Training never reads Validation or Test data and does not generate a submission.

## Predict

```bash
python predict.py --input /path/to/validation/validation.csv --image_root /path/to/validation --model_dir ./weights --output /tmp/validation_predictions.csv
```

`--sample_submission` is optional and specifies output order. The output must not already exist. `weights/` contains the retrained full-Train models; `--model_dir ./weights_retrained` uses a fresh retraining. Run prediction only on the data you explicitly supply.

## Provenance

`compare.py` and `feature_track.py` originate from `feature/minseong` commit `80d09f39a9b8ef3796ef0bcb5d1729721996135c`. `model.py` and the specialist algorithm originate from `feature/jungwon` commit `d8754b287404c0ff0f9bf105d824040a36b010c9`. The original public 0.9448 model is a different candidate; no leaderboard score is claimed for this package.
