# Public 0.9448 candidate

Saved-weight inference for the submitted sensor/image HistGradientBoosting ensemble with the `feature/jungwon` 4/5 CatBoost expert. The reported 0.9448 Public score was supplied by the team; it is not a measurement of the full Test set. On the repeatedly used full Validation set this exact model scores 0.8296 macro F1; a single 280Ah experiment accounts for most newly introduced expert errors. Private performance is unknown.

Install the pinned dependencies in `requirements.txt`, then run from any working directory:

```bash
python submission_packages/public_9448/test.py \
  --input /path/to/test.csv \
  --image_root /path/to/test \
  --sample_submission /path/to/sample_submission.csv \
  --output /path/to/new_predictions.csv
```

Image paths are relative to `image_root`. Omit `--sample_submission` to preserve input row order. The output must not already exist. Inference uses only provided current and past rows in each experiment, does not fit preprocessing or models, and produces the two columns `sample_id,label`. Supplying an incomplete prefix changes the available history; it must contain the available past observations for the relevant experiment.

## Model and provenance

- `weights/model.joblib`: independently trained sensor HistGradientBoosting (60%) and RGB image HistGradientBoosting (40%) probabilities, trained on Train only after excluding eight Train experiments containing observed stage skips. See `weights/training_metadata.json`. The image representation is 348 features calculated from the current image.
- `weights/expert.cbm`: teammate's 93-feature CatBoost 4/5 expert, retrieved from `feature/jungwon` commit `d8754b287404c0ff0f9bf105d824040a36b010c9`. If the base argmax is 4 or 5, the expert replaces it with its 4/5 prediction; all other base predictions are retained.
- `feature_track.py`, `compare.py` and `team_model.py` preserve the feature implementations used for the original evaluation. The additional experiment and training functions in these files are not called during inference.

This package does not contain the dataset or submitted prediction CSV. The model has not been independently evaluated by nested group CV after choosing this combination. Test labels are unavailable; do not interpret agreement between models as correctness.
