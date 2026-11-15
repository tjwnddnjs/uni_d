# Stable dynamic router

CatBoost sensor and image models with a 4/5 specialist. This directory contains the original fitted weights and inference code. Install the versions in `requirements.txt` and run:

```bash
python test.py --input /path/to/input.csv --image_root /path/to/images_root --output /path/to/predictions.csv --model_dir .
```

Pass `--sample_submission /path/to/sample_submission.csv` to use that file's sample order. Image paths in the input CSV are resolved relative to `--image_root`.

The original training scripts are included for reference. Their data, image-feature cache and anchor model paths are specific to the original workspace, and `train.py` also runs test inference; adjust them before retraining in a different environment.
