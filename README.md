**Project Overview**
- **Description:**: Training and tuning scripts for a Seattle weather classifier using scikit-learn and MLflow. The repository saves artifacts and an MLflow-format model suitable for building a Docker image.

**Requirements**
- **Python:**: 3.10+ recommended
- **Libraries:**: mlflow, scikit-learn, pandas, numpy, joblib, scipy (installed by CI workflow)

**Quick Start**
- **Run training script:**: From the repository root run:

```bash
python "MLProject (folder)/modeling_tuning.py" --data_path weather_preprocessing/seattle_weather_processed.csv --out_dir artifacts --n-iter 30
```

- **Run a short test (dry-run):**: set `--n-iter 1` to speed up:

```bash
python "MLProject (folder)/modeling_tuning.py" --data_path weather_preprocessing/seattle_weather_processed.csv --out_dir artifacts --n-iter 1
```

**CLI arguments**
- **--data_path:**: Path to the processed CSV dataset (default auto-searches `weather_preprocessing/seattle_weather_processed.csv`).
- **--out_dir:**: Directory to save artifacts and MLflow-format model (default `artifacts`).
- **--n-iter, --test-size, --random-state:**: Hyperparameter tuning and split options.

**MLflow & Artifacts**
- **Local tracking:**: If `MLFLOW_TRACKING_URI` is not set, the script uses `file:./mlruns` (local folder).
- **Artifacts:**: classification reports and joblib models are written to `--out_dir`.
- **MLflow model for Docker:**: The best model is saved in MLflow format at `--out_dir/model` so CI can run `mlflow models build-docker --model-uri artifacts/model`.

**GitHub Actions (CI)**
- The workflow runs the training script directly (no separate MLProject folder required). It installs dependencies, runs the script, uploads `artifacts/`, and builds a Docker image from `artifacts/model`.

**Notes / Troubleshooting**
- **Data path resolution:**: The script will try the given `--data_path` and, if missing, will resolve it relative to the script directory so it works when run from the repository root.
- **DagsHub integration:**: Optional and non-fatal — the script will continue if DagsHub package or token is missing (useful in CI).

If you want, I can also add a short `requirements.txt` or pin exact versions for CI reproducibility.
