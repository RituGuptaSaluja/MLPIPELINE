# Data Pipeline with DVC and MLflow for Machine Learning

## Overview
Simple, reproducible ML pipeline using:
- DVC for data/version/experiment pipeline management
- MLflow for experiment tracking and model registry

This README describes repo layout, setup, core commands, and recommended workflow.

## Repo layout (suggested)
- data/
    - raw/           # raw inputs (not tracked by git; tracked by DVC)
    - processed/
- src/
    - data_preparation.py
    - features.py
    - train.py
    - evaluate.py
- params.yaml      # tunable parameters tracked by DVC
- dvc.yaml         # DVC pipeline stages
- models/          # (DVC or MLflow artifact)
- README.md

## Prerequisites
- Git
- Python 3.8+
- pipenv/venv/conda (recommended)
- DVC (>=2.x)
- MLflow (>=1.x)
- Remote storage for DVC (S3/GCS/Azure/SSH/HTTP) or DVC cloud

## Getting started (quick)
1. Clone and enter repo
     - git clone <repo> && cd <repo>
2. Create Python env and install deps
     - python -m venv .venv
     - source .venv/bin/activate   # or .venv\Scripts\activate on Windows
     - pip install -r requirements.txt
3. Initialize DVC and remote
     - dvc init
     - dvc remote add -d origin s3://my-bucket/path   # or other storage
     - git add .dvc/config .gitignore && git commit -m "init dvc"
4. Add raw data and push
     - dvc add data/raw/dataset.csv
     - git add data/raw/dataset.csv.dvc && git commit -m "add raw data"
     - dvc push

## Pipeline design
Use params.yaml to store hyperparameters. Example params.yaml:
```yaml
train:
    epochs: 10
    lr: 0.001
data:
    seed: 42
```

Create a declarative pipeline (dvc.yaml):
- stages: prepare -> featurize -> train -> evaluate
- each stage declares deps, outs, cmd; run with dvc repro

Example minimal dvc stage (train):
```yaml
stages:
    train:
        cmd: python src/train.py --params-file params.yaml
        deps:
            - src/train.py
            - params.yaml
            - data/processed
        outs:
            - models/model.pkl
```

Run pipeline:
- dvc repro
- dvc metrics show
- dvc push

## MLflow integration
- Configure tracking URI (local server or remote)
    - export MLFLOW_TRACKING_URI=http://localhost:5000
- Start UI locally:
    - mlflow ui --port 5000
- In train.py use MLflow API:
    - mlflow.start_run()
    - mlflow.log_param(...)
    - mlflow.log_metric(...)
    - mlflow.log_artifact(...) or mlflow.sklearn.log_model(...)

Example snippet inside training script:
```python
with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.sklearn.log_model(model, "model")
```

## Experiment workflow
1. Update params.yaml or code
2. dvc repro
3. dvc push
4. Commit changes (including .dvc files)
5. Use MLflow UI to compare runs and register models
6. Tag git commits for reproducibility

## Best practices
- Track small metadata and pipelines in Git; large data/artifacts in DVC remote
- Pin params in params.yaml and include param changes in commits
- Use MLflow for artifact storage and model registry (optionally store model artifacts in DVC)
- Use CI (GitHub Actions) to run dvc repro, tests and push artifacts to remote on merge
- Seed randomness and log environment (pip freeze, Python version) for reproducibility

## Useful commands summary
- dvc init
- dvc add data/raw/...
- dvc remote add -d origin <url>
- dvc push / dvc pull
- dvc repro
- dvc metrics show
- mlflow ui
- git add/commit/push

## Further reading
- DVC docs: https://dvc.org/doc
- MLflow docs: https://mlflow.org/docs/latest

Contribute by adding pipeline stages, CI, and automated model promotion to a production registry.