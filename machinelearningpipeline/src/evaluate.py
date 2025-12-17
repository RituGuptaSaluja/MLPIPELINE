import os
from pathlib import Path
import yaml
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import mlflow
from urllib.parse import urlparse

# Optional: set MLflow tracking via environment variables (these were present in the
# original file). Keep them or configure your own tracking server as needed.
os.environ.setdefault('MLFLOW_TRACKING_URI', "https://dagshub.com/RituGuptaSaluja/machinelearningpipeline.mlflow")
os.environ.setdefault('MLFLOW_TRACKING_USERNAME', "RituGuptaSaluja")
os.environ.setdefault('MLFLOW_TRACKING_PASSWORD', "a51131764ded2704687082f6f7c7078fa98b08d2")

## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(pima_diabetes_synthetic_path, model_path):
    data= pd.read_csv(pima_diabetes_synthetic_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/RituGuptaSaluja/machinelearningpipeline.mlflow")

    ## Load the model from disk
    model=pickle.load(open(model_path, 'rb'))
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    ## Log metrics to MLflow

    mlflow.log.metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate(params["data"], params["model"])
    