import os
from urllib.parse import urlparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn


# Optional: set MLflow tracking via environment variables (these were present in the
# original file). Keep them or configure your own tracking server as needed.
os.environ.setdefault('MLFLOW_TRACKING_URI', "https://dagshub.com/RituGuptaSaluja/machinelearningpipeline.mlflow")
os.environ.setdefault('MLFLOW_TRACKING_USERNAME', "RituGuptaSaluja")
os.environ.setdefault('MLFLOW_TRACKING_PASSWORD', "a51131764ded2704687082f6f7c7078fa98b08d2")


def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search


## Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


def train(pima_diabetes_synthetic_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(pima_diabetes_synthetic_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Ensure mlflow uses the configured tracking URI
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

    ## start the mlflow run
    with mlflow.start_run():
        ##Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
        signature = infer_signature(X_train, y_train)

        ## Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        ## Get the best model from grid search
        best_model = grid_search.best_estimator_

        ## predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        ## Log additional metrics
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_param("best_n_estimators", int(grid_search.best_params_["n_estimators"]))
        mlflow.log_param("best_max_depth", str(grid_search.best_params_["max_depth"]))
        mlflow.log_param("best_min_samples_split", int(grid_search.best_params_["min_samples_split"]))
        mlflow.log_param("best_min_samples_leaf", int(grid_search.best_params_["min_samples_leaf"]))

        ## log the confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(cr), "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Try to register model if the tracking server supports model registry.
        # Some hosted tracking servers (e.g., DagsHub) may not support the model
        # registry endpoint; in that case we fall back to logging the model
        # as a run artifact (non-registered) to avoid crashing.
        try:
            if tracking_url_type_store != "file":
                # attempt registry-enabled logging
                mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
            else:
                mlflow.sklearn.log_model(best_model, "model", signature=signature)
        except Exception as e:
            print("Model registry not supported or failed to register model; falling back. Error:", e)
            try:
                mlflow.sklearn.log_model(best_model, "model", signature=signature)
            except Exception as e2:
                print("Failed to log model to MLflow run artifacts:", e2)

    ## create the directory to save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    filename = model_path
    pickle.dump(best_model, open(filename, 'wb'))

    print(f"Trained model saved to {model_path}")


if __name__ == "__main__":
    train(params["data"], params["model"], params.get("random_state", 42), params.get("n_estimators", 100), params.get("max_depth", None))