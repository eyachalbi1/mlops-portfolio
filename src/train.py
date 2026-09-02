"""Train multiple models, track with MLflow, register best model."""
import argparse
import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def get_models(scale_pos_weight: float) -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


def log_model_safe(model, name: str) -> None:
    """Log model to MLflow, compatible with MLflow 2.x and 3.x."""
    if "XGB" in type(model).__name__:
        mlflow.xgboost.log_model(model, artifact_path=name)
    else:
        mlflow.sklearn.log_model(model, artifact_path=name)


def train(train_path: str, test_path: str, model_dir: str) -> None:
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    # Compute class imbalance ratio for XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = round(neg / pos, 2) if pos > 0 else 1.0
    print(f"Class ratio neg/pos: {scale_pos_weight}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churn-prediction")

    best_f1, best_name, best_model = 0.0, "", None

    for name, model in get_models(scale_pos_weight).items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            log_model_safe(model, name=name)

            print(f"{name}: {metrics}")

            if metrics["f1"] > best_f1:
                best_f1, best_name, best_model = metrics["f1"], name, model

    # Save best model locally
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir) / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # Save metrics for CI quality gate
    with open("metrics.json", "w") as f:
        json.dump({"best_model": best_name, "f1": best_f1}, f)

    # Register best model in MLflow registry (skip if server unavailable)
    try:
        runs = mlflow.search_runs(
            experiment_names=["churn-prediction"],
            filter_string=f"tags.mlflow.runName = '{best_name}'",
            order_by=["metrics.f1 DESC"],
        )
        if not runs.empty:
            model_uri = f"runs:/{runs.iloc[0]['run_id']}/{best_name}"
            mlflow.register_model(model_uri, "churn-model")
    except Exception as exc:
        print(f"[warn] Model registry skipped: {exc}")

    print(f"Best model: {best_name} (F1={best_f1}), saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train(args.train, args.test, args.model_dir)
