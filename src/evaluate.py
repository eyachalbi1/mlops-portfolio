"""Evaluate a saved model and print metrics."""
import argparse
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(model_path: str, test_path: str) -> None:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    test = pd.read_csv(test_path)
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_test, y_prob):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--test", default="data/processed/test.csv")
    args = parser.parse_args()
    evaluate(args.model, args.test)
