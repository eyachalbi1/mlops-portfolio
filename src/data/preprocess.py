"""Preprocessing pipeline: load raw churn CSV, clean, encode, split, save."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess(input_path: str, output_dir: str) -> None:
    df = pd.read_csv(input_path)

    # Drop customerID, fix TotalCharges
    df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Encode binary/categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    train["Churn"] = y_train.values
    test = X_test.copy()
    test["Churn"] = y_test.values

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"Saved {len(train)} train rows and {len(test)} test rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/churn.csv")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    preprocess(args.input, args.output)
