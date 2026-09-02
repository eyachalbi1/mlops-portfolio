"""Tests for preprocessing pipeline."""
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 24, 60],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 20.25],
        "TotalCharges": ["29.85", "1889.5", ""],
        "Churn": ["No", "No", "Yes"],
    })
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return path


def test_preprocess_creates_files(raw_csv: Path, tmp_path: Path) -> None:
    from src.data.preprocess import preprocess
    out = tmp_path / "processed"
    preprocess(str(raw_csv), str(out))
    assert (out / "train.csv").exists()
    assert (out / "test.csv").exists()


def test_preprocess_no_missing(raw_csv: Path, tmp_path: Path) -> None:
    from src.data.preprocess import preprocess
    out = tmp_path / "processed"
    preprocess(str(raw_csv), str(out))
    train = pd.read_csv(out / "train.csv")
    assert train.isnull().sum().sum() == 0


def test_preprocess_churn_column(raw_csv: Path, tmp_path: Path) -> None:
    from src.data.preprocess import preprocess
    out = tmp_path / "processed"
    preprocess(str(raw_csv), str(out))
    train = pd.read_csv(out / "train.csv")
    assert "Churn" in train.columns
    assert set(train["Churn"].unique()).issubset({0, 1})
