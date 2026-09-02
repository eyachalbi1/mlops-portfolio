"""Tests for model training and evaluation."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def trained_model(tmp_path: Path):
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path, X, y


def test_model_predict_shape(trained_model) -> None:
    path, X, _ = trained_model
    with open(path, "rb") as f:
        model = pickle.load(f)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_model_predict_proba_range(trained_model) -> None:
    path, X, _ = trained_model
    with open(path, "rb") as f:
        model = pickle.load(f)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_model_binary_output(trained_model) -> None:
    path, X, _ = trained_model
    with open(path, "rb") as f:
        model = pickle.load(f)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})
