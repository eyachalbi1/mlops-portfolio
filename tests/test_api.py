"""Tests for FastAPI endpoints."""
import pickle
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

SAMPLE_PAYLOAD = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 50.0,
    "TotalCharges": 600.0,
    "gender": 1,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
}


@pytest.fixture(autouse=True)
def mock_model(tmp_path: Path, monkeypatch):
    """Create a real model and patch MODEL_PATH + module globals before tests."""
    import pandas as pd

    import src.api.main as api_module

    X = pd.DataFrame(np.random.rand(50, 19), columns=list(SAMPLE_PAYLOAD.keys()))
    y = np.random.randint(0, 2, 50)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    monkeypatch.setattr(api_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(api_module, "_model", model)
    monkeypatch.setattr(api_module, "_model_name", type(model).__name__)
    monkeypatch.setattr(api_module, "_loaded_at", "2024-01-01T00:00:00")


@pytest.fixture
def client():
    from src.api.main import app
    # use_lifespan=False: model already patched via mock_model fixture
    return TestClient(app, raise_server_exceptions=True)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid(client: TestClient) -> None:
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "probability" in body
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_input(client: TestClient) -> None:
    resp = client.post("/predict", json={"bad_field": "value"})
    assert resp.status_code == 422


def test_model_info(client: TestClient) -> None:
    resp = client.get("/model-info")
    assert resp.status_code == 200
    assert "model" in resp.json()
