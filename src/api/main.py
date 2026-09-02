"""FastAPI serving: /health, /predict, /model-info + Prometheus metrics."""
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

MODEL_PATH = Path("models/model.pkl")

_model: Any = None
_model_name: str = "unknown"
_loaded_at: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_name, _loaded_at
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _model_name = type(_model).__name__
        _loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    yield
    # Cleanup on shutdown (nothing to do here)


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


class PredictRequest(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info() -> dict:
    return {"model": _model_name, "loaded_at": _loaded_at, "path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = pd.DataFrame([request.model_dump()])
    prediction = int(_model.predict(data)[0])
    probability = round(float(_model.predict_proba(data)[0][1]), 4)

    return PredictResponse(prediction=prediction, probability=probability, model=_model_name)
