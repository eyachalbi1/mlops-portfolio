"""Generate a synthetic Telco-like churn dataset for CI and demos.

Usage:
    python scripts/generate_synthetic_data.py --rows 1000 --output data/raw/churn.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int, output: str) -> None:
    rng = np.random.default_rng(seed)

    n = rows
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 120, n).round(2)
    total = (tenure * monthly + rng.uniform(-10, 10, n)).clip(0).round(2)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges": monthly,
        "TotalCharges": [str(v) if rng.random() > 0.003 else " " for v in total],  # ~0.3% missing
    })

    # Churn: realistic signal — contract type, charges, tenure, internet service
    churn_prob = (
        0.03
        + 0.35 * (df["Contract"] == "Month-to-month").astype(float)
        + 0.15 * (df["MonthlyCharges"] > 70).astype(float)
        + 0.10 * (df["InternetService"] == "Fiber optic").astype(float)
        + 0.08 * (df["TechSupport"] == "No").astype(float)
        + 0.08 * (df["OnlineSecurity"] == "No").astype(float)
        - 0.20 * (df["tenure"] > 24).astype(float)
        - 0.15 * (df["Contract"] == "Two year").astype(float)
        - 0.10 * (df["Partner"] == "Yes").astype(float)
    ).clip(0.01, 0.95)
    df["Churn"] = np.where(rng.random(n) < churn_prob, "Yes", "No")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Generated {n} rows -> {output}  (churn rate: {churn_rate:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/raw/churn.csv")
    args = parser.parse_args()
    generate(args.rows, args.seed, args.output)
