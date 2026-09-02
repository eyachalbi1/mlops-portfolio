.PHONY: install data pipeline train api test lint drift docker-up docker-down clean

install:
	pip install -r requirements.txt

# Generate synthetic dataset (no Kaggle needed)
data:
	python scripts/generate_synthetic_data.py --rows 2000 --output data/raw/churn.csv

# Full DVC pipeline: preprocess + train + evaluate + drift
pipeline:
	dvc repro

# Train only (without DVC)
train:
	python src/data/preprocess.py --input data/raw/churn.csv --output data/processed
	python src/train.py

# Run API locally
api:
	uvicorn src.api.main:app --reload --port 8000

# Tests
test:
	pytest tests/ -v --tb=short

# Lint
lint:
	ruff check src/ tests/ scripts/ monitoring/

# Drift report
drift:
	python monitoring/drift_report.py

# Docker stack
docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

# MLflow server (local)
mlflow:
	mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
