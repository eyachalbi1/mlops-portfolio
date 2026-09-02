"""Generate Evidently data drift report: reference (train) vs current (test)."""
import argparse
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report


def generate_report(train_path: str, test_path: str, output_path: str) -> None:
    reference = pd.read_csv(train_path)
    current = pd.read_csv(test_path)

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference, current_data=current)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    print(f"Drift report saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output", default="monitoring/drift_report.html")
    args = parser.parse_args()
    generate_report(args.train, args.test, args.output)
