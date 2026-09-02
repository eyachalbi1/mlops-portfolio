"""CI gate: fail if model F1 is below threshold.

Default threshold: 0.75 (real Kaggle dataset).
Override with env var QUALITY_GATE_F1 for synthetic data in CI.
"""
import json
import os
import sys

# Real dataset threshold: 0.75 | Synthetic CI threshold: 0.45
THRESHOLD_F1 = float(os.getenv("QUALITY_GATE_F1", "0.75"))

with open("metrics.json") as f:
    metrics = json.load(f)

f1 = metrics.get("f1", 0)
print(f"Model : {metrics.get('best_model')}")
print(f"F1    : {f1}")
print(f"Gate  : {THRESHOLD_F1}")

if f1 < THRESHOLD_F1:
    print(f"FAIL: F1 {f1:.4f} < threshold {THRESHOLD_F1}")
    sys.exit(1)

print("PASS: model quality gate passed.")
