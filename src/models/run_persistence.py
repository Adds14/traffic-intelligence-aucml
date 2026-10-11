"""Run Persistence Baseline Model Script.

Loads processed PeMS04 data, runs persistence baseline, evaluates MAE and RMSE
at 15min, 30min, and 60min horizons, saves predictions and metrics, and displays
a formatted comparison table.
"""

import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.persistence import PersistenceBaseline
from src.evaluation.metrics import format_results_table


def main():
    print("=" * 70)
    print("PeMS04 Traffic Forecasting: Persistence Baseline Model Evaluation")
    print("=" * 70)
    
    data_dir = os.path.join(project_root, "data", "processed")
    output_dir = os.path.join(project_root, "outputs")
    
    horizon_steps = [2, 5, 11]  # 0-indexed: step 2 (15min), step 5 (30min), step 11 (60min)
    
    baseline = PersistenceBaseline(name="Persistence", forecast_len=12)
    metrics, pred_path, metrics_path = baseline.run_and_save(
        data_dir=data_dir,
        output_dir=output_dir,
        horizon_steps=horizon_steps
    )
    
    print("\n" + "=" * 70)
    print("Evaluation Results Summary")
    print("=" * 70)
    table_str = format_results_table(metrics, model_names=["Persistence"])
    print(table_str)
    print("=" * 70)
    print(f"Predictions saved to: {pred_path}")
    print(f"Metrics saved to:     {metrics_path}")
    print("=" * 70)
    
    return metrics


if __name__ == "__main__":
    main()
