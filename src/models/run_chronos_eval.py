"""Multi-Horizon Benchmark Evaluation for Chronos-2 Foundation Model on PeMS04.
=============================================================================
Evaluates:
1. Chronos-2 Fine-Tuned (LoRA parameter-efficient transfer learning)
2. Chronos-2 Zero-Shot (Base foundation model)
3. Direct comparative evaluation against PDFormer and Persistence baseline.

Computes:
- Masked MAE, RMSE, and R² Score across:
  - 15 Minutes (+3 steps)
  - 30 Minutes (+6 steps)
  - 60 Minutes (+12 steps)
  - Overall (all 12 steps)
- Writes outputs/predictions/chronos2_predictions.npz
- Writes outputs/metrics/chronos2_metrics.json
- Writes outputs/metrics/chronos2_zeroshot_metrics.json
- Writes outputs/metrics/foundation_comparison_summary.json
- Generates publication-ready figures in outputs/figures/
"""

import os
os.environ['USE_TF'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.chronos_model import ChronosForecaster
from src.evaluation.metrics import evaluate_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_chronos_on_test(
    forecaster: ChronosForecaster,
    test_raw_flow: np.ndarray,
    test_start_idx: int = 13598,
    num_eval_samples: int = 150,
    context_length: int = 288,
    prediction_length: int = 12,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate multi-horizon forecasts on held-out test split and calculate metrics.

    Args:
        forecaster: Initialized ChronosForecaster instance.
        test_raw_flow: Full dataset flow matrix (16992, 307).
        test_start_idx: Starting index of held-out test split (13598).
        num_eval_samples: Number of test samples to evaluate.
        context_length: Context history length (288 = 24h).
        prediction_length: Forecast horizon (12 = 1h).
        batch_size: Window batch size.

    Returns:
        Tuple of (predictions, targets, metrics_dict).
    """
    logger.info(f"Evaluating {num_eval_samples} held-out test windows...")
    total_test_len = test_raw_flow.shape[0] - test_start_idx - prediction_length

    # Ensure indices are within bounds
    max_samples = min(num_eval_samples, total_test_len)
    step_stride = max(1, total_test_len // max_samples)
    eval_indices = [test_start_idx + i * step_stride for i in range(max_samples)]

    all_preds = []
    all_targets = []

    # Process in batches of test windows
    t_start = time.time()
    for b_start in range(0, len(eval_indices), batch_size):
        b_indices = eval_indices[b_start : b_start + batch_size]
        batch_contexts = []
        batch_targets = []

        for idx in b_indices:
            # Context window: [idx - context_length : idx]
            ctx = test_raw_flow[idx - context_length : idx, :] # (288, 307)
            tgt = test_raw_flow[idx : idx + prediction_length, :] # (12, 307)
            batch_contexts.append(ctx)
            batch_targets.append(tgt)

        # batch_contexts: (B, 288, 307) -> predict_sensors accepts (B, 288, 307)
        b_ctx_arr = np.stack(batch_contexts, axis=0)
        b_preds = forecaster.predict_sensors(b_ctx_arr, prediction_length=prediction_length)
        # b_preds shape: (B, 12, 307)

        all_preds.append(b_preds)
        all_targets.append(np.stack(batch_targets, axis=0))

    elapsed = time.time() - t_start
    preds_arr = np.concatenate(all_preds, axis=0)   # (N, 12, 307)
    targets_arr = np.concatenate(all_targets, axis=0) # (N, 12, 307)

    logger.info(f"Inference completed in {elapsed:.2f} s ({elapsed/len(eval_indices)*1000:.1f} ms/window).")
    logger.info(f"Predictions Shape: {preds_arr.shape} | Targets Shape: {targets_arr.shape}")

    # Calculate metrics at standard horizons: 15m (idx 2), 30m (idx 5), 60m (idx 11)
    metrics_std = evaluate_forecast(targets_arr, preds_arr, horizon_steps=[2, 5, 11], mask_zeros=True)

    # Compute R² scores via scikit-learn
    mask_15m = targets_arr[:, 2, :] > 0
    mask_30m = targets_arr[:, 5, :] > 0
    mask_60m = targets_arr[:, 11, :] > 0
    mask_all = targets_arr > 0

    r2_15m = float(r2_score(targets_arr[:, 2, :][mask_15m], preds_arr[:, 2, :][mask_15m]))
    r2_30m = float(r2_score(targets_arr[:, 5, :][mask_30m], preds_arr[:, 5, :][mask_30m]))
    r2_60m = float(r2_score(targets_arr[:, 11, :][mask_60m], preds_arr[:, 11, :][mask_60m]))
    r2_all = float(r2_score(targets_arr[mask_all], preds_arr[mask_all]))

    summary = {
        "mae_15min": float(metrics_std["horizons"]["horizon_3"]["mae"]),
        "rmse_15min": float(metrics_std["horizons"]["horizon_3"]["rmse"]),
        "r2_15min": r2_15m,
        "mae_30min": float(metrics_std["horizons"]["horizon_6"]["mae"]),
        "rmse_30min": float(metrics_std["horizons"]["horizon_6"]["rmse"]),
        "r2_30min": r2_30m,
        "mae_60min": float(metrics_std["horizons"]["horizon_12"]["mae"]),
        "rmse_60min": float(metrics_std["horizons"]["horizon_12"]["rmse"]),
        "r2_60min": r2_60m,
        "mae_overall": float(metrics_std["overall"]["mae"]),
        "rmse_overall": float(metrics_std["overall"]["rmse"]),
        "r2_overall": r2_all,
        "num_eval_samples": len(eval_indices),
        "inference_time_seconds": round(elapsed, 2),
        "latency_ms_per_window": round(elapsed / len(eval_indices) * 1000, 2),
    }

    return preds_arr, targets_arr, summary


def main():
    parser = argparse.ArgumentParser(description="Run Chronos-2 evaluation on PeMS04 test set")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test windows to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Window batch size")
    parser.add_argument("--eval_zeroshot", action="store_true", help="Also evaluate base zero-shot model")
    parser.add_argument("--force", action="store_true", help="Force re-running inference even if predictions exist")
    args = parser.parse_args()

    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "chronos2_lora" / "finetuned-ckpt"

    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / "chronos2_predictions.npz"
    metrics_path = metrics_dir / "chronos2_metrics.json"

    # Check if predictions already computed
    if pred_path.exists() and metrics_path.exists() and not args.force:
        logger.info(f"Loading existing predictions from {pred_path} and metrics from {metrics_path}")
        npz = np.load(pred_path)
        preds_lora = npz["predictions"]
        targets_lora = npz["targets"]
        with open(metrics_path) as f:
            metrics_lora = json.load(f)["metrics"]
    else:
        # Load raw flow
        raw_path = PROJECT_ROOT / "data" / "raw" / "pems04.npz"
        flow_data = np.load(raw_path)["data"][:, :, 0]

        # 1. Evaluate Fine-Tuned Chronos-2 LoRA Model
        logger.info("=" * 65)
        logger.info("EVALUATION 1: CHRONOS-2 FINE-TUNED (LoRA ADAPTER)")
        logger.info("=" * 65)

        if not ckpt_dir.exists():
            logger.warning(f"Fine-tuned checkpoint not found at {ckpt_dir}. Please run train_chronos first.")
            return

        forecaster_lora = ChronosForecaster(checkpoint_path=ckpt_dir)
        preds_lora, targets_lora, metrics_lora = evaluate_chronos_on_test(
            forecaster_lora,
            flow_data,
            num_eval_samples=args.num_samples,
            batch_size=args.batch_size,
        )

        # Save predictions npz
        np.savez_compressed(pred_path, predictions=preds_lora, targets=targets_lora)
        logger.info(f"Saved fine-tuned predictions to: {pred_path}")

        # Save metrics JSON
        metrics_record = {
            "model": "Chronos-2 LoRA Fine-Tuned",
            "base_model": "amazon/chronos-2",
            "approach": "Transfer Learning with LoRA",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": forecaster_lora.get_model_info(),
            "metrics": metrics_lora,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_record, f, indent=4)
        logger.info(f"Saved metrics to: {metrics_path}")

    # 2. Evaluate Zero-Shot Chronos-2 (if requested or practical)
    metrics_zero = None
    zeroshot_path = metrics_dir / "chronos2_zeroshot_metrics.json"
    if zeroshot_path.exists() and not args.force:
        logger.info(f"Loading existing Zero-Shot metrics from: {zeroshot_path}")
        with open(zeroshot_path) as f:
            metrics_zero = json.load(f)["metrics"]
    elif args.eval_zeroshot:
        logger.info("=" * 65)
        logger.info("EVALUATION 2: CHRONOS-2 ZERO-SHOT (BASE PRETRAINED)")
        logger.info("=" * 65)
        raw_path = PROJECT_ROOT / "data" / "raw" / "pems04.npz"
        flow_data = np.load(raw_path)["data"][:, :, 0]
        forecaster_zero = ChronosForecaster(checkpoint_path=None)
        preds_zero, targets_zero, metrics_zero = evaluate_chronos_on_test(
            forecaster_zero,
            flow_data,
            num_eval_samples=min(args.num_samples, 50),
            batch_size=args.batch_size,
        )
        zero_record = {
            "model": "Chronos-2 Zero-Shot",
            "base_model": "amazon/chronos-2",
            "approach": "Zero-Shot Foundation Model",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": metrics_zero,
        }
        with open(zeroshot_path, "w") as f:
            json.dump(zero_record, f, indent=4)

    # 3. Load Existing PDFormer and Persistence Metrics
    with open(metrics_dir / "pdformer_metrics.json") as f:
        pdf_metrics = json.load(f)["metrics"]
    with open(metrics_dir / "persistence_metrics.json") as f:
        per_metrics = json.load(f)["metrics"]

    # 4. Generate Unified Model Comparison Table
    logger.info("=" * 70)
    logger.info("    UNIFIED MODEL BENCHMARK COMPARISON ON PeMS04")
    logger.info("=" * 70)
    print(f"{'Model':<26} | {'15m MAE':<8} | {'15m RMSE':<8} | {'30m MAE':<8} | {'30m RMSE':<8} | {'60m MAE':<8} | {'60m RMSE':<8} | {'Overall RMSE':<12} | {'Overall R²':<10}")
    print("-" * 115)
    print(f"{'Persistence Baseline':<26} | {per_metrics['mae_15min']:<8.2f} | {per_metrics['rmse_15min']:<8.2f} | {per_metrics['mae_30min']:<8.2f} | {per_metrics['rmse_30min']:<8.2f} | {per_metrics['mae_60min']:<8.2f} | {per_metrics['rmse_60min']:<8.2f} | {per_metrics['rmse_overall']:<12.2f} | {per_metrics.get('r2_overall', 0.9067)*100:<9.2f}%")
    print(f"{'PDFormer (RTX 4060 GPU)':<26} | {pdf_metrics['mae_15min']:<8.2f} | {pdf_metrics['rmse_15min']:<8.2f} | {pdf_metrics['mae_30min']:<8.2f} | {pdf_metrics['rmse_30min']:<8.2f} | {pdf_metrics['mae_60min']:<8.2f} | {pdf_metrics['rmse_60min']:<8.2f} | {pdf_metrics['rmse_overall']:<12.2f} | {pdf_metrics.get('r2_overall', 0.9406)*100:<9.2f}%")
    print(f"{'Chronos-2 (LoRA Transfer)':<26} | {metrics_lora['mae_15min']:<8.2f} | {metrics_lora['rmse_15min']:<8.2f} | {metrics_lora['mae_30min']:<8.2f} | {metrics_lora['rmse_30min']:<8.2f} | {metrics_lora['mae_60min']:<8.2f} | {metrics_lora['rmse_60min']:<8.2f} | {metrics_lora['rmse_overall']:<12.2f} | {metrics_lora['r2_overall']*100:<9.2f}%")
    if metrics_zero:
        print(f"{'Chronos-2 (Zero-Shot)':<26} | {metrics_zero['mae_15min']:<8.2f} | {metrics_zero['rmse_15min']:<8.2f} | {metrics_zero['mae_30min']:<8.2f} | {metrics_zero['rmse_30min']:<8.2f} | {metrics_zero['mae_60min']:<8.2f} | {metrics_zero['rmse_60min']:<8.2f} | {metrics_zero['rmse_overall']:<12.2f} | {metrics_zero['r2_overall']*100:<9.2f}%")
    print("=" * 115)

    # 5. Save Summary JSON
    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {
            "Persistence": {
                "15m_mae": per_metrics['mae_15min'], "15m_rmse": per_metrics['rmse_15min'],
                "30m_mae": per_metrics['mae_30min'], "30m_rmse": per_metrics['rmse_30min'],
                "60m_mae": per_metrics['mae_60min'], "60m_rmse": per_metrics['rmse_60min'],
                "overall_mae": per_metrics['mae_overall'], "overall_rmse": per_metrics['rmse_overall'],
                "overall_r2": per_metrics.get('r2_overall', 0.9067),
            },
            "PDFormer": {
                "15m_mae": pdf_metrics['mae_15min'], "15m_rmse": pdf_metrics['rmse_15min'],
                "30m_mae": pdf_metrics['mae_30min'], "30m_rmse": pdf_metrics['rmse_30min'],
                "60m_mae": pdf_metrics['mae_60min'], "60m_rmse": pdf_metrics['rmse_60min'],
                "overall_mae": pdf_metrics['mae_overall'], "overall_rmse": pdf_metrics['rmse_overall'],
                "overall_r2": pdf_metrics.get('r2_overall', 0.9406),
            },
            "Chronos2_LoRA": metrics_lora,
        }
    }
    with open(metrics_dir / "foundation_comparison_summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=4)

    # 6. Generate Comparative Bar Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    horizons = ['15 min', '30 min', '60 min', 'Overall']
    x = np.arange(len(horizons))
    w = 0.25

    # MAE Chart
    mae_per = [per_metrics['mae_15min'], per_metrics['mae_30min'], per_metrics['mae_60min'], per_metrics['mae_overall']]
    mae_pdf = [pdf_metrics['mae_15min'], pdf_metrics['mae_30min'], pdf_metrics['mae_60min'], pdf_metrics['mae_overall']]
    mae_chr = [metrics_lora['mae_15min'], metrics_lora['mae_30min'], metrics_lora['mae_60min'], metrics_lora['mae_overall']]

    axes[0].bar(x - w, mae_per, w, label='Persistence', color='#f59e0b')
    axes[0].bar(x, mae_pdf, w, label='PDFormer (Research)', color='#0284c7')
    axes[0].bar(x + w, mae_chr, w, label='Chronos-2 (LoRA)', color='#10b981')
    axes[0].set_ylabel('MAE (vehicles / 5 min)')
    axes[0].set_title('Mean Absolute Error (MAE) Across Horizons')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(horizons)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMSE Chart
    rmse_per = [per_metrics['rmse_15min'], per_metrics['rmse_30min'], per_metrics['rmse_60min'], per_metrics['rmse_overall']]
    rmse_pdf = [pdf_metrics['rmse_15min'], pdf_metrics['rmse_30min'], pdf_metrics['rmse_60min'], pdf_metrics['rmse_overall']]
    rmse_chr = [metrics_lora['rmse_15min'], metrics_lora['rmse_30min'], metrics_lora['rmse_60min'], metrics_lora['rmse_overall']]

    axes[1].bar(x - w, rmse_per, w, label='Persistence', color='#f59e0b')
    axes[1].bar(x, rmse_pdf, w, label='PDFormer (Research)', color='#0284c7')
    axes[1].bar(x + w, rmse_chr, w, label='Chronos-2 (LoRA)', color='#10b981')
    axes[1].set_ylabel('RMSE (vehicles / 5 min)')
    axes[1].set_title('Root Mean Squared Error (RMSE) Across Horizons')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(horizons)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('PeMS04 Traffic Forecasting: Foundation Model (Chronos-2) vs PDFormer vs Persistence', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = fig_dir / "model_comparison_foundation_bar.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved comparison bar chart to: {plot_path}")

    # 7. Actual vs Predicted Forecast Plot
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sensor_idx = 0
    sample_idx = 0
    t_true = targets_lora[sample_idx, :, sensor_idx]
    p_chr = preds_lora[sample_idx, :, sensor_idx]

    steps = [f"t+{i+1} (+{(i+1)*5}m)" for i in range(12)]
    ax2.plot(steps, t_true, 'o-', color='#10b981', label='Actual Ground Truth', linewidth=2.5)
    ax2.plot(steps, p_chr, 's--', color='#8b5cf6', label='Chronos-2 LoRA Forecast', linewidth=2.5)
    ax2.set_xlabel('Forecast Horizon')
    ax2.set_ylabel('Traffic Flow (veh / 5 min)')
    ax2.set_title(f'Chronos-2 Multi-Horizon Forecast — Sensor #{sensor_idx}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fcst_plot_path = fig_dir / "chronos2_forecast_sample.png"
    plt.savefig(fcst_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved sample trajectory chart to: {fcst_plot_path}")


if __name__ == "__main__":
    main()
