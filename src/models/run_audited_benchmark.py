"""
Comprehensive Audited Benchmark Execution on Leakage-Safe PeMS04 Split.
=======================================================================
Evaluates all four models against the EXACT SAME held-out test windows:
1. Persistence Baseline
2. PDFormer (Jiang et al., AAAI 2023)
3. Amazon Chronos-2 (Zero-Shot Base)
4. Amazon Chronos-2 (LoRA Fine-Tuned)

Preserves original benchmark files while outputting to outputs/audited/.
"""

import os
os.environ['USE_TF'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pdformer import PDFormer
from src.models.chronos_model import ChronosForecaster
from src.evaluation.metrics import evaluate_forecast


def run_audited_benchmark(
    num_unified_samples: int = 100,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 70)
    print("AUCML INDEPENDENT BENCHMARK AUDIT: LEAKAGE-SAFE EVALUATION")
    print("=" * 70)
    print(f"Hardware Compute Device: {device.upper()}")
    
    audited_data_dir = PROJECT_ROOT / "data" / "processed_audited"
    out_dir = PROJECT_ROOT / "outputs" / "audited"
    metrics_dir = out_dir / "metrics"
    pred_dir = out_dir / "predictions"
    fig_dir = out_dir / "figures"
    
    for d in [metrics_dir, pred_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Load Audited Test Set
    test_x_norm = np.load(audited_data_dir / "test_x.npy")       # (3388, 12, 307)
    test_x_unnorm = np.load(audited_data_dir / "test_x_unnorm.npy") # (3388, 12, 307)
    test_y_unnorm = np.load(audited_data_dir / "test_y_unnorm.npy") # (3388, 12, 307)
    scaler_data = np.load(audited_data_dir / "scaler.npz")
    scaler_mean = float(scaler_data["mean"])
    scaler_std = float(scaler_data["std"])
    
    total_test_windows = len(test_y_unnorm)
    print(f"Total Valid Leakage-Safe Test Windows: {total_test_windows}")
    print(f"Scaler: Mean = {scaler_mean:.4f}, Std = {scaler_std:.4f}")
    
    # 2. Select Unified Test Indices (evenly spaced across the 3388 windows)
    stride = max(1, total_test_windows // num_unified_samples)
    unified_indices = [i * stride for i in range(num_unified_samples)]
    print(f"Unified Comparison Windows: {len(unified_indices)} (stride={stride})")
    
    y_test_subset = test_y_unnorm[unified_indices]  # Shape: (N, 12, 307)
    x_test_subset_unnorm = test_x_unnorm[unified_indices]
    x_test_subset_norm = test_x_norm[unified_indices]
    
    # =========================================================================
    # MODEL 1: PERSISTENCE BASELINE
    # =========================================================================
    print("\n--- Evaluating Model 1: Persistence Baseline ---")
    t0 = time.time()
    # Forecast equals last historical observation repeated 12 times
    per_pred_all = np.repeat(test_x_unnorm[:, -1:, :], 12, axis=1) # (3388, 12, 307)
    per_latency_all = (time.time() - t0) / total_test_windows * 1000.0
    
    per_pred_subset = per_pred_all[unified_indices]
    
    # Evaluate Persistence on full test set and on subset
    per_metrics_full = evaluate_forecast(test_y_unnorm, per_pred_all, [2, 5, 11], mask_zeros=True)
    per_metrics_sub = evaluate_forecast(y_test_subset, per_pred_subset, [2, 5, 11], mask_zeros=True)
    
    # Calculate R2 on subset via sklearn
    per_r2_15m = r2_score(y_test_subset[:, 2, :].flatten(), per_pred_subset[:, 2, :].flatten())
    per_r2_30m = r2_score(y_test_subset[:, 5, :].flatten(), per_pred_subset[:, 5, :].flatten())
    per_r2_60m = r2_score(y_test_subset[:, 11, :].flatten(), per_pred_subset[:, 11, :].flatten())
    per_r2_all = r2_score(y_test_subset.flatten(), per_pred_subset.flatten())
    
    print(f"Persistence (Subset N={num_unified_samples}): 15m MAE={per_metrics_sub['horizons']['horizon_3']['mae']:.2f}, 30m MAE={per_metrics_sub['horizons']['horizon_6']['mae']:.2f}, 60m MAE={per_metrics_sub['horizons']['horizon_12']['mae']:.2f}, Overall RMSE={per_metrics_sub['overall']['rmse']:.2f}, R2={per_r2_all*100:.2f}%")
    
    # =========================================================================
    # MODEL 2: PDFORMER (PyTorch Checkpoint)
    # =========================================================================
    print("\n--- Evaluating Model 2: PDFormer (ST-Transformer) ---")
    adj_matrix = np.load(PROJECT_ROOT / "data" / "processed" / "adj_matrix.npy")
    pdf_model = PDFormer(
        num_nodes=307, in_dim=1, history_len=12, forecast_len=12,
        d_model=32, num_heads=2, num_layers=2, adj_matrix=adj_matrix
    )
    
    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "pdformer_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    pdf_model.load_state_dict(state_dict)
    pdf_model.to(device)
    pdf_model.eval()
    
    # Run forward pass on full test set
    pdf_preds_all = []
    t_start_pdf = time.time()
    with torch.no_grad():
        for i in range(0, total_test_windows, 64):
            batch_x = torch.tensor(test_x_norm[i : i + 64], dtype=torch.float32).to(device)
            out_norm = pdf_model(batch_x)
            pdf_preds_all.append(out_norm.cpu().numpy())
    pdf_elapsed = time.time() - t_start_pdf
    pdf_latency_per_window = (pdf_elapsed / total_test_windows) * 1000.0
    
    pdf_pred_norm_all = np.concatenate(pdf_preds_all, axis=0)
    pdf_pred_unnorm_all = pdf_pred_norm_all * scaler_std + scaler_mean
    pdf_pred_unnorm_all = np.maximum(pdf_pred_unnorm_all, 0.0)
    
    pdf_pred_subset = pdf_pred_unnorm_all[unified_indices]
    
    # Evaluate PDFormer on full test set and subset
    pdf_metrics_full = evaluate_forecast(test_y_unnorm, pdf_pred_unnorm_all, [2, 5, 11], mask_zeros=True)
    pdf_metrics_sub = evaluate_forecast(y_test_subset, pdf_pred_subset, [2, 5, 11], mask_zeros=True)
    
    pdf_r2_15m = r2_score(y_test_subset[:, 2, :].flatten(), pdf_pred_subset[:, 2, :].flatten())
    pdf_r2_30m = r2_score(y_test_subset[:, 5, :].flatten(), pdf_pred_subset[:, 5, :].flatten())
    pdf_r2_60m = r2_score(y_test_subset[:, 11, :].flatten(), pdf_pred_subset[:, 11, :].flatten())
    pdf_r2_all = r2_score(y_test_subset.flatten(), pdf_pred_subset.flatten())
    
    print(f"PDFormer (Full N=3388): Latency={pdf_latency_per_window:.2f} ms/window")
    print(f"PDFormer (Subset N={num_unified_samples}): 15m MAE={pdf_metrics_sub['horizons']['horizon_3']['mae']:.2f}, 30m MAE={pdf_metrics_sub['horizons']['horizon_6']['mae']:.2f}, 60m MAE={pdf_metrics_sub['horizons']['horizon_12']['mae']:.2f}, Overall RMSE={pdf_metrics_sub['overall']['rmse']:.2f}, R2={pdf_r2_all*100:.2f}%")
    
    # =========================================================================
    # MODEL 3: CHRONOS-2 ZERO-SHOT (Pretrained Foundation Base)
    # =========================================================================
    print("\n--- Evaluating Model 3: Chronos-2 Zero-Shot Base ---")
    chr_zero = ChronosForecaster(checkpoint_path=None, device=device)
    
    t0_zero = time.time()
    zero_preds = []
    for i in range(0, len(unified_indices), batch_size):
        b_x = x_test_subset_unnorm[i : i + batch_size] # (B, 12, 307)
        # Pass to predict_sensors (context_length = 12 steps, matching PDFormer exactly)
        pred_b = chr_zero.predict_sensors(b_x, prediction_length=12, is_normalized=False)
        zero_preds.append(pred_b)
    chr_zero_elapsed = time.time() - t0_zero
    chr_zero_latency = (chr_zero_elapsed / len(unified_indices)) * 1000.0
    
    chr_zero_pred = np.concatenate(zero_preds, axis=0) if len(zero_preds) > 1 else zero_preds[0]
    chr_zero_pred = np.maximum(chr_zero_pred, 0.0)
    
    chr_zero_metrics = evaluate_forecast(y_test_subset, chr_zero_pred, [2, 5, 11], mask_zeros=True)
    chr_zero_r2_15m = r2_score(y_test_subset[:, 2, :].flatten(), chr_zero_pred[:, 2, :].flatten())
    chr_zero_r2_30m = r2_score(y_test_subset[:, 5, :].flatten(), chr_zero_pred[:, 5, :].flatten())
    chr_zero_r2_60m = r2_score(y_test_subset[:, 11, :].flatten(), chr_zero_pred[:, 11, :].flatten())
    chr_zero_r2_all = r2_score(y_test_subset.flatten(), chr_zero_pred.flatten())
    
    print(f"Chronos-2 Zero-Shot (Subset N={num_unified_samples}): Latency={chr_zero_latency:.1f} ms/window")
    print(f"  15m MAE={chr_zero_metrics['horizons']['horizon_3']['mae']:.2f}, 30m MAE={chr_zero_metrics['horizons']['horizon_6']['mae']:.2f}, 60m MAE={chr_zero_metrics['horizons']['horizon_12']['mae']:.2f}, Overall RMSE={chr_zero_metrics['overall']['rmse']:.2f}, R2={chr_zero_r2_all*100:.2f}%")
    
    # Clean GPU memory before loading LoRA adapter
    del chr_zero
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # =========================================================================
    # MODEL 4: CHRONOS-2 LoRA FINE-TUNED
    # =========================================================================
    print("\n--- Evaluating Model 4: Chronos-2 LoRA Fine-Tuned ---")
    lora_ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "chronos2_lora" / "finetuned-ckpt"
    chr_lora = ChronosForecaster(checkpoint_path=lora_ckpt_dir, device=device)
    assert chr_lora.is_fine_tuned is True, "LoRA adapter failed to report fine-tuned status"
    
    t0_lora = time.time()
    lora_preds = []
    for i in range(0, len(unified_indices), batch_size):
        b_x = x_test_subset_unnorm[i : i + batch_size]
        pred_b = chr_lora.predict_sensors(b_x, prediction_length=12, is_normalized=False)
        lora_preds.append(pred_b)
    chr_lora_elapsed = time.time() - t0_lora
    chr_lora_latency = (chr_lora_elapsed / len(unified_indices)) * 1000.0
    
    chr_lora_pred = np.concatenate(lora_preds, axis=0) if len(lora_preds) > 1 else lora_preds[0]
    chr_lora_pred = np.maximum(chr_lora_pred, 0.0)
    
    chr_lora_metrics = evaluate_forecast(y_test_subset, chr_lora_pred, [2, 5, 11], mask_zeros=True)
    chr_lora_r2_15m = r2_score(y_test_subset[:, 2, :].flatten(), chr_lora_pred[:, 2, :].flatten())
    chr_lora_r2_30m = r2_score(y_test_subset[:, 5, :].flatten(), chr_lora_pred[:, 5, :].flatten())
    chr_lora_r2_60m = r2_score(y_test_subset[:, 11, :].flatten(), chr_lora_pred[:, 11, :].flatten())
    chr_lora_r2_all = r2_score(y_test_subset.flatten(), chr_lora_pred.flatten())
    
    print(f"Chronos-2 LoRA (Subset N={num_unified_samples}): Latency={chr_lora_latency:.1f} ms/window")
    print(f"  15m MAE={chr_lora_metrics['horizons']['horizon_3']['mae']:.2f}, 30m MAE={chr_lora_metrics['horizons']['horizon_6']['mae']:.2f}, 60m MAE={chr_lora_metrics['horizons']['horizon_12']['mae']:.2f}, Overall RMSE={chr_lora_metrics['overall']['rmse']:.2f}, R2={chr_lora_r2_all*100:.2f}%")
    
    # Save Unified Predictions NPZ
    np.savez_compressed(
        pred_dir / "audited_benchmark_predictions.npz",
        targets=y_test_subset,
        inputs=x_test_subset_unnorm,
        persistence=per_pred_subset,
        pdformer=pdf_pred_subset,
        chronos2_zeroshot=chr_zero_pred,
        chronos2_lora=chr_lora_pred,
        unified_indices=np.array(unified_indices)
    )
    print(f"\nSaved audited prediction tensors to: {pred_dir / 'audited_benchmark_predictions.npz'}")
    
    # Build Unified Comparison Dictionary
    comparison_summary = {
        "audit_metadata": {
            "dataset": "Caltrans PeMS04",
            "eval_type": "Leakage-Safe Held-Out Chronological Test Evaluation",
            "test_target_range": [13593, 16991],
            "total_test_windows": total_test_windows,
            "evaluated_test_windows": len(unified_indices),
            "stride": stride,
            "scaler": {"mean": scaler_mean, "std": scaler_std},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": device
        },
        "models": {
            "Persistence": {
                "type": "Statistical Autoregressive Baseline",
                "15m_mae": float(per_metrics_sub['horizons']['horizon_3']['mae']),
                "15m_rmse": float(per_metrics_sub['horizons']['horizon_3']['rmse']),
                "15m_r2": float(per_r2_15m),
                "30m_mae": float(per_metrics_sub['horizons']['horizon_6']['mae']),
                "30m_rmse": float(per_metrics_sub['horizons']['horizon_6']['rmse']),
                "30m_r2": float(per_r2_30m),
                "60m_mae": float(per_metrics_sub['horizons']['horizon_12']['mae']),
                "60m_rmse": float(per_metrics_sub['horizons']['horizon_12']['rmse']),
                "60m_r2": float(per_r2_60m),
                "overall_mae": float(per_metrics_sub['overall']['mae']),
                "overall_rmse": float(per_metrics_sub['overall']['rmse']),
                "r2_overall": float(per_r2_all),
                "latency_ms_per_window": float(per_latency_all)
            },
            "PDFormer": {
                "type": "Spatio-Temporal Graph Transformer (AAAI '23)",
                "15m_mae": float(pdf_metrics_sub['horizons']['horizon_3']['mae']),
                "15m_rmse": float(pdf_metrics_sub['horizons']['horizon_3']['rmse']),
                "15m_r2": float(pdf_r2_15m),
                "30m_mae": float(pdf_metrics_sub['horizons']['horizon_6']['mae']),
                "30m_rmse": float(pdf_metrics_sub['horizons']['horizon_6']['rmse']),
                "30m_r2": float(pdf_r2_30m),
                "60m_mae": float(pdf_metrics_sub['horizons']['horizon_12']['mae']),
                "60m_rmse": float(pdf_metrics_sub['horizons']['horizon_12']['rmse']),
                "60m_r2": float(pdf_r2_60m),
                "overall_mae": float(pdf_metrics_sub['overall']['mae']),
                "overall_rmse": float(pdf_metrics_sub['overall']['rmse']),
                "r2_overall": float(pdf_r2_all),
                "latency_ms_per_window": float(pdf_latency_per_window),
                "full_test_metrics": {
                    "overall_mae": float(pdf_metrics_full['overall']['mae']),
                    "overall_rmse": float(pdf_metrics_full['overall']['rmse'])
                }
            },
            "Chronos-2_ZeroShot": {
                "type": "Amazon Chronos-2 Foundation Model (Zero-Shot)",
                "15m_mae": float(chr_zero_metrics['horizons']['horizon_3']['mae']),
                "15m_rmse": float(chr_zero_metrics['horizons']['horizon_3']['rmse']),
                "15m_r2": float(chr_zero_r2_15m),
                "30m_mae": float(chr_zero_metrics['horizons']['horizon_6']['mae']),
                "30m_rmse": float(chr_zero_metrics['horizons']['horizon_6']['rmse']),
                "30m_r2": float(chr_zero_r2_30m),
                "60m_mae": float(chr_zero_metrics['horizons']['horizon_12']['mae']),
                "60m_rmse": float(chr_zero_metrics['horizons']['horizon_12']['rmse']),
                "60m_r2": float(chr_zero_r2_60m),
                "overall_mae": float(chr_zero_metrics['overall']['mae']),
                "overall_rmse": float(chr_zero_metrics['overall']['rmse']),
                "r2_overall": float(chr_zero_r2_all),
                "latency_ms_per_window": float(chr_zero_latency)
            },
            "Chronos-2_LoRA": {
                "type": "Amazon Chronos-2 with LoRA Transfer Learning",
                "15m_mae": float(chr_lora_metrics['horizons']['horizon_3']['mae']),
                "15m_rmse": float(chr_lora_metrics['horizons']['horizon_3']['rmse']),
                "15m_r2": float(chr_lora_r2_15m),
                "30m_mae": float(chr_lora_metrics['horizons']['horizon_6']['mae']),
                "30m_rmse": float(chr_lora_metrics['horizons']['horizon_6']['rmse']),
                "30m_r2": float(chr_lora_r2_30m),
                "60m_mae": float(chr_lora_metrics['horizons']['horizon_12']['mae']),
                "60m_rmse": float(chr_lora_metrics['horizons']['horizon_12']['rmse']),
                "60m_r2": float(chr_lora_r2_60m),
                "overall_mae": float(chr_lora_metrics['overall']['mae']),
                "overall_rmse": float(chr_lora_metrics['overall']['rmse']),
                "r2_overall": float(chr_lora_r2_all),
                "latency_ms_per_window": float(chr_lora_latency)
            }
        }
    }
    
    with open(metrics_dir / "audited_comparison_summary.json", "w") as f:
        json.dump(comparison_summary, f, indent=4)
    print(f"Saved audited comparison summary to: {metrics_dir / 'audited_comparison_summary.json'}")
    
    # Save individual model metrics
    for m_name, m_key in [("persistence_audited_metrics.json", "Persistence"),
                          ("pdformer_audited_metrics.json", "PDFormer"),
                          ("chronos2_zeroshot_audited_metrics.json", "Chronos-2_ZeroShot"),
                          ("chronos2_lora_audited_metrics.json", "Chronos-2_LoRA")]:
        with open(metrics_dir / m_name, "w") as f:
            json.dump({"model": m_key, "metrics": comparison_summary["models"][m_key]}, f, indent=4)
            
    # Generate Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h_labels = ['15 min', '30 min', '60 min', 'Overall']
    x = np.arange(len(h_labels))
    w = 0.2
    
    colors = ['#f59e0b', '#38bdf8', '#818cf8', '#c084fc']
    model_keys = ['Persistence', 'PDFormer', 'Chronos-2_ZeroShot', 'Chronos-2_LoRA']
    labels = ['Persistence', 'PDFormer', 'Chronos-2 Base', 'Chronos-2 LoRA']
    
    # MAE Chart
    for idx, (mk, lbl, c) in enumerate(zip(model_keys, labels, colors)):
        m = comparison_summary['models'][mk]
        vals = [m['15m_mae'], m['30m_mae'], m['60m_mae'], m['overall_mae']]
        axes[0].bar(x + (idx - 1.5) * w, vals, w, label=lbl, color=c)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(h_labels)
    axes[0].set_ylabel('MAE (veh / 5 min)')
    axes[0].set_title('Audited MAE Comparison (Exact Same Held-Out Targets)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RMSE Chart
    for idx, (mk, lbl, c) in enumerate(zip(model_keys, labels, colors)):
        m = comparison_summary['models'][mk]
        vals = [m['15m_rmse'], m['30m_rmse'], m['60m_rmse'], m['overall_rmse']]
        axes[1].bar(x + (idx - 1.5) * w, vals, w, label=lbl, color=c)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(h_labels)
    axes[1].set_ylabel('RMSE (veh / 5 min)')
    axes[1].set_title('Audited RMSE Comparison (Exact Same Held-Out Targets)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "audited_model_comparison_bar.png", dpi=150)
    plt.close()
    
    # Print Audited Results Table
    print("\n" + "=" * 80)
    print("AUDITED RESULTS TABLE: EXACT SAME HELD-OUT TARGET OBSERVATIONS")
    print("=" * 80)
    print(f"{'Model':<24} | {'15m MAE':<7} | {'15m RMSE':<8} | {'30m MAE':<7} | {'30m RMSE':<8} | {'60m MAE':<7} | {'60m RMSE':<8} | {'Overall RMSE':<12} | {'R2':<6} | {'Latency':<9}")
    print("-" * 115)
    for mk, lbl in zip(model_keys, labels):
        m = comparison_summary['models'][mk]
        print(f"{lbl:<24} | {m['15m_mae']:<7.2f} | {m['15m_rmse']:<8.2f} | {m['30m_mae']:<7.2f} | {m['30m_rmse']:<8.2f} | {m['60m_mae']:<7.2f} | {m['60m_rmse']:<8.2f} | {m['overall_rmse']:<12.2f} | {m['r2_overall']*100:<5.1f}% | {m['latency_ms_per_window']:<6.1f} ms")
    print("=" * 80)
    
    return comparison_summary


if __name__ == "__main__":
    run_audited_benchmark()
