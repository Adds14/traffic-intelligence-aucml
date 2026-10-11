"""
Standalone PDFormer Execution, Training, and Evaluation Script
=============================================================
Trains PDFormer with optimal batch size 32, computes multi-horizon metrics,
saves checkpoint, predictions, and metrics JSON.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 3 dirnames: file -> models -> src -> traffic-congestion-ml
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.pdformer import PDFormer
from src.evaluation.metrics import evaluate_forecast


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    pred_dir = os.path.join(output_dir, 'predictions')
    fig_dir = os.path.join(output_dir, 'figures')

    for d in [checkpoint_dir, metrics_dir, pred_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 65)
    print("PDFormer Training & Evaluation (Optimized Batch Size: 32)")
    print("=" * 65)

    # 1. Load data
    train_x = np.load(os.path.join(data_dir, 'train_x.npy'))
    train_y = np.load(os.path.join(data_dir, 'train_y.npy'))
    val_x = np.load(os.path.join(data_dir, 'val_x.npy'))
    val_y = np.load(os.path.join(data_dir, 'val_y.npy'))
    test_x = np.load(os.path.join(data_dir, 'test_x.npy'))
    test_y = np.load(os.path.join(data_dir, 'test_y.npy'))
    adj_matrix = np.load(os.path.join(data_dir, 'adj_matrix.npy'))
    scaler = np.load(os.path.join(data_dir, 'scaler.npz'))

    mean_val = float(scaler['mean'])
    std_val = float(scaler['std'])

    # Representative subsets for fast CPU execution
    max_train = 600
    max_val = 150
    batch_size = 32
    epochs = 2

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x[:max_train], dtype=torch.float32),
                      torch.tensor(train_y[:max_train], dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_x[:max_val], dtype=torch.float32),
                      torch.tensor(val_y[:max_val], dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    # 2. Instantiate Model
    num_nodes = train_x.shape[2]
    model = PDFormer(
        num_nodes=num_nodes,
        in_dim=1,
        history_len=12,
        forecast_len=12,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=2,
        dropout=0.1,
        adj_matrix=adj_matrix
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_ckpt = os.path.join(checkpoint_dir, 'pdformer_best.pt')

    print(f"Training on {max_train} windows for {epochs} epochs...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        t_losses = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            t_losses.append(loss.item())

        tr_loss = float(np.mean(t_losses))

        model.eval()
        v_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx)
                v_loss = criterion(pred, by)
                v_losses.append(v_loss.item())

        va_loss = float(np.mean(v_losses))
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_ckpt)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Time: {time.time()-t0:.1f}s")

    print(f"Training finished in {time.time()-t_start:.1f}s.")

    # 3. Test Set Inference (Subsample 600 representative test windows for fast CPU inference)
    print("\nRunning held-out test evaluation...")
    model.load_state_dict(torch.load(best_ckpt))
    model.eval()

    test_sub = test_x[:600]
    test_y_sub = test_y[:600]

    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_sub, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    preds = []
    with torch.no_grad():
        for (bx,) in test_loader:
            pred = model(bx)
            preds.append(pred.cpu().numpy())

    y_pred_norm = np.concatenate(preds, axis=0)

    # Invert normalization back to original physical scale
    y_pred_orig = y_pred_norm * std_val + mean_val
    y_true_orig = test_y_sub * std_val + mean_val

    # Save predictions
    pred_path = os.path.join(pred_dir, 'pdformer_predictions.npz')
    np.savez_compressed(pred_path, predictions=y_pred_orig, targets=y_true_orig)
    print(f"Saved test predictions to {pred_path}")

    # 4. Metrics Evaluation
    results = evaluate_forecast(y_true_orig, y_pred_orig, [2, 5, 11], mask_zeros=True)

    # Write metrics JSON
    metrics_record = {
        'model': 'PDFormer',
        'paper': 'Jiang et al., AAAI 2023',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'metrics': results,
        'per_horizon': {
            'horizon_3': {
                'MAE': results['horizons']['horizon_3']['mae'],
                'RMSE': results['horizons']['horizon_3']['rmse'],
                'horizon_step': 3,
                'horizon_minutes': 15
            },
            'horizon_6': {
                'MAE': results['horizons']['horizon_6']['mae'],
                'RMSE': results['horizons']['horizon_6']['rmse'],
                'horizon_step': 6,
                'horizon_minutes': 30
            },
            'horizon_12': {
                'MAE': results['horizons']['horizon_12']['mae'],
                'RMSE': results['horizons']['horizon_12']['rmse'],
                'horizon_step': 12,
                'horizon_minutes': 60
            }
        },
        'overall': {
            'MAE': results['overall']['mae'],
            'RMSE': results['overall']['rmse']
        }
    }

    metrics_path = os.path.join(metrics_dir, 'pdformer_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_record, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    # 5. Save Plots
    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), history['train_loss'], marker='o', label='Train Loss', color='steelblue')
    plt.plot(range(1, epochs + 1), history['val_loss'], marker='s', label='Val Loss', color='coral')
    plt.title('PDFormer Training Convergence (PeMS04)', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Smooth L1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'pdformer_training_curve.png'), dpi=150)
    plt.close()

    # Model comparison bar chart
    with open(os.path.join(metrics_dir, 'persistence_metrics.json')) as f:
        p_data = json.load(f)

    p_h = p_data['metrics']['horizons']
    p_ov = p_data['metrics']['overall']

    horizons = ['15 min', '30 min', '60 min', 'Overall']
    p_maes = [p_h['horizon_3']['mae'], p_h['horizon_6']['mae'], p_h['horizon_12']['mae'], p_ov['mae']]
    m_maes = [results['horizons']['horizon_3']['mae'], results['horizons']['horizon_6']['mae'], results['horizons']['horizon_12']['mae'], results['overall']['mae']]

    p_rmses = [p_h['horizon_3']['rmse'], p_h['horizon_6']['rmse'], p_h['horizon_12']['rmse'], p_ov['rmse']]
    m_rmses = [results['horizons']['horizon_3']['rmse'], results['horizons']['horizon_6']['rmse'], results['horizons']['horizon_12']['rmse'], results['overall']['rmse']]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(4)
    w = 0.35

    axes[0].bar(x - w/2, p_maes, w, label='Persistence Baseline', color='#7f7f7f', alpha=0.85)
    axes[0].bar(x + w/2, m_maes, w, label='PDFormer', color='#2ca02c', alpha=0.85)
    axes[0].set_title('Mean Absolute Error (MAE) Comparison', fontsize=12)
    axes[0].set_ylabel('MAE (veh/5min)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(horizons)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - w/2, p_rmses, w, label='Persistence Baseline', color='#7f7f7f', alpha=0.85)
    axes[1].bar(x + w/2, m_rmses, w, label='PDFormer', color='#1f77b4', alpha=0.85)
    axes[1].set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=12)
    axes[1].set_ylabel('RMSE (veh/5min)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(horizons)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'model_comparison_bar_chart.png'), dpi=150)
    plt.close()

    print("\n" + "=" * 65)
    print("PDFormer Final Test Performance (veh/5min):")
    print("=" * 65)
    print(f"15-min (Step  3): MAE = {results['horizons']['horizon_3']['mae']:.2f} | RMSE = {results['horizons']['horizon_3']['rmse']:.2f}")
    print(f"30-min (Step  6): MAE = {results['horizons']['horizon_6']['mae']:.2f} | RMSE = {results['horizons']['horizon_6']['rmse']:.2f}")
    print(f"60-min (Step 12): MAE = {results['horizons']['horizon_12']['mae']:.2f} | RMSE = {results['horizons']['horizon_12']['rmse']:.2f}")
    print(f"Overall         : MAE = {results['overall']['mae']:.2f} | RMSE = {results['overall']['rmse']:.2f}")
    print("=" * 65)


if __name__ == '__main__':
    main()
