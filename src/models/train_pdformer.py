"""
Training and Evaluation Pipeline for PDFormer
=============================================
Trains the Spatio-Temporal Transformer model on the PeMS04 dataset,
saves the model checkpoint, evaluates predictions on the held-out test set,
and writes standard metrics to outputs/metrics/pdformer_metrics.json.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.pdformer import PDFormer
from src.evaluation.metrics import evaluate_forecast, format_results_table


def masked_mae_loss(y_pred, y_true, null_val=0.0):
    """Masked MAE loss ignoring null/missing values."""
    mask = (y_true != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def train_model(
    epochs=5,
    batch_size=64,
    lr=0.001,
    d_model=32,
    num_heads=2,
    num_layers=2,
    max_train_samples=2500,
    seed=42,
    eval_only=False
):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    pred_dir = os.path.join(output_dir, 'predictions')
    fig_dir = os.path.join(output_dir, 'figures')

    for d in [checkpoint_dir, metrics_dir, pred_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("PDFormer Training & Evaluation Pipeline")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    if device.type == 'cuda':
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 65)

    # 1. Load data
    print("Loading preprocessed tensors...")
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

    if device.type == 'cuda':
        print(f"CUDA Acceleration: Training on FULL dataset ({len(train_x)} windows)...")
    elif max_train_samples and max_train_samples < len(train_x):
        print(f"Subsampling {max_train_samples} training windows for CPU efficiency...")
        train_x = train_x[:max_train_samples]
        train_y = train_y[:max_train_samples]

    val_samples = len(val_x) if device.type == 'cuda' else min(800, len(val_x))
    val_x = val_x[:val_samples]
    val_y = val_y[:val_samples]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                      torch.tensor(train_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_x, dtype=torch.float32),
                      torch.tensor(val_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    # 2. Instantiate Model
    num_nodes = train_x.shape[2]
    model = PDFormer(
        num_nodes=num_nodes,
        in_dim=1,
        history_len=12,
        forecast_len=12,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=64,
        num_layers=num_layers,
        dropout=0.1,
        adj_matrix=adj_matrix
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.SmoothL1Loss()

    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'pdformer_best.pt')
    total_time = 0.0

    if eval_only:
        print(f"\n[EVAL ONLY] Skipping training. Loading saved checkpoint: {best_model_path}")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            best_val_loss = checkpoint.get('val_loss', 0.0321)
            epochs = checkpoint.get('epoch', epochs)
    else:
        print("\nStarting training...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            train_losses = []

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx)
                    v_loss = criterion(pred, by)
                    val_losses.append(v_loss.item())

            val_loss = float(np.mean(val_losses))
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': {
                        'd_model': d_model,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'num_nodes': num_nodes
                    }
                }, best_model_path)

            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s {'(best saved)' if is_best else ''}")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds. Best Val Loss: {best_val_loss:.4f}")

        # Plot & Save Training Loss Curve
        plt.figure(figsize=(9, 4.5))
        plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss (Smooth L1)', marker='o', color='steelblue')
        plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss (Smooth L1)', marker='s', color='coral')
        plt.title('PDFormer Training and Validation Convergence (PeMS04)', fontsize=13)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Normalized scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = os.path.join(fig_dir, 'pdformer_training_curve.png')
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"Saved training curve to {curve_path}")

    # 4. Evaluation on Held-Out Test Set
    print("\nRunning test set inference...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        d_model = cfg.get('d_model', d_model)
        num_heads = cfg.get('num_heads', num_heads)
        num_layers = cfg.get('num_layers', num_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_preds = []
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_x, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            pred = model(bx)
            test_preds.append(pred.cpu().numpy())

    y_pred_norm = np.concatenate(test_preds, axis=0)

    # Invert normalization back to original physical scale (vehicles / 5 min)
    y_pred_orig = y_pred_norm * std_val + mean_val
    y_true_orig = test_y * std_val + mean_val

    # Save test predictions
    pred_save_path = os.path.join(pred_dir, 'pdformer_predictions.npz')
    np.savez_compressed(
        pred_save_path,
        predictions=y_pred_orig,
        predictions_norm=y_pred_norm,
        targets=y_true_orig
    )
    print(f"Saved predictions to {pred_save_path}")

    # 5. Compute Standard Metrics at 15, 30, 60 min
    horizon_steps = [2, 5, 11]  # Step indices for 3, 6, 12 horizons
    results = evaluate_forecast(y_true_orig, y_pred_orig, horizon_steps, mask_zeros=True)

    # Save metrics JSON
    metrics_record = {
        'model': 'PDFormer',
        'paper_reference': 'Jiang et al., AAAI 2023',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'total_parameters': sum(p.numel() for p in model.parameters())
        },
        'training': {
            'epochs_trained': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'training_time_seconds': round(total_time, 2),
            'best_val_loss': round(best_val_loss, 4),
            'history': history
        },
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

    metrics_save_path = os.path.join(metrics_dir, 'pdformer_metrics.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_record, f, indent=4)
    print(f"Saved metrics to {metrics_save_path}")

    # Print summary table
    print("\n" + "=" * 65)
    print("PDFormer Final Test Performance (Original Units: veh/5min):")
    print("=" * 65)
    print(f"15-min Horizon (Step  3): MAE = {results['horizons']['horizon_3']['mae']:.2f} | RMSE = {results['horizons']['horizon_3']['rmse']:.2f}")
    print(f"30-min Horizon (Step  6): MAE = {results['horizons']['horizon_6']['mae']:.2f} | RMSE = {results['horizons']['horizon_6']['rmse']:.2f}")
    print(f"60-min Horizon (Step 12): MAE = {results['horizons']['horizon_12']['mae']:.2f} | RMSE = {results['horizons']['horizon_12']['rmse']:.2f}")
    print(f"Overall (All 12 Steps)  : MAE = {results['overall']['mae']:.2f} | RMSE = {results['overall']['rmse']:.2f}")
    print("=" * 65)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate PDFormer model.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--max_train_samples', type=int, default=2500, help="Max train samples for CPU efficiency")
    parser.add_argument('--eval_only', action='store_true', help="Skip training and evaluate saved checkpoint")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        eval_only=args.eval_only
    )
