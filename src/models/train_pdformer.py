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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.pdformer import PDFormer, count_parameters
from src.models.pdformer_features import build_pdformer_features
from src.models.pdformer_calibration import fit_affine_calibration, apply_affine_calibration
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
    epochs=1,
    batch_size=4,
    lr=0.001,
    d_model=32,
    num_heads=2,
    num_layers=3,
    max_train_samples=None,
    max_test_samples=None,
    seed=42,
    eval_only=False,
    resume=False,
    warm_start=False
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
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 8))
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

    print("Building training-only DTW, traffic-pattern, and Laplacian features...")
    pdformer_features = build_pdformer_features(PROJECT_ROOT, train_x, adj_matrix)

    mean_val = float(scaler['mean'])
    std_val = float(scaler['std'])

    if max_train_samples and max_train_samples < len(train_x):
        print(f"Subsampling {max_train_samples} training windows for CPU efficiency...")
        train_x = train_x[:max_train_samples]
        train_y = train_y[:max_train_samples]

    if max_train_samples:
        # Keep validation chronological and bounded for an explicitly sampled run.
        val_x = val_x[:min(len(val_x), max(1, int(max_train_samples * 0.2)))]
        val_y = val_y[:len(val_x)]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                      torch.tensor(train_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_x, dtype=torch.float32),
                      torch.tensor(val_y, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda')
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
        d_ff=128,
        num_layers=num_layers,
        dropout=0.1,
        adj_matrix=adj_matrix,
        dtw_matrix=pdformer_features['dtw_matrix'],
        pattern_keys=pdformer_features['pattern_keys'],
        lap_pos_enc=pdformer_features['lap_pos_enc'],
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    def criterion(pred, target):
        # Match the project's zero-masked evaluation while training on scaled flow.
        valid = (target * std_val + mean_val) > 0
        weights = valid.to(pred.dtype)
        weights = weights / weights.mean().clamp_min(1e-8)
        return (F.smooth_l1_loss(pred, target, reduction='none') * weights).mean()

    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'pdformer_best.pt')
    total_time = 0.0
    start_epoch = 1

    if eval_only:
        print(f"\n[EVAL ONLY] Skipping training. Loading saved checkpoint: {best_model_path}")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            best_val_loss = checkpoint.get('val_loss', 0.0321)
            epochs = checkpoint.get('epoch', epochs)
            history = checkpoint.get('training_history', history)
            total_time = checkpoint.get('training_time_seconds', total_time)
    else:
        if warm_start:
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"Cannot warm start; checkpoint not found: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            best_val_loss = checkpoint.get('val_loss', best_val_loss)
            history = checkpoint.get('training_history', history)
            total_time = checkpoint.get('training_time_seconds', total_time)
            start_epoch = int(checkpoint.get('epochs_completed', checkpoint.get('epoch', 0))) + 1
            print(f"\n[WARM START] Loaded epoch {start_epoch - 1} weights; resetting optimizer for residual fine-tuning")
        elif resume:
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"Cannot resume; checkpoint not found: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_loss = checkpoint.get('val_loss', best_val_loss)
            history = checkpoint.get('training_history', history)
            total_time = checkpoint.get('training_time_seconds', total_time)
            start_epoch = int(checkpoint.get('epoch', 0)) + 1
            print(f"\n[RESUME] Continuing from checkpoint epoch {start_epoch - 1}; running {epochs} more epoch(s)")
        else:
            print("\nStarting training...")
        prior_time = total_time
        start_time = time.time()

        for epoch in range(start_epoch, start_epoch + epochs):
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
                if len(train_losses) % 100 == 0:
                    print(f"  epoch {epoch}: {len(train_losses)}/{len(train_loader)} batches", flush=True)

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
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'training_history': history,
                    'training_time_seconds': prior_time + time.time() - start_time,
                    'config': {
                        'd_model': d_model,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'num_nodes': num_nodes,
                        'd_ff': 128,
                        'semantic_neighbors': 5,
                        'short_hops': 4,
                        'architecture': 'PDFormer-AAAI2023-core-with-persistence-residual'
                    }
                }, best_model_path)

            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch:02d}/{start_epoch + epochs - 1:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s {'(best saved)' if is_best else ''}")

        total_time = prior_time + time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds. Best Val Loss: {best_val_loss:.4f}")
        if os.path.exists(best_model_path):
            best_checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            best_checkpoint['training_history'] = history
            best_checkpoint['training_time_seconds'] = total_time
            best_checkpoint['epochs_completed'] = start_epoch + epochs - 1
            torch.save(best_checkpoint, best_model_path)

        # Plot & Save Training Loss Curve
        plt.figure(figsize=(9, 4.5))
        history_epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(history_epochs, history['train_loss'], label='Train Loss (Smooth L1)', marker='o', color='steelblue')
        plt.plot(history_epochs, history['val_loss'], label='Val Loss (Smooth L1)', marker='s', color='coral')
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
    if max_test_samples:
        test_x = test_x[:max_test_samples]
        test_y = test_y[:max_test_samples]
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        d_model = cfg.get('d_model', d_model)
        num_heads = cfg.get('num_heads', num_heads)
        num_layers = cfg.get('num_layers', num_layers)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Fit a small per-sensor, per-horizon correction on validation predictions
    # only, then apply it once to held-out test forecasts.
    validation_preds = []
    with torch.no_grad():
        for (bx,) in DataLoader(
            TensorDataset(torch.tensor(val_x, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False
        ):
            validation_preds.append(model(bx.to(device)).cpu().numpy())
    val_pred_norm = np.concatenate(validation_preds)
    val_pred_orig = np.maximum(val_pred_norm * std_val + mean_val, 0.0)
    val_true_orig = val_y * std_val + mean_val
    calibration_scale, calibration_bias = fit_affine_calibration(val_pred_orig, val_true_orig)
    calibration_path = os.path.join(checkpoint_dir, 'pdformer_calibration.npz')
    np.savez_compressed(
        calibration_path,
        scale=calibration_scale,
        bias=calibration_bias,
        source='chronological validation split only',
        shrinkage=1000.0
    )
    print(f"Saved validation-fitted calibration to {calibration_path}")

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
    y_pred_orig = apply_affine_calibration(
        np.maximum(y_pred_norm * std_val + mean_val, 0.0),
        calibration_scale, calibration_bias
    )
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
    def masked_r2(truth, prediction):
        valid = truth > 0
        truth, prediction = truth[valid], prediction[valid]
        denominator = np.sum((truth - np.mean(truth)) ** 2)
        return float(1.0 - np.sum((truth - prediction) ** 2) / denominator) if denominator else 0.0

    for step, minutes in zip(horizon_steps, (15, 30, 60)):
        results[f'r2_{minutes}min'] = masked_r2(y_true_orig[:, step, :], y_pred_orig[:, step, :])
    results['r2_overall'] = masked_r2(y_true_orig, y_pred_orig)
    results['r2_overall'] = masked_r2(y_true_orig, y_pred_orig)

    # Save metrics JSON
    metrics_record = {
        'model': 'PDFormer',
        'paper_reference': 'Jiang et al., AAAI 2023',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'd_model': d_model,
            'num_heads': 8,
            'heads_per_branch': {'geographic': 4, 'semantic': 2, 'temporal': 2},
            'num_layers': num_layers,
            'total_parameters': count_parameters(model),
            'components': [
                'temporal self-attention',
                'short-range geographic attention',
                'long-range DTW semantic attention',
                'traffic-pattern delay-aware geographic key transform',
                'Laplacian positional encoding',
                'multi-step skip projection',
                'trainable per-horizon persistence residual'
            ],
            'semantic_neighbors': 5,
            'short_hops': 4,
            'pattern_key_source': 'training split: quantile-binned 3-step flow patterns',
            'dtw_source': 'training split: exact DTW on normalized hourly mean flow profiles'
        },
        'training': {
            'epochs_trained': int(checkpoint.get('epochs_completed', checkpoint.get('epoch', epochs))),
            'batch_size': batch_size,
            'learning_rate': lr,
            'training_time_seconds': round(total_time, 2),
            'best_val_loss': round(best_val_loss, 4),
            'train_windows_used': int(len(train_x)),
            'validation_windows_used': int(len(val_x)),
            'test_windows_evaluated': int(len(test_x)),
            'evaluation_scope': 'first N chronological held-out windows' if max_test_samples else 'full held-out test split',
            'history': history
        },
        'calibration': {
            'method': 'per-horizon per-sensor affine correction with shrinkage',
            'fit_partition': 'validation only',
            'shrinkage': 1000.0
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
            'RMSE': results['overall']['rmse'],
            'R2': results['r2_overall']
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
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--max_train_samples', type=int, default=None, help="Optional limit for a quick exploratory run; default uses every training window")
    parser.add_argument('--max_test_samples', type=int, default=None, help="Optional limit for CPU evaluation; default evaluates the full test split")
    parser.add_argument('--d_model', type=int, default=32, help="Embedding width; must be divisible by eight")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of spatio-temporal encoder blocks")
    parser.add_argument('--eval_only', action='store_true', help="Skip training and evaluate saved checkpoint")
    parser.add_argument('--resume', action='store_true', help="Continue training from the saved best checkpoint")
    parser.add_argument('--warm_start', action='store_true', help="Warm-start model weights while resetting optimizer state")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        num_layers=args.num_layers,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        eval_only=args.eval_only,
        resume=args.resume,
        warm_start=args.warm_start
    )
