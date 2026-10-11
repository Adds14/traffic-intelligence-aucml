"""Persistence Baseline Model for PeMS04 Traffic Forecasting.

Persistence forecasting sets all future horizons equal to the last observed
value in the historical window. Both predictions and ground truth are inverse-normalized
to the original traffic flow scale (vehicles/5min) before metrics evaluation.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from src.evaluation.metrics import evaluate_forecast, format_results_table


class PersistenceBaseline:
    """Persistence forecasting: predict future = last observed value.
    
    For each sample, the prediction at ALL future horizons equals
    the last value in the history window (x[:, -1, :]).
    """
    
    def __init__(self, name: str = 'Persistence', forecast_len: int = 12):
        self.name = name
        self.forecast_len = forecast_len
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate persistence predictions.
        
        Args:
            x: (num_samples, history_len, num_sensors) - normalized input
               or (num_samples, history_len, num_sensors, num_features)
        
        Returns:
            predictions: (num_samples, forecast_len, num_sensors)
            Each future step = x[:, -1, :] (last observed value)
        """
        # If multi-feature input, select target channel 0 (flow)
        if x.ndim == 4:
            last_obs = x[:, -1:, :, 0]
        else:
            last_obs = x[:, -1:, :]  # (num_samples, 1, num_sensors)
            
        predictions = np.repeat(last_obs, self.forecast_len, axis=1)  # repeat forecast_len times
        return predictions
    
    def evaluate(
        self,
        test_x: np.ndarray,
        test_y: np.ndarray,
        scaler: Any,
        horizon_steps: Optional[List[int]] = None
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Run prediction, inverse-normalize, compute metrics.
        
        IMPORTANT: Both predictions and targets must be inverse-normalized
        before computing metrics, so results are in original units (vehicles/5min).
        
        Args:
            test_x: Normalized test inputs (num_samples, history_len, num_sensors)
            test_y: Normalized test targets (num_samples, forecast_len, num_sensors)
            scaler: Object or dict/npz containing 'mean' and 'std'
            horizon_steps: List of horizon step indices (default: [2, 5, 11])
            
        Returns:
            metrics_dict: Dictionary with per-horizon and overall metrics
            pred_orig: Predictions in original scale (veh/5min)
            true_orig: Ground truth in original scale (veh/5min)
        """
        if horizon_steps is None:
            horizon_steps = [2, 5, 11]
            
        predictions = self.predict(test_x)
        
        # Extract mean and std
        if isinstance(scaler, (str, os.PathLike)):
            loaded = np.load(scaler)
            mean, std = float(loaded['mean']), float(loaded['std'])
        elif hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
            mean, std = float(scaler.mean), float(scaler.std)
        elif isinstance(scaler, dict) or hasattr(scaler, '__getitem__'):
            mean, std = float(scaler['mean']), float(scaler['std'])
        else:
            raise ValueError(f"Unsupported scaler format: {type(scaler)}")
            
        # Inverse normalize predictions and targets
        pred_orig = predictions * std + mean
        true_orig = test_y * std + mean
        
        metrics = evaluate_forecast(
            y_true=true_orig,
            y_pred=pred_orig,
            horizon_steps=horizon_steps,
            mask_zeros=True
        )
        
        return metrics, pred_orig, true_orig
    
    def run_and_save(
        self,
        data_dir: str = "data/processed",
        output_dir: str = "outputs",
        horizon_steps: Optional[List[int]] = None
    ) -> Tuple[Dict[str, Any], str, str]:
        """Complete baseline execution pipeline.
        
        Load processed data, predict, evaluate, save results.
        Save to outputs/predictions/persistence_predictions.npz
        Save to outputs/metrics/persistence_metrics.json
        
        Returns:
            metrics: Computed metrics dictionary
            pred_path: Path to saved predictions npz
            metrics_path: Path to saved metrics json
        """
        if horizon_steps is None:
            horizon_steps = [2, 5, 11]
            
        test_x_path = os.path.join(data_dir, "test_x.npy")
        test_y_path = os.path.join(data_dir, "test_y.npy")
        scaler_path = os.path.join(data_dir, "scaler.npz")
        
        if not os.path.exists(test_x_path) or not os.path.exists(test_y_path):
            raise FileNotFoundError(f"Processed test data not found in {data_dir}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            
        print(f"Loading test data from {data_dir}...")
        test_x = np.load(test_x_path)
        test_y = np.load(test_y_path)
        scaler_data = np.load(scaler_path)
        
        print(f"Test X shape: {test_x.shape}, Test Y shape: {test_y.shape}")
        print(f"Running {self.name} baseline predictions and evaluation...")
        metrics, pred_orig, true_orig = self.evaluate(
            test_x=test_x,
            test_y=test_y,
            scaler=scaler_data,
            horizon_steps=horizon_steps
        )
        
        pred_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        pred_path = os.path.join(pred_dir, "persistence_predictions.npz")
        metrics_path = os.path.join(metrics_dir, "persistence_metrics.json")
        
        print(f"Saving predictions to: {pred_path}")
        np.savez_compressed(
            pred_path,
            predictions=pred_orig,
            targets=true_orig
        )
        
        metrics_record = {
            "model": self.name,
            "timestamp": datetime.now().isoformat(),
            "data_dir": data_dir,
            "horizons_evaluated": horizon_steps,
            "metrics": metrics
        }
        
        print(f"Saving metrics to: {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(metrics_record, f, indent=4)
            
        return metrics, pred_path, metrics_path
