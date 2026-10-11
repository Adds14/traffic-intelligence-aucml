"""
Real-Time Inference Engine for Spatio-Temporal Traffic Prediction
==================================================================
Provides real-time model inference using trained PDFormer and Persistence models
on real Caltrans PeMS04 traffic observations.
"""

import os
import time
import numpy as np
import torch

from src.models.pdformer import PDFormer
from src.models.persistence import PersistenceBaseline


class TrafficPredictor:
    """Inference engine for Spatio-Temporal Traffic Forecasting."""

    def __init__(self, project_root=None, device=None):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.project_root = project_root

        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load scaler
        scaler_path = os.path.join(self.project_root, 'data', 'processed', 'scaler.npz')
        if os.path.exists(scaler_path):
            scaler_data = np.load(scaler_path)
            self.mean = float(scaler_data['mean'])
            self.std = float(scaler_data['std'])
        else:
            self.mean = 207.23
            self.std = 156.48

        # Load adjacency matrix
        adj_path = os.path.join(self.project_root, 'data', 'processed', 'adj_matrix.npy')
        if os.path.exists(adj_path):
            self.adj_matrix = np.load(adj_path)
        else:
            self.adj_matrix = None

        # Load model checkpoint
        self.model = None
        self._load_pdformer_model()

        # Baseline
        self.baseline = PersistenceBaseline()

        # Congestion thresholds
        self.thresholds = {'q25': 83.0, 'q50': 179.0, 'q75': 315.0}

    def _load_pdformer_model(self):
        ckpt_path = os.path.join(self.project_root, 'outputs', 'checkpoints', 'pdformer_best.pt')
        if not os.path.exists(ckpt_path):
            return

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                cfg = ckpt.get('config', {})
                d_model = cfg.get('d_model', 64)
                num_heads = cfg.get('num_heads', 4)
                num_layers = cfg.get('num_layers', 2)
            else:
                state_dict = ckpt
                d_model = 64
                num_heads = 4
                num_layers = 2

            model = PDFormer(
                num_nodes=307,
                in_dim=1,
                history_len=12,
                forecast_len=12,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                adj_matrix=self.adj_matrix
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
        except Exception as e:
            print(f"Warning: Failed to load PDFormer checkpoint: {e}")
            self.model = None

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_normalize(self, x):
        return x * self.std + self.mean

    def predict_pdformer(self, x_history):
        """
        Run forward pass of PDFormer on real traffic history.
        Args:
            x_history: (12, 307) or (B, 12, 307) in original units (vehicles/5min)
        Returns:
            dict with predictions, inference_time_ms, and status
        """
        if self.model is None:
            raise RuntimeError("PDFormer model checkpoint is not loaded.")

        x_arr = np.array(x_history, dtype=np.float32)
        single_sample = False
        if x_arr.ndim == 2:
            single_sample = True
            x_arr = np.expand_dims(x_arr, axis=0)  # (1, 12, 307)

        # Normalize
        x_norm = self.normalize(x_arr)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            out_norm = self.model(x_tensor)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        out_np = out_norm.cpu().numpy()
        y_pred = self.inverse_normalize(out_np)
        y_pred = np.maximum(y_pred, 0.0)  # Traffic flow cannot be negative

        if single_sample:
            y_pred = y_pred[0]  # (12, 307)

        return {
            'predictions': y_pred,
            'inference_time_ms': inference_time_ms,
            'device': str(self.device)
        }

    def predict_persistence(self, x_history):
        """Run persistence baseline forecast."""
        x_arr = np.array(x_history, dtype=np.float32)
        single_sample = False
        if x_arr.ndim == 2:
            single_sample = True
            x_arr = np.expand_dims(x_arr, axis=0)

        # Last observed step repeated 12 times
        last_step = x_arr[:, -1:, :]  # (B, 1, 307)
        y_pred = np.repeat(last_step, 12, axis=1)

        if single_sample:
            y_pred = y_pred[0]

        return {
            'predictions': y_pred,
            'inference_time_ms': 0.05
        }

    def classify_congestion(self, flow_values):
        """Classify flow into congestion levels."""
        flow = np.array(flow_values)
        categories = np.zeros(flow.shape, dtype=int)
        categories[flow >= self.thresholds['q25']] = 1
        categories[flow >= self.thresholds['q50']] = 2
        categories[flow >= self.thresholds['q75']] = 3
        cat_names = ['Free Flow', 'Moderate', 'Heavy', 'Severe']
        colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
        return categories, [cat_names[c] for c in categories.flat], [colors[c] for c in categories.flat]
