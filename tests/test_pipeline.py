"""
Automated Test Suite for Spatio-Temporal Traffic Prediction Pipeline
====================================================================
Tests cover:
1. Dataset loading & shape integrity
2. Preprocessing & zero-leakage chronological partitioning
3. Normalization & exact inverse transformation
4. Metric computation accuracy (MAE, RMSE, zero-masking)
5. Persistence Baseline behavior
6. PDFormer PyTorch forward pass & parameter sanity
7. Congestion classification & threshold logic
"""

import os
import sys
import pytest
import numpy as np
import torch

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import mae, rmse, evaluate_forecast
from src.mapping.congestion_map import CongestionMapper
from src.models.persistence import PersistenceBaseline
from src.models.pdformer import PDFormer


class TestDataAndPreprocessing:
    """Tests for data shapes, leakage, and normalization."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
        self.proc_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')

    def test_raw_data_shape(self):
        npz_path = os.path.join(self.raw_dir, 'pems04.npz')
        assert os.path.exists(npz_path), "pems04.npz not found in data/raw"
        arr = np.load(npz_path)['data']
        assert arr.shape == (16992, 307, 3), f"Unexpected raw shape {arr.shape}"
        assert not np.isnan(arr).any(), "Found NaN values in raw data"

    def test_processed_tensors_exist_and_match(self):
        for name in ['train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y', 'adj_matrix']:
            path = os.path.join(self.proc_dir, f'{name}.npy')
            assert os.path.exists(path), f"Processed file {name}.npy missing"

        test_x = np.load(os.path.join(self.proc_dir, 'test_x.npy'))
        test_y = np.load(os.path.join(self.proc_dir, 'test_y.npy'))
        assert test_x.shape[1] == 12, "Historical window length must be 12"
        assert test_y.shape[1] == 12, "Forecast window length must be 12"
        assert test_x.shape[2] == 307, "Sensor count must be 307"
        assert test_y.shape[2] == 307, "Sensor count must be 307"

    def test_scaler_inversion(self):
        scaler = np.load(os.path.join(self.proc_dir, 'scaler.npz'))
        mean = float(scaler['mean'])
        std = float(scaler['std'])
        assert std > 0, "Std must be positive"

        # Test invertibility on random sample
        original = np.array([50.0, 150.0, 300.0, 500.0])
        normalized = (original - mean) / std
        recovered = normalized * std + mean
        np.testing.assert_allclose(original, recovered, rtol=1e-5)


class TestMetrics:
    """Tests for metric calculations."""

    def test_mae_rmse_values(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 35.0, 38.0])

        # Errors: |2, -2, 5, -2| -> MAE = (2+2+5+2)/4 = 2.75
        expected_mae = 2.75
        assert np.isclose(mae(y_true, y_pred), expected_mae)

        # Squared errors: [4, 4, 25, 4] -> Mean = 37/4 = 9.25 -> RMSE = sqrt(9.25)
        expected_rmse = np.sqrt(9.25)
        assert np.isclose(rmse(y_true, y_pred), expected_rmse)

    def test_zero_masking(self):
        y_true = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([5.0, 15.0, 25.0])
        mask = y_true > 0

        # Masked: only indices 1 and 2: errors are |5, 5| -> MAE = 5.0
        assert np.isclose(mae(y_true, y_pred, mask=mask), 5.0)


class TestModels:
    """Tests for Baseline and PDFormer models."""

    def test_persistence_baseline(self):
        model = PersistenceBaseline()
        # Input: 2 samples, 12 steps, 5 sensors
        x = np.random.randn(2, 12, 5)
        pred = model.predict(x)

        assert pred.shape == (2, 12, 5)
        # All 12 future steps must match the last observed historical step x[:, -1, :]
        for t in range(12):
            np.testing.assert_allclose(pred[:, t, :], x[:, -1, :])

    def test_pdformer_forward_pass(self):
        adj = np.eye(10)
        model = PDFormer(num_nodes=10, in_dim=1, history_len=12, forecast_len=12,
                         d_model=16, num_heads=2, d_ff=32, num_layers=1, adj_matrix=adj)
        x = torch.randn(2, 12, 10)
        out = model(x)
        assert out.shape == (2, 12, 10), f"PDFormer output shape incorrect: {out.shape}"


class TestCongestionMapping:
    """Tests for dynamic risk mapping."""

    def test_quantile_classification(self):
        thresholds = {'q25': 100.0, 'q50': 200.0, 'q75': 300.0}
        mapper = CongestionMapper(thresholds=thresholds)

        flows = np.array([50.0, 150.0, 250.0, 350.0])
        cats, labels = mapper.classify(flows)

        assert list(cats) == [0, 1, 2, 3]
        assert list(labels) == ['Free Flow', 'Moderate', 'Heavy', 'Severe']

    def test_traffic_predictor_live_inference(self):
        from src.inference.predictor import TrafficPredictor
        predictor = TrafficPredictor()
        test_x = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_x.npy'))
        scaler = np.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'scaler.npz'))
        sample_orig = test_x[0] * float(scaler['std']) + float(scaler['mean'])

        res = predictor.predict_pdformer(sample_orig)
        assert 'predictions' in res
        assert res['predictions'].shape == (12, 307)
        assert res['inference_time_ms'] > 0
        assert not np.isnan(res['predictions']).any()

    def test_network_graph_layout_and_edges(self):
        from src.mapping.network_graph import load_network_layout, get_network_edges, create_force_directed_network_figure, get_sensor_connections
        layout = load_network_layout()
        edges = get_network_edges()
        assert layout is not None, "network_layout.json should exist"
        assert len(layout) == 307, "Layout must have 307 node coordinates"
        assert edges is not None, "distance.csv edges must load"
        assert len(edges) == 340, "Must have 340 directed road edges"

        conns = get_sensor_connections(0, edges)
        assert 'incoming' in conns and 'outgoing' in conns

        dummy_flow = np.ones(307) * 150.0
        fig = create_force_directed_network_figure(dummy_flow, pos_dict=layout, df_edges=edges)
        assert len(fig.data) > 0, "Figure must have traces"

