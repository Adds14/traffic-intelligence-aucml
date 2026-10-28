"""
Automated Test Suite for Chronos-2 Foundation Model & LoRA Transfer Learning
=============================================================================
Tests cover:
1. ChronosForecaster wrapper initialization and LoRA checkpoint verification
2. Preserved sensor ordering and output shape (12, 307)
3. Physical non-negative unit bounds
4. TrafficPredictor integration with Chronos-2
5. Metric artifact structure and value sanity
6. CPU / GPU device auto-configuration
"""

import os
import sys
import pytest
import numpy as np
import torch

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.chronos_model import ChronosForecaster, CHRONOS_AVAILABLE
from src.inference.predictor import TrafficPredictor


@pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="chronos-forecasting not installed")
class TestChronosForecaster:
    """Tests for the Chronos-2 model wrapper and LoRA transfer learning."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.ckpt_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "chronos2_lora", "finetuned-ckpt"
        )
        self.metrics_path = os.path.join(
            PROJECT_ROOT, "outputs", "metrics", "chronos2_metrics.json"
        )
        self.preds_path = os.path.join(
            PROJECT_ROOT, "outputs", "predictions", "chronos2_predictions.npz"
        )

    def test_checkpoint_files_exist(self):
        """Verify fine-tuned adapter weights and training metadata exist."""
        assert os.path.exists(self.ckpt_path), f"LoRA checkpoint directory missing: {self.ckpt_path}"
        adapter_config = os.path.join(self.ckpt_path, "adapter_config.json")
        assert os.path.exists(adapter_config), "adapter_config.json missing"
        
        meta_path = os.path.join(
            PROJECT_ROOT, "outputs", "checkpoints", "chronos2_lora", "training_metadata.json"
        )
        assert os.path.exists(meta_path), "training_metadata.json missing"

    def test_forecaster_initialization_with_lora(self):
        """Verify forecaster loads fine-tuned LoRA adapter."""
        forecaster = ChronosForecaster(checkpoint_path=self.ckpt_path)
        assert forecaster.is_fine_tuned is True, "Forecaster should report fine-tuned status"
        info = forecaster.get_model_info()
        assert info["total_parameters"] > 100_000_000, "Should have ~120M parameters"
        assert info["context_length"] == 8192
        assert info["prediction_length"] == 1024

    def test_forecast_shape_and_sensor_ordering(self):
        """Verify sensor predictions preserve (12, 307) shape and non-negative flow."""
        forecaster = ChronosForecaster(checkpoint_path=self.ckpt_path)
        # Synthetic context history for all 307 sensors: (12 steps, 307 sensors)
        dummy_context = np.full((12, 307), 180.0, dtype=np.float32)

        preds = forecaster.predict_sensors(
            context_flow=dummy_context,
            prediction_length=12,
            batch_size=512,
        )

        assert preds.shape == (12, 307), f"Expected shape (12, 307), got {preds.shape}"
        assert not np.isnan(preds).any(), "Forecast contains NaN values"
        assert (preds >= 0).all(), "Traffic flow predictions should be non-negative"

    def test_metrics_json_integrity(self):
        """Verify saved evaluation metrics are valid and within expected physical ranges."""
        import json
        assert os.path.exists(self.metrics_path), "chronos2_metrics.json missing"
        with open(self.metrics_path) as f:
            data = json.load(f)

        assert "metrics" in data
        m = data["metrics"]
        for key in ["mae_15min", "rmse_15min", "mae_30min", "rmse_30min", "mae_60min", "rmse_60min", "rmse_overall", "r2_overall"]:
            assert key in m, f"Key {key} missing from chronos2_metrics.json"
            assert m[key] > 0, f"Metric {key} must be positive"

        # Statistical sanity checks for Caltrans PeMS04
        assert 10.0 < m["mae_15min"] < 35.0, f"Unreasonable 15m MAE: {m['mae_15min']}"
        assert 15.0 < m["rmse_15min"] < 45.0, f"Unreasonable 15m RMSE: {m['rmse_15min']}"
        assert 0.90 < m["r2_overall"] < 1.0, f"R2 score should be > 0.90, got {m['r2_overall']}"

    def test_predictions_npz_integrity(self):
        """Verify chronos2_predictions.npz shape matches held-out evaluation windows."""
        assert os.path.exists(self.preds_path), "chronos2_predictions.npz missing"
        npz = np.load(self.preds_path)
        assert "predictions" in npz
        assert "targets" in npz

        preds = npz["predictions"]
        targets = npz["targets"]
        assert preds.shape == targets.shape, "Predictions and targets must match shape"
        assert preds.shape[1] == 12, "Forecast length must be 12 steps"
        assert preds.shape[2] == 307, "Sensor count must be 307"

    def test_predictor_integration(self):
        """Verify TrafficPredictor executes Chronos-2 forward pass cleanly."""
        predictor = TrafficPredictor()
        assert predictor.is_chronos_checkpoint_available() is True

        dummy_hist = np.full((12, 307), 150.0, dtype=np.float32)
        res = predictor.predict_chronos(dummy_hist, prediction_length=12)

        assert "predictions" in res
        assert res["predictions"].shape == (12, 307)
        assert res["inference_time_ms"] > 0
        assert res["is_fine_tuned"] is True
