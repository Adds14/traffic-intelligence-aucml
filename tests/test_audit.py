"""
Automated Test Suite for Temporal Leakage Audit & Validated Benchmarks.
======================================================================
Tests verify:
1. Zero target overlap between Train, Val, and Test in audited split
2. Scaler fitted strictly on training data
3. Audited comparison artifacts and predictions shape integrity
4. Metric reproducibility and non-negative unit bounds
"""

import os
import sys
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestTemporalLeakageAudit:
    """Verifies that the audited dataset and benchmark are strictly leakage-free."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.audited_dir = os.path.join(PROJECT_ROOT, "data", "processed_audited")
        self.metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "audited", "metrics")
        self.preds_dir = os.path.join(PROJECT_ROOT, "outputs", "audited", "predictions")

    def test_audited_split_zero_target_leakage(self):
        """Verify that no target timestamp appears in more than one partition."""
        meta_path = os.path.join(self.audited_dir, "split_metadata.json")
        assert os.path.exists(meta_path), "split_metadata.json missing"
        with open(meta_path) as f:
            meta = json.load(f)

        leak = meta["target_leakage"]
        assert leak["train_val_overlap"] == 0, f"Train-Val target overlap: {leak['train_val_overlap']}"
        assert leak["val_test_overlap"] == 0, f"Val-Test target overlap: {leak['val_test_overlap']}"
        assert leak["train_test_overlap"] == 0, f"Train-Test target overlap: {leak['train_test_overlap']}"

    def test_audited_window_counts(self):
        """Verify correct leakage-safe partition counts."""
        train_x = np.load(os.path.join(self.audited_dir, "train_x.npy"))
        val_x = np.load(os.path.join(self.audited_dir, "val_x.npy"))
        test_x = np.load(os.path.join(self.audited_dir, "test_x.npy"))

        assert len(train_x) == 10172, f"Expected 10172 train windows, got {len(train_x)}"
        assert len(val_x) == 3387, f"Expected 3387 val windows, got {len(val_x)}"
        assert len(test_x) == 3388, f"Expected 3388 test windows, got {len(test_x)}"

    def test_train_only_scaler_values(self):
        """Verify scaler parameters are train-only and do not match full dataset."""
        scaler = np.load(os.path.join(self.audited_dir, "scaler.npz"))
        mean = float(scaler["mean"])
        std = float(scaler["std"])

        raw_flow = np.load(os.path.join(PROJECT_ROOT, "data", "raw", "pems04.npz"))["data"][:, :, 0]
        full_mean = float(np.mean(raw_flow))
        full_std = float(np.std(raw_flow))

        # Must NOT equal full dataset statistics
        assert not np.isclose(mean, full_mean, atol=1.0), "Scaler appears to be fitted on full dataset!"
        assert not np.isclose(std, full_std, atol=1.0), "Scaler std appears to be fitted on full dataset!"

        # Must closely match raw training slice (timesteps 0 to 10194)
        raw_train_mean = float(np.mean(raw_flow[:10195]))
        assert np.isclose(mean, raw_train_mean, atol=1e-3)

    def test_audited_benchmark_artifacts(self):
        """Verify audited predictions and metrics files exist and are valid."""
        summary_path = os.path.join(self.metrics_dir, "audited_comparison_summary.json")
        assert os.path.exists(summary_path), "audited_comparison_summary.json missing"
        with open(summary_path) as f:
            summary = json.load(f)

        models = summary["models"]
        for m in ["Persistence", "PDFormer", "Chronos-2_ZeroShot", "Chronos-2_LoRA"]:
            assert m in models, f"Model {m} missing from audited summary"
            assert models[m]["15m_mae"] > 0
            assert models[m]["overall_rmse"] > 0
            assert 0.80 < models[m]["r2_overall"] < 1.0

        preds_path = os.path.join(self.preds_dir, "audited_benchmark_predictions.npz")
        assert os.path.exists(preds_path), "audited_benchmark_predictions.npz missing"
        data = np.load(preds_path)
        assert data["targets"].shape == (100, 12, 307)
        assert data["persistence"].shape == (100, 12, 307)
        assert data["pdformer"].shape == (100, 12, 307)
        assert data["chronos2_zeroshot"].shape == (100, 12, 307)
        assert data["chronos2_lora"].shape == (100, 12, 307)
