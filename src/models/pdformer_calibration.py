"""Validation-fitted per-horizon, per-sensor affine forecast calibration."""

import numpy as np


def fit_affine_calibration(predictions, targets, shrinkage=1000.0):
    """Fit a lightly regularized affine correction from validation forecasts.

    Arrays use physical units and shape (samples, forecast_steps, sensors).
    Corrections shrink toward identity to limit sensor-specific overfit.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("predictions and targets must share (samples, steps, sensors) shape")

    _, steps, sensors = predictions.shape
    scale = np.ones((steps, sensors), dtype=np.float32)
    bias = np.zeros((steps, sensors), dtype=np.float32)
    for step in range(steps):
        for sensor in range(sensors):
            x = predictions[:, step, sensor]
            y = targets[:, step, sensor]
            valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x, y = x[valid], y[valid]
            variance = x.var() if x.size else 0.0
            if x.size < 2 or variance < 1e-8:
                continue
            raw_scale = np.mean((x - x.mean()) * (y - y.mean())) / variance
            raw_bias = y.mean() - raw_scale * x.mean()
            weight = x.size / (x.size + shrinkage)
            scale[step, sensor] = 1.0 + weight * (raw_scale - 1.0)
            bias[step, sensor] = weight * raw_bias
    return scale, bias


def apply_affine_calibration(predictions, scale, bias):
    """Apply saved calibration and enforce nonnegative traffic flow."""
    predictions = np.asarray(predictions, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if predictions.shape[-2:] != scale.shape or scale.shape != bias.shape:
        raise ValueError("calibration shape must match (forecast_steps, sensors)")
    return np.maximum(predictions * scale + bias, 0.0)
