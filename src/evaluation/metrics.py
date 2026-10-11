"""Evaluation Metrics Module for Traffic Congestion Prediction.

Computes MAE and RMSE across multi-horizon forecasts for traffic flow.
Supports masking zero/invalid observations to avoid skewing error metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Mean Absolute Error. Optionally mask zero/invalid values.
    
    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.
        mask: Boolean mask where True indicates valid values to include.
        
    Returns:
        float: Calculated MAE.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """Root Mean Squared Error. Optionally mask zero/invalid values.
    
    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.
        mask: Boolean mask where True indicates valid values to include.
        
    Returns:
        float: Calculated RMSE.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_steps: Optional[List[int]] = None,
    mask_zeros: bool = True
) -> Dict[str, Any]:
    """Evaluate traffic flow predictions at specific horizon steps.
    
    Args:
        y_true: (num_samples, forecast_len, num_sensors) - in ORIGINAL scale.
        y_pred: (num_samples, forecast_len, num_sensors) - in ORIGINAL scale.
        horizon_steps: list of 0-based step indices (e.g., [2, 5, 11] for 15min, 30min, 60min).
                       If None, defaults to [2, 5, 11] or up to available length.
        mask_zeros: if True, exclude zero-flow observations (y_true > 0) from metrics.
        
    Returns:
        dict with per-horizon and overall metrics.
    """
    if horizon_steps is None:
        horizon_steps = [2, 5, 11]
    
    forecast_len = y_true.shape[1]
    horizon_mapping = {
        2: ("Horizon 3 (15 min)", 15, 3),
        5: ("Horizon 6 (30 min)", 30, 6),
        11: ("Horizon 12 (60 min)", 60, 12),
    }
    
    results: Dict[str, Any] = {
        "horizons": {},
        "overall": {},
    }
    
    # Per-horizon evaluation
    for step_idx in horizon_steps:
        if step_idx >= forecast_len:
            continue
        
        y_t_step = y_true[:, step_idx, :]
        y_p_step = y_pred[:, step_idx, :]
        
        mask = (y_t_step > 0) if mask_zeros else None
        
        step_mae = mae(y_t_step, y_p_step, mask=mask)
        step_rmse = rmse(y_t_step, y_p_step, mask=mask)
        
        label, minutes, horizon_num = horizon_mapping.get(
            step_idx,
            (f"Step {step_idx + 1} ({(step_idx + 1) * 5} min)", (step_idx + 1) * 5, step_idx + 1)
        )
        
        horizon_key = f"horizon_{horizon_num}"
        results["horizons"][horizon_key] = {
            "step_index": step_idx,
            "horizon_step": horizon_num,
            "horizon_minutes": minutes,
            "label": label,
            "mae": step_mae,
            "rmse": step_rmse,
        }
        # Flat convenience keys
        results[f"mae_{minutes}min"] = step_mae
        results[f"rmse_{minutes}min"] = step_rmse
    
    # Overall evaluation across all forecast steps
    overall_mask = (y_true > 0) if mask_zeros else None
    overall_mae = mae(y_true, y_pred, mask=overall_mask)
    overall_rmse = rmse(y_true, y_pred, mask=overall_mask)
    
    results["overall"] = {
        "mae": overall_mae,
        "rmse": overall_rmse,
    }
    results["mae_overall"] = overall_mae
    results["rmse_overall"] = overall_rmse
    
    return results


def format_results_table(
    results_dict: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
    model_names: Optional[Union[str, List[str]]] = None
) -> str:
    """Format evaluation results as a comparison table string.
    
    Args:
        results_dict: Either a single model's result dict or a nested dict
                      mapping model name to its result dict.
        model_names: Name or list of names of models.
        
    Returns:
        str: Formatted markdown comparison table.
    """
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Normalize input into dict of {model_name: metrics_dict}
    models_data: Dict[str, Dict[str, Any]] = {}
    if model_names is not None and len(model_names) == 1 and "horizons" in results_dict:
        models_data[model_names[0]] = results_dict
    elif model_names is not None:
        for name in model_names:
            if name in results_dict:
                models_data[name] = results_dict[name]
    else:
        # Check if results_dict has model names as top-level keys
        if "horizons" in results_dict:
            models_data["Model"] = results_dict
        else:
            models_data = results_dict
            
    header = (
        "| Model | 15min MAE | 15min RMSE | 30min MAE | 30min RMSE | 60min MAE | 60min RMSE | Overall MAE | Overall RMSE |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    
    rows = []
    for model_name, res in models_data.items():
        mae_15 = res.get("mae_15min", res.get("horizons", {}).get("horizon_3", {}).get("mae", float("nan")))
        rmse_15 = res.get("rmse_15min", res.get("horizons", {}).get("horizon_3", {}).get("rmse", float("nan")))
        
        mae_30 = res.get("mae_30min", res.get("horizons", {}).get("horizon_6", {}).get("mae", float("nan")))
        rmse_30 = res.get("rmse_30min", res.get("horizons", {}).get("horizon_6", {}).get("rmse", float("nan")))
        
        mae_60 = res.get("mae_60min", res.get("horizons", {}).get("horizon_12", {}).get("mae", float("nan")))
        rmse_60 = res.get("rmse_60min", res.get("horizons", {}).get("horizon_12", {}).get("rmse", float("nan")))
        
        mae_ovr = res.get("mae_overall", res.get("overall", {}).get("mae", float("nan")))
        rmse_ovr = res.get("rmse_overall", res.get("overall", {}).get("rmse", float("nan")))
        
        rows.append(
            f"| {model_name} | {mae_15:.2f} | {rmse_15:.2f} | {mae_30:.2f} | {rmse_30:.2f} | {mae_60:.2f} | {rmse_60:.2f} | {mae_ovr:.2f} | {rmse_ovr:.2f} |"
        )
        
    return "\n".join([header] + rows)
