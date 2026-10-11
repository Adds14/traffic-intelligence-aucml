"""Data Preprocessing Pipeline for PeMS04 Traffic Dataset.

Loads raw PeMS04 flow data, generates sliding window samples (12 history, 12 forecast),
splits chronologically (60% train, 20% val, 20% test), fits Z-score normalization
on the training set only, and saves processed arrays and scaler.
"""

import os
import yaml
import numpy as np


def load_raw_data(raw_path: str) -> np.ndarray:
    """Load raw PeMS04 dataset from npz file.
    
    Returns:
        np.ndarray: Array of shape (T, num_sensors, num_features).
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    data = np.load(raw_path)["data"]
    return data


def create_sliding_windows(
    data: np.ndarray,
    history_window: int = 12,
    forecast_window: int = 12,
    target_channel: int = 0
):
    """Generate sliding window samples for multi-step forecasting.
    
    Args:
        data: (T, num_sensors, num_features)
        history_window: Number of historical time steps (default 12)
        forecast_window: Number of future time steps to forecast (default 12)
        target_channel: Channel index for forecasting target (default 0 for flow)
        
    Returns:
        X: (num_samples, history_window, num_sensors)
        Y: (num_samples, forecast_window, num_sensors)
    """
    target_data = data[..., target_channel]  # (T, num_sensors)
    num_timesteps = target_data.shape[0]
    num_samples = num_timesteps - history_window - forecast_window + 1
    
    X = []
    Y = []
    for i in range(num_samples):
        x_window = target_data[i : i + history_window]
        y_window = target_data[i + history_window : i + history_window + forecast_window]
        X.append(x_window)
        Y.append(y_window)
        
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def split_data(X: np.ndarray, Y: np.ndarray, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """Chronologically split data into train, val, test sets."""
    num_samples = len(X)
    num_train = round(num_samples * train_ratio)
    num_val = round(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    train_x, train_y = X[:num_train], Y[:num_train]
    val_x, val_y = X[num_train : num_train + num_val], Y[num_train : num_train + num_val]
    test_x, test_y = X[num_train + num_val:], Y[num_train + num_val:]
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def preprocess_pems04(
    raw_path: str = "data/raw/pems04.npz",
    output_dir: str = "data/processed",
    history_window: int = 12,
    forecast_window: int = 12,
    target_channel: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
):
    """Full preprocessing pipeline: load -> window -> split -> normalize -> save."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading raw data from: {raw_path}")
    raw_data = load_raw_data(raw_path)
    print(f"Raw data shape: {raw_data.shape}")
    
    print(f"Creating sliding windows (history={history_window}, forecast={forecast_window})...")
    X, Y = create_sliding_windows(
        raw_data,
        history_window=history_window,
        forecast_window=forecast_window,
        target_channel=target_channel
    )
    print(f"Total window samples: X={X.shape}, Y={Y.shape}")
    
    print(f"Splitting data chronologically ({train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {(1 - train_ratio - val_ratio)*100:.0f}%)...")
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data(
        X, Y, train_ratio=train_ratio, val_ratio=val_ratio
    )
    print(f"Train samples: {len(train_x)}")
    print(f"Val samples:   {len(val_x)}")
    print(f"Test samples:  {len(test_x)}")
    
    # Fit scaler on train_x only
    mean = float(np.mean(train_x))
    std = float(np.std(train_x))
    print(f"Scaler fit on train: mean={mean:.4f}, std={std:.4f}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.npz")
    np.savez(scaler_path, mean=mean, std=std)
    print(f"Saved scaler to: {scaler_path}")
    
    # Normalize datasets
    norm_train_x = (train_x - mean) / std
    norm_train_y = (train_y - mean) / std
    norm_val_x = (val_x - mean) / std
    norm_val_y = (val_y - mean) / std
    norm_test_x = (test_x - mean) / std
    norm_test_y = (test_y - mean) / std
    
    # Save processed files
    np.save(os.path.join(output_dir, "train_x.npy"), norm_train_x)
    np.save(os.path.join(output_dir, "train_y.npy"), norm_train_y)
    np.save(os.path.join(output_dir, "val_x.npy"), norm_val_x)
    np.save(os.path.join(output_dir, "val_y.npy"), norm_val_y)
    np.save(os.path.join(output_dir, "test_x.npy"), norm_test_x)
    np.save(os.path.join(output_dir, "test_y.npy"), norm_test_y)
    
    print(f"Successfully saved all processed datasets to: {output_dir}")
    return {
        "train_x": norm_train_x.shape,
        "train_y": norm_train_y.shape,
        "val_x": norm_val_x.shape,
        "val_y": norm_val_y.shape,
        "test_x": norm_test_x.shape,
        "test_y": norm_test_y.shape,
        "mean": mean,
        "std": std
    }


if __name__ == "__main__":
    preprocess_pems04()
