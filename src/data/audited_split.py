"""
Leakage-Safe Chronological Partitioning for PeMS04.
===================================================
Enforces strict disjoint target intervals:
- Total timesteps: 16,992 (0 to 16991).
- Training Target Period: [12, 10194] -> 10,172 windows (origin 11 to 10182).
- Validation Target Period: [10195, 13592] -> 3,387 windows (origin 10194 to 13580).
- Test Target Period: [13593, 16991] -> 3,388 windows (origin 13592 to 16979).

Zero target overlap between any partitions.
Historical input windows legitimately use preceding observations available at forecast origin.
"""

import os
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "pems04.npz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_audited"


def create_leakage_safe_splits(history_len=12, forecast_len=12):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading raw PeMS04 from: {RAW_PATH}")
    raw_data = np.load(RAW_PATH)["data"]  # (16992, 307, 3)
    flow = raw_data[:, :, 0]  # Flow channel (16992, 307)
    total_steps, num_sensors = flow.shape
    
    # 60% / 20% / 20% temporal boundaries on raw timeline
    # 16992 * 0.6 = 10195.2 -> 10195
    # 16992 * 0.2 = 3398.4 -> 3398
    train_cutoff = 10195  # Timestamps 0 to 10194
    val_cutoff = 10195 + 3398  # 13593 -> Timestamps 10195 to 13592
    # Test timestamps: 13593 to 16991
    
    print(f"Raw Timeline Boundaries:")
    print(f"  • Train observation interval: [0, {train_cutoff - 1}] ({train_cutoff} timestamps)")
    print(f"  • Val observation interval:   [{train_cutoff}, {val_cutoff - 1}] ({val_cutoff - train_cutoff} timestamps)")
    print(f"  • Test observation interval:  [{val_cutoff}, {total_steps - 1}] ({total_steps - val_cutoff} timestamps)")
    
    # Fit scaler strictly on raw training observations: flow[:train_cutoff]
    scaler_mean = float(np.mean(flow[:train_cutoff]))
    scaler_std = float(np.std(flow[:train_cutoff]))
    print(f"\nFitted Scaler (Train-Only, Unwindowed Raw Series):")
    print(f"  • Mean: {scaler_mean:.6f}")
    print(f"  • Std:  {scaler_std:.6f}")
    
    # Generate Training Windows
    # Target must be contained strictly in [0, train_cutoff - 1] -> target end <= train_cutoff - 1
    # Target: [origin + 1 : origin + 1 + forecast_len]
    # origin + forecast_len <= train_cutoff - 1 => origin <= train_cutoff - 1 - forecast_len
    # First origin: history_len - 1 (so input [0 : history_len] is valid)
    train_x, train_y = [], []
    train_origins = []
    for origin in range(history_len - 1, train_cutoff - forecast_len):
        x = flow[origin - history_len + 1 : origin + 1]
        y = flow[origin + 1 : origin + 1 + forecast_len]
        train_x.append(x)
        train_y.append(y)
        train_origins.append(origin)
    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)
    
    # Generate Validation Windows
    # Target must start at train_cutoff (first target is [train_cutoff : train_cutoff + forecast_len])
    # So first origin is train_cutoff - 1
    # Target must end <= val_cutoff - 1
    # origin + forecast_len <= val_cutoff - 1 => origin <= val_cutoff - 1 - forecast_len
    val_x, val_y = [], []
    val_origins = []
    for origin in range(train_cutoff - 1, val_cutoff - forecast_len):
        x = flow[origin - history_len + 1 : origin + 1]
        y = flow[origin + 1 : origin + 1 + forecast_len]
        val_x.append(x)
        val_y.append(y)
        val_origins.append(origin)
    val_x = np.array(val_x, dtype=np.float32)
    val_y = np.array(val_y, dtype=np.float32)
    
    # Generate Test Windows
    # Target must start at val_cutoff (first target is [val_cutoff : val_cutoff + forecast_len])
    # So first origin is val_cutoff - 1
    # Target must end <= total_steps - 1
    test_x, test_y = [], []
    test_origins = []
    for origin in range(val_cutoff - 1, total_steps - forecast_len):
        x = flow[origin - history_len + 1 : origin + 1]
        y = flow[origin + 1 : origin + 1 + forecast_len]
        test_x.append(x)
        test_y.append(y)
        test_origins.append(origin)
    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.float32)
    
    # Verify zero target overlap
    train_target_set = set()
    for o in train_origins:
        train_target_set.update(range(o + 1, o + 1 + forecast_len))
        
    val_target_set = set()
    for o in val_origins:
        val_target_set.update(range(o + 1, o + 1 + forecast_len))
        
    test_target_set = set()
    for o in test_origins:
        test_target_set.update(range(o + 1, o + 1 + forecast_len))
        
    overlap_train_val = train_target_set.intersection(val_target_set)
    overlap_val_test = val_target_set.intersection(test_target_set)
    overlap_train_test = train_target_set.intersection(test_target_set)
    
    print(f"\nWindow Generation Audit:")
    print(f"  • Train Windows: {train_x.shape[0]} | Target Range: [{min(train_target_set)}, {max(train_target_set)}]")
    print(f"  • Val Windows:   {val_x.shape[0]} | Target Range: [{min(val_target_set)}, {max(val_target_set)}]")
    print(f"  • Test Windows:  {test_x.shape[0]} | Target Range: [{min(test_target_set)}, {max(test_target_set)}]")
    print(f"  • Target Overlap Train <-> Val:  {len(overlap_train_val)} (Should be 0)")
    print(f"  • Target Overlap Val <-> Test:   {len(overlap_val_test)} (Should be 0)")
    print(f"  • Target Overlap Train <-> Test: {len(overlap_train_test)} (Should be 0)")
    
    assert len(overlap_train_val) == 0, "Train and Val share targets!"
    assert len(overlap_val_test) == 0, "Val and Test share targets!"
    assert len(overlap_train_test) == 0, "Train and Test share targets!"
    
    # Normalize inputs and targets using train-only scaler
    norm_train_x = (train_x - scaler_mean) / scaler_std
    norm_train_y = (train_y - scaler_mean) / scaler_std
    norm_val_x = (val_x - scaler_mean) / scaler_std
    norm_val_y = (val_y - scaler_mean) / scaler_std
    norm_test_x = (test_x - scaler_mean) / scaler_std
    norm_test_y = (test_y - scaler_mean) / scaler_std
    
    # Save artifacts into data/processed_audited/
    np.save(OUTPUT_DIR / "train_x.npy", norm_train_x)
    np.save(OUTPUT_DIR / "train_y.npy", norm_train_y)
    np.save(OUTPUT_DIR / "val_x.npy", norm_val_x)
    np.save(OUTPUT_DIR / "val_y.npy", norm_val_y)
    np.save(OUTPUT_DIR / "test_x.npy", norm_test_x)
    np.save(OUTPUT_DIR / "test_y.npy", norm_test_y)
    
    # Also save unnormalized test targets for direct ground-truth comparison
    np.save(OUTPUT_DIR / "test_y_unnorm.npy", test_y)
    np.save(OUTPUT_DIR / "test_x_unnorm.npy", test_x)
    
    # Save scaler
    np.savez(
        OUTPUT_DIR / "scaler.npz",
        mean=np.array(scaler_mean, dtype=np.float32),
        std=np.array(scaler_std, dtype=np.float32),
    )
    
    metadata = {
        "dataset": "Caltrans PeMS04 (Audited Leakage-Free Split)",
        "total_timesteps": total_steps,
        "num_sensors": num_sensors,
        "history_len": history_len,
        "forecast_len": forecast_len,
        "scaler": {
            "mean": scaler_mean,
            "std": scaler_std,
            "fit_method": "raw training observations flow[:10195]"
        },
        "partitions": {
            "train": {
                "num_windows": int(train_x.shape[0]),
                "origin_range": [int(train_origins[0]), int(train_origins[-1])],
                "target_range": [int(min(train_target_set)), int(max(train_target_set))]
            },
            "val": {
                "num_windows": int(val_x.shape[0]),
                "origin_range": [int(val_origins[0]), int(val_origins[-1])],
                "target_range": [int(min(val_target_set)), int(max(val_target_set))]
            },
            "test": {
                "num_windows": int(test_x.shape[0]),
                "origin_range": [int(test_origins[0]), int(test_origins[-1])],
                "target_range": [int(min(test_target_set)), int(max(test_target_set))]
            }
        },
        "target_leakage": {
            "train_val_overlap": len(overlap_train_val),
            "val_test_overlap": len(overlap_val_test),
            "train_test_overlap": len(overlap_train_test)
        }
    }
    
    with open(OUTPUT_DIR / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSuccessfully generated and saved leakage-safe dataset to {OUTPUT_DIR}")
    return metadata


if __name__ == "__main__":
    create_leakage_safe_splits()
