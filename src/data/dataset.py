"""
PeMS04 Traffic Dataset Loading and Preprocessing Pipeline.

Provides the PeMS04Dataset class for loading, chronological splitting,
z-score normalizing, and creating sequence windows for spatio-temporal
traffic flow forecasting models.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset


class PeMS04Dataset:
    """Load and preprocess PeMS04 traffic data for spatio-temporal modeling."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize dataset with configuration settings.

        Args:
            config_path: Path to configuration YAML file.
        """
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = self._resolve_path(config_path)
        self.config = self._load_config(self.config_path)

        # Paths
        dataset_cfg = self.config.get("dataset", {})
        paths_cfg = self.config.get("paths", {})

        raw_file_rel = dataset_cfg.get("raw_file", "data/raw/pems04.npz")
        adj_file_rel = dataset_cfg.get("adj_file", "data/raw/distance.csv")
        proc_dir_rel = paths_cfg.get("processed_data", "data/processed/")

        self.raw_path = self._resolve_path(raw_file_rel)
        self.adj_path = self._resolve_path(adj_file_rel)
        self.processed_dir = self._resolve_path(proc_dir_rel)

        # Split ratios
        split_cfg = self.config.get("splitting", {})
        self.train_ratio = split_cfg.get("train_ratio", 0.6)
        self.val_ratio = split_cfg.get("val_ratio", 0.2)
        self.test_ratio = split_cfg.get("test_ratio", 0.2)

        # Sequence lengths
        preproc_cfg = self.config.get("preprocessing", {})
        self.history_len = preproc_cfg.get("history_window", 12)
        # Default forecast len: max horizon or 12
        horizons = preproc_cfg.get("forecast_horizons", [3, 6, 12])
        self.forecast_len = max(horizons) if horizons else 12

        # Target feature configuration
        self.target_idx = dataset_cfg.get("target_feature_index", 0)
        self.num_sensors = dataset_cfg.get("num_sensors", 307)

        # Cache variables
        self.raw_data: Optional[np.ndarray] = None
        self.adj_matrix: Optional[np.ndarray] = None
        self.scaler: Optional[Dict[str, float]] = None

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to project root if not absolute."""
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def _load_config(self, config_path: Path) -> dict:
        """Load YAML configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_raw_data(self) -> np.ndarray:
        """Load pems04.npz and extract data array.

        Returns:
            np.ndarray: Raw traffic data array with shape (16992, 307, 3),
                        corresponding to (num_timesteps, num_sensors, num_features).
                        Features: [flow, occupancy, speed].
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")

        loaded = np.load(self.raw_path)
        if "data" not in loaded:
            raise KeyError(f"'data' key not found in {self.raw_path}")

        data = loaded["data"].astype(np.float32)
        self.raw_data = data
        return data

    def load_adjacency(self, threshold: Optional[float] = None) -> np.ndarray:
        """Load distance.csv and build Gaussian kernel adjacency matrix (307, 307).

        Gaussian kernel formula:
            A[i, j] = exp(-dist^2 / sigma^2) if dist <= threshold

        Sensor IDs from the edge list are mapped to zero-based matrix indices.

        Args:
            threshold: Optional maximum distance threshold to retain an edge.
                       If None, all edges in distance.csv are included.

        Returns:
            np.ndarray: Adjacency matrix of shape (307, 307), dtype float32.
        """
        if not self.adj_path.exists():
            raise FileNotFoundError(f"Distance file not found: {self.adj_path}")

        df = pd.read_csv(self.adj_path)
        required_cols = {"from", "to", "cost"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"distance.csv must contain columns {required_cols}, found {df.columns}")

        # Unique sensor mapping
        unique_sensors = sorted(list(set(df["from"]).union(set(df["to"]))))
        sensor_to_idx = {sid: idx for idx, sid in enumerate(unique_sensors)}
        num_nodes = max(len(unique_sensors), self.num_sensors)

        distances = df["cost"].values.astype(np.float64)
        sigma = float(np.std(distances))
        if sigma == 0:
            sigma = 1.0

        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for _, row in df.iterrows():
            u = int(row["from"])
            v = int(row["to"])
            dist = float(row["cost"])

            if threshold is not None and dist > threshold:
                continue

            i = sensor_to_idx.get(u, u)
            j = sensor_to_idx.get(v, v)
            if i < num_nodes and j < num_nodes:
                weight = float(np.exp(- (dist ** 2) / (sigma ** 2)))
                adj[i, j] = weight

        self.adj_matrix = adj
        return adj

    def get_target(self) -> np.ndarray:
        """Extract flow (channel 0) as target array.

        Returns:
            np.ndarray: Traffic flow array of shape (16992, 307), dtype float32.
        """
        if self.raw_data is None:
            self.load_raw_data()
        assert self.raw_data is not None
        return self.raw_data[:, :, self.target_idx]

    def chronological_split(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data chronologically into train, validation, and test sets.

        Maintains temporal ordering:
            train: [0 : train_end]
            val:   [train_end : val_end]
            test:  [val_end : ]

        Args:
            data: Input array where first dimension is time steps.

        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        n_samples = len(data)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    def compute_normalization(self, train_data: np.ndarray) -> Dict[str, float]:
        """Compute mean and std from training data only to prevent leakage.

        Args:
            train_data: Training set traffic flow array.

        Returns:
            Dict[str, float]: Scaler dictionary with 'mean' and 'std'.
        """
        mean = float(np.mean(train_data))
        std = float(np.std(train_data))
        if std == 0:
            std = 1.0

        scaler = {"mean": mean, "std": std}
        self.scaler = scaler
        return scaler

    def normalize(self, data: np.ndarray, scaler: Dict[str, float]) -> np.ndarray:
        """Apply z-score standardization: (data - mean) / std.

        Args:
            data: Traffic data array.
            scaler: Dictionary containing 'mean' and 'std'.

        Returns:
            np.ndarray: Normalized array of same shape and float32 dtype.
        """
        return ((data - scaler["mean"]) / scaler["std"]).astype(np.float32)

    def inverse_normalize(
        self, data: np.ndarray, scaler: Dict[str, float]
    ) -> np.ndarray:
        """Reverse z-score standardization: data * std + mean.

        Args:
            data: Normalized traffic data array.
            scaler: Dictionary containing 'mean' and 'std'.

        Returns:
            np.ndarray: Re-scaled traffic data in original units (vehicles/5min).
        """
        return (data * scaler["std"] + scaler["mean"]).astype(np.float32)

    def create_sequences(
        self,
        data: np.ndarray,
        history_len: Optional[int] = None,
        forecast_len: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for input X and target Y.

        Sequence extraction:
            X[i] = data[i : i + history_len]
            Y[i] = data[i + history_len : i + history_len + forecast_len]

        Zero data leakage is preserved since Y starts strictly after X ends.

        Args:
            data: 2D array of shape (timesteps, num_sensors).
            history_len: Number of historical time steps (default: self.history_len = 12).
            forecast_len: Number of forecast time steps (default: self.forecast_len = 12).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                X: shape (num_samples, history_len, num_sensors)
                Y: shape (num_samples, forecast_len, num_sensors)
        """
        h_len = history_len if history_len is not None else self.history_len
        f_len = forecast_len if forecast_len is not None else self.forecast_len

        total_steps = len(data)
        num_samples = total_steps - h_len - f_len + 1

        if num_samples <= 0:
            raise ValueError(
                f"Data length ({total_steps}) is too short for history_len ({h_len}) "
                f"+ forecast_len ({f_len})."
            )

        num_nodes = data.shape[1]
        x = np.zeros((num_samples, h_len, num_nodes), dtype=np.float32)
        y = np.zeros((num_samples, f_len, num_nodes), dtype=np.float32)

        for i in range(num_samples):
            x[i] = data[i : i + h_len]
            y[i] = data[i + h_len : i + h_len + f_len]

        return x, y

    def prepare_data(self) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """Execute full preprocessing pipeline and persist processed artifacts.

        Steps:
            1. Load raw PeMS04 data.
            2. Build Gaussian kernel adjacency matrix.
            3. Extract target feature (flow, channel 0).
            4. Chronologically split into train (60%), val (20%), test (20%).
            5. Compute z-score parameters strictly from training data.
            6. Apply normalization to all splits.
            7. Generate sliding window sequences (X: 12 steps, Y: 12 steps).
            8. Save all arrays and metadata into data/processed/.

        Returns:
            dict containing:
                'train_x': np.ndarray,
                'train_y': np.ndarray,
                'val_x': np.ndarray,
                'val_y': np.ndarray,
                'test_x': np.ndarray,
                'test_y': np.ndarray,
                'scaler': dict,
                'adj_matrix': np.ndarray
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load raw data and target
        self.load_raw_data()
        target = self.get_target()

        # 2. Build adjacency matrix
        adj = self.load_adjacency()

        # 3. Chronological split
        train_raw, val_raw, test_raw = self.chronological_split(target)

        # 4. Fit scaler on train only
        scaler = self.compute_normalization(train_raw)

        # 5. Normalize
        train_norm = self.normalize(train_raw, scaler)
        val_norm = self.normalize(val_raw, scaler)
        test_norm = self.normalize(test_raw, scaler)

        # 6. Sliding window sequences
        train_x, train_y = self.create_sequences(train_norm)
        val_x, val_y = self.create_sequences(val_norm)
        test_x, test_y = self.create_sequences(test_norm)

        # 7. Persist to data/processed/
        np.save(self.processed_dir / "train_x.npy", train_x)
        np.save(self.processed_dir / "train_y.npy", train_y)
        np.save(self.processed_dir / "val_x.npy", val_x)
        np.save(self.processed_dir / "val_y.npy", val_y)
        np.save(self.processed_dir / "test_x.npy", test_x)
        np.save(self.processed_dir / "test_y.npy", test_y)
        np.save(self.processed_dir / "adj_matrix.npy", adj)

        np.savez(
            self.processed_dir / "scaler.npz",
            mean=np.array(scaler["mean"], dtype=np.float32),
            std=np.array(scaler["std"], dtype=np.float32),
        )

        return {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y,
            "scaler": scaler,
            "adj_matrix": adj,
        }

    def get_torch_dataloaders(
        self,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Utility method to obtain PyTorch DataLoaders.

        Args:
            batch_size: Mini-batch size.
            shuffle_train: Whether to shuffle training batches.
            num_workers: Number of worker processes.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        data_dict = self.prepare_data()

        train_dataset = TensorDataset(
            torch.from_numpy(data_dict["train_x"]),
            torch.from_numpy(data_dict["train_y"]),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(data_dict["val_x"]),
            torch.from_numpy(data_dict["val_y"]),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(data_dict["test_x"]),
            torch.from_numpy(data_dict["test_y"]),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader
