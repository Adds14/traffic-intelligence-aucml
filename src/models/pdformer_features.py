"""Training-only semantic graph and traffic-pattern inputs for PDFormer."""

from pathlib import Path

import numpy as np


def _daily_profiles(train_flow, steps_per_day=288, bins=24):
    days = len(train_flow) // steps_per_day
    x = train_flow[:days * steps_per_day].reshape(days, steps_per_day, train_flow.shape[1])
    x = x.reshape(days, bins, steps_per_day // bins, train_flow.shape[1]).mean(axis=2)
    profile = x.mean(axis=0).T.astype(np.float32)  # nodes, hour-of-day
    profile -= profile.mean(axis=1, keepdims=True)
    profile /= np.maximum(profile.std(axis=1, keepdims=True), 1e-6)
    return profile


def _pairwise_dtw(profiles):
    """Exact dynamic-time-warping distances for short, hourly node profiles."""
    n, width = profiles.shape
    inf = np.full((n, n), np.inf, dtype=np.float32)
    previous = np.full((width + 1, n, n), np.inf, dtype=np.float32)
    previous[0] = inf
    previous[0, np.arange(n), np.arange(n)] = 0.0
    # Vectorize each dynamic-programming cell across every node pair.
    for i in range(1, width + 1):
        current = np.full((width + 1, n, n), np.inf, dtype=np.float32)
        for j in range(1, width + 1):
            cost = np.abs(profiles[:, i - 1, None] - profiles[None, :, j - 1])
            current[j] = cost + np.minimum(np.minimum(previous[j], current[j - 1]), previous[j - 1])
        previous = current
    distances = previous[width]
    distances /= width
    return distances


def _pattern_prototypes(train_x, num_patterns=8, pattern_len=3):
    # Training histories only. Quantile-binned local shapes supply robust
    # representative patterns without fitting on validation or test data.
    windows = np.asarray(train_x[:, -pattern_len:, :], dtype=np.float32).transpose(2, 0, 1)
    n, count, _ = windows.shape
    keys = np.empty((n, num_patterns, pattern_len), dtype=np.float32)
    for node in range(n):
        order = np.argsort(windows[node].mean(axis=1), kind="stable")
        for pattern, group in enumerate(np.array_split(order, num_patterns)):
            keys[node, pattern] = windows[node, group].mean(axis=0) if len(group) else 0.0
    return keys


def build_pdformer_features(project_root, train_x, adj_matrix, *, force=False):
    """Build/load DTW similarity, short-hop graph, Laplacian PE and patterns.

    All time-series-derived features use only the chronological training
    partition. Cache is tied to the input training tensor shape and scaler stats.
    """
    project_root = Path(project_root)
    data_dir = project_root / "data" / "processed"
    cache_path = data_dir / "pdformer_features.npz"
    scaler = np.load(data_dir / "scaler.npz")
    cache_key = (len(train_x), train_x.shape[1], train_x.shape[2],
                 round(float(scaler["mean"]), 5), round(float(scaler["std"]), 5))
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        saved_key = tuple(cached["cache_key"].tolist())
        if saved_key == cache_key:
            return {key: cached[key] for key in ("dtw_matrix", "pattern_keys", "lap_pos_enc")}

    raw_path = project_root / "data" / "raw" / "pems04.npz"
    raw = np.load(raw_path)["data"][:, :, 0].astype(np.float32)
    train_end = int(len(raw) * 0.6)
    train_flow = raw[:train_end]
    profiles = _daily_profiles(train_flow)
    dtw = _pairwise_dtw(profiles)

    # Symmetric normalized graph Laplacian eigenvectors (omit trivial vector).
    adjacency = np.maximum(np.asarray(adj_matrix, dtype=np.float64),
                           np.asarray(adj_matrix, dtype=np.float64).T)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-8))
    laplacian = np.eye(len(degree)) - inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    lap_pos_enc = eigenvectors[:, 1:9].astype(np.float32)
    if lap_pos_enc.shape[1] < 8:
        lap_pos_enc = np.pad(lap_pos_enc, ((0, 0), (0, 8 - lap_pos_enc.shape[1])))
    pattern_keys = _pattern_prototypes(train_x)
    np.savez_compressed(cache_path, dtw_matrix=dtw, pattern_keys=pattern_keys,
                        lap_pos_enc=lap_pos_enc, cache_key=np.asarray(cache_key))
    return {"dtw_matrix": dtw, "pattern_keys": pattern_keys, "lap_pos_enc": lap_pos_enc}
