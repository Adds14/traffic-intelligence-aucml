# PeMS04 Traffic Dataset Documentation

## Overview

The **PeMS04** dataset is a standard benchmark for spatio-temporal traffic forecasting. It is collected by the California Department of Transportation (Caltrans) Performance Measurement System (**PeMS**) in real-time across major freeway segments.

- **Geographic Area:** District 4 (San Francisco Bay Area, California, USA)
- **Time Period:** January 1, 2018 – February 28, 2018 (59 continuous days)
- **Temporal Resolution:** 5-minute aggregation intervals (12 steps per hour, 288 steps per day)
- **Total Timesteps:** 16,992 (59 days × 288 intervals/day)
- **Sensors (Nodes):** 307 loop detectors
- **Features:** 3 channels per sensor at each timestep:
  1. `flow`: Traffic volume (vehicles per 5 minutes)
  2. `occupancy`: Proportion of time detector was occupied ($0.0 \le \text{occupancy} \le 1.0$)
  3. `speed`: Traffic speed (miles per hour, mph)
- **Primary Target:** Traffic flow (`channel 0`)
- **Adjacency / Graph Topology:** 340 directed edges with physical road network distances
- **Geographic Coordinates:** Not included in the standard ASTGCN benchmark release (graph structure defined purely by sensor distance connectivity)

---

## Academic Citation

PeMS04 was curated and standardized for spatio-temporal graph neural networks by Guo et al. (AAAI 2019):

```bibtex
@inproceedings{guo2019astgcn,
  title={Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting},
  author={Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={922--929},
  year={2019}
}
```

---

## Dataset Statistics

The statistics below were empirically computed across all 16,992 timesteps and 307 sensors ($5,216,544$ measurements per channel):

| Metric | Traffic Flow (veh / 5min) | Occupancy (ratio) | Speed (mph) | Road Distance (meters / cost) |
| :--- | :--- | :--- | :--- | :--- |
| **Count** | 5,216,544 | 5,216,544 | 5,216,544 | 340 directed edges |
| **Mean** | 211.7008 | 0.0528 | 63.4706 | 410.3006 |
| **Std Dev ($\sigma$)** | 158.0684 | 0.0495 | 8.3557 | 257.5187 |
| **Min** | 0.0000 | 0.0000 | 3.0000 | 3.2000 |
| **25th Percentile** | 81.0000 | 0.0213 | 62.5000 | 328.7750 |
| **Median (50th)** | 180.0000 | 0.0443 | 65.6000 | 367.1500 |
| **75th Percentile** | 319.0000 | 0.0691 | 67.8000 | 422.3000 |
| **95th Percentile** | 508.0000 | 0.1543 | 70.8000 | 785.4500 |
| **99th Percentile** | 632.0000 | 0.2458 | 72.8000 | 1297.0000 |
| **Max** | 919.0000 | 0.7716 | 85.2000 | 2712.1000 |
| **Zero Values** | 82,935 (1.59%) | 83,044 (1.59%) | 0 (0.00%) | N/A |
| **Missing / NaNs** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 |

---

## Data Preprocessing and Leak-Free Pipeline

### 1. Chronological Splitting
Traffic data exhibits strong temporal autocorrelations and daily/weekly periodicity. To prevent temporal lookahead leakage, data is split strictly chronologically:
- **Training Set (60%):** Timesteps 0 to 10,194 (10,195 timesteps, approx. 35.4 days)
- **Validation Set (20%):** Timesteps 10,195 to 13,592 (3,398 timesteps, approx. 11.8 days)
- **Test Set (20%):** Timesteps 13,593 to 16,991 (3,399 timesteps, approx. 11.8 days)

### 2. Normalization (Fit on Train Only)
Z-score standardization is fitted exclusively on the training split:
$$\mu_{\text{train}} = 207.2369, \quad \sigma_{\text{train}} = 156.4841$$
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Validation and test sets are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ without re-estimating parameters.

### 3. Sliding Window Sequences
For each split, sliding window sequences of length 12 are generated:
- **Input ($X$):** Historical 12 steps (60 minutes) $\to$ shape `(num_samples, 12, 307)`
- **Target ($Y$):** Future 12 steps (60 minutes) $\to$ shape `(num_samples, 12, 307)`
- **Evaluation Horizons:** 3 steps (15 min), 6 steps (30 min), 12 steps (60 min)
- Zero leakage guarantee: Target sequence $Y[i]$ begins at index $i+12$, directly following historical input $X[i]$ ($i$ to $i+11$). Splitting is performed prior to sequence formation so windows never cross split boundaries.

### Resulting Dataset Shapes
| Split | Timesteps | Sequence Samples ($N$) | Input ($X$) Shape | Target ($Y$) Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 10,195 | 10,172 | `(10172, 12, 307)` | `(10172, 12, 307)` |
| **Validation**| 3,398 | 3,375 | `(3375, 12, 307)` | `(3375, 12, 307)` |
| **Test** | 3,399 | 3,376 | `(3376, 12, 307)` | `(3376, 12, 307)` |

---

## Graph Topology Construction

Adjacency matrix $A \in \mathbb{R}^{307 \times 307}$ is constructed from `data/raw/distance.csv` using a Gaussian threshold kernel:

$$A_{i,j} = \exp\left(-\frac{\text{dist}(i,j)^2}{\sigma^2}\right)$$

where $\sigma = 257.5187$ is the standard deviation of road network distances across all 340 directed edges. Original sensor IDs in the edge list are mapped bijectively to zero-indexed node positions $[0, 306]$.

---

## File Structure

```
data/
├── README.md                 # This documentation
├── raw/
│   ├── pems04.npz           # Raw multi-channel array: 'data' (16992, 307, 3)
│   └── distance.csv         # Directed road connectivity: from, to, cost (340 rows)
└── processed/
    ├── train_x.npy          # Normalized training inputs (10172, 12, 307)
    ├── train_y.npy          # Normalized training targets (10172, 12, 307)
    ├── val_x.npy            # Normalized validation inputs (3375, 12, 307)
    ├── val_y.npy            # Normalized validation targets (3375, 12, 307)
    ├── test_x.npy           # Normalized test inputs (3376, 12, 307)
    ├── test_y.npy           # Normalized test targets (3376, 12, 307)
    ├── adj_matrix.npy       # Gaussian kernel adjacency matrix (307, 307)
    └── scaler.npz           # Fitted normalization params: mean, std
```
