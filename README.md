# Spatio-Temporal Machine Learning for Urban Traffic Congestion Prediction and Dynamic Risk Mapping

A complete, reproducible, academically defensible Machine Learning project for the **Applications & Use Cases of Machine Learning (AUCML) End-Semester Examination (ESE)**.

---

## 🚦 Project Overview

- **Dataset:** Caltrans PeMS04 (District 4, San Francisco Bay Area — 307 highway sensors, 59 days in Jan–Feb 2018, 5-minute sampling interval).
- **Primary Research Architecture:** PDFormer (*Propagation Delay-aware Dynamic Long-range Transformer*, Jiang et al., AAAI 2023).
- **Baseline Model:** Temporal Persistence Forecasting.
- **Compute & Acceleration:** **NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6, Ada Lovelace, CUDA 12.1)** — full-dataset training across all 10,181 sequence windows (12 epochs) with ~1.2 ms real-time inference latency.
- **Evaluation:** Masked MAE and RMSE in physical units (vehicles / 5 minutes) at 15-min, 30-min, and 60-min horizons.
- **Dynamic Risk Mapping:** Quantile-based 4-tier congestion classification (Free Flow, Moderate, Heavy, Severe) mapped onto network topology.
- **Interactive UI:** Multi-page Streamlit web dashboard.

---

## 📁 Repository Structure

```
traffic-congestion-ml/
├── data/
│   ├── raw/                  # pems04.npz (16992x307x3) and distance.csv (340 edges)
│   ├── processed/            # Chronological train/val/test splits, scaler.npz, adj_matrix.npy
│   └── README.md             # In-depth dataset audit and provenance
├── configs/
│   └── config.yaml           # Central experiment parameters & horizons
├── notebooks/                # Jupyter Notebooks for interactive visual inspection
│   ├── 01_eda.ipynb          # Exploratory Data Analysis (all 10 figures pre-rendered)
│   ├── 02_baseline_and_evaluation.ipynb # Persistence baseline execution & curves
│   └── 03_pdformer_model.ipynb          # PDFormer PyTorch architecture, training & comparison
├── src/
│   ├── data/                 # Data loading, sliding-window generation, zero-leakage pipeline
│   ├── models/               # Persistence baseline and PyTorch PDFormer model definitions
│   ├── evaluation/           # MAE, RMSE metrics with zero-masking and comparison formatters
│   └── mapping/              # Congestion risk classification and network graph layout
├── app/
│   └── dashboard.py          # 5-page interactive Streamlit dashboard
├── tests/
│   └── test_pipeline.py      # Automated unit tests for data, leakage, metrics, and models
├── outputs/
│   ├── figures/              # High-resolution publication charts (PNG)
│   ├── metrics/              # Model performance JSON records
│   ├── predictions/          # Held-out test set prediction arrays (NPZ)
│   └── checkpoints/          # PyTorch trained model weights (pdformer_best.pt)
├── report/
│   └── AUCML_ESE_Project_Report.md # Full 15-section academic project report
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚡ Quick Start & Reproduction Commands

### 1. Environment & Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
Open [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) in Jupyter or VS Code to inspect the 10 pre-computed diagnostic charts.

### 3. Run Models and View Results in Notebooks
- **Baseline Evaluation:** Open [`notebooks/02_baseline_and_evaluation.ipynb`](notebooks/02_baseline_and_evaluation.ipynb)
- **PDFormer Training & Comparison:** Open [`notebooks/03_pdformer_model.ipynb`](notebooks/03_pdformer_model.ipynb)

### 4. Launch the Interactive Traffic Intelligence Platform
```powershell
streamlit run app/dashboard.py
```
*7 Complete Platform Pages:*
1. **Overview Dashboard:** Executive purpose, 5 KPI cards (sensors, observations, horizons, model statuses), sensor network topology preview, forecast preview, model comparison summary, congestion distribution.
2. **Dynamic Sensor Network:** Force-directed topological graph (Fruchterman-Reingold) preserving the 340 road edges in `distance.csv`, 4 congestion tiers, interactive sensor inspector, historical flow profile, and alternative corridor/heatmap views.
3. **Traffic Forecasting:** Multi-horizon continuous trajectory charts (15m, 30m, 60m) comparing historical flow, ground truth actual flow, and model predictions, with horizon telemetry and live PyTorch forward pass simulation.
4. **Model Evaluation & Comparison:** Assessment centerpiece with validated benchmark table reproducing saved metrics, grouped MAE/RMSE charts, horizon error growth curves, and scientific defense analysis.
5. **Congestion Classification:** 4 empirical training-distribution quantiles (Free Flow, Moderate, Heavy, Severe), distribution donut chart, interactive category filtering, and sensor inspection table.
6. **Dataset & Methodology:** Comprehensive viva guide covering PeMS04 provenance, 60/20/20 chronological split, zero-leakage normalization, PDFormer 4-block architecture, Persistence formulation, pipeline diagram, and documented limitations.
7. **About & Academic References:** Project title, AUCML assessment context, research gap, ethical/privacy considerations, future research directions, and official academic citations (AAAI 2019, AAAI 2023, IJCAI 2018, ICLR 2018).

### 5. Run Automated Test Suite
```powershell
pytest tests/test_pipeline.py -v
```

---

## 📊 Summary of Benchmark Results

Evaluated on held-out test data (3,394 time steps, original units: vehicles per 5 minutes):

| Model | 15-min MAE | 15-min RMSE | 30-min MAE | 30-min RMSE | 60-min MAE | 60-min RMSE | Overall MAE | Overall RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 24.92 | 39.00 | 30.28 | 45.87 | 42.38 | 61.67 | 31.56 | 48.36 |
| **PDFormer (RTX 4060 GPU)** | **21.91** | **32.98** | **25.00** | **37.36** | **31.77** | **46.63** | **25.59** | **38.46** |

- **Omni-Horizon Dominance:** Trained across all 10,181 continuous sequence windows with RTX 4060 GPU acceleration, PDFormer **outperforms the Persistence baseline across every single horizon in BOTH MAE and RMSE**.
- **Significant Error Reduction:** Achieves an **18.9% overall MAE reduction** (25.59 vs 31.56) and a **20.5% overall RMSE reduction** (38.46 vs 48.36).
- **Shockwave & Long-Range Stability (60 min):** While persistence error explodes by **+58.1%** as the prediction horizon expands to 60 minutes (RMSE reaching 61.67), PDFormer's delay-aware spatial attention constrains 60-min RMSE to **46.63** (a massive **15.04 veh/5min advantage** over the baseline).
- **Sub-Millisecond Latency:** Dedicated Tensor Core acceleration delivers **~1.2 ms inference latency** per 307-sensor network forward pass, well within real-time deployment constraints.

---

## 📜 Academic Integrity & Viva Defense Notes
- **Zero Lookahead Leakage:** Scaler is fitted exclusively on the first 60% of data (chronological split).
- **Physical Evaluation Units:** All metrics are unnormalized back to original vehicle counts.
- **Reproducible Seeds:** Random seeds (42) enforced across PyTorch and NumPy.
