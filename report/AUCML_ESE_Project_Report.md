# Spatio-Temporal Machine Learning for Urban Traffic Congestion Prediction and Dynamic Risk Mapping

**Course:** Applications & Use Cases of Machine Learning (AUCML)  
**Deliverable:** End-Semester Examination (ESE) Project Report  
**Author:** Student / Research Engineer  
**Institution:** Academic Engineering Department  
**Dataset:** California Department of Transportation (Caltrans) PeMS04  
**Primary Architectures:** Persistence Baseline & PDFormer (Propagation Delay-aware Dynamic Long-range Transformer)  

---

## 1. Abstract

Urban traffic congestion represents a multi-billion dollar societal and economic burden that severely deteriorates urban mobility, increases carbon emissions, and complicates emergency logistics. Modern intelligent transportation systems (ITS) require accurate, multi-step spatio-temporal traffic forecasts to facilitate proactive signal management, route optimization, and congestion mitigation. In this work, we present a complete, reproducible, and mathematically rigorous machine learning framework for network-scale traffic flow forecasting and dynamic congestion risk mapping using the California Performance Measurement System District 4 (PeMS04) benchmark. 

The dataset comprises 16,992 temporal intervals recorded across 307 inductive loop sensors in the San Francisco Bay Area freeway network. We formulate the forecasting task under a strict zero-leakage chronological split (60% training, 20% validation, 20% testing) across 15-minute, 30-minute, and 60-minute prediction horizons. We benchmark the temporal persistence baseline against the Propagation Delay-aware Dynamic Long-range Transformer (PDFormer) architecture, which integrates geographic-masked spatial self-attention and temporal self-attention. Full-scale model training was accelerated on an NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6, CUDA 12.1) across all 10,181 continuous sequence windows. Predictions are dynamically mapped onto sensor-level congestion categories using empirical training-distribution quantiles and rendered via a high-performance interactive Streamlit dashboard. Experimental results on held-out test data demonstrate that PDFormer decisively outperforms the Persistence baseline across every single horizon in both MAE (25.59 vs. 31.56, an 18.9% error reduction) and RMSE (38.46 vs. 48.36, a 20.5% error reduction), capturing non-linear topological propagation delays and providing resilient long-range traffic stability with sub-2ms real-time inference latency.

---

## 2. Introduction & Real-World Problem Statement

Rapid urbanization and expanding vehicular densities have rendered conventional reactive traffic control methods obsolete. Urban freeways exhibit complex spatio-temporal dynamics governed by two fundamental phenomena:
1. **Temporal Non-Linearity:** Traffic exhibits strong cyclic patterns (morning and evening rush hours, weekday vs. weekend shifts) superimposed with stochastic fluctuations caused by bottlenecks, merges, and lane closures.
2. **Spatial Topological Propagation:** Upstream congestions propagate backward as shockwaves along road corridors, while downstream queues restrict upstream inflow according to highway capacity constraints.

Conventional transportation planning often relies on static historical averages or naive auto-regressive formulations that neglect the underlying topological network structure. This project addresses the challenge of building an end-to-end Machine Learning pipeline that ingests continuous inductive loop sensor streams, prevents lookahead data leakage, produces calibrated multi-horizon predictions in physical engineering units (vehicles per 5 minutes), and maps network states into interpretable congestion risk levels for operational stakeholders.

---

## 3. Objectives, Scope, Stakeholders, and Relevance

### 3.1 Primary Objectives
- **Data Engineering:** Clean, audit, and preprocess high-dimensional spatio-temporal traffic tensors without temporal or spatial data leakage.
- **Baseline Formulation:** Establish a mathematically sound Persistence Baseline predicting future states from the most recent historical observation across all forecast horizons.
- **Research Model Implementation:** Implement and evaluate PDFormer, capturing spatial dependencies constrained by physical road distances alongside multi-head temporal attention.
- **Risk Mapping & Decision Support:** Classify continuous traffic volume into four operational risk categories (Free Flow, Moderate, Heavy, Severe) and visualize network health through dynamic spatial graph layouts.
- **Interactive Delivery:** Deploy a production-grade Streamlit web interface and comprehensive Jupyter Notebooks for academic defense and live inspection.

### 3.2 Target Stakeholders
- **Department of Transportation (DOT) Traffic Operators:** Monitor freeway corridors, anticipate bottleneck formation 15–60 minutes in advance, and modulate variable speed limits (VSL).
- **Municipal Emergency Dispatchers:** Reroute emergency medical and fire services away from predicted high-risk shockwaves.
- **Navigation & Mobility Services:** Ingest sensor-level flow forecasts to optimize dynamic multi-modal routing algorithms.

---

## 4. Literature Review and Research Gap

Traditional time-series models such as ARIMA and vector autoregression (VAR) model temporal autocorrelation but fail to scale to hundreds of interconnected road segments. Classical machine learning approaches (e.g., Support Vector Regression, Random Forests, Gradient Boosted Trees) treat spatial sensors independently or flatten adjacency relationships, destroying spatial invariance.

The advent of Spatio-Temporal Graph Neural Networks (ST-GNNs) catalyzed a paradigm shift:
- **ASTGCN (Guo et al., AAAI 2019):** Introduced spatial and temporal attention mechanisms combined with Chebyshev graph convolutions on PeMS04 and PeMS08.
- **STSGCN (Song et al., AAAI 2020):** Proposed localized spatial-temporal synchronous graphs to capture localized spatio-temporal correlations.
- **PDFormer (Jiang et al., AAAI 2023):** Recognized that traffic disturbances do not propagate instantaneously; rather, information travels across road networks with distinct time delays dependent on physical distance and traffic speed. PDFormer introduces geographic-masked spatial attention and dynamic long-range semantic attention to explicitly model delay-aware propagation.

**Research Gap Addressed in this Study:** While research papers report benchmark metrics on normalized tensors, practical engineering implementations often neglect strict zero-leakage normalization fitting, fail to report unnormalized physical error units, lack baseline execution before deep modeling, or omit practical dynamic risk mapping layers for non-technical dispatchers.

---

## 5. PeMS04 Dataset & Exploratory Analysis

### 5.1 Dataset Provenance & Specifications
The PeMS04 benchmark is collected from Caltrans Performance Measurement System (PeMS) District 4 (San Francisco Bay Area):
- **Temporal Span:** January 1, 2018 to February 28, 2018 (59 consecutive days).
- **Sampling Interval:** Aggregated into 5-minute epochs (12 records per hour, 288 records per day).
- **Total Timesteps ($T$):** $59 \times 288 = 16,992$ timesteps.
- **Spatial Nodes ($N$):** 307 inductive loop detector stations.
- **Feature Channels ($F$):**
  1. *Channel 0 — Traffic Flow:* Total vehicle count per 5-minute window (Range: 0 to 919 vehicles).
  2. *Channel 1 — Lane Occupancy:* Proportion of time sensor is covered ($0.0 \le \text{occ} \le 0.77$).
  3. *Channel 2 — Speed:* Average vehicular velocity ($3.0 \le \text{speed} \le 85.2$ mph, Mean: 63.5 mph).
- **Topological Adjacency:** 340 directed road network edges with physical distance measurements (Range: 30.1 to 4,874.5 meters).

### 5.2 Key Exploratory Findings
- **Data Completeness:** Verified 0 NaN/null values. Exactly 1.59% of observations recorded zero flow, occurring predominantly between 01:00 AM and 04:00 AM.
- **Diurnal Periodicity:** Pronounced morning peak (07:00–09:00 AM) and evening peak (04:00–07:00 PM) on weekdays.
- **Weekend Variance:** Weekday profiles exhibit twin sharp commuter peaks, whereas weekend traffic displays a unimodal, delayed plateau (11:00 AM to 05:00 PM).
- **Fundamental Flow-Speed Curve:** Visualized the inverse relationship: at high flow rates (>350 veh/5min), average speeds drop sharply from 65 mph down to 25–40 mph, signaling congestion onset.

---

## 6. Preprocessing & Data Leakage Prevention

To ensure academic validity and eliminate data leakage, we enforce the following engineering protocols:
1. **Chronological Splitting:**
   - Training Set: First 60% ($10,181$ sequence windows, ~35 days).
   - Validation Set: Middle 20% ($3,394$ sequence windows, ~12 days).
   - Testing Set: Held-out final 20% ($3,394$ sequence windows, ~12 days).
2. **Strict Scaler Isolation:**
   - Normalization parameters are calculated exclusively on the training partition:
     $$\mu_{\text{train}} = 207.23 \text{ veh/5min}, \quad \sigma_{\text{train}} = 156.48 \text{ veh/5min}$$
   - The validation and test sets are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ with zero knowledge of future statistical distributions.
3. **Sliding Window Sequence Formation:**
   - Historical Input Window: $T_{\text{in}} = 12$ steps (60 minutes).
   - Forecast Target Window: $T_{\text{out}} = 12$ steps (60 minutes ahead).
   - For sample index $i$, inputs are strictly $X_i = [t_{i}, \dots, t_{i+11}]$ and targets are $Y_i = [t_{i+12}, \dots, t_{i+23}]$, ensuring zero lookahead overlap.
4. **Inverse Normalization Protocol:**
   - All loss functions during training operate on standardized tensors, but all reported evaluation metrics (MAE, RMSE) are calculated on inverse-transformed values:
     $$\hat{Y}_{\text{orig}} = \hat{Y}_{\text{norm}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$$

---

## 7. System Architecture & Methodology

The complete end-to-end architecture comprises five modular layers:

```
+-----------------------------------------------------------------------+
|                         PeMS04 Raw Ingestion                          |
|         pems04.npz (16992 x 307 x 3) + distance.csv (340 edges)       |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                 Zero-Leakage Preprocessing Engine                     |
|  - Chronological 60/20/20 Split  - Train-Only Z-Score Standardization  |
|  - Sliding Window Tensor Slicing - Gaussian Kernel Adjacency Matrix   |
+-----------------------------------------------------------------------+
            |                                           |
            v                                           v
+------------------------+             +--------------------------------+
|  Persistence Baseline  |             |      PDFormer Architecture     |
|   Last Historical Step |             |  - Spatio-Temporal Embedding   |
|   Repeated 12 Horizons |             |  - Geographic Spatial Attn     |
|                        |             |  - Temporal Self-Attention     |
|                        |             |  - Multi-Horizon Linear Head   |
+------------------------+             +--------------------------------+
            \                                           /
             \---------------------\ /-----------------/
                                    v
+-----------------------------------------------------------------------+
|                    Unified Evaluation & Inversion                     |
|   - Inverse Z-Score Transform back to original units (veh/5min)       |
|   - Masked MAE and RMSE at 15-min, 30-min, 60-min, and Overall        |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|             Dynamic Risk Mapping & Streamlit Dashboard                |
|   - Quantile-Based Congestion Thresholds (Free, Moderate, Heavy, etc.)|
|   - Interactive Sensor Network Topology & Flow Heatmaps               |
|   - Multi-Page Web Interface with Live Prediction Inspection          |
+-----------------------------------------------------------------------+
```

---

## 8. Model Formulation: Persistence Baseline & PDFormer

### 8.1 Persistence Baseline
The persistence model serves as the standard competitive benchmark for high-frequency traffic forecasting. For a given input sequence $X \in \mathbb{R}^{B \times T_{\text{in}} \times N}$, the persistence prediction $\hat{Y}_{t+k}$ at horizon $k \in \{1, \dots, 12\}$ is defined as:
$$\hat{Y}_{t+k} = X_{t} \quad \forall k \in \{1, 2, \dots, 12\}$$
This baseline exploits the high temporal autocorrelation of highway traffic, proving strong at immediate horizons ($t+1$) while exhibiting error accumulation at extended horizons ($t+12$).

### 8.2 PDFormer Architecture
PDFormer (*Jiang et al., AAAI 2023*) addresses the spatial-temporal propagation delay inherent in road networks through four interconnected modules:

1. **Spatio-Temporal Embedding Layer:**
   Projects the single-channel flow feature into $d_{\text{model}} = 32$ latent space and adds learnable spatial node embeddings $E_{\text{node}} \in \mathbb{R}^{1 \times 1 \times N \times d_{\text{model}}}$ and temporal positional embeddings $E_{\text{temp}} \in \mathbb{R}^{1 \times T \times 1 \times d_{\text{model}}}$:
   $$H^{(0)} = \text{Linear}(X) + E_{\text{node}} + E_{\text{temp}}$$

2. **Geographic Masked Spatial Attention:**
   Restricts multi-head self-attention to physically connected or topologically proximal highway nodes using an additive spatial penalty derived from adjacency matrix $A$:
   $$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{geo}}\right) V$$
   where $M_{\text{geo}}[i, j] = 0$ if nodes $i$ and $j$ share road connectivity and $-\infty$ otherwise.

3. **Temporal Self-Attention:**
   Applies multi-head self-attention across the 12 historical time slots to capture velocity-dependent information propagation delays across temporal horizons.

4. **Multi-Horizon Forecast Projection:**
   Flattens the temporal latent dimensions and projects through a two-layer feed-forward network to output the full 12-step future forecast matrix $\hat{Y} \in \mathbb{R}^{B \times 12 \times 307}$.

---

## 9. Experimental Setup & Evaluation Metrics

### 9.1 Evaluation Protocol
Models are evaluated on the exact same 3,394 held-out test sequences. All metrics are computed in physical measurement units (**vehicles per 5 minutes**) with zero observations masked out to avoid zero-division and distortion.

### 9.2 Mathematical Metric Formulations
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{|\Omega|} \sum_{(t, n) \in \Omega} |Y_{t, n} - \hat{Y}_{t, n}|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{|\Omega|} \sum_{(t, n) \in \Omega} (Y_{t, n} - \hat{Y}_{t, n})^2}$$
where $\Omega = \{(t, n) \mid Y_{t, n} > 0\}$ denotes the set of valid non-zero traffic observations.

Evaluations are performed at three standard operational horizons:
- **Horizon 3 (15 Minutes Ahead):** Immediate routing and tactical control.
- **Horizon 6 (30 Minutes Ahead):** Medium-range corridor management.
- **Horizon 12 (60 Minutes Ahead):** Strategic urban demand management.
- **Overall:** Mean error across all 12 forecast steps.

### 9.3 Hardware & Computational Training Environment
- **Compute Architecture:** Dedicated **NVIDIA GeForce RTX 4060 Laptop GPU** featuring 8.0 GB GDDR6 VRAM (Ada Lovelace architecture, Compute Capability 8.9).
- **Deep Learning Framework:** PyTorch 2.5.1 compiled with native CUDA 12.1 runtime support (`torch-2.5.1+cu121`).
- **Optimization Hyperparameters:** Trained across the **entire 10,181 continuous training window sequence** for 12 epochs using Smooth L1 Loss, Adam optimizer ($\eta = 0.001$, weight decay $10^{-4}$), batch size 64, and a ReduceLROnPlateau scheduler.
- **VRAM Utilization:** ~5.2 GB allocated during backpropagation across the 307-node spatio-temporal attention graph.
- **Training Throughput:** Total training converged in 605.4 seconds (~10.1 minutes, ~50.4 s/epoch). Real-time inference latency is **~1.2 ms** per 307-sensor network forward pass.

---

## 10. Results and Comparative Analysis

### 10.1 Quantitative Performance Comparison

| Model | 15-min MAE | 15-min RMSE | 30-min MAE | 30-min RMSE | 60-min MAE | 60-min RMSE | Overall MAE | Overall RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 24.92 | 39.00 | 30.28 | 45.87 | 42.38 | 61.67 | 31.56 | 48.36 |
| **PDFormer (RTX 4060 GPU)** | **21.91** | **32.98** | **25.00** | **37.36** | **31.77** | **46.63** | **25.59** | **38.46** |

*(Units: vehicles per 5 minutes on held-out PeMS04 test partition)*

### 10.2 Empirical Observations & Analysis
1. **Omni-Horizon Predictive Superiority:** Powered by GPU acceleration on the complete 10,181-window training distribution, PDFormer outperforms the Persistence baseline across **every single forecast horizon in both MAE and RMSE**. Overall, PDFormer slashes MAE by **18.9%** (25.59 vs. 31.56 veh/5min) and RMSE by **20.5%** (38.46 vs. 48.36 veh/5min).
2. **Mitigating Long-Range Error Cascades (Shockwave Modeling):** Highway traffic exhibits strong local momentum, allowing naive persistence to perform passably at immediate 15-minute intervals. However, as the forecast horizon extends to 60 minutes, persistence error escalates sharply (+58.1% in RMSE from 39.00 to 61.67 veh/5min) as historical readings become stale. In stark contrast, PDFormer's delay-aware spatial attention constrains 60-minute RMSE to **46.63 veh/5min**, securing a massive **15.04 veh/5min advantage** over persistence.
3. **Topological Spatial Regularization:** By enforcing geographic adjacency masking ($M_{\text{geo}}$) derived from the 340 physical highway connections, PDFormer isolates road corridors and models directional shockwave dynamics without suffering from spurious correlation noise across disconnected sensors.
4. **Real-Time Operational Deployment:** With an inference latency of **1.2 ms** on the RTX 4060 GPU, PDFormer easily qualifies for sub-second ITS control loops and dynamic variable speed limit modulation.

---

## 11. Dynamic Sensor-Level Congestion Mapping

To bridge the gap between numerical flow forecasts and real-world traffic management, we implement a dynamic risk classification layer in `src/mapping/congestion_map.py`:

### 11.1 Flow-Based Congestion Thresholds
Empirical quantiles derived from the training data distribution ($N = 10,181$ intervals) establish four objective operational tiers:
- **Free Flow (Level 0 - Green):** $\text{Flow} < 83.0 \text{ veh/5min}$ (Low density, unrestricted travel)
- **Moderate (Level 1 - Yellow):** $83.0 \le \text{Flow} < 179.0 \text{ veh/5min}$ (Stable flow, light commuter volume)
- **Heavy (Level 2 - Orange):** $179.0 \le \text{Flow} < 315.0 \text{ veh/5min}$ (Approaching highway capacity, speed reduction)
- **Severe (Level 3 - Red):** $\text{Flow} \ge 315.0 \text{ veh/5min}$ (Near-capacity / bottleneck breakdown, high congestion risk)

### 11.2 Topological Graph Visualization
Because the standard PeMS04 research release omits latitude/longitude coordinates, our mapping module constructs a force-directed network layout using `distance.csv` road connections. Sensors are dynamically color-coded by predicted congestion level for any chosen future timestamp and forecast horizon.

---

## 12. Ethical and Operational Considerations

- **Safety-Critical Disclaimer:** This predictive system forecasts physical traffic volume (vehicle count per 5-minute interval). It does **not** predict vehicular collisions, road fatalities, or individual vehicle trajectories.
- **Fairness & Geographic Routing:** Over-reliance on localized congestion maps in commercial navigation apps can divert freeway overflow into residential arterial corridors ("cut-through traffic"), creating neighborhood safety hazards. Operators must couple forecasts with municipal access policies.
- **Privacy Assurance:** Inductive loop detectors aggregate vehicular pulse counts without capturing license plates, facial recognition, or telematics identities, completely protecting commuter anonymity.

---

## 13. Limitations and Future Scope

1. **Distributed Multi-Node Scaling:** While the current implementation operates with GPU acceleration on a dedicated NVIDIA GeForce RTX 4060 GPU (~1.2 ms inference latency), scaling to state-wide highway networks encompassing tens of thousands of loop detectors across multi-year temporal horizons will benefit from distributed multi-GPU data parallelism.
2. **Missing Sensor Lat/Lon:** PeMS04 standard benchmark files include topological distances but omit GPS coordinates. Future work involves scraping the Caltrans PeMS web portal metadata to project nodes directly onto OpenStreetMap / GIS layers.
3. **Multi-Target Joint Forecasting:** While flow was the primary target, future models can jointly predict speed, occupancy, and flow simultaneously in a multi-task learning head to compute direct Level-of-Service (LOS) ratings.

---

## 14. Conclusion

This project successfully establishes a complete, academically defensible, and fully reproducible machine learning system for urban traffic congestion forecasting on the Caltrans PeMS04 benchmark. By enforcing strict chronological splitting, training-isolated Z-score standardization, baseline verification, GPU-accelerated PyTorch PDFormer integration (NVIDIA GeForce RTX 4060, CUDA 12.1), and dynamic sensor risk mapping, we demonstrated that Spatio-Temporal Transformers decisively outperform temporal persistence degradation across all 15 to 60-minute horizons (achieving an 18.9% overall MAE reduction and a 20.5% overall RMSE reduction). All components—including Jupyter notebooks, automated test suites, metrics JSON records, and an interactive Streamlit dashboard—are fully implemented and verified on real data.

---

## 15. References

1. **Jiang, J., Han, C., Zhao, W. X., & Wang, J. (2023).** *PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction.* Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI-23), 37(4), 4365–4373.
2. **Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019).** *Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (ASTGCN).* Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 33(01), 922–929.
3. **Song, C., Lin, Y., Guo, S., & Wan, H. (2020).** *Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Diversity Modeling (STSGCN).* AAAI-20, 34(01), 814–821.
4. **Chen, C., Petty, K., Skabardonis, A., Varaiya, P., & Bertini, R. (2001).** *Freeway Performance Measurement System (PeMS): Mining Loop Detector Data.* Transportation Research Record, 1748(1), 96–102.
5. **Vaswani, A., et al. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008.
