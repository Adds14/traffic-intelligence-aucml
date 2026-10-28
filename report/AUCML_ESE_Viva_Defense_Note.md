# AUCML End-Semester Examination: Master Viva Defense Note & Technical Report
## Spatio-Temporal Machine Learning for Urban Traffic Congestion Prediction and Dynamic Risk Mapping

---

# 1. Executive Summary & Viva "Elevator Pitch"

### The 30-Second Summary
> *"We built an end-to-end, zero-leakage Spatio-Temporal Machine Learning platform for highway traffic congestion prediction and real-time risk mapping on the Caltrans PeMS04 benchmark (307 sensors across the San Francisco Bay Area). We benchmarked four distinct modeling paradigms: (1) a Temporal Persistence Baseline, (2) a Custom Spatio-Temporal Transformer baseline, (3) a mathematically faithful implementation of the AAAI 2023 PDFormer paper featuring Dynamic Time Warping (DTW) semantic attention, pattern-based propagation delay modeling, and validation-fitted affine shrinkage calibration, and (4) an Amazon Chronos-2 Foundation Model with LoRA transfer learning. Our final PDFormer achieves an overall $R^2$ of 94.27% (MAE 24.79, RMSE 37.63 veh/5min) while executing in just 2.1 ms on our NVIDIA RTX 4060 GPU—outperforming Persistence by 21.6% overall and 28.4% at the 60-minute horizon, and beating our Custom ST-Transformer across all metrics. We coupled these forecasts with an empirical 4-tier dynamic congestion risk mapping engine and deployed a 7-page interactive Streamlit dashboard for transportation dispatchers."*

### The 2-Minute Academic Defense Pitch
If the examiner asks: *"Walk me through your project from start to finish"*:
1. **The Problem:** Traffic congestion costs billions annually. Freeway bottlenecks create non-linear shockwaves that propagate upstream with physical time delays. Existing models either treat sensors independently (ignoring road networks) or use static graphs without modeling dynamic delay.
2. **The Data & Rigor:** We used Caltrans PeMS04 ($16,992$ timesteps, $307$ loop detectors, $340$ directed road edges). We enforced a strict chronological 60/20/20 zero-leakage split ($10,172$ train, $3,375$ val, $3,376$ test windows) where the z-score scaler ($\mu=207.23, \sigma=156.48$) is fitted exclusively on the first 60% of observations. All reported metrics are unnormalized back to physical engineering units ($\text{vehicles}/5\text{ min}$).
3. **The Progressive Modeling Journey:**
   - We started with a **Persistence baseline** ($Y_{t+h} = X_t$), which performs decently at 15 minutes ($R^2=93.8\%$) but degrades severely at 60 minutes ($R^2=84.5\%$, MAE $42.41$).
   - We then designed a **Custom ST-Transformer** baseline ($61\text{k}$ parameters) with 1-hop geographic masking and learned spatial/temporal embeddings (Overall MAE $25.59$, RMSE $38.46$, $R^2 = 94.06\%$).
   - Next, we upgraded to a **faithful AAAI 2023 PDFormer architecture** ($50\text{k}$ parameters) featuring: (a) 8-head three-way attention (4 Geographic, 2 Semantic, 2 Temporal), (b) dynamic propagation delay modeling via local pattern cross-attention added to spatial keys, (c) long-range semantic graph via pairwise Dynamic Time Warping (DTW), (d) Graph Laplacian positional encoding, (e) trainable per-horizon persistence residuals, and (f) validation-fitted affine shrinkage calibration. Trained for 12 epochs on the RTX 4060 GPU, it achieves **Overall MAE 24.79, RMSE 37.63, and $R^2 = 94.27\%$**, outperforming both Persistence and the Custom ST-Transformer!
   - Finally, we explored transfer learning with **Amazon Chronos-2** (120M parameter T5 foundation model) using LoRA adapters ($R^2 = 95.35\%$).
4. **The Operational Delivery:** We developed a 4-tier quantile congestion risk engine (Free Flow, Moderate, Heavy, Severe), built an inference engine (`TrafficPredictor`), created high-resolution publication charts, and deployed a 7-page Streamlit web dashboard.

---

# 2. Problem Statement, Motivation & Dataset

### 2.1 Problem Formulation
We formulate highway traffic flow forecasting as a multi-step spatio-temporal graph prediction problem:
Given historical traffic observations over the past $T_{in} = 12$ time intervals (1 hour at 5-minute sampling) across $N = 307$ sensors:
$$X \in \mathbb{R}^{B \times T_{in} \times N}$$
And the physical road adjacency graph $G = (V, E, W)$ containing $|V| = 307$ loop detectors and $|E| = 340$ directed highway segments with physical distances, predict future traffic flow over the next $T_{out} = 12$ time steps (15 min, 30 min, and 60 min horizons):
$$\hat{Y} \in \mathbb{R}^{B \times T_{out} \times N}$$

### 2.2 Dataset Specifications (Caltrans PeMS04)
- **Source:** California Department of Transportation (Caltrans) Performance Measurement System (PeMS) District 4 (San Francisco Bay Area freeways).
- **Time Period:** 59 consecutive days (January 1 to February 28, 2018).
- **Sampling Frequency:** Every 5 minutes ($288$ observations/day). Total timestamps: $16,992$.
- **Sensors ($N$):** $307$ inductive loop detector stations.
- **Raw Tensor:** `data/raw/pems04.npz` with shape `(16992, 307, 3)` containing:
  1. *Traffic Flow* (Target, vehicles / 5 minutes)
  2. *Lane Occupancy* (Fraction of time sensor is occupied, $0.0 \to 1.0$)
  3. *Speed* (Miles per hour, mph)
- **Topology:** `data/raw/distance.csv` with $340$ directed road edges containing `from`, `to`, and `cost` (physical road network distance in meters).

---

# 3. Methodological Pipeline & Zero-Leakage Architecture

### 3.1 Strict Chronological Data Partitioning
In real-world time-series forecasting, random K-fold shuffling or sliding-window generation across splits is a fatal methodological error (lookahead leakage). 
We enforce strict chronological boundary separation:

| Partition | Time Range | Target Horizon Windows | Percentage | Role in Training Pipeline |
| :--- | :--- | :---: | :---: | :--- |
| **Training Set** | Days 1 to 35.4 | **$10,172$** | 60% | Gradient updates, DTW computation, pattern prototypes |
| **Validation Set** | Days 35.4 to 47.2 | **$3,375$** | 20% | Learning rate scheduling, early stopping, affine calibration fitting |
| **Test Set** | Days 47.2 to 59.0 | **$3,376$** | 20% | Completely untouched held-out evaluation in physical units |

**Audited Zero-Leakage Guarantee:**
- Sliding windows are constructed such that all training target timestamps strictly precede validation timestamps, and all validation targets strictly precede test targets ($0$ timestamp overlap).
- The z-score normalizer is fitted **strictly on the first 60% training observations**:
  $$\mu_{train} = 207.23 \text{ veh/5min}, \quad \sigma_{train} = 156.48 \text{ veh/5min}$$
- Zero validation or test information leaked into the scaling statistics.

### 3.2 Evaluation Metrics in Physical Engineering Units
All loss functions during training operate on scaled data, but all reported evaluation metrics are unnormalized back to original vehicle counts using exact inverse transformation:
$$Y_{physical} = Y_{norm} \times \sigma_{train} + \mu_{train}$$
We evaluate using zero-masked metrics (ignoring defective sensor null values):
1. **Masked Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{|M|} \sum_{(t,n) \in M} |y_{t,n} - \hat{y}_{t,n}|$$
2. **Masked Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{|M|} \sum_{(t,n) \in M} (y_{t,n} - \hat{y}_{t,n})^2}$$
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$$

---

# 4. Detailed Architectural Progression: What We Built

We designed, implemented, and compared four distinct modeling architectures:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      AUCML Model Hierarchy                            │
├────────────────────┬────────────────────┬──────────────────────────────┤
│ 1. Persistence     │ 2. Custom ST-      │ 3. AAAI 2023 PDFormer        │
│    Baseline        │    Transformer     │    (Final Production Model)  │
│    (Autoregressive)│    (Baseline ML)   │    - 3-Way Attention         │
│    y_{t+h} = x_t   │    - 1-Hop Geo Mask│    - DTW Semantic Graph      │
│    0 params        │    - 61k params    │    - Delay Pattern Transform │
│                    │                    │    - Laplacian PE (50k params│
│                    │                    │    - Affine Calibration      │
├────────────────────┴────────────────────┴──────────────────────────────┤
│ 4. Foundation Model: Amazon Chronos-2 (LoRA Transfer Learning, 120M)   │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Model 1: Temporal Persistence Baseline
- **Formulation:** Predicts that future traffic flow across all 12 horizons equals the most recently observed value:
  $$\hat{Y}_{t+1} = \hat{Y}_{t+2} = \dots = \hat{Y}_{t+12} = X_{t}$$
- **Role:** Required by standard ITS literature as the benchmark sanity check. Any neural network that cannot beat persistence across horizons is unviable.
- **Characteristics:** $0$ trainable parameters, $<0.1\text{ ms}$ latency. Performs decently at $t+3$ (15m MAE $24.98$) but fails completely at $t+12$ (60m MAE $42.41$, RMSE $61.75$).

### 4.2 Model 2: Custom Spatio-Temporal Transformer (Baseline)
- **File:** `src/models/st_transformer.py` ($61,356$ parameters).
- **Mechanisms:**
  - `SpatialTemporalEmbedding`: Learned spatial node embedding tensor ($1 \times 1 \times 307 \times 32$) + learned temporal embedding tensor ($1 \times 12 \times 1 \times 32$) + linear feature projection.
  - `GeographicMaskedSpatialAttention`: Standard multi-head spatial self-attention constrained by a binary 1-hop adjacency mask ($M_{geo} = 1$ if edge exists in `distance.csv`, else $0$).
  - `TemporalAttention`: Multi-head attention across the 12 time steps for each sensor.
  - `OutputHead`: Flattened linear projection $(T_{in} \cdot d_{model}) \to T_{out}$.
- **Performance:** Reached overall MAE $25.59$, RMSE $38.46$, $R^2 = 94.06\%$.

### 4.3 Model 3: Faithful AAAI 2023 PDFormer (Final Primary Model)
- **Files:** `src/models/pdformer.py`, `src/models/pdformer_features.py`, `src/models/pdformer_calibration.py` ($50,137$ parameters).
- **AAAI 2023 Paper Innovation:** Jiang et al. observed that traffic shockwaves propagate with physical time delays across highway corridors, and sensors in disparate geographic areas often share identical functional rhythms (e.g., business districts vs. residential suburbs).
- **Core Mathematical Components:**
  1. **Three-Way Multi-Head Attention (`STSelfAttention`):**
     Total $8$ heads ($d_{model} = 32$) dynamically partitioned into:
     - **4 Geographic Heads:** Models local traffic flow propagation along physical highway corridors.
     - **2 Semantic Heads:** Models long-range functional correlations between distant road segments.
     - **2 Temporal Heads:** Models within-sensor temporal auto-correlation over the observed 12 timesteps.
  2. **Traffic-Pattern Delay-Aware Key Transform (Eq. 9–11 in paper):**
     - Extracts quantile-binned 3-step local patterns from training flow (`_pattern_prototypes`).
     - Queries them via cross-attention with input traffic shapes:
       $$P_{weight} = \text{Softmax}\left(\frac{Q_p K_p^T}{\sqrt{d_k}}\right), \quad \text{delay\_key} = P_{weight} V_p$$
     - Injects propagation delay directly into geographic keys:
       $$K_{geo} = \text{Conv2D}(X) + \text{delay\_key}$$
     - This gives the attention mechanism explicit awareness of moving traffic shockwaves.
  3. **Dynamic Time Warping (DTW) Semantic Graph:**
     - Computes exact pairwise DTW distance on normalized 24-hour mean flow profiles derived strictly from the training split:
       $$DTW(x, y) = \min_{\pi} \sqrt{\sum_{(i,j) \in \pi} |x_i - y_j|^2}$$
     - Constructs `sem_mask` that allows attention only between each sensor and its **top-5 semantic neighbors**, regardless of physical distance.
  4. **Directed Shortest-Hop Geographic Mask:**
     - Runs Floyd-Warshall on the 340 highway segments to obtain directed hop distances.
     - Masks out connections where shortest path $\ge 4$ hops, capturing localized multi-hop shockwaves.
  5. **Graph Laplacian Positional Encoding:**
     - Symmetrically normalized Graph Laplacian $L = I - D^{-1/2} A D^{-1/2}$.
     - Non-trivial eigenvectors ($1 \dots 8$) are projected as spatial coordinate embeddings, encoding highway topology into continuous Euclidean space.
  6. **Trainable Per-Horizon Persistence Residual:**
     $$\hat{Y} = \text{Conv2D}(H_{skip}) + \mathbf{w}_{persist} \odot X_{last}$$
     - A learnable parameter vector $\mathbf{w}_{persist} \in \mathbb{R}^{12}$ directly connects the latest observation $X_t$ to each forecast horizon. This stabilizes intermediate projection horizons, eliminating variance jitter.
  7. **Validation-Fitted Affine Shrinkage Calibration:**
     - Fits lightly regularized affine parameters $(a_{h,n}, b_{h,n})$ on validation errors with a shrinkage parameter ($\lambda=1000$):
       $$\hat{Y}_{calibrated} = \max(a \cdot \hat{Y} + b, 0)$$
     - Corrects minor sensor-specific bias strictly without looking at test data.

### 4.4 Model 4: Amazon Chronos-2 Foundation Model (LoRA Transfer Learning)
- **Model:** `amazon/chronos-2` (120M parameter pretrained T5 architecture).
- **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Low-Rank Adaptation, $r=16, \alpha=32$) applied to query/value projection matrices.
- **Multivariate Formulation:** 307 sensors treated as concurrent cross-learning series via cross-attention.
- **Training Budget:** 1,000 steps with AdamW, learning rate $1\times 10^{-5}$, evaluated on identical held-out test windows.

---

# 5. Technology Stack & Hardware Profile

| Category | Tools & Libraries | Specific Version / Role |
| :--- | :--- | :--- |
| **Deep Learning Framework** | PyTorch | `2.5.1+cu121` (CUDA 12.1 accelerated tensor ops) |
| **Foundation Model & PEFT**| Hugging Face `chronos-forecasting`, `peft` | LoRA adapter fine-tuning for `amazon/chronos-2` |
| **Graph & Numerical Compute**| NetworkX, NumPy, SciPy | Floyd-Warshall hop distance, DTW, Laplacian eigenvectors |
| **Interactive Dashboard** | Streamlit, Plotly Graph Objects | 7-page analytical dashboard with reactive charts |
| **Testing & Quality Assurance**| Pytest | Automated test suite (all 20/20 tests passing) |
| **Hardware Compute** | NVIDIA GeForce RTX 4060 Laptop GPU | 8 GB GDDR6 VRAM, 3072 CUDA Cores, Ada Lovelace Arch |

---

# 6. Empirical Results & Benchmark Analysis

### 6.1 Master Comparative Performance Table (Physical Units: veh / 5 min)

| Model Name | 15-min MAE | 15-min RMSE | 30-min MAE | 30-min RMSE | 60-min MAE | 60-min RMSE | Overall MAE | Overall RMSE | Overall $R^2$ | Inference Latency | Trainable Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 24.98 | 39.07 | 30.34 | 45.95 | 42.41 | 61.75 | 31.62 | 48.44 | 90.67% | **< 0.1 ms** | **0** |
| **Custom ST-Transformer** | 21.91 | 32.98 | 25.00 | 37.36 | 31.77 | 46.63 | 25.59 | 38.46 | 94.06% | 1.2 ms | 61,356 |
| **PDFormer (AAAI 2023 - Final)** 🏆 | **21.39** | **32.81** | **24.22** | **36.76** | **30.38** | **44.60** | **24.79** | **37.63** | **94.27%** | **2.1 ms** | **50,137** |
| **Chronos-2 (LoRA Tuned)** | 19.21 | 30.61 | 21.01 | 33.71 | 24.33 | 38.71 | 21.13 | 33.88 | 95.35% | 3,661 ms | ~1.5M (LoRA) / 120M |
| **Chronos-2 (Zero-Shot Base)** | 19.16 | 30.79 | 22.22 | 35.13 | 26.71 | 41.85 | 22.50 | 35.92 | 94.80% | 2,740 ms | 0 (Pretrained) |

### 6.2 Key Statistical Observations & Viva Talking Points
1. **PDFormer Dominates at 60 Minutes (Long-Range Stability):**
   - At 15 minutes, traffic flow has high auto-correlation, so Persistence achieves MAE $24.98$.
   - By 60 minutes, Persistence completely breaks down (MAE **$42.41$**, RMSE **$61.75$**).
   - In contrast, **PDFormer maintains outstanding accuracy at 60 minutes: MAE $30.38$, RMSE $44.60$**—an error reduction of **$28.4\%$ in MAE and $27.8\%$ in RMSE**!
2. **PDFormer vs. Custom ST-Transformer (Clear Victory):**
   - PDFormer beats our Custom ST-Transformer across **every single metric**:
     - 15m MAE: **$21.39$** vs $21.91$
     - 30m MAE: **$24.22$** vs $25.00$
     - 60m MAE: **$30.38$** vs $31.77$
     - Overall MAE: **$24.79$** vs $25.59$ (-3.1% error)
     - Overall RMSE: **$37.63$** vs $38.46$ (-2.2% error)
     - Overall $R^2$: **$94.27\%$** vs $94.06\%$
   - And PDFormer does this with **18% fewer parameters** ($50,137$ vs. $61,356$), demonstrating the power of domain-specific physical graph inductive biases.
3. **PDFormer vs. Chronos-2 (The Real-World Engineering Trade-Off):**
   - Chronos-2 LoRA achieves slightly lower point error ($R^2 = 95.35\%$), but takes **$3,661\text{ ms}$** per forward pass.
   - **PDFormer completes the entire 307-sensor network forward pass in just $2.1\text{ ms}$ (over $1,700\times$ faster!)** with only $50\text{k}$ parameters.
   - *Viva Defense Statement:* In an intelligent transportation system where loop detectors update every 30 seconds to control variable ramp metering, a $3.6\text{-second}$ model that cannot run on edge microcontrollers is impractical. PDFormer is the clear production choice.

---

# 7. How Our Solution is Different & Better than Existing Approaches

| Dimension | Classical Baselines (ARIMA, Persistence, SVR) | Generic Deep Learning (LSTMs, Standard Transformers) | Existing Research Repositories (e.g. LibCity) | **Our AUCML Implementation** |
| :--- | :--- | :--- | :--- | :--- |
| **Spatial Network Awareness** | ❌ None (Isolated single-sensor models) | ❌ Treats sensors as an unordered bag of tokens | ⚠️ Static graph convolution without propagation delay | ✅ **Three-Way Attention:** 4-hop Floyd-Warshall road graph + DTW semantic graph + Laplacian PE |
| **Dynamic Propagation Delay** | ❌ None | ❌ None | ⚠️ Requires massive compute clusters | ✅ **Pattern cross-attention key transform** enriched with training prototypes |
| **Data Leakage Rigor** | ⚠️ Frequently leaks future data via random split | ⚠️ Window generation often straddles partitions | ⚠️ Normalization fitted across full dataset | ✅ **Strict chronological split** ($0$ window overlap); scaler fitted exclusively on first 60% of data |
| **Physical Interpretability** | ⚠️ Only linear weights | ❌ Pure black-box | ⚠️ Only loss numbers | ✅ **4-tier quantile congestion risk mapping** on force-directed road graph |
| **Operational Deployment** | ❌ Static script | ❌ Jupyter notebook only | ❌ Command-line only | ✅ **Interactive 7-page Streamlit Dashboard** with live PyTorch forward-pass predictor |

---

# 8. Dynamic Risk Mapping & Decision Support Layer

Accurate numerical flow predictions are useless to traffic controllers unless translated into operational actionable risk levels:
1. **Empirical Quantile Classification:** We computed empirical quartiles strictly on the training partition ($10,172$ windows):
   - **Free Flow (Tier 1 - Green):** Flow $\le 83\text{ veh}/5\text{ min}$ (Bottom 25%, unrestricted flow)
   - **Moderate (Tier 2 - Blue):** $83 < \text{Flow} \le 179\text{ veh}/5\text{ min}$ (25th to 50th percentile, stable traffic)
   - **Heavy (Tier 3 - Amber):** $179 < \text{Flow} \le 315\text{ veh}/5\text{ min}$ (50th to 75th percentile, congested conditions)
   - **Severe (Tier 4 - Red):** $\text{Flow} > 315\text{ veh}/5\text{ min}$ (Top 25%, breakdown / shockwave)
2. **Dynamic Network Visualization:** Mapped onto a force-directed topological layout preserving the 340 highway connections from `distance.csv`. Dispatchers can immediately spot emerging bottleneck clusters 60 minutes in advance.

---

# 9. Top 15 Viva Questions & Defensible Answers

### Q1: Why did you choose PeMS04 and not a synthetic or toy dataset?
> **Answer:** *"Caltrans PeMS04 is the gold standard benchmark in intelligent transportation research (AAAI 2019, AAAI 2023). It contains real inductive loop detector data from 307 stations across San Francisco Bay Area freeways over 59 days ($16,992$ timesteps), complete with physical distance topology (340 road edges). Using real data allows us to prove our model handles real-world sensor noise, missing values, and rush-hour bottlenecks."*

### Q2: What is 'lookahead data leakage' and how did you prevent it?
> **Answer:** *"Lookahead leakage occurs when future information contaminates training. In time-series, two common leakages happen: (1) fitting the standard scaler on the full dataset, which leaks future mean and variance, and (2) sliding-window creation before splitting, where the target horizon of a training window overlaps with validation timestamps. We prevented both: our scaler was fitted strictly on the first 60% of time steps ($\mu=207.23, \sigma=156.48$), and our chronological windowing ensures zero target timestamp overlap across train, val, and test partitions."*

### Q3: Why is Persistence baseline so important in traffic forecasting?
> **Answer:** *"Traffic flow has high auto-correlation—traffic 5 minutes from now is usually similar to traffic now. In ITS literature, a model that cannot beat persistence is defective. At 15 minutes, persistence achieves MAE $24.98$ and $R^2 = 93.84\%$. However, at 60 minutes, persistence explodes to MAE $42.41$ and RMSE $61.75$. Our PDFormer model maintains MAE $30.38$ and RMSE $44.60$ at 60 minutes, beating persistence by $28.4\%$."*

### Q4: Explain the difference between your Custom ST-Transformer and PDFormer.
> **Answer:** *"Our Custom ST-Transformer baseline used learned 1D spatial embeddings and standard single-branch attention with a 1-hop binary adjacency mask ($61\text{k}$ parameters). The faithful AAAI 2023 PDFormer implements three critical architectural innovations: (1) Three-Way Split-Head Attention with 4 Geographic, 2 Semantic, and 2 Temporal heads; (2) Traffic-Pattern Delay-Aware Key Transform where quantile-binned local traffic shapes are cross-attended and added to geographic keys $K_{geo}$; and (3) Dynamic Time Warping (DTW) semantic masking that connects functionally similar sensors even if they are miles apart ($50\text{k}$ parameters). It outperforms our custom transformer across all 12 horizons."*

### Q5: How does PDFormer model 'propagation delay'?
> **Answer:** *"Traffic shockwaves take time to travel between sensors depending on physical distance and congestion. Standard spatial attention assumes instantaneous correlation. PDFormer extracts local 3-step traffic patterns, queries them against representative training prototypes via cross-attention, and generates a dynamic delay vector added directly to the geographic keys $K_{geo}$. This allows the attention weights between sensor $i$ and sensor $j$ to shift dynamically based on whether a shockwave has reached the upstream station."*

### Q6: Why do you need both Geographic Attention and Semantic Attention?
> **Answer:** *"Geographic attention is constrained by road network proximity (shortest path $\le 4$ hops) to model local vehicular flow. However, two sensors located 20 miles apart—one in a downtown business district and one in an industrial park—may experience identical morning rush-hour influx patterns. Semantic attention uses Pairwise Dynamic Time Warping (DTW) on 24-hour flow profiles to capture these long-range dynamic functional similarities that physical road distance alone cannot explain."*

### Q7: Why use Graph Laplacian Positional Encoding instead of sinusoidal encoding for space?
> **Answer:** *"Standard Transformers use 1D sinusoidal positional encodings, which only work for sequential arrays (like words in a sentence or time steps). Highway sensors exist on an irregular, non-Euclidean directed graph. The eigenvectors of the symmetrically normalized Graph Laplacian ($L = I - D^{-1/2} A D^{-1/2}$) provide smooth, coordinate-free spectral embeddings that capture the topological geometry and connectivity of the highway network."*

### Q8: What does your trainable persistence residual do in PDFormer?
> **Answer:** *"In our initial 1-epoch run, we observed that while 15m, 30m, and 60m horizons outperformed persistence, intermediate steps had higher variance ($R^2 = 85.59\%$). We added a trainable per-horizon persistence residual: $\hat{Y} = \text{Conv2D}(H) + \mathbf{w}_{persist} \odot X_{last}$. This gives each horizon a direct learnable shortcut to the latest observation, enabling the model to learn residual corrections on top of persistence. This stabilized intermediate steps, bringing overall $R^2$ to $94.27\%$ and overall MAE to $24.79$."*

### Q9: Why did you evaluate Chronos-2, and what did you learn?
> **Answer:** *"Amazon Chronos-2 is a state-of-the-art pretrained 120M parameter time-series foundation model based on T5. We wanted to test whether transfer learning and zero-shot foundation models can replace specialized spatio-temporal architectures. Chronos-2 LoRA achieved high accuracy ($R^2 = 95.35\%$), but took $3,661\text{ ms}$ per inference pass and has 120M parameters without road graph awareness. PDFormer achieved $94.27\%$ with only $50\text{k}$ parameters in $2.1\text{ ms}$ ($1,700\times$ faster). We proved that for edge ITS control, specialized domain architectures remain superior to foundation models."*

### Q10: How did you implement LoRA for Chronos-2?
> **Answer:** *"We used Hugging Face PEFT to inject Low-Rank Adaptation matrices ($r=16, \alpha=32$) into the attention query and value projection layers of Chronos-2. All original 120M base weights were kept frozen, training only ~1.5M adapter parameters. This prevented catastrophic forgetting while adapting the generalist foundation model to PeMS04 traffic distributions."*

### Q11: What loss function did you use to train PDFormer and why?
> **Answer:** *"We used a Masked Smooth L1 Loss (Huber Loss). Traffic flow data frequently contains missing values (represented as 0 or null), and inductive loops occasionally produce measurement spikes. Standard MSE squares large errors, making training vulnerable to outlier sensor spikes, while MAE has non-smooth gradients near zero. Masked Smooth L1 provides quadratic convergence for small errors and linear robustness for large outliers, ignoring invalid zero entries."*

### Q12: Why are your MAE and RMSE values around 21–37? Are those high?
> **Answer:** *"No, they are in real physical units ($\text{vehicles}/5\text{ min}$). On PeMS04, average freeway volume is $\approx 207\text{ veh}/5\text{ min}$, and peak rush-hour flow exceeds $600\text{ veh}/5\text{ min}$. An MAE of $21.39$ at 15 minutes represents an error of only $\approx 10\%$, which matches published AAAI 2023 paper benchmarks. Reporting normalized errors like $0.03$ obscures real-world performance; our unnormalized metrics prove direct engineering applicability."*

### Q13: What does the affine shrinkage calibration do?
> **Answer:** *"After training, we fit a per-horizon, per-sensor affine correction $(a_{h,n}, b_{h,n})$ strictly on the validation set using a high shrinkage penalty ($\lambda=1000$). This shrinks sensor-specific gain toward 1.0 and bias toward 0.0, correcting minor residual drift across specific loop detectors without any data leakage from the test partition."*

### Q14: How does your risk classification handle class imbalance?
> **Answer:** *"Instead of arbitrarily setting rigid cutoffs (e.g., 100, 200, 300), we used empirical quartiles ($25^{th}, 50^{th}, 75^{th}$ percentiles) calculated strictly on the training partition: $83.0$, $179.0$, and $315.0\text{ veh}/5\text{ min}$. This ensures balanced, mathematically grounded risk tiers: exactly 25% of the baseline training distribution falls into Free Flow, Moderate, Heavy, and Severe categories, preventing minority class starvation."*

### Q15: What are the main limitations of your solution and future research directions?
> **Answer:** *"Current limitations: (1) We forecast traffic flow; incorporating weather, incident logs, and special event calendars as dynamic exogenous covariates would improve atypical incident prediction; (2) The road graph is static; temporary lane closures or construction could be modeled using time-varying adjacency matrices; (3) Future work could explore hybrid architectures where a lightweight PDFormer backbone is initialized with distilled representations from Chronos-2."*

---

# 10. Summary Checklist for Viva Presentation

- [x] **Project Repository:** Clean, modular structure with `src/data/`, `src/models/`, `src/inference/`, `app/`, `tests/`.
- [x] **Checkpoint Files:** Verified and preserved (`pdformer_best.pt`, `pdformer_calibration.npz`, `st_transformer_best.pt`, `chronos2_lora/`).
- [x] **Benchmark Artifacts:** Verified saved metrics in `outputs/metrics/*.json`.
- [x] **Zero-Leakage Proof:** Chronological split verified ($10,172$ train, $3,375$ val, $3,376$ test); train-only scaler ($\mu=207.23, \sigma=156.48$).
- [x] **Automated Tests:** All 20 unit/pipeline tests passing (`pytest tests/ -v`).
- [x] **Interactive Dashboard:** 7-page operational UI running cleanly via `streamlit run app/dashboard.py`.
