"""PDFormer-style spatio-temporal transformer for traffic forecasting.

The encoder follows the public AAAI 2023 PDFormer implementation's core
mechanisms: temporal attention, short-range geographic attention, long-range
semantic attention, and a traffic-pattern feature transform that enriches the
geographic keys.  The project keeps its standalone PyTorch/data pipeline and
builds the semantic and pattern inputs from its chronological training split.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbedding(nn.Module):
    """Flow, within-window position, and Laplacian node embeddings."""

    def __init__(self, num_nodes, d_model, lap_pos_enc, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(1, d_model)
        self.node_embedding = nn.Linear(lap_pos_enc.shape[-1], d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("lap_pos_enc", torch.as_tensor(lap_pos_enc, dtype=torch.float32))
        self.num_nodes = num_nodes
        self.d_model = d_model

    def forward(self, x):
        # x: batch, time, node
        _, steps, nodes = x.shape
        position = torch.arange(steps, device=x.device, dtype=x.dtype)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pos = torch.zeros(steps, self.d_model, device=x.device, dtype=x.dtype)
        pos[:, 0::2] = torch.sin(position[:, None] * div_term)
        pos[:, 1::2] = torch.cos(position[:, None] * div_term[: pos[:, 1::2].shape[-1]])
        spatial = self.node_embedding(self.lap_pos_enc[:nodes]).view(1, 1, nodes, self.d_model)
        return self.dropout(self.value_embedding(x.unsqueeze(-1)) + pos.view(1, steps, 1, -1) + spatial)


class STSelfAttention(nn.Module):
    """Three-way attention plus PDFormer's pattern-based delay transform."""

    def __init__(self, dim, geo_heads=4, sem_heads=2, temporal_heads=2, dropout=0.1):
        super().__init__()
        total_heads = geo_heads + sem_heads + temporal_heads
        if dim % total_heads:
            raise ValueError("d_model must be divisible by the sum of attention heads")
        self.geo_heads, self.sem_heads, self.temporal_heads = geo_heads, sem_heads, temporal_heads
        self.head_dim = dim // total_heads
        self.geo_dim = self.head_dim * geo_heads
        self.sem_dim = self.head_dim * sem_heads
        self.temporal_dim = self.head_dim * temporal_heads
        self.scale = self.head_dim ** -0.5
        self.pattern_embedding = nn.Linear(3, dim)

        def conv(out_dim):
            return nn.Conv2d(dim, out_dim, kernel_size=1)

        self.geo_q, self.geo_k, self.geo_v = conv(self.geo_dim), conv(self.geo_dim), conv(self.geo_dim)
        self.sem_q, self.sem_k, self.sem_v = conv(self.sem_dim), conv(self.sem_dim), conv(self.sem_dim)
        self.temp_q, self.temp_k, self.temp_v = conv(self.temporal_dim), conv(self.temporal_dim), conv(self.temporal_dim)
        self.pattern_q = nn.Linear(dim, self.geo_dim)
        self.pattern_k = nn.Linear(dim, self.geo_dim)
        self.pattern_v = nn.Linear(dim, self.geo_dim)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _conv(x, layer):
        return layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def _spatial_branch(self, x, q_layer, k_layer, v_layer, heads, dim, mask):
        b, t, n, _ = x.shape
        q, k, v = (self._conv(x, layer) for layer in (q_layer, k_layer, v_layer))
        q = q.reshape(b, t, n, heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(b, t, n, heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(b, t, n, heads, self.head_dim).permute(0, 1, 3, 2, 4)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.view(1, 1, 1, n, n), torch.finfo(scores.dtype).min)
        weights = self.dropout(scores.softmax(dim=-1))
        out = weights @ v
        return out.transpose(2, 3).reshape(b, t, n, dim)

    def forward(self, x, x_patterns, pattern_keys, geo_mask, sem_mask):
        b, t, n, _ = x.shape
        # Temporal attention is per sensor across the observed time window.
        tq, tk, tv = (self._conv(x, layer) for layer in (self.temp_q, self.temp_k, self.temp_v))
        tq = tq.permute(0, 2, 1, 3).reshape(b, n, t, self.temporal_heads, self.head_dim).transpose(2, 3)
        tk = tk.permute(0, 2, 1, 3).reshape(b, n, t, self.temporal_heads, self.head_dim).transpose(2, 3)
        tv = tv.permute(0, 2, 1, 3).reshape(b, n, t, self.temporal_heads, self.head_dim).transpose(2, 3)
        tw = self.dropout(((tq @ tk.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        temporal = (tw @ tv).transpose(2, 3).reshape(b, n, t, self.temporal_dim).transpose(1, 2)

        # Pattern queries attend over training-derived patterns for each node.
        # Their values are added to the geographic keys, modeling propagation delay.
        px = self.pattern_embedding(x_patterns)
        pk = self.pattern_embedding(pattern_keys)
        pq, pkey, pvalue = self.pattern_q(px), self.pattern_k(pk), self.pattern_v(pk)
        pweights = ((pq.unsqueeze(-2) @ pkey.unsqueeze(0).unsqueeze(0).transpose(-2, -1)) * self.scale).softmax(-1)
        delay_key = (pweights @ pvalue.unsqueeze(0).unsqueeze(0)).squeeze(-2)

        geo_k = self._conv(x, self.geo_k) + delay_key
        # Geographic attention uses the delay-enriched keys.
        gq = self._conv(x, self.geo_q).reshape(b, t, n, self.geo_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        gk = geo_k.reshape(b, t, n, self.geo_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        gv = self._conv(x, self.geo_v).reshape(b, t, n, self.geo_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        gs = (gq @ gk.transpose(-2, -1)) * self.scale
        gs = gs.masked_fill(geo_mask.view(1, 1, 1, n, n), torch.finfo(gs.dtype).min)
        gw = self.dropout(gs.softmax(-1))
        geographic = (gw @ gv).transpose(2, 3).reshape(b, t, n, self.geo_dim)

        semantic = self._spatial_branch(x, self.sem_q, self.sem_k, self.sem_v,
                                        self.sem_heads, self.sem_dim, sem_mask)
        return self.dropout(self.projection(torch.cat((temporal, geographic, semantic), dim=-1)))


class STEncoderBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, mlp_ratio=2, attention_heads=(4, 2, 2)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = STSelfAttention(dim, *attention_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout))

    def forward(self, x, x_patterns, pattern_keys, geo_mask, sem_mask):
        x = x + self.attention(self.norm1(x), x_patterns, pattern_keys, geo_mask, sem_mask)
        return x + self.mlp(self.norm2(x))


class PDFormer(nn.Module):
    """PDFormer core; inputs and outputs are (batch, time, sensors)."""

    def __init__(self, num_nodes=307, in_dim=1, history_len=12, forecast_len=12,
                 d_model=32, num_heads=2, d_ff=128, num_layers=3, dropout=0.1,
                 adj_matrix=None, dtw_matrix=None, pattern_keys=None, lap_pos_enc=None,
                 semantic_neighbors=5, short_hops=4):
        super().__init__()
        del in_dim, num_heads  # The original implementation uses a single flow feature and split heads.
        self.num_nodes, self.history_len, self.forecast_len = num_nodes, history_len, forecast_len
        if d_model % 8 == 0:
            attention_heads = (4, 2, 2)
        elif d_model % 4 == 0:
            attention_heads = (2, 1, 1)
        else:
            raise ValueError("d_model must be divisible by four for the spatial/temporal attention heads")
        if lap_pos_enc is None:
            lap_pos_enc = np.eye(num_nodes, min(8, num_nodes), dtype=np.float32)
        if pattern_keys is None:
            pattern_keys = np.zeros((num_nodes, 1, 3), dtype=np.float32)
        if adj_matrix is None:
            adj_matrix = np.eye(num_nodes, dtype=np.float32)
        if dtw_matrix is None:
            dtw_matrix = np.eye(num_nodes, dtype=np.float32)

        geo_mask, sem_mask = self._build_masks(adj_matrix, dtw_matrix, semantic_neighbors, short_hops)
        self.register_buffer("geo_mask", torch.as_tensor(geo_mask, dtype=torch.bool))
        self.register_buffer("sem_mask", torch.as_tensor(sem_mask, dtype=torch.bool))
        self.register_buffer("pattern_keys", torch.as_tensor(pattern_keys, dtype=torch.float32))
        self.embedding = DataEmbedding(num_nodes, d_model, lap_pos_enc, dropout)
        self.layers = nn.ModuleList([STEncoderBlock(d_model, dropout, max(2, d_ff // d_model), attention_heads)
                                     for _ in range(num_layers)])
        skip_dim = d_model * 2
        self.skip_convs = nn.ModuleList([nn.Conv2d(d_model, skip_dim, 1) for _ in range(num_layers)])
        self.time_projection = nn.Conv2d(history_len, forecast_len, 1)
        self.output_projection = nn.Conv2d(skip_dim, 1, 1)
        # Learn a per-horizon persistence residual. This gives each forecast
        # step a direct path from the latest observation, including horizons
        # whose nonlinear output head may become inactive.
        self.persistence_scale = nn.Parameter(torch.zeros(forecast_len))

    @staticmethod
    def _build_masks(adj_matrix, dtw_matrix, semantic_neighbors, short_hops):
        adjacency = np.asarray(adj_matrix) > 0
        n = adjacency.shape[0]
        # Directed shortest hop distances, matching the road-edge direction.
        hops = np.where(adjacency, 1.0, np.inf).astype(np.float32)
        np.fill_diagonal(hops, 0.0)
        for k in range(n):
            hops = np.minimum(hops, hops[:, k:k + 1] + hops[k:k + 1, :])
        geo_mask = hops >= short_hops
        np.fill_diagonal(geo_mask, False)
        dtw = np.asarray(dtw_matrix, dtype=np.float32)
        allowed = np.argsort(dtw, axis=1)[:, :max(1, semantic_neighbors)]
        sem_mask = np.ones((n, n), dtype=bool)
        sem_mask[np.arange(n)[:, None], allowed] = False
        np.fill_diagonal(sem_mask, False)
        return geo_mask, sem_mask

    def forward(self, x):
        if x.ndim == 4:
            x = x[..., 0]
        if x.shape[-1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} sensor features, received {x.shape[-1]}")
        b, t, n = x.shape
        if t != self.history_len or n != self.num_nodes:
            raise ValueError(f"Expected input (batch,{self.history_len},{self.num_nodes}), got {tuple(x.shape)}")
        padded = F.pad(x, (0, 0, 2, 0))
        x_patterns = torch.stack([padded[:, i:i + t, :] for i in range(3)], dim=-1)
        h = self.embedding(x)
        skip = 0
        for layer, skip_conv in zip(self.layers, self.skip_convs):
            h = layer(h, x_patterns, self.pattern_keys, self.geo_mask, self.sem_mask)
            skip = skip + skip_conv(h.permute(0, 3, 2, 1))
        # Project history steps to forecast steps, then channels to flow.
        y = self.time_projection(F.relu(skip.permute(0, 3, 2, 1)))
        y = self.output_projection(F.relu(y.permute(0, 3, 2, 1)))
        y = y.permute(0, 3, 2, 1).squeeze(-1).contiguous()
        return y + self.persistence_scale.view(1, self.forecast_len, 1) * x[:, -1:, :]


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
