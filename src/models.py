"""ORIEON model definitions — Section 4 of the approved protocol.

Groups:
  A. Tree baselines are constructed in train.py (RandomForest, LightGBM).
  B. Deep baselines: MLP, LSTM, GRU        (input: 20 x 15 sequence)
  C. GNN-only candidates: GCN, GraphSAGE, GAT, Static-Learned, Dynamic
  D. Hybrids: {Dynamic, Static-Learned}-GNN + {GRU, LSTM}

All neural models share: WIDTH=48 (graph node width), prediction head
Dropout(0.2) -> Linear(in, 32) -> ReLU -> Linear(32, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_FEATURES = 15
LOOKBACK = 20
WIDTH = 48
HIDDEN = 64
GOLD_NODE_IDX = 0  # Gold_Close is feature index 0


def make_head(in_dim: int) -> nn.Sequential:
    """Shared prediction head for every neural model."""
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


# ---------------------------------------------------------------------------
# Group B — deep baselines
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Flatten the 20x15 window -> Linear(300,128) -> ReLU -> Dropout(0.2)
    -> Linear(128,64) -> ReLU -> head."""

    def __init__(self, n_features: int = N_FEATURES, lookback: int = LOOKBACK):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback * n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = make_head(64)

    def forward(self, x):  # x: (B, T, F)
        b = x.shape[0]
        z = self.net(x.reshape(b, -1))
        return self.head(z).squeeze(-1)


class LSTMModel(nn.Module):
    """nn.LSTM(15, hidden 64, batch_first=True), last time step -> head."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = make_head(hidden)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class GRUModel(nn.Module):
    """nn.GRU(15, hidden 64, batch_first=True), last time step -> head."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = HIDDEN):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.head = make_head(hidden)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


# ---------------------------------------------------------------------------
# Graph encoder shared by Groups C and D
# ---------------------------------------------------------------------------
class DailyGraphEncoder(nn.Module):
    """Per day, treat the 15 features as 15 nodes.

    h = tanh(value_proj(x) + node_embedding)              (WIDTH = 48)
    message passing -> z  (variant-specific)
    daily readout = concat(mean-pool(z), z[Gold_Close node])
                  -> Linear(96, 48) -> Tanh
    """

    VARIANTS = ("gcn", "sage", "gat", "static", "dynamic")

    def __init__(self, variant: str, n_features: int = N_FEATURES,
                 width: int = WIDTH, gcn_adjacency: torch.Tensor | None = None):
        super().__init__()
        assert variant in self.VARIANTS, f"unknown variant {variant}"
        self.variant = variant
        self.n = n_features
        self.width = width

        self.value_proj = nn.Linear(1, width)
        self.node_embedding = nn.Parameter(torch.randn(n_features, width) * 0.1)

        self.graph_proj = nn.Linear(width, width)
        if variant in ("sage", "static", "dynamic"):
            self.self_proj = nn.Linear(width, width)
        if variant == "gat":
            # Additive attention: two linear maps to scalars, summed pairwise.
            self.att_left = nn.Linear(width, 1)
            self.att_right = nn.Linear(width, 1)
        if variant == "static":
            # Learnable n x n adjacency logits; identity added at forward time.
            self.adj_logits = nn.Parameter(torch.zeros(n_features, n_features))
        if variant == "dynamic":
            self.q_proj = nn.Linear(width, width)
            self.k_proj = nn.Linear(width, width)

        # GCN: fixed adjacency from training-row Pearson correlation,
        # diagonal 1, symmetric degree normalization D^-1/2 A D^-1/2.
        if variant == "gcn":
            assert gcn_adjacency is not None, "GCN requires gcn_adjacency"
            self.register_buffer("gcn_adjacency", gcn_adjacency)

        self.readout = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.Tanh(),
        )

    # -- message passing ----------------------------------------------------
    def message_pass(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, n, width) -> z: (B, n, width)"""
        n = self.n
        if self.variant == "gcn":
            z = torch.relu(self.graph_proj(torch.einsum("ij,bjk->bik", self.gcn_adjacency, h)))
        elif self.variant == "sage":
            eye = torch.eye(n, device=h.device, dtype=h.dtype)
            neigh = (torch.ones(n, n, device=h.device, dtype=h.dtype) - eye) / (n - 1)
            neighbour_mean = torch.einsum("ij,bjk->bik", neigh, h)
            z = torch.relu(self.self_proj(h) + self.graph_proj(neighbour_mean))
        elif self.variant == "gat":
            s = self.att_left(h) + self.att_right(h).transpose(1, 2)  # (B, n, n)
            a = torch.softmax(s, dim=-1)
            z = torch.relu(self.graph_proj(torch.bmm(a, h)))
        elif self.variant == "static":
            eye = torch.eye(n, device=h.device, dtype=h.dtype)
            a = torch.softmax(self.adj_logits + eye, dim=-1)
            z = torch.relu(self.graph_proj(torch.einsum("ij,bjk->bik", a, h))
                           + self.self_proj(h))
        else:  # dynamic
            q, k = self.q_proj(h), self.k_proj(h)
            eye = torch.eye(n, device=h.device, dtype=h.dtype)
            a = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.width ** 0.5) + eye,
                              dim=-1)
            z = torch.relu(self.graph_proj(torch.bmm(a, h)) + self.self_proj(h))
        return z

    # -- daily readout ------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n) -> daily readout sequence (B, T, width)."""
        b, t, n = x.shape
        h = torch.tanh(self.value_proj(x.reshape(b * t, n, 1)) + self.node_embedding)
        z = self.message_pass(h) + h                            # residual link (B*T, n, width)
        pooled = torch.cat([z.mean(dim=1), z[:, GOLD_NODE_IDX]], dim=-1)  # (B*T, 2w)
        r = self.readout(pooled)                                # (B*T, width)
        return r.reshape(b, t, self.width)


def compute_gcn_adjacency(train_features) -> torch.Tensor:
    """Fixed adjacency = |Pearson correlation| of the 15 features computed on
    TRAINING rows only, diagonal set to 1, symmetric degree normalization."""
    import numpy as np

    corr = np.corrcoef(np.asarray(train_features), rowvar=False)
    a = np.abs(corr)
    np.fill_diagonal(a, 1.0)
    d = a.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    a_norm = a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return torch.tensor(a_norm, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Group C — GNN-only candidates
# ---------------------------------------------------------------------------
class GraphOnlyModel(nn.Module):
    """Daily graph encoding -> mean over the 20 days -> head."""

    def __init__(self, variant: str, **kw):
        super().__init__()
        self.encoder = DailyGraphEncoder(variant, **kw)
        self.head = make_head(2 * WIDTH)

    def forward(self, x):  # x: (B, T, F)
        r = self.encoder.encode(x)          # (B, T, width)
        z = torch.cat([r.mean(dim=1), r[:, -1]], dim=-1)  # mean + last day
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Group D — hybrids (graph encoding + recurrent temporal layer)
# ---------------------------------------------------------------------------
class GraphTemporalModel(nn.Module):
    """Sequence of 20 daily 48-dim readouts -> recurrent layer (hidden 64,
    batch_first) -> last hidden state -> head."""

    def __init__(self, variant: str, rnn_type: str, **kw):
        super().__init__()
        assert variant in ("static", "dynamic")
        assert rnn_type in ("gru", "lstm")
        self.encoder = DailyGraphEncoder(variant, **kw)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(WIDTH, HIDDEN, batch_first=True)
        self.head = make_head(HIDDEN)

    def forward(self, x):  # x: (B, T, F)
        r = self.encoder.encode(x)          # (B, T, width)
        out, _ = self.rnn(r)
        return self.head(out[:, -1]).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
