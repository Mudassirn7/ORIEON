"""ORIEON model definitions — GRN-MLP Ensemble project.

Models:
  Benchmarks : RandomForest & LightGBM (built in train.py), MLP, LSTM
  Proposed   : GRNMLPEnsemble — Gated Residual Network branch + MLP branch,
               fused by a per-sample learned gate (dynamic gated blending).

All neural models take a (B, LOOKBACK, n_features) window and predict the
scaled next-day log-return (a scalar).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOOKBACK = 20
HIDDEN = 64


def make_head(in_dim: int) -> nn.Sequential:
    """Shared prediction head for the benchmark neural models."""
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


# ---------------------------------------------------------------------------
# Benchmarks — deep
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Flatten the 20xF window -> Linear(-,128) -> ReLU -> Dropout(0.2)
    -> Linear(128,64) -> ReLU -> head."""

    def __init__(self, n_features: int, lookback: int = LOOKBACK):
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
    """nn.LSTM(F, hidden 64, batch_first=True), last time step -> head."""

    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = make_head(hidden)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


# ---------------------------------------------------------------------------
# Proposed — GRN-MLP Ensemble with dynamic gated blending
# ---------------------------------------------------------------------------
class GLU(nn.Module):
    """Gated Linear Unit: split into (a, b); output a * sigmoid(b)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRNBlock(nn.Module):
    """Gated Residual Network block (TFT-style, no static context):

        h = ELU(W1 x);  h = Dropout(ELU(W2 h))
        out = LayerNorm(x + GLU(h))

    The GLU gate decides per sample how much of the transformed signal to
    let through — suppressing noisy features in calm regimes and amplifying
    them during news shocks.
    """

    def __init__(self, dim: int, hidden: int = HIDDEN, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.glu = GLU(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.drop(F.elu(self.fc2(h)))
        return self.norm(x + self.glu(h))


class GRNModel(nn.Module):
    """Standalone Gated Residual Network (benchmark): flattened window ->
    input projection -> 2 stacked GRN blocks -> shared head.

    Yeh GRN-MLP ensemble ka dynamic branch akela hai — comparison ke liye,
    taake gate-filtering ka individual contribution napa ja sake."""

    def __init__(self, n_features: int, lookback: int = LOOKBACK,
                 hidden: int = HIDDEN):
        super().__init__()
        self.in_proj = nn.Linear(lookback * n_features, hidden)
        self.grn1 = GRNBlock(hidden, hidden)
        self.grn2 = GRNBlock(hidden, hidden)
        self.head = make_head(hidden)

    def forward(self, x):  # x: (B, T, F)
        b = x.shape[0]
        h = F.elu(self.in_proj(x.reshape(b, -1)))
        h = self.grn2(self.grn1(h))
        return self.head(h).squeeze(-1)


class GRNMLPEnsemble(nn.Module):
    """Proposed architecture.

    Branch 1 (GRN): input projection -> 2 stacked GRN blocks -> Linear(-,1).
        Dynamic component — gates filter features conditioned on the market
        state encoded in the window.
    Branch 2 (MLP): Linear(-,128) -> ReLU -> Dropout -> Linear(128,64) -> ReLU
        -> Linear(64,1).  Static non-linear feature interactions.
    Fusion: w = sigmoid(Linear(flat window)) per sample;
        prediction = w * grn + (1 - w) * mlp   (dynamic gated blending).
    """

    def __init__(self, n_features: int, lookback: int = LOOKBACK,
                 hidden: int = HIDDEN):
        super().__init__()
        d_in = lookback * n_features
        # GRN branch
        self.in_proj = nn.Linear(d_in, hidden)
        self.grn1 = GRNBlock(hidden, hidden)
        self.grn2 = GRNBlock(hidden, hidden)
        self.grn_out = nn.Linear(hidden, 1)
        # MLP branch
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.mlp_out = nn.Linear(64, 1)
        # Gated blending
        self.gate = nn.Sequential(nn.Linear(d_in, 1), nn.Sigmoid())

    def forward(self, x):  # x: (B, T, F)
        b = x.shape[0]
        flat = x.reshape(b, -1)
        g = self.grn_out(self.grn2(self.grn1(F.elu(self.in_proj(flat)))))
        m = self.mlp_out(self.mlp(flat))
        w = self.gate(flat)
        return (w * g + (1.0 - w) * m).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
