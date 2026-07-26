"""
Compact state-space (SSM) temporal model — a drop-in alternative to GRUDetector.

This is a real-valued **diagonal SSM** (S4D / LRU style), implemented in pure
PyTorch so it runs on CPU/MPS without CUDA selective-scan kernels (i.e. not the
full Mamba kernel). Each block applies a stable diagonal linear recurrence

    h_t = a ⊙ h_{t-1} + (1 - a) ⊙ (W x_t),   a = sigmoid(·) ∈ (0, 1)
    y_t = out(h_t) + D ⊙ x_t

followed by a GELU MLP, with residual connections and LayerNorm — the standard
SSM-block shape. The sequence is short (T=16) so a sequential scan is fine.

Interface matches GRUDetector exactly: ``forward(x, h=None) -> (logits, h)`` so
it is a drop-in replacement in trainer/eval/A-GEM.
"""
import torch
import torch.nn as nn


class DiagSSM(nn.Module):
    """Stable diagonal linear state-space layer (leaky-integrator recurrence)."""

    def __init__(self, d_model: int):
        super().__init__()
        # a = sigmoid(log_a); init so channels span a range of memory horizons
        self.log_a = nn.Parameter(torch.linspace(0.5, 3.0, d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)   # skip / feedthrough

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (B, T, d)
        a = torch.sigmoid(self.log_a)                      # (d,) in (0,1)
        u = self.in_proj(x)                                # (B, T, d)
        B, T, d = u.shape
        h = x.new_zeros(B, d)
        ys = []
        for t in range(T):
            h = a * h + (1.0 - a) * u[:, t]                # diagonal recurrence
            ys.append(h)
        y = torch.stack(ys, dim=1)                         # (B, T, d)
        return self.out_proj(y) + self.D * x


class SSMBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SSMDetector(nn.Module):
    """Diagonal-SSM temporal classifier, drop-in for GRUDetector."""

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 48,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,   # accepted for API parity (ignored)
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = max(1, num_layers)
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.inp = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [SSMBlock(hidden_dim, dropout) for _ in range(self.num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, h=None):
        z = self.inp(x)                       # (B, T, hidden)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        logits = self.cls(z[:, -1])           # causal last-step pooling (like GRU)
        return logits, h
