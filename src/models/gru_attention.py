import torch
import torch.nn as nn


class GRUAttentionDetector(nn.Module):
    """GRU -> MultiheadAttention (self-attention) -> pooling -> Linear classifier.

    Drop-in replacement for ``GRUDetector``: ``forward(x, h=None)`` returns the
    tuple ``(logits, h)`` where ``x`` is ``(B, T, feature_dim)``, ``logits`` is
    ``(B, num_classes)``, and ``h`` is the GRU hidden state.

    Pipeline
    --------
    1. GRU over the frame sequence  -> ``(B, T, H)`` (H = hidden_dim * dirs)
    2. ``nn.MultiheadAttention`` self-attention over the GRU outputs
       (query = key = value = GRU sequence, ``batch_first=True``)
    3. Pool the attended sequence to ``(B, H)``.
       Pooling strategy: **mean-pool** over the time dimension. Mean-pooling is
       used (instead of taking the last step like GRUDetector) because the
       self-attention already lets every position aggregate global context, so
       averaging gives every frame an equal vote into the final representation
       and is robust to varying sequence length T.
    4. ``nn.Linear`` -> ``num_classes`` logits.
    """

    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        num_classes=48,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        num_heads=4,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_heads = num_heads

        self.gru = nn.GRU(
            feature_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Dimension of the GRU output (and the attention embedding dim).
        attn_dim = hidden_dim * (2 if bidirectional else 1)

        # MultiheadAttention requires embed_dim to be divisible by num_heads.
        # Fall back to a divisor of attn_dim if the requested head count does
        # not divide evenly, so the model stays robust to arbitrary dims.
        if attn_dim % num_heads != 0:
            num_heads = self._largest_divisor_leq(attn_dim, num_heads)
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cls = nn.Linear(attn_dim, num_classes)

    @staticmethod
    def _largest_divisor_leq(value: int, upper: int) -> int:
        for d in range(min(upper, value), 0, -1):
            if value % d == 0:
                return d
        return 1

    def forward(self, x, h=None):
        # 1. GRU over the sequence.
        out, h = self.gru(x, h)                       # out: (B, T, attn_dim)

        # 2. Self-attention over the GRU output sequence.
        attended, _ = self.attn(out, out, out)        # (B, T, attn_dim)

        # 3. Mean-pool over time -> (B, attn_dim).
        pooled = attended.mean(dim=1)

        # 4. Classify.
        logits = self.cls(pooled)                     # (B, num_classes)

        return logits, h
