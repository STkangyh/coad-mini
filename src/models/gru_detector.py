import torch
import torch.nn as nn

class GRUDetector(nn.Module):
    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        num_classes=87,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            feature_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        cls_dim = hidden_dim * (2 if bidirectional else 1)
        self.cls = nn.Linear(
            cls_dim,
            num_classes
        )

    def forward(self, x, h=None):

        out, h = self.gru(x, h)

        logits = self.cls(out[:, -1])

        return logits, h
