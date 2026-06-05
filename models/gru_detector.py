import torch
import torch.nn as nn

class GRUDetector(nn.Module):
    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        num_classes=87
    ):
        super().__init__()

        self.gru = nn.GRU(
            feature_dim,
            hidden_dim,
            batch_first=True
        )

        self.cls = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x, h=None):

        out, h = self.gru(x, h)

        logits = self.cls(out[:, -1])

        return logits, h