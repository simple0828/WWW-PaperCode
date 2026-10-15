from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionMLP(nn.Module):
    """Match exp MLP: project token hidden states to text_embedding_dim."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x
