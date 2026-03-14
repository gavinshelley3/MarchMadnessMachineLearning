from __future__ import annotations

import torch
from torch import nn


class TabularNN(nn.Module):
    """Simple feedforward network for matchup classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        dims = (input_dim, *hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.model(x)
        return logits.squeeze(-1)


def build_model(input_dim: int) -> TabularNN:
    return TabularNN(input_dim=input_dim)


class AdvancementNN(nn.Module):
    """Multi-output network for predicting advancement milestones."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        dims = (input_dim, *hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def build_advancement_model(input_dim: int, output_dim: int) -> AdvancementNN:
    return AdvancementNN(input_dim=input_dim, output_dim=output_dim)
