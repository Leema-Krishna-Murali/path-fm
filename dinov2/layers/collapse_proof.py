"""
Collapse-Proof projector module derived from the reference implementation in
https://github.com/emsansone/CPLearn.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CollapseProofProjector(nn.Module):
    """
    Projector used by Collapse-Proof learning. The layer first applies a linear layer followed
    by batch-normalisation and Tanh, then projects onto a fixed random ±1 codebook and rescales
    the output as described in the original implementation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.tanh = nn.Tanh()

        random_codes = torch.randint(0, 2, (hidden_dim, output_dim), dtype=torch.float32)
        random_codes = 2 * random_codes - 1
        self.register_buffer("weights", random_codes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape [batch_size, in_dim].

        Returns:
            torch.Tensor: Projected features with shape [batch_size, output_dim].
        """
        z = self.projector(x)
        z = self.tanh(z)
        z = z @ self.weights

        batch_size, code_dim = z.shape
        scale = self.hidden_dim / (
            math.sqrt(batch_size) * math.log((1.0 - self.epsilon * (code_dim - 1.0)) / self.epsilon)
        )

        return z / scale
