from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class CollapseProofProjector(nn.Module):
    """
    Thin wrapper around the projector proposed in:
    "Collapse-Proof Non-Contrastive Self-Supervised Learning" (Sansone et al., 2025).

    The projector first maps backbone features through a linear layer and batch normalization,
    followed by a ``tanh`` activation. The activated features are then multiplied by a fixed
    random ±1 matrix whose entries are sampled once at construction time. Finally, the output
    is scaled with the normalization term described in the paper (Eq. 10).

    Args:
        input_dim: Dimensionality of the incoming backbone representation.
        hidden_dim: Hidden dimensionality of the projector (``m`` in the paper).
        output_dim: Dimensionality of the projected codes (``k`` in the paper).
        epsilon: Small constant used in the normalization factor.
        random_sign_seed: Optional deterministic seed used to sample the ±1 random matrix.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        epsilon: float = 1e-8,
        random_sign_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("hidden_dim and output_dim must be positive integers.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.activation = nn.Tanh()

        generator = torch.Generator(device="cpu")
        if random_sign_seed is not None:
            generator.manual_seed(random_sign_seed)

        sign_matrix = torch.randint(
            low=0,
            high=2,
            size=(hidden_dim, output_dim),
            generator=generator,
            dtype=torch.float32,
        )
        sign_matrix.mul_(2.0).sub_(1.0)  # map {0, 1} -> {-1, 1}
        self.register_buffer("sign_matrix", sign_matrix, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute the collapse-proof codes for a batch of features.

        Args:
            x: Backbone features with shape ``(batch_size, input_dim)``.

        Returns:
            Projected and normalized codes with shape ``(batch_size, output_dim)``.
        """
        if x.ndim != 2:
            raise ValueError("CollapseProofProjector expects 2D inputs of shape (batch, dim).")

        y = self.projector(x)
        y = self.activation(y)
        y = y @ self.sign_matrix.to(y.dtype)
        norm = self._normalization_factor(batch_size=y.shape[0], code_dim=y.shape[1], device=y.device, dtype=y.dtype)
        return y / norm

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the intermediate embeddings (pre random sign projection) used for evaluation.
        """
        if x.ndim != 2:
            raise ValueError("CollapseProofProjector expects 2D inputs of shape (batch, dim).")
        y = self.projector(x)
        return self.activation(y)

    def _normalization_factor(
        self,
        *,
        batch_size: int,
        code_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute the normalization factor that rescales the output codes.
        """
        numerator = torch.tensor(float(self.hidden_dim), device=device, dtype=dtype)
        denominator = math.sqrt(batch_size) * math.log((1.0 - self.epsilon * (code_dim - 1.0)) / self.epsilon)
        return numerator / denominator
