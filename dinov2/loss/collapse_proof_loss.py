import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("dinov2")


class CollapseProofLoss(nn.Module):
    """
    Optional Collapse-Proof regularizer.

    If the `cplearn` package is available and `backend == "cplearn"` (or "auto"),
    the module will try to delegate to the official implementation. Otherwise it
    falls back to a simple, effective anti-collapse objective inspired by VICReg:

    - variance loss encourages per-dimension std >= var_target
    - covariance loss penalizes off-diagonal covariance
    - mean loss penalizes non-zero mean (optional)

    Expected input: a 2D tensor of shape (batch, dim), typically the student
    CLS representations before the DINO head.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",  # one of {auto, cplearn, builtin}
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        mean_weight: float = 0.0,
        var_target: float = 1.0,
        eps: float = 1e-4,
        cplearn_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.mean_weight = mean_weight
        self.var_target = var_target
        self.eps = eps
        self._cplearn_available = False
        self._cplearn_loss = None

        if backend in {"auto", "cplearn"}:
            try:
                import importlib

                cplearn = importlib.import_module("cplearn")
                # Try common entry points; adjust if project exposes a different API.
                # We intentionally keep this dynamic to avoid hard dependency.
                build = getattr(cplearn, "build_loss", None)
                if callable(build):
                    self._cplearn_loss = build(**(cplearn_kwargs or {}))
                else:
                    # Fallback: try a direct class if provided as kwarg
                    cls = (cplearn_kwargs or {}).get("class_name")
                    if cls is not None and hasattr(cplearn, cls):
                        self._cplearn_loss = getattr(cplearn, cls)(**((cplearn_kwargs or {}).get("args", {})))

                if self._cplearn_loss is not None:
                    self._cplearn_available = True
                    logger.info("Collapse-Proof: using cplearn backend")
                else:
                    logger.warning(
                        "Collapse-Proof: cplearn found but no known factory; falling back to builtin backend"
                    )
            except Exception as exc:  # noqa: BLE001
                if backend == "cplearn":
                    logger.warning(
                        f"Collapse-Proof: requested cplearn backend but import failed: {exc}; using builtin backend"
                    )
                else:
                    logger.info("Collapse-Proof: cplearn not available; using builtin backend")

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        if representations.ndim != 2:
            representations = representations.flatten(1)

        if self._cplearn_available and self._cplearn_loss is not None:
            # Delegate to cplearn implementation
            try:
                return self._cplearn_loss(representations)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Collapse-Proof: cplearn loss failed at runtime ({exc}); using builtin fallback for this step"
                )

        return self._builtin_loss(representations)

    def _builtin_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Center
        mean = x.mean(dim=0)
        x_centered = x - mean

        # Variance per dimension
        std = x_centered.var(dim=0, unbiased=False).add(self.eps).sqrt()
        var_term = torch.relu(self.var_target - std).mean()

        # Covariance (off-diagonal)
        n = x_centered.shape[0]
        cov = (x_centered.T @ x_centered) / max(n - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_term = (off_diag.pow(2).sum() / x_centered.shape[1])

        # Mean penalty (optional)
        mean_term = mean.pow(2).mean()

        total = (
            self.var_weight * var_term
            + self.cov_weight * cov_term
            + self.mean_weight * mean_term
        )
        return total
