from __future__ import annotations

import torch


def collapse_proof_loss(z1: torch.Tensor, z2: torch.Tensor, *, beta: float = 0.05) -> torch.Tensor:
    """
    Collapse-proof loss from Sansone et al. (2025).

    This is a direct port of the public implementation released with CPLearn.

    Args:
        z1: Projected features for view 1 with shape ``(batch_size, code_dim)``.
        z2: Projected features for view 2 with shape ``(batch_size, code_dim)``.
        beta: Temperature scaling coefficient of the cross-view term.
    """
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape.")

    z1 = z1 - torch.max(z1, dim=1, keepdim=True)[0]
    z2 = z2 - torch.max(z2, dim=1, keepdim=True)[0]

    code_dim = z1.shape[1]

    p_z1 = z1.softmax(dim=1)
    p_z2 = z2.softmax(dim=1)

    marginal_z1 = code_dim * p_z1.mean(dim=0, keepdim=True)
    marginal_z2 = code_dim * p_z2.mean(dim=0, keepdim=True)

    loss1 = (p_z2 * marginal_z2.log()).sum(dim=1).mean()
    loss2 = -(p_z2 * (beta * z1 - beta * z1.logsumexp(dim=1, keepdim=True))).sum(dim=1).mean()
    loss3 = (p_z1 * marginal_z1.log()).sum(dim=1).mean()
    loss4 = -(p_z1 * (beta * z2 - beta * z2.logsumexp(dim=1, keepdim=True))).sum(dim=1).mean()

    return 0.5 * (loss1 + loss2) + 0.5 * (loss3 + loss4)
