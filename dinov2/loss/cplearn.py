"""
Copyright 2025 CPLearn team.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import torch


def cplearn_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """
    Computes CPLearn loss given batches of projected features z1 and z2, corresponding to
    different augmented views of the same images. This implementation mirrors the reference
    implementation from https://github.com/emsansone/CPLearn.

    Args:
        z1: Tensor containing projected features from view 1 with shape [batch_size, dim].
        z2: Tensor containing projected features from view 2 with shape [batch_size, dim].
        beta: Scaling factor for cross-view coupling terms.

    Returns:
        torch.Tensor: Scalar tensor with the CPLearn loss value.
    """

    # shift logits for numerical stability
    z1 = z1 - torch.max(z1, dim=1, keepdim=True)[0]
    z2 = z2 - torch.max(z2, dim=1, keepdim=True)[0]

    _, c = z1.shape

    # entropy regularization for z2
    loss1 = (z2.softmax(1) * torch.log(c * z2.softmax(1).mean(0, keepdim=True))).sum(1).mean()
    # cross-view coupling terms
    loss2 = -(
        z2.softmax(1) * (beta * z1) - z2.softmax(1) * (beta * z1.logsumexp(1, keepdim=True))
    ).sum(1).mean()

    # symmetric terms swapping the two views
    loss3 = (z1.softmax(1) * torch.log(c * z1.softmax(1).mean(0, keepdim=True))).sum(1).mean()
    loss4 = -(
        z1.softmax(1) * (beta * z2) - z1.softmax(1) * (beta * z2.logsumexp(1, keepdim=True))
    ).sum(1).mean()

    loss = 0.5 * (loss1 + loss2) + 0.5 * (loss3 + loss4)
    return loss
