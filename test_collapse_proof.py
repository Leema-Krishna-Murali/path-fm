import torch

from dinov2.layers import CollapseProofProjector
from dinov2.loss import cplearn_loss_func


def test_cplearn_loss_returns_scalar_and_is_finite():
    torch.manual_seed(42)
    z1 = torch.randn(16, 32)
    z2 = torch.randn(16, 32)

    loss = cplearn_loss_func(z1, z2, beta=0.5)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_collapse_proof_projector_shape_and_gradients():
    torch.manual_seed(0)
    batch_size, in_dim, hidden_dim, out_dim = 12, 64, 128, 256
    projector = CollapseProofProjector(in_dim, hidden_dim, out_dim)

    inputs = torch.randn(batch_size, in_dim, requires_grad=True)
    outputs = projector(inputs)

    assert outputs.shape == (batch_size, out_dim)
    unique_codes = projector.weights.unique()
    assert torch.allclose(unique_codes.sort().values, torch.tensor([-1.0, 1.0]))

    outputs.sum().backward()
    assert inputs.grad is not None
    assert projector.projector[0].weight.grad is not None
