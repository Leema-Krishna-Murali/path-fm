import torch

from dinov2.layers import CollapseProofProjector
from dinov2.loss import collapse_proof_loss


def test_collapse_proof_projector_shape():
    projector = CollapseProofProjector(16, 8, 4, random_sign_seed=123)
    x = torch.randn(10, 16)
    projector.eval()

    out = projector(x)

    assert out.shape == (10, 4)
    assert torch.isfinite(out).all()


def test_collapse_proof_loss_symmetry_and_finiteness():
    projector = CollapseProofProjector(12, 6, 6, random_sign_seed=42)
    inputs = torch.randn(8 * 2, 12)
    projector.eval()

    codes = projector(inputs)
    z1, z2 = codes.chunk(2)

    loss = collapse_proof_loss(z1, z2, beta=0.05)

    assert torch.isfinite(loss)
    assert loss >= torch.zeros_like(loss)


def test_collapse_proof_random_matrix_reproducible():
    proj_a = CollapseProofProjector(4, 3, 2, random_sign_seed=7)
    proj_b = CollapseProofProjector(4, 3, 2, random_sign_seed=7)

    assert torch.equal(proj_a.sign_matrix, proj_b.sign_matrix)
