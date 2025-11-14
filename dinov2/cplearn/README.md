# Collapse-Proof Integration

CPLearn “Collapse-Proof” objective into the Path-FM (DINOv2) codebase.


## 1. Collapse-Proof Building Blocks

- [ ] Port `cplearn_loss_func` from `CPLearn/solo/losses/cplearn.py` into `dinov2/loss/cplearn.py`.
- [ ] Port the projector logic from `CPLearn/solo/methods/cplearn.py` into `dinov2/layers/collapse_proof.py`.
- [ ] Expose the new modules through `dinov2/loss/__init__.py` and `dinov2/layers/__init__.py`.
- [ ] Run unit smoke tests for both new modules.

## 2. Implement Training Logic

- [ ] Update `dinov2/train/ssl_meta_arch.py` to:
  - Detect `cfg.collapse_proof` settings.
  - Instantiate Collapse-Proof heads for both student and teacher.
  - Compute Collapse-Proof loss from the two global crops and fold it into the training loss accumulator.
- [ ] Ensure the projector participates in FSDP wrapping and EMA updates without breaking existing heads.
- [ ] Verify `loss_dict` now reports `collapse_proof_loss` when enabled.

## 3. Configuration

- [ ] Extend `dinov2/configs/ssl_default_config.yaml` with default Collapse-Proof hyper-parameters and FSDP precision entries.
- [ ] Update at least one train config (e.g. `dinov2/configs/train/vits14_reg4.yaml`) to enable Collapse-Proof and set CPLearn-inspired projector sizes.
- [ ] Document recommended overrides (loss weight, beta, projector dimensions) for different backbones.

## 4. Testing & Validation

- [ ] Add `test_collapse_proof.py` containing:
  - A numerical sanity check for `cplearn_loss_func`.
  - Gradient/shape checks for `CollapseProofProjector`.
- [ ] Install pytest if missing (`python3 -m pip install pytest`).
- [ ] Run `python3 -m pytest test_collapse_proof.py` and fix any failures.
- [ ] Execute a short training dry-run (few iterations) to confirm no runtime errors with Collapse-Proof enabled.

## 5. Optional

- [ ] Port CPLearn evaluation scripts (embedding extraction, collapse diagnostics) into `scripts/` for deeper analysis.
- [ ] Benchmark training throughput with and without Collapse-Proof to assess overhead.

