# Collapse-Proof Integration Guide

This walkthrough explains how Collapse-Proof learning from [CPLearn](https://github.com/emsansone/CPLearn) was grafted into the Path-FM codebase (DINOv2 fork) and what you need to do end-to-end to reproduce or adjust the integration.

## 1. Vendor the CPLearn building blocks

- **Loss**: `dinov2/loss/cplearn.py` ports `cplearn_loss_func` verbatim.  
  *Test:* `test_collapse_proof.py::test_cplearn_loss_returns_scalar_and_is_finite` validates numerical stability.

- **Projector**: `dinov2/layers/collapse_proof.py` implements the CPLearn projector (linear → BN → tanh → fixed ±1 codes).  
  *Test:* `test_collapse_proof.py::test_collapse_proof_projector_shape_and_gradients` checks shape, codebook values, and gradients.

Exports were wired through the existing package initialisers (`dinov2/loss/__init__.py`, `dinov2/layers/__init__.py`).

## 2. Extend the SSL meta-architecture

`dinov2/train/ssl_meta_arch.py` now:
- Detects `cfg.collapse_proof.loss_weight > 0`.
- Instantiates `CollapseProofProjector` for student/teacher module dicts so FSDP + EMA keep working.
- Computes the Collapse-Proof loss from the two global crops (before the DINO head) and adds it to the global loss accumulator.

*Self-check:* Run a tiny forward pass in a debugger and ensure `loss_dict["collapse_proof_loss"]` appears when the feature is enabled.

## 3. Teach the config system about Collapse-Proof

`dinov2/configs/ssl_default_config.yaml` now contains defaults:
- `collapse_proof` group with hyper-parameters (`loss_weight`, `beta`, projector sizes, `epsilon`).
- FSDP precision options for the new head under both `compute_precision.teacher` and `compute_precision.student`.

Override values per experiment in training configs, e.g. `dinov2/configs/train/vits14_reg4.yaml`.

## 4. Update experiment configs

`dinov2/configs/train/vits14_reg4.yaml` illustrates how to enable Collapse-Proof:

```yaml
collapse_proof:
  loss_weight: 1.0
  beta: 1.0
  proj_hidden_dim: 2048
  proj_output_dim: 16384
```

Tune these to match backbone capacity or replicate the CPLearn paper.

## 5. Verify with tests

New unit tests live in `test_collapse_proof.py`. Run them after installing pytest:

```bash
python3 -m pip install pytest      # once per environment
python3 -m pytest test_collapse_proof.py
```

## 6. Train

Launch training as before; Collapse-Proof activates automatically when `collapse_proof.loss_weight > 0`:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=34001 --nproc_per_node=1 \
  dinov2/train/train.py \
  --config-file ./dinov2/configs/train/vits14_reg4.yaml \
  --output-dir ./output_cplearn
```

`wandb` logs now include `collapse_proof_loss`.

## 7. Next steps

- Sweep `beta`, projector dimensions, and `loss_weight` to balance DINOv2 and Collapse-Proof objectives.
- If you disable the DINO or iBOT heads, you can reduce compute by turning off their config flags while keeping Collapse-Proof active.
- Port additional CPLearn evaluation scripts (embedding extraction, collapse metrics) if needed; this guide covers training integration only.
