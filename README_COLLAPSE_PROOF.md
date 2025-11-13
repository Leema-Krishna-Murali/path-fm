# Collapse-Proof Integration Guide

This document explains how the collapse-proof objective from [CPLearn](https://github.com/emsansone/CPLearn) is integrated into the DINOv2-based Path-FM codebase, and the tasks you need to follow to reproduce or extend the integration end-to-end.

## 1. Prerequisites

- Install the repository dependencies (base or development extras) so that PyTorch, xFormers, and other runtime components are available.
- (Optional) Install tooling for testing and linting:
  ```bash
  pip install -r requirements-dev.txt
  ```
- Ensure you have access to GPUs if you plan to run full training; the included short test config can run on CPU for smoke tests.

## 2. Implementation Overview

The integration consists of four building blocks:

1. **Collapse-Proof Projector** – `dinov2/layers/collapse_projector.py`
   - Implements the non-trainable random ±1 projection and the tanh-based projector described in the paper.
   - Provides both training (`forward`) and embedding-only (`embeddings`) pathways.

2. **Collapse-Proof Loss** – `dinov2/loss/collapse_proof.py`
   - Adds the reference loss function ported from the CPLearn codebase.

3. **Training Architecture Wiring** – `dinov2/train/ssl_meta_arch.py`
   - Instantiates the projector when `collapse_proof.enabled: true`.
   - Computes the loss on the student’s global CLS tokens and adds it to the total self-supervised objective.
   - Ensures the projector participates in FSDP wrapping, EMA updates, and stream synchronization.

4. **Configuration Surface** – `dinov2/configs/ssl_default_config.yaml` and `dinov2/configs/train/short_test_collapse.yaml`
   - Introduces a dedicated `collapse_proof` block with projector dimensions, β, ε, and seeding.
   - Adds per-module mixed-precision settings for FSDP.
   - Provides a ready-to-use short training recipe that enables the new head.

## 3. Step-by-Step Tasks

1. **Add the Collapse-Proof modules**
   - Create the projector and loss files as described above.
   - Export them via `dinov2/layers/__init__.py` and `dinov2/loss/__init__.py`.

2. **Update the SSL meta architecture**
   - Detect the `collapse_proof` flag in the configuration.
   - Attach the projector to both student and teacher `ModuleDict` objects.
   - Backpropagate the weighted collapse-proof loss alongside DINO/iBOT losses.

3. **Extend configuration defaults**
   - Update the default YAML to include options for the new head and its FSDP policies.
   - Add or adjust training configs (e.g., `short_test_collapse.yaml`) to toggle the feature.

4. **Create validation tests**
   - Add unit tests (see `test_collapse_proof.py`) covering the projector shape, deterministic sign matrix, and loss finiteness.
   - Optional: craft integration tests that run a single forward/backward pass with the short config.

5. **Document and verify**
   - Run the provided tests to confirm the implementation:
     ```bash
     python3 -m pytest -k collapse_proof
     ```
     (Install `pytest` if it is not already available.)
   - Launch a smoke-training run to validate wiring:
     ```bash
     python3 -m dinov2.train.train \
       --config-file dinov2/configs/train/short_test_collapse.yaml \
       --output-dir /tmp/dinov2-collapse-test
     ```

## 4. Operational Notes

- The projector’s random sign matrix is seeded for reproducibility and stored as a buffer so that checkpoints keep the same codes.
- The current wiring still requires DINO/iBOT heads; pure collapse-proof runs would need additional refactoring to bypass the multi-head attention block.
- Adjust projector dimensions (`proj_hidden_dim`, `proj_output_dim`) to balance compute with representation quality—larger values follow the original paper more closely but increase memory usage.

## 5. Next Steps

- Cross-check monitored metrics (e.g., `collapse_proof_loss`) in Weights & Biases dashboards during training.
- Port evaluation scripts (linear probe, collapse diagnostics) from CPLearn if you need parity with the paper’s experiments.
- Experiment with alternative scheduling for the collapse-proof loss weight or β to match your dataset or architecture.

By following these steps, you can integrate, configure, and validate collapse-proof training within the Path-FM DINOv2 stack.
