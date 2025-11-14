# Collapse-Proof Integration Playbook

This checklist outlines every task required to bring the CPLearn “Collapse-Proof” objective into the Path-FM (DINOv2) codebase. Two contributors can divide the items between them; suggested ownership fields are left blank so you can fill in your initials.

> **Tip:** Tasks are grouped in dependency order. Finish each section before moving to the next to avoid rework.

---

## 1. Environment & Source Prep

- [ ] *(Owner: __)* Clone the upstream CPLearn repository locally for reference.
- [ ] *(Owner: __)* Create or update the Path-FM Python environment (`python3 -m venv .venv && source .venv/bin/activate`).
- [ ] *(Owner: __)* Install core dependencies: `pip install -r requirements.txt` plus `torch`, `omegaconf`, and `pytest`.
- [ ] *(Owner: __)* Confirm GPU availability (`nvidia-smi`) and NCCL/XFormers prerequisites (Path-FM already requires these).

## 2. Vendor Collapse-Proof Building Blocks

- [ ] *(Owner: __)* Port `cplearn_loss_func` from `CPLearn/solo/losses/cplearn.py` into `dinov2/loss/cplearn.py`.
- [ ] *(Owner: __)* Port the projector logic from `CPLearn/solo/methods/cplearn.py` into `dinov2/layers/collapse_proof.py`.
- [ ] *(Owner: __)* Expose the new modules through `dinov2/loss/__init__.py` and `dinov2/layers/__init__.py`.
- [ ] *(Owner: __)* Run unit smoke tests for both new modules (see §5).

## 3. Wire Training Logic

- [ ] *(Owner: __)* Update `dinov2/train/ssl_meta_arch.py` to:
  - Detect `cfg.collapse_proof` settings.
  - Instantiate Collapse-Proof heads for both student and teacher.
  - Compute Collapse-Proof loss from the two global crops and fold it into the training loss accumulator.
- [ ] *(Owner: __)* Ensure the projector participates in FSDP wrapping and EMA updates without breaking existing heads.
- [ ] *(Owner: __)* Verify `loss_dict` now reports `collapse_proof_loss` when enabled.

## 4. Configuration Plumbing

- [ ] *(Owner: __)* Extend `dinov2/configs/ssl_default_config.yaml` with default Collapse-Proof hyper-parameters and FSDP precision entries.
- [ ] *(Owner: __)* Update at least one train config (e.g. `dinov2/configs/train/vits14_reg4.yaml`) to enable Collapse-Proof and set CPLearn-inspired projector sizes.
- [ ] *(Owner: __)* Document recommended overrides (loss weight, beta, projector dimensions) for different backbones.

## 5. Testing & Validation

- [ ] *(Owner: __)* Add `test_collapse_proof.py` containing:
  - A numerical sanity check for `cplearn_loss_func`.
  - Gradient/shape checks for `CollapseProofProjector`.
- [ ] *(Owner: __)* Install pytest if missing (`python3 -m pip install pytest`).
- [ ] *(Owner: __)* Run `python3 -m pytest test_collapse_proof.py` and fix any failures.
- [ ] *(Owner: __)* Execute a short training dry-run (few iterations) to confirm no runtime errors with Collapse-Proof enabled.

## 6. Documentation & Knowledge Transfer

- [ ] *(Owner: __)* Update or create `docs/collapse_proof_integration.md` describing architecture changes and tuning guidance.
- [ ] *(Owner: __)* Capture environment setup and training commands in the main `README.md` or project wiki.
- [ ] *(Owner: __)* Record open questions, hyper-parameter sweeps, and evaluation plans in your shared project tracker.

## 7. Optional Extras (Nice-to-Haves)

- [ ] *(Owner: __)* Port CPLearn evaluation scripts (embedding extraction, collapse diagnostics) into `scripts/` for deeper analysis.
- [ ] *(Owner: __)* Add CI coverage for the new unit tests.
- [ ] *(Owner: __)* Benchmark training throughput with and without Collapse-Proof to assess overhead.
- [ ] *(Owner: __)* Experiment with alternative projector initialisations (e.g., Gaussian codes) and document findings.

---

### Tracking & Communication

- Use your team’s kanban/issue tracker to map each checkbox to a ticket.
- Post progress updates after completing each section—especially §3 (core integration) and §5 (tests), which gate downstream work.
- When both contributors commit, ensure branch protections/lint checks cover the new files and tests.

Happy hacking! Fill in owners as you go to keep the workload balanced. If new tasks emerge, append them to this README so the integration plan stays current.***
