# HARDWARE

Single NVIDIA RTX 4060 with 16 GB VRAM. This document records every place we trade scale for tractability and why the trade does not damage the core claims.

## VRAM budget at peak training

| Item | bf16 size |
|---|---|
| PaiNN $E^{\mathrm{QM}}$ activations, batch 32, max 50 atoms | 5.0 GB |
| GINE $V^{\mathrm{couple}}$ activations | 1.5 GB |
| Optimizer states (AdamW, bf16 params + fp32 master) | 1.0 GB |
| PCD replay buffer on GPU | 0.3 GB |
| Misc cache, kernel scratch, RDKit cuda interop | 1.0 GB |
| Headroom | 2.0 GB |
| **Total** | **~10.8 GB** |

Default config in `configs/default.yaml` is sized to this budget. There is a `configs/tiny.yaml` for sanity runs (batch 8, hidden 64).

## Adaptations away from the unrestricted plan

| Component | Unrestricted plan | 4060 plan | Justification |
|---|---|---|---|
| Equivariant backbone | MACE-large (24 GB) | PaiNN hidden 128, 4 layers | PaiNN reaches within 1 kcal/mol of MACE on QM9 / SPICE-small subsets at a fraction of cost |
| QM data | Train from scratch on 2 M SPICE conformers | Use 200 k SPICE-small subset, optionally warm-start from MACE-OFF23-small public weights for the embedding only | The novelty is the Hamiltonian decomposition, not the force field. A well-validated subset suffices |
| Coupling potential | Big GIN, 6 layers | GINE hidden 256, 4 layers | Capacity matched to ChEMBL-2M density modeling |
| PCD buffer | 16k chains | 4k chains | Cheap-large-buffer effect saturates by 4k for our objective |
| Vocabulary discovery | VQ-VAE on the fly | Off in v0, optional Fig 4 ablation in v1 | Discovery is bonus novelty, not required for C1-C5 |
| Molecule size cap | 80 heavy atoms | 50 heavy atoms (covers > 95 % of ZINC drug-like) | Quadratic in atom count for radial cutoff GNNs; this is the single biggest VRAM lever |
| Mixed precision | fp32 | bf16 with fp32 master | bf16 is exact-enough for energy regression, saves 40 % memory |
| Parallel chains for sampling | 16 | 4 | Sufficient to hit ESS > 1000 in our ablations on toy systems |
| LIT-PCBA targets evaluated | 15 | 15 (unchanged, eval is CPU-bound docking) | Vina is CPU; 4060 is fine for DiffDock confidence inference |
| FEP / MM-PBSA | FEP on top-10 | MM-PBSA only on top-30 (SI) | FEP needs much more compute |

## What we do **not** compromise

- The **Boltzmann formulation** in equation (1).
- The **three lemmas** in METHOD section 2 (these are theoretical, not compute-bound).
- The **chemical-potential calibration** $\mathcal{L}_\mu$ which is the interpretability story.
- The **detailed-balance** regularizer (equation 8) which is the rigor story.
- The **head-to-head LIT-PCBA evaluation** against BBAR / RxnFlow / TargetDiff (most compute is docking, not training).

## External public weights to leverage

To stretch the 4060 further, we initialize selected modules from public checkpoints rather than from scratch:

| What | Source | Used for | Frozen? |
|---|---|---|---|
| MACE-OFF23-small atomic embeddings | https://github.com/ACEsuit/mace-off | Initialize PaiNN node embedding | Frozen for first 5k steps then unfrozen |
| ChEMBL-pretrained ChemBERTa | HuggingFace `seyonec/ChemBERTa-zinc-base-v1` | Optional warm-start of fragment-level token features for $V^{\mathrm{couple}}$ | Frozen |
| DiffDock-L | Public release | Pose validation only at evaluation time | Inference only |
| RDKit Crippen / Bickerton / SA scorer | RDKit and Ertl SA implementation | $\boldsymbol{\phi}$ feature computation | Deterministic |

## Disk and dataset sizing

| Dataset | Full size | Subset we use | Disk (preprocessed) |
|---|---|---|---|
| SPICE | 70 GB | 200 k conformers, drug-like elements only (H,C,N,O,F,P,S,Cl,Br) | ~3 GB |
| QMugs | 80 GB | 50 k molecules, single conformer per molecule | ~1 GB |
| ZINC drug-like | 9 GB | 1 M molecules, BRICS fragmented | ~2 GB |
| ChEMBL-34 | 23 GB | 500 k molecules with assay annotation | ~1.5 GB |
| LIT-PCBA | 15 MB compressed (already in `data/external`) | All 15 targets | unchanged |

Total disk for working set: under 10 GB. Fits on a typical workstation SSD.

## Wall-clock budget per phase (4060, single user)

| Phase | Wall clock |
|---|---|
| QM head pretrain (200 k SPICE samples, 30 epochs) | 24 hours |
| Coupling potential pretrain (ChEMBL 500 k, 20 epochs) | 18 hours |
| Joint fine-tune with PCD (10 k steps) | 30 hours |
| Sampling 1 k molecules per LIT-PCBA target (15 targets) | 4 hours |
| Vina docking (CPU) for 15 x 1000 molecules | 12 hours (parallelizable) |
| DiffDock confidence on top-100 per target | 3 hours |
| Per-Hamiltonian-term ablation retrain at half scale | 3 x 24 hours = 72 hours |
| **Total to finish all experiments end-to-end** | **~1.5 calendar weeks of compute** |

## Mitigations if VRAM still pinches

In order of preference:

1. Drop molecule size cap from 50 to 40 heavy atoms (covers 88 % of ZINC).
2. Reduce PaiNN hidden from 128 to 96.
3. Reduce batch from 32 to 16 with gradient accumulation = 2.
4. Move PCD buffer to CPU pinned memory and stream.
5. Disable forces in $\mathcal{L}_{\mathrm{QM}}$ for the first 5k steps.

Do not give up bf16. Do not give up parallel tempering (you can drop to 2 chains).
