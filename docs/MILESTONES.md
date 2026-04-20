# MILESTONES

A 12-week schedule sized to a single 4060 16 GB and one full-time researcher. Compute hours assume the workstation is dedicated to this project.

## Phase 0 (week 1): environment and data

- [ ] Set up conda env from `environment.yml`.
- [ ] Download SPICE-v2 drug-like subset, QMugs, ChEMBL-34, ZINC drug-like.
- [ ] Run `scripts/preprocess.py` for all five datasets.
- [ ] Sanity check: load a batch from each LMDB / NPZ, print shapes.

Exit criterion: a notebook in `notebooks/01_data_sanity.ipynb` opens every dataset, plots a histogram of molecule sizes, and prints checksum manifests.

## Phase 1 (weeks 2-3): QM head pretrain

- [ ] Implement `src/thermofrag/model/painn.py` (PaiNN backbone) and `src/thermofrag/potentials/qm.py` (energy + force head).
- [ ] Train QM head on SPICE subset, 30 epochs, bf16, batch 32.
- [ ] Evaluate on QMugs holdout, produce Fig 2 (a, b, c).
- [ ] Optionally warm-start from MACE-OFF23-small embedding to save a week.

Exit criterion: C1 holds (energy MAE < 5 kcal/mol, Spearman > 0.9). Checkpoint saved to `results/checkpoints/qm_pretrain.pt`.

## Phase 2 (weeks 4-5): coupling potential and unconditional generator

- [ ] Implement `src/thermofrag/potentials/coupling.py` (GINE on fragment graph).
- [ ] Implement `src/thermofrag/sampling/mh.py` (graph proposals + MH).
- [ ] Implement `src/thermofrag/training/pcd.py` (replay buffer).
- [ ] Train unconditional density model on ZINC, with PCD.
- [ ] Sanity figure: generated vs ZINC property histograms (logP, QED, MW, TPSA).

Exit criterion: KL(generated || ZINC) on first three property marginals below 0.05.

## Phase 3 (weeks 6-7): chemical-potential head and joint fine-tune

- [ ] Implement `src/thermofrag/potentials/external_field.py` ($\boldsymbol{\mu}$ head with Laplace last layer).
- [ ] Implement $\mathcal{L}_\mu$ via finite-difference thermodynamic integration.
- [ ] Joint fine-tune on ChEMBL-conditional, 10 k steps.
- [ ] Produce Fig 3 (temperature) and Fig 5 (chemical potential).

Exit criterion: C2 holds (Spearman vs WC > 0.6, vs Bickerton > 0.6).

## Phase 4 (weeks 8-9): downstream evaluation

- [ ] Implement `src/thermofrag/sampling/anneal.py` with parallel tempering.
- [ ] Generate 1000 ligands per LIT-PCBA target.
- [ ] Run AutoDock Vina on CPU pool (12 hours in background).
- [ ] Run DiffDock-L on top-100 per target.
- [ ] Run baselines BBAR, RxnFlow, TargetDiff on the same targets (use their public checkpoints — **see `docs/BASELINES.md` for the full execution plan**; weights are public for all three, prior "upstream-blocked" notes are stale).
- [ ] Produce Fig 7.
- [ ] Run OpenMM strain audit, produce Fig 8.

Exit criterion: C3 and C4 hold. As of 2026-04-19 C3 passes vs the LIT-PCBA 246k reference library only; generator-vs-generator arm is open and tracked in `docs/BASELINES.md`.

## Phase 5 (week 10): ablations and SI experiments

- [ ] Train three half-scale ablation models (no-QM, no-coupling, no-mu).
- [ ] Detailed-balance numerical check.
- [ ] Counterfactual-fragment-knockout audit applied to ThermoFrag (recycle from `BBAR/manuscript5`).
- [ ] Pareto reachability + OOD AUROC, produce Fig 6.

Exit criterion: C5 and C6 hold. SI tables ready.

## Phase 6 (weeks 11-12): writing

- [ ] Draft main text (8 figures, 3500 words).
- [ ] Methods section in supplement, lifting from `docs/METHOD.md` verbatim.
- [ ] SI figures S1-S6.
- [ ] Cover letter highlighting the three-paradigm-recovery property.
- [ ] Internal review pass against typical Adv Sci / NC reviewer attack lines.
- [ ] Submit.

## Risk register

| Risk | Mitigation |
|---|---|
| QM head fails to converge on 4060 in two weeks | Initialize from public MACE-OFF23-small weights; freeze for first 5k steps |
| PCD diverges (energy explosion) | Energy clipping, smaller learning rate, smaller buffer refresh fraction |
| Sampler mixing too slow at low $\beta$ | Add more parallel-tempering chains or shorten chain length per evaluation |
| LIT-PCBA baselines run too slowly | Use published baseline numbers where reproducible; document any number that comes from publication |
| Joint fine-tune destabilizes the QM head | Use smaller learning rate on QM parameters in joint phase, or freeze QM head entirely |
| DiffDock confidence inference OOMs on 4060 | Run on a subset (top-30 instead of top-100), document in SI |

## Definition of done for the paper

All six claims (C1-C6) hold. All eight main figures and six SI figures are in `results/figures/` with reproduction scripts in `scripts/`. A single command `bash scripts/reproduce_all.sh` regenerates every figure from cached intermediates.
