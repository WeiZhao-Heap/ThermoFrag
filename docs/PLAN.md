# PLAN

## Why this project exists

Five prior manuscripts in `BBAR/manuscript{1..5}` audit a 2023 fragment generator (BBAR) without changing its architecture. Reviewers reject these as incremental tooling around an outside model. ThermoFrag is the from-scratch replacement: a generator that is itself novel from physical principles, with the audit work from the prior manuscripts retained as a Discussion / SI section so no effort is wasted.

## The first-principles diagnosis we are answering

Current multi-objective small-molecule generators have four coupled defects:

1. Property conditioning is implemented as a wished-for vector concatenated with noise, with no physical meaning.
2. Step-wise autoregressive softmax produces myopic decisions and no global equilibrium guarantee.
3. Loss is a scalarized sum over independently learned property predictors, breaking the thermodynamic coupling that makes the properties non-independent in reality.
4. The fragment vocabulary is fixed by retrosynthesis rules (BRICS), so the generator cannot discover chemically appropriate units for new objectives.

ThermoFrag answers all four in one model, derived from a single Boltzmann formulation.

## Falsifiable scientific claims

C1. ThermoFrag's internal energy head agrees with DFT single-point energies on a held-out QM benchmark (target Spearman > 0.9, energy MAE < 5 kcal/mol).

C2. The learned chemical potential vector reproduces the empirical Wildman-Crippen logP weights and Bickerton QED weights to Spearman > 0.6, demonstrating that the conditioning carries a physical meaning rather than being a black-box vector.

C3. On 15 LIT-PCBA targets, ThermoFrag generated ligands have higher mean Vina docking scores and higher DiffDock pose confidence than BBAR, RxnFlow, and TargetDiff baselines (paired Wilcoxon p < 0.01 on at least 10 / 15 targets).

C4. Generated molecules show lower post-relaxation strain energy (OpenMM / GAFF) than baselines (paired effect size d > 0.3) without any post-hoc filtering.

C5. The chemical-potential uncertainty (Laplace approximation) flags out-of-distribution target requests with AUROC > 0.8 against ChEMBL Pareto-frontier holdout.

C6. Ablations: removing the QM head collapses C1 and C4. Removing the coupling potential collapses C2. Removing the chemical-potential head collapses C3 and C5. So all three terms in the Hamiltonian are independently necessary.

If C1, C3, C4 hold simultaneously, the paper has its three legs (physics fidelity, downstream utility, sanity preservation). C2 is the interpretability story. C5 is the safety story. C6 is the methods rigor.

## Hard non-goals

- No wet-lab validation. All claims are computational.
- No FEP. MM-PBSA at most, on a top-30 subset, in SI only.
- No protein-specific training (no SBDD per-pocket fine-tuning). Pocket only enters at evaluation time via Vina/DiffDock.
- No fancy 3D diffusion such as DiffDock-pose decoder. Coordinate updates are Langevin only.

## Audience and venue strategy

Primary submission target: Advanced Science (the same journal that published BBAR). This makes the "we generalize BBAR with proper physics" framing land cleanly. Secondary target: Nature Communications (Computational and Systems Biology section).

The cover letter pitches the unified Boltzmann formulation as the key contribution and the three-paradigm-recovery property as the technical novelty.
