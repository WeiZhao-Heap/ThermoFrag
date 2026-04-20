# FIGURES

Eight planned main-text figures. Each is paired with the falsifiable claim it serves (see PLAN.md).

## Fig 1. Conceptual framework

Type: schematic, no data.

Panels:
- (a) The Boltzmann formulation, equation (1), drawn as a graphical model.
- (b) Hamiltonian decomposition with three colored boxes: $E^{\mathrm{QM}}$, $V^{\mathrm{couple}}$, $-\boldsymbol{\mu}\cdot\boldsymbol{\phi}$.
- (c) Three limit regimes drawn as corners of a triangle: BBAR limit, ML force-field limit, density-modeling limit. ThermoFrag sits in the interior.

Tooling: TikZ or Inkscape. Source kept in `results/figures/fig1/`.

## Fig 2. QM consistency

Claim: C1.

Panels:
- (a) Scatter plot of predicted vs DFT single-point energy on QMugs holdout (50 k molecules).
- (b) Force MAE histogram, broken down by element.
- (c) Spearman correlation table for SPICE-test, QMugs-test, and a 500-molecule conformational diversity probe.

Data: trained checkpoint, `scripts/eval_qm.py`.

## Fig 3. Temperature-controlled exploration / precision tradeoff

Claim: positioning, not a numbered claim. Establishes that the temperature knob does what statistical mechanics says.

Panels:
- (a) Diversity (internal Tanimoto, lower is more diverse) vs $\beta$ on a continuous sweep $\beta\in[0.1, 10]$.
- (b) Property-target hit rate vs $\beta$ on three target conditions.
- (c) Comparison: BBAR's softmax-temperature hack does not produce a smooth tradeoff (overlay).

Data: `scripts/sweep_temperature.py`.

## Fig 4. Vocabulary discovery vs BRICS (optional / SI-fallback)

Claim: bonus novelty (only if v1 includes vocabulary discovery).

Panels:
- (a) UMAP of learned fragment codebook vs BRICS centroids.
- (b) Downstream MMD between generator and ChEMBL-test, comparing the two vocabularies.
- (c) Top-10 learned fragments alongside their BRICS nearest neighbors, with chemical commentary.

Data: `scripts/vocab_compare.py`.

## Fig 5. Chemical-potential interpretability

Claim: C2.

Panels:
- (a) $\mu_{\mathrm{logP}}$ across a 2D logP-QED grid, overlaid with Wildman-Crippen empirical contributions.
- (b) Spearman-correlation bar plot: ThermoFrag $\mu$ vs WC, vs Bickerton, vs SA, vs TPSA reference weights. Include a baseline showing BBAR's condition embedding has no such correspondence.
- (c) Marginal substitution rate $\partial\mu_i/\partial y_j$ as a heatmap, with chemist commentary on each off-diagonal block.

Data: `scripts/eval_mu.py`. Re-uses the WC-correlation infrastructure from BBAR `manuscript5/` if available.

## Fig 6. Pareto reachability and uncertainty

Claim: C5.

Panels:
- (a) Pareto frontier on logP x QED x SA in ChEMBL holdout; ThermoFrag generated cloud overlaid.
- (b) $\sigma_\mu$ (Laplace) field on the same axes, showing that the model's own uncertainty rises in the holdout-empty corners.
- (c) ROC curve for using $\|\sigma_\mu\|$ as an OOD detector on synthetic out-of-distribution targets.

Data: `scripts/eval_pareto.py`.

## Fig 7. LIT-PCBA docking head-to-head

Claim: C3.

Panels:
- (a) For 15 LIT-PCBA targets: paired box plot of mean Vina top-10 score, ThermoFrag vs BBAR vs RxnFlow vs TargetDiff vs Pocket2Mol.
- (b) DiffDock pose-confidence distribution for the same molecules.
- (c) Two example targets with overlay of best generated pose vs reference inhibitor (PyMOL render).

Data: `scripts/eval_litpcba.py`. Vina runs are CPU; DiffDock uses GPU inference.

## Fig 8. Physical realism audit

Claim: C4.

Panels:
- (a) Strain energy after one MMFF94 minimization, distribution per generator.
- (b) Torsional angle distribution at rotatable bonds, Wasserstein distance to PDB ligand reference.
- (c) Ring strain distribution by ring size.

Data: `scripts/eval_strain.py`. Uses OpenMM + GAFF.

## Auxiliary (SI only)

- S1. Detailed-balance numerical check: histogram of Metropolis acceptance vs predicted equilibrium ratio.
- S2. Per-Hamiltonian-term ablation table (claim C6).
- S3. Compute and wall-clock comparison vs published baselines.
- S4. Counterfactual-fragment-knockout audit of ThermoFrag, recycled from `BBAR/manuscript{1,5}`. This makes the prior work visible and re-purposed.
- S5. Effect of the parallel-tempering chain count on ESS.
- S6. SPICE element coverage and the impact of restricting to drug-like elements.
