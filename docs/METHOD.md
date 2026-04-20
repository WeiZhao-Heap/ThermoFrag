# METHOD

This document is the mathematical reference. Code in `src/thermofrag/` follows the symbols and equation numbers used here.

## 1. Generative model

Let $m=(V_m,E_m)$ be a molecular graph assembled from fragments in vocabulary $\mathcal{F}$, and let $\mathbf{x}\in\mathbb{R}^{|V_m|\times 3}$ be its atomic coordinates. Let $y\in\mathbb{R}^{K}$ be a target property vector (logP, QED, SA, optional pIC50 surrogate, ...).

The conditional generative distribution is

$$
p_\theta(m,\mathbf{x}\mid y) \;=\; \frac{1}{Z_\theta(y,\beta)}\,\exp\!\Big(-\beta\,\mathcal{H}_\theta(m,\mathbf{x};y)\Big) \tag{1}
$$

with Hamiltonian

$$
\mathcal{H}_\theta(m,\mathbf{x};y) \;=\; E_\theta^{\mathrm{QM}}(m,\mathbf{x}) \;+\; V_\theta^{\mathrm{couple}}(m) \;-\; \boldsymbol{\mu}_\theta(y)^{\!\top}\boldsymbol{\phi}(m,\mathbf{x}). \tag{2}
$$

The three terms have non-overlapping responsibilities:

- $E_\theta^{\mathrm{QM}}$: a learned ML potential, trained against quantum-chemistry single-point energies. This is the physics anchor.
- $V_\theta^{\mathrm{couple}}$: a graph potential learned from chemical databases. This captures statistics that QM alone does not (e.g., bioavailability bias of ChEMBL).
- $\boldsymbol{\mu}_\theta(y)$: a calibrated chemical-potential vector. $\boldsymbol{\phi}$ are pre-computable property features (Crippen logP, Bickerton QED, SA score, surrogate pIC50, MW, TPSA, etc.).

## 2. Limit recovery (positioning lemmas)

**Lemma 1** (BBAR limit). As $\beta\to\infty$, equation (1) concentrates on $\arg\min_{(m,\mathbf{x})}\mathcal{H}_\theta$, which under the standard fragment-by-fragment factorization recovers a chain of greedy block selections. With $E^{\mathrm{QM}}=V^{\mathrm{couple}}=0$ and $\mu$ identified with BBAR's condition embedding output, this reduces to BBAR's argmax block head.

**Lemma 2** (ML force-field limit). With $V^{\mathrm{couple}}=0$ and $\mu=0$, equation (1) is $p\propto e^{-\beta E^{\mathrm{QM}}}$, the Boltzmann distribution under a learned force field, recovering ANI / MACE-style sampling.

**Lemma 3** (data-density limit). With $E^{\mathrm{QM}}=0$ and $\mu=0$, equation (1) reduces to an EBM density model on $\mathcal{F}$-graphs, recovering Hamiltonian-flow style EBMs.

These three lemmas are the positioning argument: ThermoFrag is the unique convex combination of three previously separate paradigms.

## 3. Loss decomposition

$$
\mathcal{L}(\theta) \;=\; \mathcal{L}_{\mathrm{QM}} \;+\; \lambda_1 \mathcal{L}_{\mathrm{couple}} \;+\; \lambda_2 \mathcal{L}_{\mu} \;+\; \lambda_3 \mathcal{L}_{\mathrm{DB}} \;+\; \lambda_4 \mathcal{L}_{\mathrm{vocab}}. \tag{3}
$$

### 3.1 QM regression

$$
\mathcal{L}_{\mathrm{QM}} \;=\; \mathbb{E}_{(m,\mathbf{x},E^*,\mathbf{F}^*)\sim\mathcal{D}_{\mathrm{QM}}}\Big[(E_\theta^{\mathrm{QM}}-E^*)^2 \;+\; \alpha_F\,\|\nabla_{\mathbf{x}}E_\theta^{\mathrm{QM}}-\mathbf{F}^*\|^2\Big]. \tag{4}
$$

Energy is centered per-element to avoid a trivial atom-counting solution. Forces are supervised when available (SPICE provides them).

### 3.2 Coupling potential (persistent contrastive divergence)

$$
\mathcal{L}_{\mathrm{couple}} \;=\; \mathbb{E}_{m\sim p_{\mathrm{data}}}[V_\theta^{\mathrm{couple}}(m)] \;-\; \mathbb{E}_{m\sim p_\theta}[V_\theta^{\mathrm{couple}}(m)]. \tag{5}
$$

The negative samples come from a persistent replay buffer of MCMC chains (Tieleman 2008). Buffer size 4096 graphs, refresh 5 percent per step.

### 3.3 Chemical-potential calibration

The thermodynamic identity $\mu_i=\partial F/\partial y_i$ where $F=-\beta^{-1}\log Z$ implies

$$
\mathbb{E}_{p_\theta(\cdot\mid y)}[\phi_i] \;=\; -\partial_{y_i}\log Z_\theta(y) \;=\; \beta\,\partial_{y_i}\langle\mathcal{H}\rangle. \tag{6}
$$

We enforce the practical version

$$
\mathcal{L}_{\mu} \;=\; \mathbb{E}_{y\sim p(y)}\,\Big\|\boldsymbol{\mu}_\theta(y) \;-\; \nabla_y\,\widehat{F}_\theta(y)\Big\|^2 \tag{7}
$$

with $\widehat{F}_\theta(y)$ estimated by a single-step thermodynamic-integration finite difference using the buffer in 3.2. This makes $\boldsymbol{\mu}$ a verifiable physical quantity, not a black-box conditioner.

### 3.4 Detailed-balance regularizer

For the proposal kernel $q(m'\mid m)$ used by the sampler (section 4),

$$
\mathcal{L}_{\mathrm{DB}} \;=\; \mathbb{E}\!\Big[\big(\,\log p_\theta(m\!\to\!m') + \mathcal{H}(m) - \log p_\theta(m'\!\to\!m) - \mathcal{H}(m')\big)^2\Big]. \tag{8}
$$

This makes the learned proposal compatible with the equilibrium target.

### 3.5 Vocabulary discovery (optional, can be turned off in v0)

A VQ-VAE with codebook size 1024 over fragment embeddings learns $\mathcal{F}$ jointly. The reconstruction loss $\mathcal{L}_{\mathrm{vocab}}$ is the standard VQ-VAE objective with EMA codebook updates. Initialized from BRICS centroids.

## 4. Sampler

A single sampling sweep at temperature $\beta$ alternates two updates.

### 4.1 Discrete graph update (Metropolis-Hastings)

Propose $m'\sim q(\cdot\mid m)$, where $q$ uniformly chooses among fragment-add, fragment-swap, and fragment-delete. Accept with probability

$$
A(m\!\to\!m') \;=\; \min\!\Big(1,\;\frac{q(m\mid m')}{q(m'\mid m)}\,\exp\!\big(-\beta[\mathcal{H}(m')-\mathcal{H}(m)]\big)\Big). \tag{9}
$$

### 4.2 Coordinate update (Langevin)

$$
\mathbf{x}_{t+1} \;=\; \mathbf{x}_t \;-\; \eta\,\nabla_{\mathbf{x}}\mathcal{H}_\theta(m,\mathbf{x}_t;y) \;+\; \sqrt{2\eta/\beta}\,\boldsymbol{\xi}_t,\qquad \boldsymbol{\xi}_t\sim\mathcal{N}(0,I). \tag{10}
$$

### 4.3 Annealing

Linear inverse-temperature schedule $\beta_t=\beta_0+\gamma t$ for $t\in[0,T]$. $\beta_0=0.5$, $\beta_T=5.0$, $T=200$ for default sampling. Parallel-tempering with 4 chains, swap every 20 steps.

### 4.4 Initialization

Initial $m$ from a small BRICS seed; initial $\mathbf{x}$ from RDKit ETKDG with one MMFF94 minimization step.

## 5. Architecture summary

| Component | Backbone | Parameters | VRAM (4060) |
|---|---|---|---|
| $E^{\mathrm{QM}}$ | PaiNN, hidden 128, 4 layers, cutoff 5 Å | 1.0 M | 1.5 GB at batch 32 |
| $V^{\mathrm{couple}}$ | GINE, hidden 256, 4 layers + readout MLP | 0.6 M | 0.4 GB |
| $\mu(y)$ | 2-layer MLP (256 hidden) + Laplace last layer | 0.07 M | negligible |
| Sampler buffer | 4096 graphs at avg 30 atoms | n/a | 0.3 GB |
| Total training | mixed precision bf16, batch 32 | ~2 M | ~9 GB peak |

See `docs/HARDWARE.md` for the rationale.

## 6. Inference time budget

Per generated molecule at default annealing schedule on 4060: roughly 0.5 s for 200 sweeps with parallel tempering. 1000 molecules per evaluation run takes about 9 minutes. LIT-PCBA at 15 targets x 200 molecules is achievable in under one hour.

## 7. Evaluation metrics summary

| Claim | Metric | Tool |
|---|---|---|
| C1 QM | Energy MAE, force MAE, Spearman | Held-out SPICE / QMugs split |
| C2 $\mu$ | Spearman vs Wildman-Crippen, vs Bickerton | RDKit Crippen, RDKit QED |
| C3 docking | Vina top-10 mean, DiffDock confidence | AutoDock Vina, DiffDock |
| C4 strain | OpenMM single-point post-MMFF94 minimization $\Delta E$ | OpenMM + GAFF |
| C5 OOD | AUROC of $\|\sigma_{\mu}\|$ vs OOD label | Laplace last-layer |
| C6 ablation | Per-claim drop after removing each Hamiltonian term | Full retrain at smaller scale |

Detailed protocols are in `docs/FIGURES.md`.
