# ThermoFrag

A statistical-mechanical, SE(3)-equivariant fragment-assembly generative model for multi-objective small-molecule design.

Target venue: Advanced Science / Nature Communications (small-molecule generative-AI track).

Hardware target: a single NVIDIA RTX 4060 16 GB. All design choices in this repo are scaled to fit this constraint without sacrificing the first-principles claim.

---

## One-paragraph pitch

Existing multi-objective molecular generators treat property targets as opaque conditioning vectors and reward weights, which produces molecules that violate Pareto coherence and physical chemistry once they leave the model. ThermoFrag rewrites fragment-based 3D generation as conditional sampling from a Boltzmann distribution whose Hamiltonian decomposes into a QM-grounded internal energy, a learned coupling potential, and a property external field with a calibrated chemical potential. The sampler satisfies detailed balance, so generated molecules are unbiased samples from a physically meaningful target distribution rather than ad hoc autoregressive outputs. In the zero-temperature limit ThermoFrag recovers BBAR-style greedy fragment assembly; with the QM term off it recovers data-driven density modeling; with the property field off it recovers an unconditional 3D generator. ThermoFrag is therefore a strict generalization of three existing paradigms with new physics put in by hand.

---

## Repository map

```
ThermoFrag/
  README.md                  This file.
  docs/                      Planning docs. Read in order: PLAN, METHOD, HARDWARE, DATA, FIGURES, MILESTONES.
  src/thermofrag/            Library source. Skeleton with TODOs.
    model/                   Equivariant backbone, Hamiltonian heads, chemical-potential head.
    sampling/                Discrete MH on fragment graph + Langevin on coordinates, temperature annealing.
    potentials/              QM, coupling, external-field decomposition.
    training/                Loss assembly, optimizer, persistent contrastive divergence buffer.
    data/                    Dataset wrappers (SPICE, QMugs, ZINC, ChEMBL, LIT-PCBA).
    utils/                   Logging, seeding, I/O.
  vendor/                    Vendored BBAR fragmentation/transform code (BRICS).
  data/
    raw/                     Untouched downloads.
    processed/               Cached preprocessed tensors.
    external/                Files copied from BBAR (LIT-PCBA, ZINC, fragment library).
  configs/                   YAML run configs.
  scripts/                   Entrypoints: preprocess, train, sample, eval.
  notebooks/                 Exploratory notebooks.
  results/                   Metrics, figures, model checkpoints (gitignored except summaries).
  tests/                     Pytest unit tests.
```

---

## Quick start (when code is filled in)

```bash
conda env create -f environment.yml
conda activate thermofrag
python scripts/preprocess.py --config configs/default.yaml
python scripts/train.py     --config configs/default.yaml
python scripts/sample.py    --config configs/default.yaml --n 1000
python scripts/eval.py      --config configs/default.yaml
```

---

## Reading order for a new contributor

1. `docs/PLAN.md` for the overall strategy and falsifiable claims.
2. `docs/METHOD.md` for the math.
3. `docs/HARDWARE.md` for the 4060-specific scaling decisions (this is what makes the project tractable).
4. `docs/DATA.md` for which datasets and which subsets, with download URLs.
5. `docs/FIGURES.md` for the eight planned main-text figures and what each must prove.
6. `docs/MILESTONES.md` for the week-by-week schedule.
