# DATA

All datasets, where to get them, what subset we use, and how they are preprocessed.

## Datasets used

### 1. SPICE (QM training for $E^{\mathrm{QM}}$)

- Source: https://github.com/openmm/spice-dataset, current release v2.
- Size: ~2 M conformers with DFT energies and forces.
- Subset: drug-like subset, 200 k conformers, restricted to elements H, C, N, O, F, P, S, Cl, Br. Filtered to molecules with up to 50 heavy atoms.
- Preprocessing: see `scripts/preprocess_spice.py` (TODO). Stored as a single `.npz` per shard in `data/processed/spice_shards/`.
- Why: pretraining target for the QM head. Public, well-curated, force labels are present.

### 2. QMugs (held-out QM evaluation)

- Source: https://www.research-collection.ethz.ch/handle/20.500.11850/482129
- Size: ~660 k molecules with multiple conformers, GFN2-xTB and DFT energies.
- Subset: 50 k molecules, one conformer each, used as held-out test for Fig 2.
- Why: independent QM benchmark for C1 (no leakage with SPICE).

### 3. ZINC drug-like (fragment library construction and unconditional density)

- Source: https://zinc20.docking.org/, drug-like subset.
- Local copy: vendored from BBAR in `data/external/ZINC.tar`.
- Subset: 1 M molecules, BRICS fragmented using vendored code in `vendor/bbar_fragmentation/`.
- Output: `data/processed/fragment_library.parquet` (fragments + frequencies + RDKit canonical SMILES).
- Why: fragment vocabulary and unconditional density baseline for $V^{\mathrm{couple}}$.

### 4. ChEMBL-34 (multi-property conditional training and chemical-potential calibration)

- Source: https://chembl.gitbook.io/chembl-interface-documentation/downloads
- Size: ~2.4 M compounds, ~20 M activity records.
- Subset: 500 k molecules with at least one IC50 / Ki annotation, after standardization with the ChEMBL structure pipeline.
- Properties cached: logP (Crippen), QED (Bickerton), SA (Ertl), TPSA, MW, HBA, HBD, rotatable-bonds, surrogate pIC50 from a pretrained ChemBERTa-regression head.
- Why: provides $(m, y)$ pairs for $\mathcal{L}_\mu$ and the coupling potential.

### 5. LIT-PCBA (downstream docking benchmark, Fig 7)

- Local copy: `data/external/LIT-PCBA.tar.gz` (vendored from BBAR).
- 15 targets, prepared receptors and decoys included.
- We use it only at evaluation time. Generated ligands are docked with AutoDock Vina (CPU) and pose-confidence-rescored with DiffDock-L (GPU inference).
- Why: a community-standard external benchmark, identical evaluation protocol as BBAR's original paper, makes head-to-head comparison defensible.

### 6. PDB ligand strain reference (Fig 8 baseline)

- Source: a curated set of ~1000 small-molecule ligands from the PDB with resolution < 2.0 Å.
- Strain energy distribution from Pinheiro et al 2019 / RDKit + OpenMM minimization.
- Why: the empirical "physical realism" reference for what generated strain distributions should look like.

## Preprocessing pipeline

```
scripts/preprocess.py
  ├── spice    -> data/processed/spice_shards/*.npz          (atoms, coords, energies, forces)
  ├── qmugs    -> data/processed/qmugs_test.npz              (atoms, coords, energies)
  ├── zinc     -> data/processed/fragment_library.parquet    (fragment SMILES, freq, anchors)
  │           -> data/processed/zinc_unconditional.lmdb      (graph tensors)
  ├── chembl   -> data/processed/chembl_conditional.lmdb     (graph tensors + property vector y)
  └── litpcba  -> data/processed/litpcba/{target}/receptor.pdbqt etc
```

Each preprocessor writes a `manifest.json` with the dataset version, filter criteria, and a SHA256 of the produced files. This makes runs reproducible across machines.

## Property computation reference

Properties enter the model as the feature vector $\boldsymbol{\phi}(m,\mathbf{x})$.

| Property | Code | Source |
|---|---|---|
| logP | `rdkit.Chem.Crippen.MolLogP` | Crippen 1999 |
| QED | `rdkit.Chem.QED.qed` | Bickerton 2012 |
| SA | Ertl SA scorer (script vendored from RDKit Contrib) | Ertl & Schuffenhauer 2009 |
| TPSA | `rdkit.Chem.Descriptors.TPSA` | Ertl 2000 |
| MW | `rdkit.Chem.Descriptors.MolWt` | RDKit |
| HBA / HBD / RotB | RDKit Lipinski | Lipinski 1997 |
| pIC50 surrogate | ChemBERTa fine-tuned on ChEMBL IC50 | Pretrained, not differentiable in our pipeline |

## Storage layout

```
data/
  external/             # vendored as-is from BBAR, never modified
    LIT-PCBA.tar.gz
    ZINC.tar
    3CL_ZINC.tar
    bbar_library.csv
  raw/                  # downloads (gitignored), one folder per dataset
    spice/
    qmugs/
    chembl/
  processed/            # preprocessed tensors (gitignored)
    spice_shards/
    qmugs_test.npz
    fragment_library.parquet
    zinc_unconditional.lmdb
    chembl_conditional.lmdb
    litpcba/
```

Add `data/raw/` and `data/processed/` to `.gitignore`. `data/external/` is committed because it is small and provenance-locked.
