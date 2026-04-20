# BASELINES

Execution plan for the **generator-vs-generator** arm of claims **C3** (Vina docking) and **C4** (OpenMM strain). PLAN.md §C3/C4 requires paired comparison against BBAR, RxnFlow, and TargetDiff on the 15 LIT-PCBA targets. As of 2026-04-19 this arm is **open** — ThermoFrag has only been compared to the LIT-PCBA 246k reference library (`results/eval/phase5/c3_vs_litpcba.json`).

**Correction to prior status**: earlier session notes (`results/eval/claim_summary.json`, `project_phase4_status.md`, `project_phase5_status.md`) claim the baselines are "upstream-blocked" because weights are unavailable. **This is wrong** — all three repos publish pretrained weights. The blocker is engineering work, not availability. Do not re-propagate the "upstream blocker" language.

## The target state

For each baseline `b ∈ {bbar, rxnflow, targetdiff}` and each LIT-PCBA target `t`, produce the same parquet artefacts ThermoFrag already has, under a parallel directory tree:

```
results/eval/phase4_baselines/<b>/
  decoded/<t>.parquet       # columns: target, chain_idx, smiles  (≥ 1000 valid rows)
  vina/<t>.parquet          # columns: target, chain_idx, smiles, vina_score, status
  strain/<t>.parquet        # columns: target, chain_idx, smiles, e_mmff, e_gaff, strain, n_atoms, status
  manifest.json             # baseline version, weight SHA256, sampling config, wall-clock
```

The existing dock + strain drivers are generic over input parquets — once `<b>/decoded/<t>.parquet` exists, `scripts/dock_vina.py` and `scripts/eval_strain.py` run unchanged with a path override.

## Per-baseline recipes

### 1. RxnFlow (highest priority — directly pocket-conditional)

- Repo: https://github.com/SeonghwanSeo/RxnFlow
- Weights: the repo auto-downloads named checkpoints via `--pretrained_model`. Use **`qvina-unif-0-64`** (pocket-conditional QVina reward, matches our task exactly). `./weights/README.md` in the upstream repo lists every available ckpt.
- Install: `git clone` into `vendor/rxnflow/`, follow its `pyproject.toml` (PyG ≥ 2.5, which our env already satisfies).
- Sampling protocol per target `t`:
  1. Load `data/external/receptors/<t>/receptor.pdbqt` + `box.json`.
  2. Run RxnFlow generation with `--pretrained_model qvina-unif-0-64` and the pocket file, requesting ≥ 1100 molecules (20 % margin for invalid SMILES after RDKit sanitize).
  3. Take the first 1000 valid canonical SMILES. Assign `chain_idx = 0..999`.
- Output: write `results/eval/phase4_baselines/rxnflow/decoded/<t>.parquet` with columns `target, chain_idx, smiles` (and any RxnFlow-native confidence as an extra column — optional, but useful for sensitivity).
- **Pitfall**: RxnFlow samples are synthetic-route-derived — some SMILES may fail MMFF94 embedding in our Vina preprocessing. This is expected; the yield loss is part of the comparison.

New script to write: `scripts/sample_rxnflow.py` (thin wrapper around the upstream `rxnflow.sampler` or equivalent entrypoint).

### 2. BBAR (needed because the paper pitch is "we generalize BBAR")

- Repo: https://github.com/SeonghwanSeo/BBAR
- Weights: `sh ./download-weights.sh` → deposits `./test/pretrained_model/{mw,logp,tpsa,qed,3cl-affinity}.tar`. Use the **qed** and **logp** checkpoints (property-conditional) — BBAR has no pocket conditioning, so the fair protocol is to generate once per checkpoint globally and dock the same pool against every pocket.
- Install: `git clone` into `vendor/bbar_upstream/` (keep separate from the existing `vendor/bbar_fragmentation/` which is only the fragmentation utils). Follow BBAR's env pins.
- Sampling protocol:
  1. For each of {qed, logp}, generate 1500 molecules with the condition set to the same drug-like target as ThermoFrag's y-vectors (`results/eval/phase4/litpcba_targets/<t>/y_raw.npy` gives the per-target target vector; for BBAR use the aggregated mean QED / LogP over those targets).
  2. Keep the first 1000 valid SMILES per condition.
  3. For each LIT-PCBA target `t`, dock the **same** 1000-molecule pool (BBAR cannot condition on pocket, so the pool is shared across pockets; this is fine and documented below).
- Output: `results/eval/phase4_baselines/bbar/decoded/<t>.parquet` — same schema, but all 15 targets reference the same SMILES list with `chain_idx` recycled. Add a column `condition ∈ {qed, logp}` so the two-condition pools can be separated in analysis.
- **Known asymmetry**: BBAR has no pocket input. This biases the comparison in ThermoFrag's favor on C3 only because ThermoFrag's sampler is also pocket-agnostic (pocket enters only at Vina time). So the asymmetry is **zero**, and the comparison is fair. Spell this out in the paper.

New script to write: `scripts/sample_bbar.py`.

### 3. TargetDiff (most demanding baseline — 3D pocket-conditional diffusion)

- Repo: https://github.com/guanjq/targetdiff
- Weights: Google Drive folder https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz. Pull the ligand-generation checkpoint (the binding-affinity `egnn_pdbbind_v2016.pt` is not needed for C3/C4).
- Install: `git clone` into `vendor/targetdiff/`. The env needs `torch-scatter` / `torch-cluster` matching the current CUDA build — likely the single highest setup cost across the three baselines.
- Sampling protocol per target `t`:
  1. Prepare pocket input by cropping a 10 Å sphere around the cognate ligand in `data/external/receptors/<t>/cognate_ligand.pdb`. The upstream `targetdiff/scripts/data_preparation/` handles this.
  2. Run pocket-conditional sampling for 1000 molecules. Convert 3D output → canonical SMILES via RDKit (`AllChem.AssignStereochemistryFrom3D`).
- Output: same parquet schema. Keep the 3D sdf too (write `results/eval/phase4_baselines/targetdiff/poses/<t>.sdf`) — TargetDiff generates poses directly, so the same-pose-as-Vina comparison is diagnostic.
- **Important caveat**: TargetDiff **uses pocket information at generation time**, whereas ThermoFrag does not. This makes TargetDiff a **harder** baseline to beat, not an easier one. If ThermoFrag still wins, the result is stronger; if it loses, the paper should frame ThermoFrag as pocket-agnostic by design (the Boltzmann formulation is compositional, not SBDD). Do not drop TargetDiff to save face — run it and report honestly.

New script to write: `scripts/sample_targetdiff.py`.

## Unified downstream pipeline (no new scripts needed)

Once each baseline's `decoded/<t>.parquet` exists:

```bash
# Vina
for b in rxnflow bbar targetdiff; do
  for t in ADRB2 ALDH1 ESR_ago ESR_antago FEN1 GBA IDH1 KAT2A MAPK1 MTORC1 OPRK1 PKM2 PPARG TP53 VDR; do
    python scripts/dock_vina.py \
      --decoded_dir results/eval/phase4_baselines/$b/decoded \
      --out_dir     results/eval/phase4_baselines/$b/vina \
      --target $t
  done
done

# Strain
for b in rxnflow bbar targetdiff; do
  python scripts/eval_strain.py \
    --decoded_dir results/eval/phase4_baselines/$b/decoded \
    --out_dir     results/eval/phase4_baselines/$b/strain
done
```

`dock_vina.py` and `eval_strain.py` currently hard-code `results/eval/phase4/decoded` and `results/eval/phase4/vina`. You will need to add `--decoded_dir` and `--out_dir` flags to both — 10-line change each, do **not** copy the scripts.

## Statistical analysis (new script)

Write `scripts/eval_generator_vs_generator.py` consuming:

- `results/eval/phase4/vina/<t>.parquet` (ThermoFrag)
- `results/eval/phase4_baselines/<b>/vina/<t>.parquet` (each baseline)
- `results/eval/phase4/strain/<t>.parquet` and `results/eval/phase4_baselines/<b>/strain/*.parquet`

Per target, pair by `chain_idx` rank (both pools are independently sampled, so rank-pair the sorted top-k scores rather than row-match). Produce:

```
results/eval/phase5/c3_vs_generators.csv
  target, baseline, tf_top10_mean, baseline_top10_mean, tf_minus_baseline, wilcoxon_p
results/eval/phase5/c4_vs_generators.csv
  target, baseline, tf_mean_strain, baseline_mean_strain, cohens_d
results/eval/phase5/c3_c4_bars.png    # paired bar plot per target, three baselines stacked
```

### Acceptance criteria (from PLAN.md)

- **C3**: TF beats each baseline on **≥ 10 / 15 targets** at paired Wilcoxon `p < 0.01` on top-10 Vina means. If any baseline is matched on < 10 targets, document which targets fail and why.
- **C4**: TF has **lower** strain than each baseline with **Cohen's d > 0.3**. Note: the current `nomu` ablation already shows TF has **higher** strain than ZINC-random (median 13.9 vs 10.82), i.e. the sampler trades strain for property targeting. If TF also has higher strain than the three generators, **C4 as written in PLAN.md fails**, and the paper needs an honest reframing (the effect size is real but of the *opposite* sign — property-targeting comes at a small strain cost).

## Compute estimate (single 4060 + 16-core CPU workstation)

| Step | Wall clock |
|---|---|
| RxnFlow env setup + 15 × 1000 sampling | 4 h GPU |
| BBAR env setup + 2 × 1000 sampling (shared across pockets) | 2 h GPU |
| TargetDiff env setup (torch-scatter / torch-cluster) | 4 h human + 30 min compile |
| TargetDiff 15 × 1000 sampling (3D diffusion, ~6 s/molecule) | ~25 h GPU |
| Vina: 3 baselines × 15 targets × 1000 ligands (CPU, parallel) | 36 h CPU wall |
| OpenMM strain: 3 baselines × 15 × 1000 (CPU) | 20 h CPU wall |
| Paired-stats + figures | < 1 h |
| **Total** | **~5 calendar days** (most of the time is idle CPU background on Vina) |

PLAN.md risk register permits publishing "any number that comes from publication" if rerunning fails, but weights exist for all three here, so run them.

## What goes in the paper after this arm closes

- **Fig 7** becomes a 4-box-per-target plot (TF / RxnFlow / BBAR / TargetDiff / LIT-PCBA-ref) — update `scripts/eval_c3_c4_c6_summary.py` accordingly.
- **Fig 8** gains three extra strain histograms. If C4 sign flips (see above), the figure is still scientifically valid — it becomes evidence that property conditioning has a cost, which is a defensible discussion point, not a failure.
- `claim_summary.json` open_items items 1 and 2 get closed; item 3 (large PaiNN per-mol MAE on tiny molecules) stays open as reframed in the Phase-6 session-2 memory.

## What this arm does *not* cover

- **DiffDock pose confidence** (PLAN.md C3 second half). Separate driver needed — `scripts/dock_diffdock.py` does not exist yet. Out of scope for this doc; open as a follow-up after Vina closes.
- **FEP / MM-PBSA on top-30** (SI only, PLAN.md hard non-goal to keep it at MM-PBSA).
- **Pocket2Mol** (FIGURES.md Fig 7 (a) mentions it). Treat as optional stretch goal once the three primary baselines are done.

## Entry points for the next agent

1. Read this file top to bottom.
2. Read `results/eval/claim_summary.md` for the current verdict state.
3. Start with **RxnFlow** — it is lowest-friction (auto-download weights, pocket-conditional, matches C3 directly). A green RxnFlow comparison derisks the other two and the paper can survive on RxnFlow alone if TargetDiff env proves intractable.
4. Add `--decoded_dir` / `--out_dir` flags to `scripts/dock_vina.py` and `scripts/eval_strain.py` before running any baseline, so the artefact tree above is respected.
5. Check in to memory after each baseline finishes (update `project_phase4_status.md` + `project_phase5_status.md` with actual numbers, overwriting the stale "upstream-blocked" language).
