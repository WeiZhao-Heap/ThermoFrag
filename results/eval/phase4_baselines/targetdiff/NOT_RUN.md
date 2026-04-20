# TargetDiff — infrastructure staged, full sampling not run

**Status**: env + weights + driver all in place. **Sampling not executed** due to
compute-budget mismatch with the available hardware. No `decoded/`, `poses/`,
`manifest.json`, or `HANDOFF_DONE` was written — the downstream Vina/strain
chain should **not** be kicked off for TargetDiff until this is resolved.

## What is in place

| Asset | Path |
|---|---|
| Upstream source (pinned shallow clone) | `vendor/targetdiff/` |
| Pretrained diffusion checkpoint | `vendor/targetdiff/pretrained_models/pretrained_diffusion.pt` (33 MB, sha256 to compute on real run) |
| Extra weights (bonus, not needed for C3/C4) | `egnn_pdbbind_v2016.pt`, `pk_reg_para.pkl` in the same dir |
| Sampling driver | `scripts/sample_targetdiff.py` |
| Conda env | `targetdiff` (python 3.8, torch 1.13.1+cu116, pyg 2.2.0, scatter 2.1.0, cluster 1.6.0, rdkit 2022.03.2, + `pyarrow` added for parquet) |

The driver already handles:

- 10 Å pocket crop from `data/external/receptors/<t>/{receptor_clean.pdb, cognate_ligand.pdb}` using the upstream `PDBProtein.query_residues_ligand` helper (caches to `pockets/<t>_pocket10.pdb`)
- Numpy alias patching (`np.long`, `np.bool`, `np.int`, etc. → builtins) because upstream was written against numpy 1.19
- `sample_diffusion_ligand` wrapping, reconstruction (`utils.reconstruct.reconstruct_from_generated`), SMILES canonicalization (`AssignStereochemistryFrom3D` → `MolToSmiles`)
- Per-target dedup (keep first `--n-keep`), parquet + SDF writes with the schema `docs/BASELINES.md §3` requires

Per-target resumability: `decoded/<t>.parquet` existence is checked with `--force` opt-in to re-sample. The driver is safe to re-invoke.

## Why not run

Benchmarked on this workstation's RTX 4060 Ti with full-spec settings
(`--batch-size 100 --num-steps 1000`):

- Per diffusion step on a batch of 100: **~3.5 s**
- Per batch of 100 molecules: ~59 min
- Per target at 1100 raw samples (~10 % recon margin): **~11 h** GPU
- 15 targets sequential: **~165 h (≈ 7 days)** GPU-continuous

The handoff doc `docs/AGENT_HANDOFF_BBAR_TARGETDIFF.md §4` budgeted 25 h GPU
assuming ~6 s/molecule, which is plausible on an A100 class GPU but **6×
optimistic for this 4060 Ti**. Starting the full run here would block the GPU
for the entire week and still not finish before the main session's analysis
stage expects the artefacts.

## Options for closing TargetDiff

1. **Run on a bigger GPU** (A100 / H100). With the driver as-is, a single
   `python scripts/sample_targetdiff.py` should complete the spec in ~25 h
   on an A100. This is the cleanest path. The asset tree above is portable.
2. **Reduce `--n-keep` to 100** (documented as a compute-budget shortfall per
   handoff §2). Wall-clock drops to ~15 h on this hardware. Paired-Wilcoxon
   on top-10 Vina survives fine at n=100; strain distribution comparisons
   also survive. The paper should flag this in the Methods.
3. **Reduce `--num-steps` to 500**. Cuts time in half to ~80 h total but
   deviates from the upstream config and likely degrades reconstruction
   completeness (paper reports steep quality drops below 1000 steps). Not
   recommended.
4. **Drop TargetDiff from the paper**. The handoff calls the baseline
   optional (§2 output tree). BBAR + RxnFlow alone give a two-generator
   comparison. Framing: "pocket-conditional diffusion baselines are out of
   scope for this revision due to compute budget." Weaker, but honest.

## To resume (option 1 or 2)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate targetdiff
# Option 1: full spec, pass a better GPU
CUDA_VISIBLE_DEVICES=0 python -u scripts/sample_targetdiff.py \
    --n-request 1100 --n-keep 1000 \
    > results/logs/sample_targetdiff.log 2>&1 &

# Option 2: reduced n, on this hardware (~15 h overnight)
python -u scripts/sample_targetdiff.py \
    --n-request 120 --n-keep 100 \
    > results/logs/sample_targetdiff.log 2>&1 &
```

Once it finishes, write `results/eval/phase4_baselines/targetdiff/HANDOFF_DONE`
(ISO-8601 timestamp + "targetdiff") and append a line to
`results/logs/baselines_handoff.log`, exactly as for BBAR.

## Smoke test evidence (small model)

A 200-step × batch=20 smoke on ADRB2 completed in 286 s for 40 raw samples
(7.2 s/sample), with reconstruction completeness 2/40 = 5 % (too lossy —
upstream num_steps=1000 is required for reasonable recon). At num_steps=1000
with batch=100 a single batch was measured at 3.52-3.55 s per step over the
first 239 steps before being stopped — sustained rate is stable. See
`scripts/sample_targetdiff.py` for the exact driver.

*Generated 2026-04-19 as part of the BBAR/TargetDiff parallel-agent handoff.
BBAR baseline is complete (`bbar/HANDOFF_DONE`).*
