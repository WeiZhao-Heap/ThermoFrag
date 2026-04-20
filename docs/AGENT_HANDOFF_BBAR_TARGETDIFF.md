# Parallel-agent handoff — BBAR + TargetDiff baselines

**Audience**: another Claude Code agent, running **in parallel** with the current
session. Your job is the **GPU-bound** baseline work for the C3/C4 generator-
vs-generator arm of ThermoFrag. The current session is running the **CPU-bound**
RxnFlow Vina pipeline and will handle its own strain + stats at the end. Stay
off the CPU; stay off the `rxnflow` and `py310` conda envs.

Date: 2026-04-19.

## 0. Before you touch anything

Read in order (cheap, <5 min):

1. `docs/PLAN.md` §C3/C4 — scientific claims you are trying to close
2. `docs/BASELINES.md` §§1-3 — the original spec for RxnFlow/BBAR/TargetDiff
3. `results/eval/claim_summary.md` — current verdict state
4. `~/.claude/projects/-home-chaoxue-code-ThermoFrag/memory/project_phase{4,5}_status.md` — ignore any "upstream-blocked" language, that was wrong
5. `scripts/sample_rxnflow.py` — reference implementation you should mirror

## 1. Current state — do not disturb

**Running processes** (will finish in ~20 h):

| PID-ish | Command | Env |
|---|---|---|
| sample_rxnflow.py | FINISHED — 15 decoded parquets at `results/eval/phase4_baselines/rxnflow/decoded/` |  |
| dock_vina.py | RUNNING — 15 × 1000 ligands × 15 CPU workers (~10 h) | `py310` |
| eval_strain.py | chained after Vina (~7 h CPU) | `py310` |
| eval_generator_vs_generator.py | chained after strain (<1 min) | `py310` |
| chain script | `scripts/run_rxnflow_baseline_pipeline.sh` background | bash |

**Rules**:

- **Do not kill or restart** any process whose command matches `dock_vina.py`, `eval_strain.py`, `eval_generator_vs_generator.py`, or `run_rxnflow_baseline_pipeline.sh`.
- **Do not modify** any file under `results/eval/phase4_baselines/rxnflow/`. That tree is "owned" by the current session.
- **Do not write** to `claim_summary.md`, `claim_summary.json`, or `project_phase{4,5}_status.md` in memory. The current session will update them once all three baselines are done.
- **Do not install anything into** the existing `rxnflow` or `py310` conda envs. Make new envs.
- **CPU budget**: you may use up to **1 core** for light tasks (script-level). Anything heavier (Vina, strain, CPU training) is forbidden until the current session's pipeline log shows `[pipeline] DONE`.
- **GPU budget**: the GPU is free for you. Only one process should use it at a time — sequence BBAR, then TargetDiff.

Check current state at the start of each of your sessions:

```bash
tail -20 results/logs/rxnflow_pipeline.log
pgrep -af "sample_rxnflow|dock_vina|eval_strain|eval_generator_vs|run_rxnflow_baseline"
```

If `[pipeline] DONE` already appears in the log, the current session is done and
you are free to run Vina + strain on your own decoded parquets using the
ThermoFrag `py310` env. Otherwise keep off the CPU.

## 2. Scope — what to produce

Two baseline trees parallel to the RxnFlow one:

```
results/eval/phase4_baselines/bbar/
  decoded/<target>.parquet        # target, chain_idx, smiles, condition
  manifest.json
results/eval/phase4_baselines/targetdiff/
  decoded/<target>.parquet        # target, chain_idx, smiles
  poses/<target>.sdf              # 3D poses (TargetDiff generates them)
  manifest.json
```

Schema for each `decoded/<target>.parquet` (strict — dock_vina.py / eval_strain.py
depend on it):

- `target` (str) — matches `data/external/receptors/<target>/` dir name
- `chain_idx` (int) — 0..N-1, unique within target
- `smiles` (str) — canonical RDKit SMILES, sanitized
- extra columns allowed (e.g. `condition`, `proxy_score`) but must not break
  parquet type inference

Each parquet must have **≥ 1000** valid rows (upstream may need to be asked for
more to compensate for RDKit canonicalization loss). If you cannot produce
1000 for a target, document the shortfall in `manifest.json` but still write
whatever you did get — do not silently fail.

## 3. BBAR — do this first (~2 h GPU + ~30 min env setup)

**Upstream**: https://github.com/SeonghwanSeo/BBAR

### 3.1 Env

BBAR's `environment.yml` is likely torch-1.x. **Do not** install into the
existing `rxnflow` env. Create a fresh `bbar` conda env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
git clone https://github.com/SeonghwanSeo/BBAR vendor/bbar_upstream
cd vendor/bbar_upstream
# read their environment.yml or requirements.txt; build a yaml named
# vendor/bbar_upstream/environment.yml with conda env name "bbar"
# If upstream doesn't provide one, write one based on README's pin list.
conda env create -f environment.yml   # or equivalent
```

Validate: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` must show CUDA True.

### 3.2 Weights

```bash
cd vendor/bbar_upstream
sh ./download-weights.sh
ls test/pretrained_model/   # expect: {mw,logp,tpsa,qed,3cl-affinity}.tar
```

If the script fails (likely DNS), pre-clone / pre-download outside the repo
then place manually. Fall back to `gdown` on the same IDs if provided in the
repo's download script.

### 3.3 Protocol (per BASELINES.md §2)

BBAR has no pocket input. Fair protocol: generate **one pool per condition**,
dock the same pool against every pocket.

Conditions used: **`qed`** and **`logp`**. Rationale: these are the two
conditions whose y-targets we already compute for ThermoFrag. Values come from
`results/eval/phase4/litpcba_targets/<t>/y_raw.npy` aggregated over the 15
targets (use the mean across all 15 for the "drug-like" pool).

Steps:

1. Load each `.tar` checkpoint with BBAR's sampling driver (see `vendor/bbar_upstream/README.md`
   — likely a `sample.py` or `generate.py` script; the exact flag name is on
   them, but the pattern is `--ckpt <tar> --condition <value> --num <N>`).
2. For each of `{qed=0.75, logp=2.5}` (adjust if aggregated mean differs):
   - Generate 1500 molecules.
   - Canonicalize via RDKit, drop failures, dedup.
   - Keep first 1000.
3. For each LIT-PCBA target `t ∈ TARGETS_15`:
   - Write `decoded/<t>.parquet` with *both* pools concatenated, one row per
     molecule. Add column `condition ∈ {qed, logp}` so both are separable.
     `chain_idx` runs 0..1999 within the file. All 15 files carry the same
     rows — that is by design and matches `docs/BASELINES.md` §2.

`TARGETS_15` = `["ADRB2","ALDH1","ESR_ago","ESR_antago","FEN1","GBA","IDH1","KAT2A","MAPK1","MTORC1","OPRK1","PKM2","PPARG","TP53","VDR"]`.

### 3.4 Write `scripts/sample_bbar.py`

Mirror `scripts/sample_rxnflow.py` for style. It must accept `--out-dir`,
`--targets`, `--n-keep`, `--force`, and write a `manifest.json` with:

- `baseline: "bbar"`
- `checkpoints: {qed: <path + sha256>, logp: <path + sha256>}`
- `conditions: {qed: <value>, logp: <value>}`
- `n_pool_per_condition`
- `per_target`: stats dict
- `note`: "BBAR has no pocket input; the same SMILES pool is scored against all
  15 pockets. Pairing is fair because both BBAR and ThermoFrag are pocket-agnostic
  at generation time (pocket enters only at Vina)."

### 3.5 Exit criterion

```bash
ls results/eval/phase4_baselines/bbar/decoded/ | wc -l   # must be 15
python -c "import pandas as pd; df = pd.read_parquet('results/eval/phase4_baselines/bbar/decoded/ADRB2.parquet'); \
    assert len(df) >= 1000 and df['smiles'].notna().all()"
test -f results/eval/phase4_baselines/bbar/manifest.json
```

**Do NOT** run Vina or strain on the BBAR pool yet. The current session's
chain-script or a later session will do that.

## 4. TargetDiff — only if BBAR is fully green (~25 h GPU + ~4 h env setup)

**Upstream**: https://github.com/guanjq/targetdiff
**Weights**: Google Drive folder https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz

This one is the gnarliest. It needs CUDA-specific `torch-scatter` / `torch-cluster`
wheels compiled against a specific torch version — TargetDiff's repo pins an older
torch+CUDA combo. Plan on burning 4 h human-time fighting the env. Build a fresh
`targetdiff` conda env; do not share with `rxnflow`.

### 4.1 Protocol (per BASELINES.md §3)

Per target `t`:

1. Prepare pocket PDB by cropping a 10 Å sphere around `data/external/receptors/<t>/cognate_ligand.pdb`. Upstream `scripts/data_preparation/` helpers exist.
2. Run pocket-conditional sampling for ≥ 1100 molecules.
3. Convert 3D output → canonical SMILES (`Chem.MolFromMolFile(...)` then `Chem.MolToSmiles(...)`, apply `Chem.AssignStereochemistryFrom3D` first).
4. Also write the 3D SDF at `results/eval/phase4_baselines/targetdiff/poses/<t>.sdf`.
5. Dedup, keep first 1000.
6. Write parquet with columns `target, chain_idx, smiles`.

### 4.2 Write `scripts/sample_targetdiff.py`

Same style as sample_rxnflow.py / sample_bbar.py. Write a `manifest.json` with checkpoint SHA256 and env notes.

### 4.3 Exit criterion

Same parquet / manifest checks as BBAR. Additionally each `poses/<t>.sdf` must
contain ≥ 1000 conformers.

## 5. Hand-back signal

When you finish BBAR (and optionally TargetDiff), write:

```
results/eval/phase4_baselines/<b>/HANDOFF_DONE
```

… a one-line file containing an ISO-8601 timestamp and the baseline name.
Also append a line to `results/logs/baselines_handoff.log` noting what
you did:

```
2026-04-19T14:30:00+08:00  bbar DONE  decoded=15 files, pool_size=2000, manifest=yes
```

This is how the current session knows to start Vina on your parquets.

## 6. Do NOT

- Do NOT run Vina, strain, or `eval_generator_vs_generator.py`. Those are the
  current session's responsibility so they are chained with TF-reference data.
- Do NOT write to `results/eval/phase4_baselines/rxnflow/`.
- Do NOT rebuild `results/eval/phase4/` — that is the ThermoFrag reference.
- Do NOT `git commit`, `gh` anything, or touch `CLAUDE.md` / `claim_summary.md` / `project_phase*` memory files.
- Do NOT copy-paste this handoff doc content into the paper draft. It is ops-only.
- Do NOT invent filenames outside the scheme in §2 — the downstream stats
  pipeline hardcodes the path structure.

## 7. What "done" looks like to the coordinating agent

```
results/eval/phase4_baselines/
  rxnflow/{decoded,vina,strain}/*.parquet   # owned by current session
  bbar/{decoded}/*.parquet                  # owned by you
  targetdiff/{decoded,poses}/*.{parquet,sdf}  # owned by you (optional)
  */manifest.json                            # every baseline
  */HANDOFF_DONE                             # your signal files
results/logs/baselines_handoff.log           # your append-only notes
```

Once this tree exists, I will run Vina + strain on BBAR/TargetDiff and re-run
`eval_generator_vs_generator.py` with `--baselines rxnflow bbar targetdiff` to
produce the final C3/C4 result and update memory + claim_summary.

Good hunting.
