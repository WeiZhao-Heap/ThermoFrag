# MIGRATION / RESUME-ON-NEW-MACHINE

Self-contained guide for bringing ThermoFrag up on a new machine (e.g. a 3090
workstation) and continuing the paper work. Written 2026-04-20 during the final
hours of the RxnFlow C3/C4 arm on the original 4060 host.

Everything needed is in-tree except two external sources that must be downloaded
fresh (they are too large to track):
- ThermoFrag checkpoints under `results/checkpoints/` — **bring over via rsync**
- Processed LMDBs under `data/processed/` — **bring over via rsync**

PharmacoNet + RxnFlow weights are already bundled under `vendor/_cached_weights/`
and `vendor/rxnflow/weights/` so they travel with the repo.

---

## 1. What works today, what is half-done

Pipeline as of 2026-04-20 ~18:50 on the 4060 machine:

| Phase | State | Location |
|---|---|---|
| Phase 1 (QM) | C1 ✓ | `results/checkpoints/qm_recalibrated_best_large.pt` |
| Phase 2 (coupling) | ✓ | `results/checkpoints/coupling_final.pt` |
| Phase 3 (joint) | C2 ✓ | `results/checkpoints/joint_final.pt` |
| Phase 4 (downstream TF) | C5 ✓, C3 vs LIT-PCBA ref ✓ | `results/eval/phase4/`, `results/eval/phase5/` |
| **RxnFlow baseline** | decoded ✓, Vina ✓ (15/15), strain **14/15 — VDR in flight** | `results/eval/phase4_baselines/rxnflow/` |
| BBAR baseline | not started | — |
| TargetDiff baseline | not started | — |
| Generator-vs-generator stats | not started (blocked on strain + BBAR) | — |

See `results/eval/claim_summary.md` for the C1–C6 verdict table, and
`~/.claude/projects/-home-chaoxue-code-ThermoFrag/memory/` for the machine-local
memory log (not part of the repo; rebuild as needed on the new machine).

---

## 2. Tree layout that matters for migration

```
ThermoFrag/
├── docs/                       ← read these first
│   ├── PLAN.md                  scientific claims
│   ├── METHOD.md                math
│   ├── FIGURES.md               figure spec
│   ├── BASELINES.md             baseline execution plan (RxnFlow/BBAR/TargetDiff)
│   ├── AGENT_HANDOFF_BBAR_TARGETDIFF.md   task doc for a parallel agent
│   └── MIGRATION.md             THIS FILE
├── scripts/                    ← all entry points
│   ├── sample_rxnflow.py
│   ├── dock_vina.py
│   ├── eval_strain.py
│   ├── eval_generator_vs_generator.py
│   ├── run_rxnflow_baseline_pipeline.sh
│   ├── build_rxnflow_env.sh
│   └── setup_new_machine.sh      ← run this once after cloning
├── vendor/
│   ├── rxnflow/                  upstream RxnFlow (cloned, pyproject editable)
│   │   ├── environment.yml       templated with {{PROJECT_ROOT}}
│   │   ├── weights/              qvina-unif-0-64 checkpoint (27 MB, in-tree)
│   │   └── data/envs/zincfrag/   env_dir skeleton (smi + template in-tree;
│   │                             bb_fp_*.npy/bb_mask.npy gitignored — rebuild
│   │                             via scripts/build_rxnflow_env.sh, ~1 min)
│   ├── pharmaconet/              upstream PharmacoNet (cloned)
│   ├── bbar_upstream/            upstream BBAR (cloned, weights force-added)
│   ├── targetdiff/               upstream TargetDiff (cloned, staged env + NOT_RUN.md)
│   ├── bbar_fragmentation/       BBAR fragmentation utils (used by phase-2)
│   └── _cached_weights/pmnet/    tacogfn_proxy ckpt in-tree (18 MB).
│                                 pmnet.tar (139 MB) gitignored — pmnet_appl
│                                 auto-downloads it on first run
├── data/external/receptors/<t>/ 15 LIT-PCBA receptors (ship with repo)
├── data/processed/               LMDBs — bring via rsync, they are ~25 GB
├── results/                     checkpoints + eval parquets + figures
│   ├── checkpoints/              bring via rsync (~3 GB)
│   ├── eval/phase{1..5}/         TF results (already in-tree if committed)
│   └── eval/phase4_baselines/   baseline outputs
│       └── rxnflow/              in-flight: decoded ✓, vina ✓, strain partial
└── src/thermofrag/              the model code
```

---

## 3. Bootstrap on a new machine

### 3.1 Clone and create the main ThermoFrag env

```bash
# On the new 3090 machine:
git clone <your-fork-or-rsync> ~/code/ThermoFrag
cd ~/code/ThermoFrag

# Bring the heavy binary stuff over (if not in git):
rsync -av old-host:~/code/ThermoFrag/results/checkpoints/        results/checkpoints/
rsync -av old-host:~/code/ThermoFrag/data/processed/             data/processed/

# Main ThermoFrag env (py310 — torch+PyG+RDKit+meeko+openmm+Vina stack)
conda env create -f environment.yml    # the repo-root yaml
conda activate thermofrag                # name per environment.yml
```

If the repo-root `environment.yml` points at `thermofrag` but your shell
expects `py310`, symlink or rename; the code does not care about the env
name, only about finding the binaries.

### 3.2 Run the migration helper for the baseline side

```bash
bash scripts/setup_new_machine.sh
```

What it does:
1. Replants cached PharmacoNet weights from `vendor/_cached_weights/pmnet/`
   into `~/.local/share/pmnet/` (where `pmnet_appl` expects them).
2. Renders `vendor/rxnflow/environment.yml` with your actual `$PWD` as
   `{{PROJECT_ROOT}}` and builds a fresh `rxnflow` conda env from it.
3. Builds the `zincfrag` env_dir if not already present (skips if
   `vendor/rxnflow/data/envs/zincfrag/.done` exists — it should, since the
   dir travels with the repo).
4. Runs an import smoke test (`torch`, `rxnflow`, `pmnet_appl`, CUDA).

Sub-modes:
- `scripts/setup_new_machine.sh --weights-only` — only replant weights.
- `scripts/setup_new_machine.sh --env-only` — only build rxnflow env.

### 3.3 Smoke-test everything

```bash
# ThermoFrag env check
conda activate thermofrag       # or `py310`
python -c "import thermofrag, openmm, rdkit; print('tf ok')"
which vina mk_prepare_ligand.py     # both must resolve; vina must be GPU-capable host

# RxnFlow env check
~/miniconda3/envs/rxnflow/bin/python -c "import torch, rxnflow, pmnet_appl; \
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

---

## 4. Resume the RxnFlow pipeline

### 4.1 If the original strain run did NOT finish before migration

```bash
# Only re-runs the missing target(s) because eval_strain.py overwrites per-target parquet
conda activate thermofrag
python scripts/eval_strain.py \
  --decoded-dir results/eval/phase4_baselines/rxnflow/decoded \
  --out-dir     results/eval/phase4_baselines/rxnflow/strain \
  --targets VDR                  # or whichever is missing
```

### 4.2 Run the C3/C4 generator-vs-generator stats

```bash
python scripts/eval_generator_vs_generator.py --baselines rxnflow
```

Writes:
- `results/eval/phase5/c3_vs_generators.csv`
- `results/eval/phase5/c4_vs_generators.csv`
- `results/eval/phase5/c3_c4_summary.json`
- `results/eval/phase5/c3_c4_bars.png`

Acceptance thresholds are in `docs/BASELINES.md` §"Statistical analysis".

---

## 5. Continue with BBAR + TargetDiff on the 3090

`docs/AGENT_HANDOFF_BBAR_TARGETDIFF.md` is the full spec for those two
baselines. Delegate it to an agent (or run it yourself) — it outputs to
`results/eval/phase4_baselines/{bbar,targetdiff}/` in the same schema as
RxnFlow's. Once done, re-run:

```bash
python scripts/eval_generator_vs_generator.py --baselines rxnflow bbar targetdiff
```

---

## 6. Known fixes already applied — do NOT re-introduce

- **PATH bug in baseline pipeline**: `scripts/run_rxnflow_baseline_pipeline.sh`
  was initially invoking `~/miniconda3/envs/py310/bin/python` directly.
  That does **not** activate the env, so `subprocess.run(["mk_prepare_ligand.py",...])`
  inside `dock_vina.py` failed with `FileNotFoundError`. The fix is to
  `source ~/miniconda3/etc/profile.d/conda.sh; conda activate py310`
  before running. Any new baseline pipeline script must do the same.
- **RxnFlow env_dir substitution**: the upstream `qvina-unif-0-64` checkpoint
  was trained on the Enamine Catalog (private). We substitute the public
  **ZINCFrag-200k** building-block library. The env_dir ships pre-built under
  `vendor/rxnflow/data/envs/zincfrag/`. Documented in every RxnFlow
  manifest.json. Do not treat as a bug.
- **First 2 Vina parquets were garbage** (before the PATH fix) — deleted,
  re-run. If you see ADRB2 / ALDH1 with all `embed_failed` + `mk_prepare_ligand.py
  not found` rows, they are stale.
- **python 3.12 vs 3.10 mismatch**: `rxnflow` env uses Python 3.12 (pin
  from RxnFlow pyproject). The main ThermoFrag `thermofrag/py310` env is
  Python 3.10. These cannot share site-packages. The two envs are
  intentionally split; all sampling runs in `rxnflow`, all Vina/strain/stats
  runs in the ThermoFrag env.

---

## 7. What a fresh agent should read in order

1. `docs/PLAN.md` — the scientific claims C1–C6
2. `docs/BASELINES.md` — RxnFlow/BBAR/TargetDiff spec + acceptance
3. `docs/MIGRATION.md` — this file (state + bootstrap)
4. `docs/AGENT_HANDOFF_BBAR_TARGETDIFF.md` — if tasked with BBAR/TargetDiff
5. `results/eval/claim_summary.md` — latest verdict
6. `scripts/sample_rxnflow.py` — reference implementation style

After that, you should be able to continue the work unambiguously.
