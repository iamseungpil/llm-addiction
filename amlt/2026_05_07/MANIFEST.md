# AMLT 2026-05-07 — NeurIPS Rebuttal Launch Manifest

## Files

```
amlt/2026_05_07/
├── shared/
│   ├── bootstrap_addiction_node.sh   — env setup, HF tarball pull, gpu_keeper + push_ckpts launch
│   ├── gpu_keeper.py                  — defeats BSC ~17-min idle suspend (heartbeat 60s)
│   └── push_ckpts_to_hf.py            — periodic HF sync (interval 600s)
├── track0.yaml        — Track 0 W3 cross-model matched-cap (Gemma+LLaMA, GPU)
├── track0_api.yaml    — Track 0 W3 (gpt-4o-mini, claude-haiku, gemini-flash, gpt-4o re-baseline)
├── m2.yaml            — Track A1 M2 persona-decoupling (Gemma+LLaMA, GPU)
├── m2_api.yaml        — Track A1 M2 (4 API models)
├── m5.yaml            — Track A2 M5 compliance residualisation (cheap, GPU for SAE re-encode)
├── m1.yaml            — Track B M1 portfolio discriminant (Gemma+LLaMA, GPU)
├── m1_api.yaml        — Track B M1 (4 API models)
├── d.yaml             — Track D top-K removal (cheap, GPU)
└── MANIFEST.md        — this file
```

## Pre-submit checklist

- [ ] `code_snapshot/2026_05_07/code.tar.gz` uploaded to HF dataset `iamseungpil/llm-addiction-rebuttal-2026-05/code_snapshots/2026_05_07/`. Contents: full `llm-addiction/` repo including `paper_experiments/`, `sae_v3_analysis/`, `amlt/2026_05_07/`.
- [ ] HF dataset repo `iamseungpil/llm-addiction-rebuttal-2026-05` created (private OK; bot pushes results there).
- [ ] AMLT secrets configured: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` available as env vars on the API yamls (`$$VAR` substitution).
- [ ] Pre-registration committed to git: Plan v4 frozen at this commit; deviation log §13 starts here.
- [ ] Live smoke run on local Azure VM (5 games × 1 cell) confirms each runner produces valid JSON output before submitting full grid.

## Submission order (Plan v4 §6.2)

### Day 0 (2026-05-08)
- Submit `track0.yaml` (Node 1)
- Submit `track0_api.yaml` (alongside Node 1, no extra GPU need)
- Submit `m2.yaml` (Node 2)
- Submit `m2_api.yaml` (alongside Node 2)
- Submit `m5.yaml` (Node 3)
- Total nodes used: 3 GPU jobs (track0, m2, m5) + 2 API jobs sharing GPU node
- Save 1 GPU node (Node 4) as spare for failures or M3 instruction-strength appendix work

### Day 1–2
- Track 0, M2, M5 run in parallel
- Estimated wall: ~36-48h for Track 0 GPU (16 cells × 200 games × ~15-20 min/cell on H100); APIs may finish earlier
- M2: ~40-50h GPU (16 cells × 200 games)
- M5: ~3-4h total (extraction + residualise + refit + analyze)

### Day 3 — gating readout
- Run `analyze_track0.py` (already in track0.yaml command)
- Run `analyze_m2.py` (already in m2.yaml command)
- Run `analyze_m5.py` (already in m5.yaml command)
- Decision rule (Plan v4 §1bis.4 + §2.4 + §3.4):
  - Track 0 primary CI > 0 + ≥4/6 individual models pass → invoke `claim_surgery_§3.2_branches.md::W3-passes`
  - Track A1 (M2) primary CI > 0 → invoke `claim_surgery_M2_outcome_branches.md::M2-passes`
  - Track A2 (M5) joint residualisation passes → invoke `claim_surgery_M5_outcome_branches.md::M5-passes`
  - If any fail: invoke matching surgery branch and skip Track B / narrow paper claim accordingly.

### Day 4-6
- If Track 0 + A both clean: submit `m1.yaml` + `m1_api.yaml` (Track B portfolio discriminant)
- Wait for ~36-48h
- Run `analyze_m1.py` (already in m1.yaml command)
- Decision rule: Track B primary CI > 0 → M1-passes; else M1-mixed/M1-fails

### Day 6-7 (any time)
- Submit `d.yaml` (cheap; ~30 min)
- This runs regardless of Track 0/A/B outcome
- Final consolidation: gather all `_analysis.json` + `_sanity.md` outputs from `/scratch/x3415a02/data/llm-addiction/<module>/`
- Generate specification-curve appendix (Plan v4 §9ter) — runs locally on cached features, not on AMLT

## Submission commands (from local shell)

```bash
cd /home/v-seungplee/llm-addiction
amlt run --config amlt/2026_05_07/track0.yaml      track0_w3 -t msrresrchbasicvc
amlt run --config amlt/2026_05_07/track0_api.yaml  track0_w3_api -t msrresrchbasicvc
amlt run --config amlt/2026_05_07/m2.yaml          m2_persona -t msrresrchbasicvc
amlt run --config amlt/2026_05_07/m2_api.yaml      m2_persona_api -t msrresrchbasicvc
amlt run --config amlt/2026_05_07/m5.yaml          m5_residualisation -t msrresrchbasicvc
# (later, only if Day 3 gate passes)
amlt run --config amlt/2026_05_07/m1.yaml          m1_portfolio -t msrresrchbasicvc
amlt run --config amlt/2026_05_07/m1_api.yaml      m1_portfolio_api -t msrresrchbasicvc
amlt run --config amlt/2026_05_07/d.yaml           d_robustness -t msrresrchbasicvc
```

## Key constraints baked into yamls (verify before submit)

- `target.service: sing` + `target.name: msrresrchbasicvc` + `workspace_name: msra-sh-aml-ws` — per `feedback_amlt_tier_h200_h100.md`
- `sku: 80G4-H100` (NOT H200) + `sla_tier: standard` — Standard tier avoids preemption
- `max_run_duration_seconds: 604800` (7 days) — per Plan v4 §6
- `bash /scratch/code/llm-addiction/amlt/2026_05_07/shared/bootstrap_addiction_node.sh` — every yaml starts with bootstrap
- `sleep 86400` at end of every command list — per `feedback_amlt_job_preservation.md` (outer foreground exit = job pass)
- `gpu_keeper.py` background — per `feedback_bsc_idle_suspend.md` (~17-min idle suspend defeat)
- `push_ckpts_to_hf.py` background — per `feedback_hf_sync.md` (results to HF)
- NO apostrophes in `bash -c '...'` blocks — per `feedback_yaml_apostrophe_quote.md`. All quoting uses `\"` for inner double-quotes; condition names use `pG`/`pM`/`pGM` placeholder mapped to `+G`/`+M`/`+GM` via `sed` to avoid yaml `+` parsing issues.
- HF tarball pull pattern (NOT SSH-base64) — per `feedback_hf_bootstrap.md`

## Recovery patterns (per `reference_amlt_submit.md`)

- If yaml validation fails: `python -c "import yaml; yaml.safe_load(open('amlt/2026_05_07/<file>.yaml'))"` to surface error
- If `amlt run` fails with "permission denied": SC-Alt account may need refresh; `az login`
- If a job sibling fails due to a code bug: do NOT cancel queued siblings; patch HF tarball + resubmit only the failed yaml (per `feedback_amlt_queue.md`)
- If outer command exits before grid completes: AMLT marks job passed (per `feedback_amlt_job_preservation.md`); do NOT kill orchestrator. Re-run only the missing cells.

## Output paths (per Plan v4 + CLAUDE.md)

```
/scratch/x3415a02/data/llm-addiction/
├── track0_w3/
│   ├── final_<model>_cap{cap}_{mode}_<timestamp>.json
│   ├── _analysis.json
│   └── _sanity.md
├── m2_persona/
│   ├── final_<model>_<condition>_<framing>_SM_<timestamp>.json
│   ├── _analysis.json
│   └── _manipulation_check.md
├── m5_residualisation/
│   ├── directions/direction_<model>_<direction>.npz
│   ├── delta_g_dp_baseline_<model>.json
│   ├── residualised/<model>_<mode>.npz
│   └── _analysis.json
├── m1_portfolio/
│   ├── final_<model>_<condition>_<objective>_<blurb>_<timestamp>.json
│   ├── _analysis.json
│   └── _sanity.md
└── d_robustness/
    ├── d_<model>_<top|random>_K{K}_seed<seed>.json
    └── _analysis.json
```

All outputs auto-pushed to HF dataset `iamseungpil/llm-addiction-rebuttal-2026-05/results/` at 600s intervals via `push_ckpts_to_hf.py`.
