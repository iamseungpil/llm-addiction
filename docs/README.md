# docs — records

Plans, review notes, cluster job files and earlier maps, kept for the audit trail. They describe
the repository as it was when they were written: read old paths in them through
[`../PATH_MAP.md`](../PATH_MAP.md) (the tag `pre-reorg` holds that layout). The current entry point
is the [top-level README](../README.md).

| Path | What it is |
|---|---|
| `rebuttal_review/` | NeurIPS 2026 review period: reviews received, posted replies, LaTeX of the rebuttal, verified facts, and **`CAMERA_READY_MAP.md`** (every rebuttal promise mapped to the camera-ready) |
| `plans/`, `PLAN_4NODE_EXECUTION_2026_05_07.md`, `PLAN_TRACK0_W3_v5.md`, `PLAN_TRACK_L_LENGTH_CONFOUND_v1.md` | Experiment plans written before each run |
| `amlt/2026_05_07/` | Cluster job files for the May 2026 additional controls (track0, m1, m2, m5, d) |
| `drafts/` | A draft restructuring of the discussion section |
| `PAPER_CANONICAL_CODE.md` | Earlier figure → code → data map (pre-camera-ready numbering); superseded by the README and `PATH_MAP.md` |
| `MANIFEST.md`, `STRUCTURE.md` | Earlier claim → file map and repository structure notes |
| `EXPERIMENT_DESIGN_COMPARISON.md`, `ev_transparency_gambling_avoidance.md`, `token_truncation_root_cause_analysis.{md,pdf}` | Analyses from earlier stages |
| `SLURM_GUIDE.md` | Batch-job notes for the HPC cluster |
| `investment_choice_bet_constraint_cot/code_review_report.pdf` | A code review of the CoT investment-choice runner (now `archive/legacy/investment_choice_bet_constraint_cot/`) |

The design specs of the steering harness are with that experiment, in
[`experiments/08_steering/docs/`](../experiments/08_steering/docs/).
