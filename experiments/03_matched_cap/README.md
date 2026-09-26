# 03 — Matched cap

**Question.** Is the bet-size effect the freedom to choose, or just the larger range of bets a
choosing model can reach? Here both arms share the same maximum bet.

**Paper result.** Finding 4 (second half), Figure 3d; appendix Figure 18 and tables
`tab:matched-cap-intervals`, `tab:matched-cap-companion`. The choosing arm goes broke more often at
matched caps: for GPT-4o-mini, variable-arm bankruptcy is about 15/16/18% at $30/$50/$70 over all
1,600 games per cap, and on the three API models ten of twelve cap × model intervals clear zero
with none reversed.

## Design (from the code)

- Same slot machine as [01](../01_slot_machine/README.md): $100 bankroll, win probability 0.30,
  payout 3.0×.
- **First run, GPT-4o-mini** (`sm_cap_ablation/src/`): fixed arm bets exactly $30, $50 or $70 for
  up to 100 rounds; variable arm bets $5 to the cap ($10, $30, $50, $70) for up to 50 rounds; all
  32 prompt combinations × 50 games per cap; temperature 0.7, system message "rational decision
  maker", no role sentence. The variable arm was finished by the per-cap restart runners in
  `gpt_variable_max_bet_experiment/src/gpt_variable_restart_{10_30,50_70}.py`.
- **Three API models** (`track0_w3_replication/`): GPT-4o-mini, GPT-4.1-mini and Gemini-2.5-Flash
  (Claude-3.5-Haiku has been withdrawn from its API); caps $10/$30/$50/$70; prompts `BASE` and
  `GMPRW` (the paper's GMHWP); fixed vs variable; 50 games per cell; driver `src/run_mc_ladder.sh`.
  These runs carry the role sentence, as the paper discloses.
- Both harnesses share one parser, `sm_cap_ablation/src/improved_gpt_parsing.py`;
  `track0_w3_replication/tests/test_protocol_parity.py` checks the prompts and parser against the
  first run.

## Quick Start

```bash
# First run (GPT-4o-mini; GPT_API_KEY or OPENAI_API_KEY). Each script runs its full grid;
# gpt_variable_max_bet_experiment/src/gpt_variable_restart_{10_30,50_70}.py resume unfinished variable cells.
python experiments/03_matched_cap/sm_cap_ablation/src/gpt_fixed_bet_size_experiment.py
python experiments/03_matched_cap/sm_cap_ablation/src/gpt_variable_max_bet_experiment.py

# Three API models, one cell (OPENAI_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY)
python experiments/03_matched_cap/track0_w3_replication/src/run_track0_api.py \
    --provider openai --model_id gpt-4o-mini --cap 70 --mode variable \
    --prompt_combo GMPRW --persona --n_games 50 --output_dir out/mc32
# Whole grid, as run for the paper (edit the paths at its top first)
bash experiments/03_matched_cap/track0_w3_replication/src/run_mc_ladder.sh

# Parity and smoke tests (no API calls)
python -m pytest experiments/03_matched_cap/track0_w3_replication/tests -q
```

At the reorganisation commit these tests give 75 passed, 3 failed: the three OpenAI protocol
tests expect the token limit of the first run (600 for GPT-4o/GPT-4o-mini, 1024 for
GPT-4.1-mini) where `run_track0_api.py` now sends 2048. Before the move they did
not run at all (their paths pointed at the original machine).

## Folder contents

| Path | What it is | HF folder |
|---|---|---|
| `sm_cap_ablation/` | GPT-4o-mini fixed and variable runners and the shared parser | `analysis/fixed_variable_comparison/` |
| `gpt_variable_max_bet_experiment/` | Per-cap restart runners that completed the variable arm (`restart_complete_*.json`), plus the analysis and figures of the time | `analysis/fixed_variable_comparison/gpt_variable_max_bet_results/` |
| `gpt_fixed_bet_size_experiment/` | README and run log of the fixed arm (the runner itself is in `sm_cap_ablation/src/`) | `analysis/fixed_variable_comparison/gpt_fixed_bet_size_results/` |
| `track0_w3_replication/` | Three-API-model harness, driver, analysis and tests; `experiments/05_framing_worked_example` reuses its `src/` | `rebuttal_neurips_2026/matched_cap_mc32/` |

The dataset also holds other runs of the three-model harness (`experiments/track0_w3/`,
`rebuttal_neurips_2026/track0_fixed_arm_rerun/`); the paper's matched-cap panel reads
`rebuttal_neurips_2026/matched_cap_mc32/`.

The restart runners import `improved_gpt_parsing` from `/home/ubuntu/llm_addiction`, as on the
original machine; the repository copy is `sm_cap_ablation/src/improved_gpt_parsing.py`.

## Figures and tables (paper repository, private)

- `scripts/figures/fig05_matched_cap.py` (with `scripts/build_figure_data.py::build_cap_ablation`)
  builds `paper_data/fig05_matched_cap.json`;
  `scripts/figures/fig03_investment_choice_1x4.py` draws it as Figure 3d.
- `scripts/figures/figA_matched_cap_forest.py` — appendix Figure 18.
