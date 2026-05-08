# Track A1 / M2 — Persona-decoupling

Plan v4 priority 2. Reviewer attack: "+G measures persona/role uptake, not propensity." This module tests whether the +G prompt effect persists in first-person framing or only under role-play framing.

## Intent

Replicate the §3.1 slot-machine experiment on six models under two framings:

- `first_person`: ROLE_INSTRUCTION-style behavioral-economics participant.
- `role_play_gambler`: explicit role-play preamble ("imagine you are a gambler at a casino, trying to maximize your profits").

If +G survives the switch from role-play to first-person, the +G effect is propensity-like. If +G only fires under role-play, the effect is theatrical role uptake.

## Hypothesis (Plan §2.2)

```
H_M2_main:  Δ_{+G, first_person} − Δ_{+G, role_play_gambler}  >  0
            where Δ_{cond, frame} = E[risk | cond, frame] − E[risk | BASE, frame]
```

risk metric = bankruptcy rate (Stage 1 primary). Decision rule: lower 95% Wald CI > 0.

## Primary contrast

The single pre-registered test: Δ_{+G, first} − Δ_{+G, role}, fitted under the mixed-logit `risk_metric ~ condition * framing + (1 | model)` (formula in `configs/m2_config.yaml`). Cluster-robust SEs by (model, game_id). bambi/pymc soft-imported; bootstrap fallback when unavailable.

## Files

```
m2_persona_decoupling/
├── __init__.py
├── README.md
├── claim_surgery_M2_outcome_branches.md   # rebuttal §3.2 prose for 3 outcomes
├── configs/
│   └── m2_config.yaml                      # pre-registered grid + bambi formula
├── src/
│   ├── __init__.py
│   ├── prompts.py                          # framing prefix builders
│   ├── run_m2_open_weight.py               # Gemma / LLaMA bf16 launcher
│   ├── run_m2_api.py                       # OpenAI / Anthropic / Google launcher
│   ├── analyze_m2.py                       # mixed-logit + bootstrap primary CI
│   └── sanity_checks.py                    # gambling-keyword manipulation check
└── tests/
    ├── __init__.py
    └── test_m2_smoke.py                    # synthetic + GPU-skip + API-skip smokes
```

Reused (NOT reimplemented): `track0_w3_replication.src.game_logic.{SlotMachineGame, run_single_game, parse_response, create_prompt, ROLE_INSTRUCTION}`.

## Smoke commands

```bash
# Open-weight (requires CUDA)
python paper_experiments/m2_persona_decoupling/src/run_m2_open_weight.py \
  --model gemma --gpu 0 --condition +G --framing first_person --task SM \
  --n_games 5 --output_dir /tmp/m2_smoke --smoke

# API (requires OPENAI_API_KEY)
python paper_experiments/m2_persona_decoupling/src/run_m2_api.py \
  --provider openai --model_id gpt-4o-mini --condition +G \
  --framing role_play_gambler --task SM --n_games 5 \
  --output_dir /tmp/m2_smoke --smoke

# Analysis (synthetic)
pytest paper_experiments/m2_persona_decoupling/tests/test_m2_smoke.py -k synthetic
```

## Live launch (Stage 1, n=200 per cell)

For each (model, condition, framing, task) cell run with `--n_games 200` against the configured output dir (`/scratch/x3415a02/data/llm-addiction/m2_persona/`). Total grid: 6 models × 4 conditions × 2 framings × 1 primary task (SM) = 48 cells; IC and MW are robustness and only launched if SM compute permits per Plan §2.5.

After all cells write `final_*.json`:

```bash
python paper_experiments/m2_persona_decoupling/src/analyze_m2.py \
  --input_dir /scratch/x3415a02/data/llm-addiction/m2_persona/ \
  --output_path /scratch/x3415a02/data/llm-addiction/m2_persona/summary.json

python paper_experiments/m2_persona_decoupling/src/sanity_checks.py \
  --input_dir /scratch/x3415a02/data/llm-addiction/m2_persona/ \
  --output_path /scratch/x3415a02/data/llm-addiction/m2_persona/sanity.md
```

## Reading the analysis

`summary.json` per task:

- `primary_contrast.delta_gap_prob` and `.ci_low/.ci_high`: the framing-gap contrast on the probability scale with 95% bootstrap or posterior CI.
- `primary_contrast.delta_gap_logit` and `.logit_ci_low/.logit_ci_high`: same on logit scale (pre-reg primary).
- `primary_passes` + `primary_pass_scale`: True if the primary CI excludes zero on the logit scale (preferred) or probability scale (fallback when logit unavailable).
- `per_model_deltas`: each model's Δ at first_person and role_play, with bootstrap CIs (per-model heterogeneity diagnostic).
- `framing_condition_heatmap`: long-format rows for a (4 × 2) bankruptcy-rate heatmap.
- `manipulation_check`: gambling-keyword frequency per cell + the role_boost > first_boost flag (qualitative sanity).

`sanity.md`: per-model PASS/FAIL for the manipulation check + summary line "[N / 6] models pass framing manipulation". A FAIL on the manipulation check means the role-play framing is too weak to interpret a primary null; trigger Plan §2.5 fallback (re-pilot framing).

## Stage 2 trigger (Plan §2.3 / §8.3)

If Stage 1 primary CI lower bound (probability scale) is in `[-0.01, +0.02]` OR the probability-scale CI half-width exceeds `0.03`, extend the primary contrast cells to n=500 per cell. Other cells stay at n=200.

## Outcome branches

The three §3.2 prose variants are committed verbatim in `claim_surgery_M2_outcome_branches.md`. Day 5 selects the branch by the pre-registered decision rule and inserts it into the rebuttal.
