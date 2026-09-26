# M1 — Portfolio Discriminant Validity

## Intent

Plan v4 (`PLAN_4NODE_EXECUTION_2026_05_07.md` §4) Track B priority 3. Reviewer attack
target: "the `+G` autonomy effect generalises to any prompt-conditioned risk, not
gambling-specific." Tests gambling-specificity by running a matched non-gambling-but-
still-risky portfolio allocation task with the same `+G/+M` prompt manipulation on
the same 6 models that ran §3.1 SM, then estimating the cross-domain interaction.

## Hypothesis

`H_M1_main`: `+G` raises bankruptcy-equivalent risk in the gambling tasks more than
in the portfolio task.

```
risk_event ~ domain * condition + (1 | model)
domain     ∈ {gambling, portfolio}
condition  ∈ {BASE, +G}     # primary contrast on +G only
```

## Primary contrast (frozen pre-registration)

```
β_{+G × gambling}  −  β_{+G × portfolio}  >  0     (interaction term, logit scale)
decision: lower 95% Wald CI on the logit-scale interaction term > 0 → gambling-specific
```

Probability-scale marginal difference and per-model interaction CIs are reported as
secondary descriptive output. We avoid probability-scale primary because mixed-logit β
estimates are on log-odds and converting to ΔP before testing invites scale artefacts.

## Files

```
m1_portfolio_discriminant/
├── __init__.py
├── README.md
├── claim_surgery_M1_outcome_branches.md       # M1-passes / M1-mixed / M1-fails verbatim
├── configs/
│   └── m1_config.yaml                          # frozen pre-registration
├── src/
│   ├── __init__.py
│   ├── portfolio_simulator.py                  # PortfolioGame + simulate_returns + run_single_game
│   ├── prompts.py                              # build_portfolio_prompt (5 conds × 3 blurbs × 2 objectives)
│   ├── parse_allocation.py                     # robust allocation parser (Final Allocation:/JSON/keyed pairs)
│   ├── run_m1_open_weight.py                   # Gemma / LLaMA launcher (bf16, eager attention)
│   ├── run_m1_api.py                           # OpenAI / Anthropic / Google launcher
│   ├── analyze_m1.py                           # bambi domain×condition + bootstrap-logit fallback
│   └── sanity_checks.py                        # MAX_RISK positive control + asset-menu temptation check
└── tests/
    ├── __init__.py
    └── test_m1_smoke.py                        # synthetic + live-skip-if-no-resource
```

## Reuse

- Gambling-domain runs are *not* re-generated; `analyze_m1.py` walks
  `output.gambling_input_dirs` (default: Track 0 outputs at
  `/scratch/x3415a02/data/llm-addiction/track0_w3/` and §3.1 SM outputs at
  `/scratch/x3415a02/data/llm-addiction/slot_machine/`) and ingests any payload
  that records a `condition` ∈ {BASE, +G, ...}. The §3.1 SM 6-model panel already
  has BASE + +G runs with the same models we use here.

### Prompt-combo normalisation (C1)

The §3.1 SM panel writes its prompt-combo bitmasks directly into the records
(``"BASE"``, ``"G"``, ``"GM"``, ``"GMHWP"``, etc., **with no `+` prefix**), while
the M1 portfolio runners write the canonical ``"+G"`` / ``"+M"`` / ``"+GM"``
labels. ``analyze_m1._normalise_prompt_combo`` reconciles both into the M1
vocabulary and **excludes any cell that mixes the H/W/P bit flags into G/M**
because the portfolio arm has no analogue of those manipulations — including
them would contaminate the cross-domain comparison. See Plan v4 §13 deviation
log entry "M1 cross-domain prompt-combo normalisation".

### API / GPU failure handling (C6)

When the API runner exhausts its retries (or the open-weight runner exhausts
its OOM/runtime retries), the response function emits the sentinel
``__FALLBACK_API_FAILURE__`` instead of synthesising a 100%-cash allocation.
The parser (`parse_allocation.parse_allocation`) explicitly detects and rejects
this sentinel; the simulator records the round as a parse-skip and bumps the
per-game ``fallback_count``. ``analyze_m1`` reports
``fallback_rate_per_cell`` in the summary JSON; ``--exclude_high_fallback``
optionally drops games with ``fallback_count > 5`` from the primary contrast.
This avoids silently biasing per-model risk_event downward when an unstable API
forces many synthetic-cash rounds.
- `paper_experiments/track0_w3_replication/src/game_logic.SlotMachineGame` is the
  canonical SM class. M1 does not modify it. If we ever need to run *fresh* gambling-
  domain SM cells with explicit BASE/+G labels, the same class is imported.
- `portfolio_simulator.py` is **NEW** code — no portfolio analogue exists in the repo.

## Dependencies

- Conda env `llm-addiction` per `CLAUDE.md`.
- For `analyze_m1.py` mixed-logit fit: `pip install bambi pymc arviz`. The module
  loads without these (it falls back to bootstrap-pooled contrast); only the
  `fit_mixed_logit_interaction` function requires them.

## How to run a smoke test

Open-weight (5 games on Gemma, BASE condition, neutral blurb, wealth-max objective):

```
python paper_experiments/m1_portfolio_discriminant/src/run_m1_open_weight.py \
  --model gemma --gpu 0 \
  --condition BASE --objective wealth_maximisation --blurb_variant neutral \
  --smoke
```

API (5 games on GPT-4o-mini; needs `OPENAI_API_KEY`):

```
python paper_experiments/m1_portfolio_discriminant/src/run_m1_api.py \
  --provider openai --model_id gpt-4o-mini \
  --condition BASE --objective wealth_maximisation --blurb_variant neutral \
  --smoke
```

Pytest:

```
cd paper_experiments/m1_portfolio_discriminant
pytest tests/ -v
```

## How to run live (Stage 1, n=200 per cell)

The grid is 6 models × 5 conditions × 2 objectives × 3 blurb variants = 180 cells.
Each cell is one CLI invocation; AMLT yamls (added separately under `amlt/2026_05_07/
m1_track_b.yaml`) parallelise across 4 nodes.

Example single live cell (LLaMA, +G condition, wealth-max objective, salient blurb,
n=200 games):

```
python paper_experiments/m1_portfolio_discriminant/src/run_m1_open_weight.py \
  --model llama --gpu 0 \
  --condition +G --objective wealth_maximisation --blurb_variant salient_upside \
  --n_games 200
```

API live cell:

```
python paper_experiments/m1_portfolio_discriminant/src/run_m1_api.py \
  --provider anthropic --model_id claude-3-5-haiku-20241022 \
  --condition +G --objective wealth_maximisation --blurb_variant salient_upside \
  --n_games 200
```

Output: `final_{model}_{condition}_{objective}_{blurb}_{timestamp}.json` under
`output.base_dir` from the config (default `/scratch/x3415a02/data/llm-addiction/m1_portfolio/`).

## How to analyze

Once all 180 portfolio cells exist alongside the gambling-domain runs:

```
python paper_experiments/m1_portfolio_discriminant/src/analyze_m1.py \
  --portfolio_input_dir /scratch/x3415a02/data/llm-addiction/m1_portfolio/ \
  --gambling_input_dirs \
      /scratch/x3415a02/data/llm-addiction/track0_w3/ \
      /scratch/x3415a02/data/llm-addiction/slot_machine/ \
  --output_path /scratch/x3415a02/data/llm-addiction/m1_portfolio/summary.json
```

Sanity checks:

```
python paper_experiments/m1_portfolio_discriminant/src/sanity_checks.py \
  --input_dir /scratch/x3415a02/data/llm-addiction/m1_portfolio/ \
  --output_path /scratch/x3415a02/data/llm-addiction/m1_portfolio/sanity.md
```

## How to read results

`summary.json`:
- `primary_contrast`: pooled `β_{+G × gambling} − β_{+G × portfolio}` on logit scale
  with 95% CI (the gating quantity for the M1 decision rule).
- `secondary_marginal_diff`: probability-scale marginal difference with bootstrap CI.
- `per_model_interactions`: per-model logit-scale interaction with bootstrap CI;
  reveals whether the cross-domain dissociation is universal or model-dependent.
- `bambi_fit_status`: `"ok"` if the pre-registered hierarchical mixed-logit fit
  converged; otherwise the bootstrap-logit pooled estimate is reported.

`sanity.md`:
- C1 MAX_RISK positive control: PASS/FAIL per model. The portfolio task must
  register risk-seeking under MAX_RISK or the +G null is uninformative
  (Plan v4 §4.5 risk).
- C2 asset-menu temptation: PASS/FAIL per asset (leveraged_etf_3x,
  single_volatile_stock, otm_call_or_crypto). If none of these are picked
  meaningfully anywhere, the menu is functionally cash + bonds + index.

The three M1 outcome branches are pre-written in
`claim_surgery_M1_outcome_branches.md`; the appropriate one is inserted verbatim
into the rebuttal at readout.

## Open issues

- **Return-distribution parameters**: `assets[*].mean_annual` and `vol_annual` in
  `m1_config.yaml` are the most contestable choice in the module. Chosen so that
  (a) `broad_index` Sharpe is highest under a passive optimisation (rational default),
  (b) `leveraged_etf_3x` and `otm_call_or_crypto` are *plausibly* attractive when
  prompted aggressively (so MAX_RISK positive control has room to fire), and (c)
  the 50% drawdown threshold is reachable for all-in temptation allocations within
  30 monthly rounds. Documented; will revisit if `MAX_RISK` positive control fails.
- **Re-use of §3.1 SM gambling-domain runs**: `analyze_m1.load_gambling_results`
  ingests any payload with a `condition` field. The §3.1 SM panel was generated by
  `paper_experiments/slot_machine_6models/src/{llama_gemma,run_gpt5,run_claude,run_gemini}_experiment.py`
  with prompt-combo bitmasks (BASE, +G, etc.) — those need to be saved as
  `condition` in the JSON if they are not already, otherwise we must regenerate
  BASE / +G cells in the gambling domain to match the M1 portfolio cells. Verify
  before launching live.
