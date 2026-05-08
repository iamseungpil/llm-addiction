# Track 0 — W3 Cross-Model Matched-Cap Replication

## Intent

Plan v4 (`PLAN_4NODE_EXECUTION_2026_05_07.md` §1bis) promotes W3 (matched-cap N=1 generalization gap) above the prior Track A construct-validity work. The §3.2 paper claim "freedom-to-choose at root" is supported by GPT-4o only; Track 0 replicates the matched-cap protocol on the other 5 models in the §3.1 panel (Gemma-2-9b, LLaMA-3.1-8b, GPT-4o-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) plus a GPT-4o re-baseline run.

## Hypothesis

`H_W3_main`: At cap=$70, variable mode produces a higher bankruptcy rate than fixed mode in ≥4/6 models, AND the pooled mixed-logit interaction has lower 95% CI > 0.

```
bankrupt ~ condition * cap + (condition * cap | model)
condition ∈ {fixed, variable}
cap       ∈ {$10, $30, $50, $70}
```

## Primary contrast (frozen pre-registration)

```
β_primary = E[bankrupt | variable, cap=$70] − E[bankrupt | fixed, cap=$70]
decision: lower 95% CI on logit-scale β_primary > 0
```

Cluster-robust SE clustering on `(model, game_id)` is used as a frequentist cross-check.

## Files

```
track0_w3_replication/
├── __init__.py
├── README.md
├── claim_surgery_§3.2_branches.md          # W3-passes / W3-mixed / W3-fails verbatim
├── configs/
│   └── track0_config.yaml                   # frozen pre-registration
├── src/
│   ├── __init__.py
│   ├── game_logic.py                        # canonical SlotMachineGame + create_prompt + parse_response + run_single_game
│   ├── run_track0_open_weight.py            # Gemma / LLaMA launcher (bf16, eager attention)
│   ├── run_track0_api.py                    # OpenAI / Anthropic / Google launcher
│   ├── analyze_track0.py                    # bambi mixed-logit + bootstrap CIs + cluster-robust cross-check
│   └── sanity_checks.py                     # S1 (freedom-not-range) + S2 (more rounds under variable)
└── tests/
    ├── __init__.py
    └── test_track0_smoke.py                 # synthetic + live-skip-if-no-resource
```

## Dependencies

- Conda env `llm-addiction` per `CLAUDE.md`.
- For `analyze_track0.py` mixed-logit fit: `pip install bambi pymc arviz`. The module loads without these (it falls back to bootstrap-pooled contrast); only `fit_mixed_logit` requires them.
- `statsmodels` is needed for the cluster-robust SE cross-check (already in `llm-addiction` env).

## How to run a smoke test

Open-weight (5 games on Gemma at cap=$30, variable mode):

```
python paper_experiments/track0_w3_replication/src/run_track0_open_weight.py \
  --model gemma --gpu 0 --cap 30 --mode variable --smoke
```

API (5 games on GPT-4o-mini at cap=$30, variable mode; needs `OPENAI_API_KEY`):

```
python paper_experiments/track0_w3_replication/src/run_track0_api.py \
  --provider openai --model_id gpt-4o-mini --cap 30 --mode variable --smoke
```

Pytest:

```
cd paper_experiments/track0_w3_replication
pytest tests/ -v
```

## How to run live (Stage 1, n=200 per cell)

The grid is 6 models × 4 caps × 2 modes = 48 cells. Each cell is one CLI invocation; AMLT yamls in `amlt/2026_05_07/` (added in a separate skeleton iteration) parallelize across nodes.

Example single live cell (LLaMA, cap=$70, variable, 200 games):

```
python paper_experiments/track0_w3_replication/src/run_track0_open_weight.py \
  --model llama --gpu 0 --cap 70 --mode variable --n_games 200
```

API live cell:

```
python paper_experiments/track0_w3_replication/src/run_track0_api.py \
  --provider anthropic --model_id claude-3-5-haiku-20241022 --cap 70 --mode variable --n_games 200
```

Output: `final_{model}_cap{cap}_{mode}_{timestamp}.json` under `output.base_dir` from the config (default `/scratch/x3415a02/data/llm-addiction/track0_w3/`).

## How to analyze

Once all 48 cells have produced JSON files:

```
python paper_experiments/track0_w3_replication/src/analyze_track0.py \
  --input_dir /scratch/x3415a02/data/llm-addiction/track0_w3/ \
  --output_path /scratch/x3415a02/data/llm-addiction/track0_w3/summary.json
```

Sanity checks:

```
python paper_experiments/track0_w3_replication/src/sanity_checks.py \
  --input_dir /scratch/x3415a02/data/llm-addiction/track0_w3/ \
  --output_path /scratch/x3415a02/data/llm-addiction/track0_w3/sanity.md
```

## How to read results

`summary.json`:
- `primary_contrast`: pooled β_primary at the highest cap with 95% CI on probability and logit scales (key field for the W3 decision rule).
- `per_model_deltas`: per-model Δ at cap=$70 with 95% bootstrap CIs; count of models with CI excluding 0 drives the qualitative ≥4/6 secondary rule.
- `bambi_fit_status`: `"ok"` if the pre-registered hierarchical mixed-logit fit converged; otherwise the bootstrap-pooled contrast is reported.
- `cluster_robust_se_check`: frequentist cross-check with clustering on (model, game_id).
- `qualitative_secondary.passes`: True iff ≥4/6 models have positive Δ with CI excluding 0.

`sanity.md`: per-model PASS/FAIL on the two predicted signatures of the freedom-to-choose mechanism. A PASS on both is what the W3-passes branch needs; failures are reported alongside the primary contrast and decide which §3.2 surgery branch is committed.

The three §3.2 surgery branches are pre-written in `claim_surgery_§3.2_branches.md`; Day 5 selects one by the decision rule above and inserts it verbatim into the rebuttal.
