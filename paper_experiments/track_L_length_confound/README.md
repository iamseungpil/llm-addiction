# Track L — Length / Survival Confound Re-analysis

NeurIPS 2026 rebuttal artefact for the LLM-addiction paper.

This folder defends paper §3.2 against a length-confound critique: "variable
mode bankruptcy is just longer games accumulating exposure, not a real
per-decision risk shift". Track L re-analyses every round of every game in
the 6-model panel and shows that variable mode raises *per-decision*
bankruptcy hazard ~100x in the slot-machine setting, while the open-weight
investment-choice cohort shows the directional inversion the paper §3.2
P1-3 cap-confound disclosure already predicted.

## Quick result

| Dataset | n_bankrupt | RR_per_decision | 95 % CI | Holm-p | Verdict |
|---------|------------|-----------------|---------|--------|---------|
| **SM_API** (Claude / Gemini / GPT-4.1-mini / GPT-4o-mini-corrected) | 926 | 90.6 | [44.8, 183.4] | < 1e-34 | **L-passes** |
| **SM_OW** (LLaMA + Gemma v4_role) | 313 | 104 | [7.5, 1479] | 5.6e-4 | **L-passes** (pre-specified ridge fallback for separation) |
| **IC_OW** (LLaMA + Gemma v2_role, max_rounds=100) | 307 | 0.112 | [0.079, 0.158] | < 1e-34 | **L-fails** (directional inversion predicted by paper §3.2 P1-3) |

`IC_API` (4 API IC × cap × mode, max_rounds=10) is **descriptive-only**:
0/6,600 bankruptcy events make per-decision RR non-estimable.

## Layout

```
track_L_length_confound/
├── README.md                       # this file
├── PLAN_TRACK_L_LENGTH_CONFOUND_v1.md  # Plan v3.4 (latest)
├── src/
│   ├── build_round_table.py        # 4-schema parser → round_table.csv
│   └── fit_hazard.py               # cause-specific MNL + Holm + ridge fallback
├── tests/                          # unit + sanity guards
├── round_table.csv                 # 237,385 round-level rows (4 datasets)
└── track_L_results.json            # 3 primary RR readouts + Holm-adjusted p
```

## Reproducing

```bash
# 1. Pull the source data from HF (no copies stored here):
python -c "
from huggingface_hub import snapshot_download
snapshot_download('llm-addiction-research/llm-addiction', repo_type='dataset',
                  allow_patterns=['investment_choice/bet_constraint/results/*.json',
                                  'behavioral/investment_choice/v2_role_*/*.json',
                                  'behavioral/slot_machine/*_v4_role/final_*.json',
                                  'slot_machine/*/*.json',
                                  'analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_*.json'],
                  local_dir='/path/to/data')
"

# 2. Build the round-level table:
python src/build_round_table.py --data-root /path/to/data \
    --out round_table.csv

# 3. Fit the 3 primary readouts:
python src/fit_hazard.py --table round_table.csv \
    --out track_L_results.json
```

## Methodology in one screen

- **Estimand**: `RR_per_decision = exp(β_var^bankrupt)` from a cause-specific
  multinomial logit with reference category `continue` and contrast
  `C(bet_type, Treatment(reference='fixed'))`.
- **Conditioning**: `cap`, `log1p(balance_before)`, `round`, `C(model)` (when
  no quasi-separation), `C(prompt_combo)`.
- **Standard errors**: cluster-robust on
  `(dataset, file_timestamp, cap, prompt_combo, model, game_id)`.
- **Overlap restriction**: rounds 1-10; (model × balance-quartile × round)
  cells where both bet types are observed.
- **Multiplicity**: Holm step-down across the three confirmatory primaries
  (SM_API, SM_OW, IC_OW). Pooled fit reported as sensitivity only.
- **Pre-specified ridge fallback**: triggered automatically by the
  convergence/separation rule when MLE diverges (Gemma fixed-mode has
  0/5,581 bankruptcy events in SM_OW). The fallback is binary
  cause-specific logistic with mild L2 ridge — a pragmatic Firth
  approximation, since `firthlogist` requires Python <3.11.

## Codex review history

This artefact was iterated through nine codex review rounds:

- **R5** API vs open-weight segregation
- **R6** IC_API descriptive-only (no events)
- **R7** parser hardening — gpt-4o-mini dual-list alignment guards
- **R8** Holm correction across 3 primaries; transportability framing
- **R9** rebuttal narrative — IC_OW openly disclosed, SM_OW lower-CI framing
- **R10** four wording constraints (estimand, non-estimability, Holm
  definition, pre-specification audit)
- **R11** writing + experiment priority for the 7-day rebuttal window

The Plan document (PLAN_TRACK_L_LENGTH_CONFOUND_v1.md) records the full
deliberation chain.

## Status

- Phase 3 (primary fits) **complete**.
- Phase 4 (B5 forest plot, B1 Cox sensitivity, B3 pooled fit) **planned**
  per Plan v3.4 §8 minimum-viable-set.
- Body §3.4 paragraph + Appendix §X **planned** per Plan v3.4 §7.
