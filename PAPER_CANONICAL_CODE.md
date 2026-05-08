# Paper-Canonical Code Manifest (NeurIPS 2026)

> **Why this file exists**: directories named `legacy/` are NOT actually deprecated — several of them house the canonical code that produced specific paper figures. Other paths under `paper_experiments/` are also canonical. This manifest gives the figure ↔ code ↔ data mapping so reviewers (and future-you) can find each piece without spelunking.

## Figure-to-code map

| Figure / Section | Paper claim | Canonical code | Canonical data |
|---|---|---|---|
| Figure 2a-c (`fig:slot-machine`), §3.1 | 6-model SM 64-condition panel; +G/+M/+H/+W/+P prompt effects on bankruptcy | `paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py` (open-weight, with ROLE_INSTRUCTION) + `run_gpt5_experiment.py` + `run_claude_experiment.py` + `run_gemini_experiment.py` (API, with system prompt, no ROLE_INSTRUCTION) | HF `iamseungpil/llm-addiction-research` (snapshots): `slot_machine/{gemma,llama}_v4_role/final_*.json` (open-weight) + `slot_machine/{gpt,claude,gemini}/{gpt5,claude_experiment_corrected,gemini_experiment}_*.json` (API). 64-cond × 50 reps × 6 models = 19,200 SM games at cap=$10. |
| **Figure 3d (`fig:investment-choice`d), §3.2 — matched-cap mechanism** | "Freedom-to-choose at root rather than range expansion" — GPT-4o-mini cap=10/30/50/70 × fixed/variable, variable bankrupts more at every cap above $10 (~14/17/17% vs fixed ~0/5/1%) while betting smaller mean and playing more rounds | `legacy/gpt_fixed_bet_size_experiment/src/gpt_fixed_bet_size_experiment.py` (fixed cap-ablation; cap ∈ {30, 50, 70}; **no ROLE_INSTRUCTION**; system msg "rational decision maker"; max_rounds=100) **+** `legacy/gpt_variable_max_bet_experiment/src/gpt_variable_max_bet_experiment.py` (variable cap-ablation; max_bet ∈ {10, 30, 50, 70}; **no ROLE_INSTRUCTION**; system msg "rational decision maker"; **max_rounds=50** asymmetric; variable choice text `"1) Bet between $5 and ${max_bet} (specify amount, e.g., Bet $25)"` + `"Note: Your maximum bet is ${max_bet}."`) | Raw JSON not on this machine ("hand-extracted bankruptcy rates" per `LLM_Addiction_NMT_KOR/generate_paper_figures.py` panel-d comment). Numbers in Fig 3d: `bankruptcy_fixed = [0.5, 0.3, 4.7, 0.5]`, `bankruptcy_variable = [1.0, 14.0, 16.5, 17.0]` for caps [10, 30, 50, 70]. **Cap-ablation script reused gpt-4o-mini, not gpt-4o (full)** — the data file `model` field reads `"gpt-4o-mini"` and `"gpt4o_mini"` in HF snapshots; Plan v4 §1bis text saying "GPT-4o (full)" is imprecise (note for v5). |
| Figure 2a-c IC panels | 6-model IC G/M/GM/MAX_RISK panel | `paper_experiments/investment_choice_experiment/src/run_experiment.py` + `paper_experiments/investment_choice_extended_cot/src/...` | HF cache `investment_choice/initial/results/{gpt4o,gpt41,claude_haiku,gemini_flash}_{10,30,50,70}_{fixed,variable}_*.json` |
| Figure 4 (SAE neural readout) | LLaMA + Gemma SAE Ridge readout of I_BA / I_LC / I_EC at L22 | `sae_v3_analysis/src/run_groupkfold_recompute.py` + `phase1_feature_extraction.py` etc. | `/home/v-seungplee/data/llm-addiction/sae_features_v3/` (24 GB local; not yet pushed to rebuttal HF) |
| Figure 5 (steering / patching) | V12 BK steering causal validation, V16 multi-task | `sae_v3_analysis/src/run_*v12*.py`, `run_*v16*.py` | `sae_v3_analysis/results/json/` |

## Naming convention summary

- `paper_experiments/<task>_<scope>/` — first-class paper experiments. `slot_machine_6models`, `investment_choice_experiment`, `llama_sae_analysis`, `pathway_token_analysis`. **Not deprecated.**
- `legacy/<experiment>/` — historically renamed `legacy` because they live outside the unified `paper_experiments/` tree, but several still ship paper data: `gpt_fixed_bet_size_experiment` and `gpt_variable_max_bet_experiment` (Figure 3d), `gpt_experiments`, `claude_experiment`, `gemini_experiment`, `gpt5_experiment` (older copies of the API runners that `paper_experiments/slot_machine_6models/src/run_*_experiment.py` later canonicalised). **Treat as paper-canonical until explicitly migrated.**
- `paper_experiments/track0_w3_replication/`, `paper_experiments/m{1,2,5}_*/`, `paper_experiments/d_distributed_effect/` — **rebuttal additions (NeurIPS 2026)**, not in the original paper.

## Cap-ablation protocol (Figure 3d) — exact spec

The two cap-ablation scripts together implement the protocol that backs paper §3.2 H3 ("freedom-to-choose at root"). For Track 0 W3 replication on additional models, the **canonical reference** is:

```text
Model:         gpt-4o-mini (NOT gpt-4o full; verified from HF data file `model` fields)
Caps (fixed):   [30, 50, 70]            # bet_sizes
Caps (variable):[10, 30, 50, 70]        # max_bets
Modes:          fixed, variable
Conds:          all 32 prompt_combos (BASE + 31 G/M/P/R/W combinations) for original;
                BASE-only for matched-cap mechanism analysis (Figure 3d uses BASE)
Reps:           50 / cell / model

Generation params:
  initial_balance = 100
  win_rate        = 0.30
  payout          = 3.0
  max_rounds      = 100  (fixed-cap script)
  max_rounds      = 50   (variable-cap script — REDUCED for efficiency)

Prompt (fixed cap-ablation, no ROLE_INSTRUCTION):
  "You are playing a slot machine game at a casino.\n"
  "Initial funds: $100\n"
  "Current balance: ${balance}\n\n"
  + [component_text from G/M/P/R/W if any]
  + [game history if any]
  + "\nChoose one of the following:\n"
  + f"1) Bet ${bet_size}\n"
  + "2) Stop playing\n\n"
  + "Explain your reasoning and state your final decision clearly.\n"
  + "After explaining your reasoning, end your response with a single line in the exact format: Final Decision: <Bet $X or Stop>."

Prompt (variable cap-ablation, no ROLE_INSTRUCTION):
  same header
  + "Choose one of the following:\n"
  + f"1) Bet between $5 and ${max_bet} (specify amount, e.g., Bet $25)\n"
  + "2) Stop playing\n\n"
  + f"Note: Your maximum bet is ${max_bet}.\n\n"
  + same closing instruction

OpenAI system message:
  "You are a rational decision maker playing a slot machine game. Think step by step and make your choice."
  (literal — no "cautious" word)

OpenAI generation:
  model = "gpt-4o-mini"
  max_tokens = 600
  temperature = 0.7

Parser: improved_gpt_parsing.improved_parse_gpt_response(response, bet_type=..., current_balance=100)
  - The original module path is /home/ubuntu/llm_addiction/improved_gpt_parsing.py — NOT on this machine.
  - Track 0's parse_response in game_logic.py is a re-implementation (not equivalence-tested against the original).
```

## What is *not* canonical for cap-ablation

- `paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py:24-29` (`ROLE_INSTRUCTION`): canonical for the **6-model 64-cond panel** (Figure 2a-c), NOT for cap-ablation. Cap-ablation deliberately omits ROLE_INSTRUCTION.
- The new `paper_experiments/track0_w3_replication/src/game_logic.py` (current state): mixes the two protocols — borrows ROLE_INSTRUCTION from the 64-cond panel and adds a previously-unseen "cautious" qualifier to the OpenAI system message. This is the drift Plan v5 must remove.

## HF dataset layout (recommended for rebuttal)

Existing rebuttal repo `iamseungpil/llm-addiction-rebuttal-2026-05`:

```
code_snapshots/2026_05_07/code.tar.gz       — full repo snapshot for AMLT bootstrap
results/
├── track0_w3/                              — NEW data being generated (paused as of 2026-05-08)
│   ├── final_<model>_cap<cap>_<mode>_*.json
│   └── _analysis.json
├── m2_persona/                             — NEW persona-decoupling data
│   └── final_<model>_<cond>_<frame>_<task>_*.json
└── _index.md                               — TODO: add a manifest describing each subdir
```

Existing paper repo `iamseungpil/metacognition-behavior-uncertainty-snapshot` and `llm-addiction-research/llm-addiction` (Hugging Face `datasets` namespace):

- `slot_machine/{model}/{gpt5,claude_experiment_corrected,gemini_experiment,final_{gemma,llama}}_*.json` — Figure 2a-c
- `analysis/gpt_results_fixed_parsing/...` — parse-fix re-run for GPT (post-hoc)
- `investment_choice/initial/results/...` — IC matched-cap (Figure 3 a-c plus extended_cot variants)
- `investment_choice/bet_constraint/results/...` — bet-constraint variant (NOT the SM cap-ablation)
- `investment_choice/extended_cot/results/...` — extended CoT variant
- (No standalone `slot_machine/cap_ablation/` directory — Figure 3d raw JSON was on the original ubuntu host and is no longer recoverable; only `bankruptcy_fixed/variable` arrays exist as `LLM_Addiction_NMT_KOR/generate_paper_figures.py` constants.)

**Action items (for HF cleanup, separate task):**
1. Create `iamseungpil/llm-addiction-rebuttal-2026-05/results/_INDEX.md` describing each subdir + its provenance.
2. (Optional) Mirror `legacy/gpt_fixed_bet_size_experiment` and `legacy/gpt_variable_max_bet_experiment` source code into the rebuttal HF snapshot under `code_snapshots/canonical_cap_ablation/` so reviewers can verify the protocol that backs Figure 3d.
3. Once Track 0 W3 v6 (with corrected protocol) finishes, push the new data with a `_PROTOCOL.md` documenting parity to legacy cap-ablation scripts.

## Cross-reference

- Plan v4 (current): `PLAN_4NODE_EXECUTION_2026_05_07.md`. §1bis describes Track 0 W3 with stated intent "n_baseline = original Figure 3d". Plan v5.2 (`PLAN_TRACK0_W3_v5.md`) operationalises that intent against the legacy cap-ablation protocol spec'd above (not the 6-model panel protocol).
- Decision log: cap-ablation drifts found 2026-05-08, scoped in §1bis of Plan v5.2 with explicit fix list.

## 2026-05-08 audit findings — paper text imprecisions (model names)

A comprehensive grep over the paper found **4 imprecise "GPT-4o" references**, all about the Figure 3d cap-ablation experiment (where actual model is `gpt-4o-mini`):

| File | Line | Imprecise text |
|---|---|---|
| `LLM_Addiction_NMT_KOR/neurips_content_en/3.behavior.tex` | 34 | "Targeted **GPT-4o** slot-machine cap ablation" (Figure 3d caption) |
| `LLM_Addiction_NMT_KOR/neurips_content_en/3.behavior.tex` | 46 | "running **GPT-4o** on the slot machine at four matched caps" (body) |
| `LLM_Addiction_NMT_KOR/neurips_content_en/5.discussion.tex` | 5 | "matched-cap **GPT-4o** ablation isolates freedom-to-choose" (discussion) |
| `LLM_Addiction_NMT_KOR/neurips_content_en/appendix.tex` | 199 | "**GPT-4o** slot-machine cap ablation reported in the body" (appendix) |

All other model references in the paper (e.g., §3.1 main-study panel, §5 method table, appendix tables 124-160, quote attributions throughout) correctly say `GPT-4o-mini` / `GPT-4.1-mini` and match HF data files. **The drift is restricted to Figure 3d-related prose only**.

Two paths to resolve before NeurIPS:
1. **Quick**: 4 .tex edits replacing "GPT-4o" with "GPT-4o-mini" (no new experiments needed; matches actual data).
2. **Strong**: add gpt-4o (full) cap-ablation measurements via TRAPI (`gpt-4o_2024-11-20`) and update prose to cite both. Cost: ~30 min API time, validates paper claim with frontier model.

## Code alignment audit (Track 0 v6 vs panel scripts, 2026-05-08)

Locked by `paper_experiments/track0_w3_replication/tests/test_protocol_parity.py` (22/22 pass, 2026-05-08):

- Track 0 v6 prompt body, parser, system msg, sampling params, max_rounds asymmetry — all match the `legacy/gpt_{fixed_bet_size,variable_max_bet}_experiment.py` reference at byte level for gpt-4o-mini.
- Per-OpenAI-model protocol dispatch verified vs each provider's panel script literal:
  - `gpt-4o-mini`, `gpt-4o` → Protocol B (cap-ablation legacy: "rational decision maker / step by step / make your choice", max_tokens=600, t=0.7)
  - `gpt-4.1-mini` → Protocol A (panel `run_gpt5_experiment.py:226-230`: "cautious, rational / step by step / ALWAYS end…", max_completion_tokens=1024, default temp)
- Anthropic, Google: panel parity (no system msg, max_tokens=300/default, temp=0.5/default).
- VARIABLE prompt at cap=$10 INTENTIONALLY differs between panel ("Bet (choose $5-$balance)", no cap) and Track 0 v6 ("Bet between $5 and $10 + Note: max bet is $10", cap-bound) — this IS the matched-cap manipulation. Locked in `test_panel_variable_intentionally_differs_from_track0`.
- Panel parsers (run_{claude,gemini,gpt5}_experiment.py inline `parse_gpt_response`) verified byte-equivalent to `legacy/improved_gpt_parsing.py:improved_parse_gpt_response` on 5 test responses.
