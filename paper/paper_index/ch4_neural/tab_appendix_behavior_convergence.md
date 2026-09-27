# Table — `tab:appendix-behavior-convergence`

**Paper location**
Appendix \subsection{Cross-task sharing at the hidden-state and sparse-feature levels}, `neurips_content_en/appendix.tex:670`

**What the experiment asks**
Do the three games produce the same kinds of irrational betting at the behaviour level, so that a shared internal direction would even make sense?

**HF path(s) of the raw data**
Open-weight corpora only:
  - `behavioral/slot_machine/{gemma_v4_role,llama_v4_role}/final_*.json`
  - `behavioral/investment_choice/{v2_role_gemma,v2_role_llama}/*.json`
  - `behavioral/mystery_wheel/{gemma_v2_role,llama_v2_role}/*_mysterywheel_c30_*.json`

**Code that turns raw data into the printed values**
`scripts/tables/appendix_behavioural_tables.py` -> `paper_data/tables/appendix/behaviour_convergence.tex`.

**Schema map**
This float's printed values rest on round-level outcomes, and the six slot-machine corpora do not
agree on where a round outcome lives or what type it has. Reading one corpus's schema against
another does not raise an error; it silently returns "no wins".

| Corpus (HF) | Decision array | Round outcome | Type | Bet on that round |
|---|---|---|---|---|
| Gemma-2-9B `behavioral/slot_machine/gemma_v4_role/` | `decisions[]` | `history[i].win`, equivalently `decisions[i].result` | `bool` / `"W"`-`"L"` `str` | `history[i].bet` = `decisions[i].bet` |
| LLaMA-3.1-8B `behavioral/slot_machine/llama_v4_role/` | `decisions[]` | `history[i].win`, equivalently `decisions[i].result` | `bool` / `"W"`-`"L"` `str` | `history[i].bet` = `decisions[i].bet` |
| GPT-4.1-mini `slot_machine/gpt/` | `round_details[]` | `round_details[i].game_result.result` | **`dict`** | `round_details[i].game_result.bet` |
| Gemini-2.5-Flash `slot_machine/gemini/` | `round_details[]` | `round_details[i].game_result.result` | **`dict`** | `round_details[i].game_result.bet` |
| Claude-3.5-Haiku `slot_machine/claude/` | `round_details[]` | `round_details[i].game_result.result` | **`dict`** | `round_details[i].game_result.bet` |
| GPT-4o-mini `analysis/gpt_results_fixed_parsing/` | `round_details[]` | `game_history[i].result` — **no `game_result` on the round at all** | `"W"`-`"L"` `str` (`win` bool alongside) | `game_history[i].bet` |

Four disagreements, each of which has already cost this project a printed number or a manifest verdict:

1. **`game_result` is a dict, not a string.** `str(game_result).startswith("W")` returns `False` for
   every round of the GPT-4.1-mini, Gemini-2.5-Flash and Claude-3.5-Haiku corpora — a win rate of
   0.000 on three of six corpora rather than an exception. That is the defect that reached print.
2. **The decision array is longer than the outcome array.** A game's terminal round is a stop or a
   refusal and carries no outcome: `decisions[]` runs 61,955 against 59,969 `history[]` entries on
   LLaMA and 21,423 against 18,308 on Gemma, and GPT-4o-mini keeps 14,466 `round_details[]` against
   11,607 `game_history[]`. Zipping the two positionally scores those rounds as losses. Measured, that
   costs each corpus 0.01 to 0.06 of win rate — 0.2438, 0.2552, 0.2590, 0.2847, 0.2891, 0.2544 against
   the six correct rates below — which puts only GPT-4o-mini outside the window and leaves the other
   five wrong but passing. **Denominator = outcome-bearing rounds, never `len(round_details)`.**
3. **Field naming.** The open-weight corpora carry `result` on `decisions[i]` and `win` on
   `history[i]`; `decisions[i].win` does not exist. Either reading is fine; the composite
   `decisions[i].win` is not, and returns `None` for all 83,378 open-weight decision records.
4. **Prompt alphabet.** The fifth prompt module is spelled `H` (hidden patterns) in the LLaMA V4role
   run and `R` in the GPT, Claude, Gemini and Gemma runs. A `prompt_combo` filter written against one
   alphabet drops the other model's condition silently. Cross-checked against
   `scripts/tables/rebuttal_info_bottleneck.py`, which states the same split in its provenance header.

**Load invariant**
*Assertion.* Every slot-machine corpus is a 30%-win-rate task by construction, so for each of the six
corpora independently, the round-level win rate over outcome-bearing rounds must satisfy
`0.25 <= win_rate <= 0.35`. Task spec is 0.30. This is a per-corpus assertion, never a pooled one:
pooling hides a corpus that reads as all-losses behind five that do not.

*One-line command.* The dataset is gated (auto-approve), so an HF token must be in the environment
first; without one this returns a 401 rather than a FAIL, which is not an invariant result.

```sh
HF_HUB_DISABLE_XET=1 python3 -c "
import json;from huggingface_hub import hf_hub_download as D
P=[('GPT-4o-mini','analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json'),('GPT-4.1-mini','slot_machine/gpt/gpt5_experiment_20250921_174509.json'),('Gemini-2.5-Flash','slot_machine/gemini/gemini_experiment_20250920_042809.json'),('Claude-3.5-Haiku','slot_machine/claude/claude_experiment_corrected_20250925.json'),('LLaMA-3.1-8B','behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json'),('Gemma-2-9B','behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json')]
for m,p in P:
 G=json.load(open(D('llm-addiction-research/llm-addiction',p,repo_type='dataset')));G=G.get('results') or G.get('games')
 O=[e for g in G for e in (g.get('history') or g.get('game_history') or [d['game_result'] for d in (g.get('round_details') or []) if isinstance(d.get('game_result'),dict)])]
 w=sum(1 for e in O if (e.get('win') if e.get('win') is not None else str(e.get('result','')).upper().startswith('W')))
 print('%-17s n=%6d win_rate=%.4f %s'%(m,len(O),w/len(O),'OK' if .25<=w/len(O)<=.35 else 'FAIL'))
"
```

*Expected value.* Six `OK` lines. The window is the assertion; the observed tightness is the sharper
diagnostic, because a correctly loaded corpus lands within 0.004 of 0.30 and the denominator error of
point 2 does not:

| Corpus | Outcome-bearing rounds | Win rate | |
|---|---|---|---|
| GPT-4o-mini | 11,607 | 0.3039 | OK |
| GPT-4.1-mini | 16,257 | 0.3038 | OK |
| Gemini-2.5-Flash | 15,656 | 0.2984 | OK |
| Claude-3.5-Haiku | 52,263 | 0.2999 | OK |
| LLaMA-3.1-8B | 59,969 | 0.2986 | OK |
| Gemma-2-9B | 18,308 | 0.2976 | OK |

Run and passing on all six corpora, against the public release.

*Gate.* **A manifest may not claim VERIFIED while its invariant is unrun.** An unrun invariant is
NOT CHECKED, whatever the cell-by-cell agreement looks like: the defect that reached print produced
correct-looking table cells on the three corpora it read correctly, and the pooled 15.4% it printed
was a plausible number. Cell agreement is not evidence the corpus was loaded; only the invariant is.

*Scope for this float.* Of the corpora this table reads, only the two slot-machine ones are bound by the invariant above; run the two `behavioral/slot_machine/*_v4_role/` rows of it. Investment choice and mystery wheel are not 30%-win-rate tasks and have no equivalent constant. Their gate is the denominator caveat recorded under *Corpus vintage*: the fixed half of both records `bet_amount = None` at decision level, so a correct load yields 800 of 1,600 IC games and 1,600 of 3,200 MW games per model. A load that returns the full 1,600 and 3,200 has read the fixed half as though it carried bets, and is wrong.

**Corpus vintage**
Canonical V4role/V2role open-weight. `slot_machine/gemma/` and `slot_machine/llama/` (the V1 October-2025 runs) each carry a `DEPRECATION_WARNING.md` and are NOT used here. The release also mirrors the clean open-weight files at `slot_machine/gemma_v4_role/` and `slot_machine/llama_v4_role/`, identical filenames to the `behavioral/...` copies; either path is canonical, the bare `slot_machine/{gemma,llama}/` paths are not.

Denominator caveat the data forces and the caption does not state: 'all games' is exact only for slot machine. In the IC and MW corpora the fixed-bet half records `bet_amount = None` at decision level, so those rows are the variable-bet half only — 800 of 1,600 IC games and 1,600 of 3,200 MW games per model. Reproducing the printed digits requires exactly that subset, which is evidence it is the subset the paper used.

**Reproduction status**
PARTIALLY VERIFIED / PARTIALLY UNREPRODUCIBLE, 2026-08-25.
  - I_BA and I_EC: VERIFIED, 12 printed values, max deviation 0.000.
  - I_LC (6 printed values: 0.307, 0.323, 0.404, 0.166, 0.296, 0.345): UNREPRODUCIBLE. No definition tried by the regeneration script reproduces them from the released corpora; the script deliberately emits no I_LC column rather than print a number whose rule is unknown (see `ILC_DEFINITIONS_TRIED` in `scripts/tables/appendix_behavioural_tables.py`).
