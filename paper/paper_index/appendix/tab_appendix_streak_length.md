# Table — `tab:appendix-streak-length`

**Paper location**
Appendix \subsubsection{Escalation by streak length}, `neurips_content_en/appendix.tex:247`

**What the experiment asks**
After a run of exactly one, two, three, four or five losses in a row, how much does a model raise the share of its money it puts on the next spin — and does the forced-bet arm do the same?

**HF path(s) of the raw data**
Should be the six canonical slot-machine corpora pooled (see `tab_appendix_slot_comprehensive.md`). Not independently confirmed.

**Code that turns raw data into the printed values**
NOT CHECKED — no fragment in `paper_data/tables/`. The closest live implementation is the RATIO definition in `scripts/build_figure_data.py` / `scripts/figures/fig02_slot_machine.py`, which computes the same quantity for figure panels (c)/(d).

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

*Scope for this float.* All six rows apply: the table pools the six corpora. The invariant is a necessary condition here and not a sufficient one, because this table also depends on the streak-scanning rule and the escalation definition, neither of which the invariant constrains.

**Corpus vintage**
Presumed canonical (the round counts, e.g. 9,745 forced post-loss rounds at streak 1, are of the right order for the six-model pooled corpus). Not confirmed. No `DEPRECATION_WARNING.md` is implicated, but nothing rules one in either.

The structural zeros in the post-win forced rows are genuine, not missing data: a forced $10 wager cannot be raised.

**Reproduction status**
NOT CHECKED, and blocked on a definition rather than on data. No generator fragment exists in
`paper_data/tables/`, and the escalation definition used here — mean rise in bet-to-balance ratio at
exact streak length — is one of at least two the project contains for a quantity of this name.

The three rules that must be pinned before this table can be recomputed, none of which the paper
states:

  1. **Escalation definition.** RATIO (mean rise in the bet-to-balance ratio) versus LEGACY (share of
     occurrences whose next dollar bet is larger). `scripts/figures/fig02_slot_machine.py` documents
     both and records that they disagree. This table's caption implies RATIO; Figure 2(c)/(d) prints
     LEGACY.
  2. **Streak scan.** Overlapping sliding window (Figure 2's LEGACY rule) versus disjoint scan
     advancing by the streak length on a match (`appendix/fig_appendix_model_streak.md`). The two
     give different long-streak cells on the same corpus.
  3. **"Exact" streak length.** Whether a length-k row counts runs of exactly k or runs of at least k.
     The camera-ready Figure 2 caption pins the latter for that figure; nothing pins it for this table.

Until those three are fixed, a disagreement between a recompute and the printed table is a
disagreement about definitions, not evidence of a data defect.

The load invariant above is a necessary precondition and is passing on all six corpora. It is not
sufficient here: it constrains the load, and this table is blocked at points 1 to 3, which sit
downstream of the load.

