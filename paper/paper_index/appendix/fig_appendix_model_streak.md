# Figure — `fig:appendix-model-streak`

**Paper location**
Appendix \section{Model-by-model details}, `neurips_content_en/appendix.tex:507-509`; graphic `images/individual_model_streak_analysis.pdf`

**What the experiment asks**
Per API model, does it bet more after winning, after losing, or simply keep playing regardless?

**HF path(s) of the raw data**
Read off the recovered generator's loader table (see below), with one substitution the release
forces. The generator names four API corpora:
  - GPT-4o-mini      `gpt_results_corrected/gpt_corrected_complete_20250911_071013.json`
  - GPT-4.1-mini     `slot_machine/gpt/gpt5_experiment_20250921_174509.json`
  - Gemini-2.5-Flash `slot_machine/gemini/gemini_experiment_20250920_042809.json`
  - Claude-3.5-Haiku `slot_machine/claude/claude_experiment_corrected_20250925.json`

The first of those is **not in the public release.** A search of the complete file listing returns no
`gpt_results_corrected/` directory and no `gpt_corrected_complete_*` file; that GPT-4o-mini export is
an unreleased predecessor of the canonical
`analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`. The
reproduction below substitutes the canonical file. The other three paths are canonical and present.
That substitution is the most likely cause of the residual on the GPT-4o-mini column; it does not
account for the one column that genuinely disagrees, which is Gemini's.

**Code that turns raw data into the printed values**
**Generator recovered, and it writes the printed filename.** This figure was recorded in earlier
passes of this index as having no generator anywhere. The search covered the paper repo and the HF
upload; the script was in neither, because it was in the code repo (`iamseungpil/llm-addiction`) and
had been deleted there. Commit `16acccf` ("clean files") removed the whole of
`legacy/writing/table_figure/`. Git still has it:

```sh
git -C <llm-addiction> show 16acccf^:legacy/writing/table_figure/create_individual_model_streak_analysis.py
```

A verbatim copy is also checked into this repo at `scripts/figures/recovered/create_individual_model_streak_analysis.py`, beside a `README.md` recording the corpora, subset and scan rule of all seven recovered
scripts. Cite the copy; the git command above is how to re-derive it.

Its two `savefig` targets are `individual_model_streak_analysis.png` and
`individual_model_streak_analysis.pdf` — the exact asset the paper includes, no revision suffix.

**The subset.** This is the part no earlier pass recorded, and it is the part that decides the
numbers. A streak figure is almost entirely definition:

  - **Arm: variable only.** The fixed arm is excluded. A \$10-locked wager cannot be raised, so every
    fixed-arm streak contributes a structural zero to the bet-increase rate, and pooling the arms
    roughly halves the plotted rates rather than nudging them. Note that the recovered ancestor
    script does **not** apply this filter; the printed figure requires it. Measured both ways under
    *Reproduction status*: variable-only 0.0102 mean absolute error, pooled 0.0957.
  - **Streak scan: sliding, advancing by the streak length on a match.** The scan walks `i` over the
    round list, tests whether `rounds[i .. i+k-1]` are all the same outcome, and on a match advances
    `i += k`; on a miss it advances `i += 1`. So a run of six wins yields three disjoint length-2
    streaks, not five overlapping ones. A scan that always advances by 1 counts overlapping windows
    and inflates the long-streak cells; a scan that only takes maximal runs undercounts them. Neither
    matches the printed figure.
  - **The comparison pair:** current bet = `rounds[i+k-1]` (the last round of the streak), next bet =
    `rounds[i+k]`. A streak is dropped unless both bets are present and positive. "Increase" is
    strict, `next > current`, and the plotted quantity is the *share* of streak occurrences that
    increase, not the mean size of the increase.
  - **The bet field: `round_details[i].bet_amount`, not `game_result.bet`.** The two agree on the
    value where both are present (84,117 of 84,176 rounds across the three dict-schema corpora, 59
    disagreements), so this is not a question of which number is right. It is a question of which
    rounds survive: `game_result` is absent on every stop round, and absent from the **entire**
    GPT-4o-mini corpus, so a loader keyed on `game_result.bet` silently empties that model's sample
    and prints a zero rate for it. Measured against the printed values, `bet_amount` gives 0.0062 mean
    absolute error on the win-streak panels and `game_result.bet` gives 0.0898. Read `bet_amount`.
  - **Continuation:** `next_round['decision'] == 'continue'`, the second row of panels.
  - **Grid:** streak lengths 1 to 5, win streaks and loss streaks scanned separately, four models.

The HF `figA08_individual_streak/code/generate_behavioral_figures.py` attachment remains what the
previous pass found it to be: one script copied under five figure directories, drawing none of them.
It is a mislabelled attachment, not the generator.
Repo redraw at print size: `scripts/figures/appendix_behavioral_panels.py`, which recovers the
plotted series from the submitted vector PDF's bar and marker geometry rather than from raw logs.
Artwork on HF at `paper_neurips_2026/camera_ready/figures/individual_model_streak_analysis.pdf`.

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

*Scope for this float.* This figure covers the four API models, so run the four API rows of the invariant. Both open-weight rows must still be run before any six-model sibling float claims VERIFIED off the same loader.

**Corpus vintage**
**Canonical, with one unreleased predecessor named — no longer undetermined.** The loader table is a
hard-coded dictionary, so the roster is read rather than inferred. Three of the four paths are the
canonical files. The fourth, the GPT-4o-mini entry, is the unreleased `gpt_results_corrected/` export
described above; it is a predecessor of the canonical fixed-parsing file, not one of the deprecated
corpora.

Two open questions this closes. The abandoned GPT-5-mini run
(`slot_machine/gpt/archived_gpt5mini_20250921/`, stopped at 9.4% after a 60% API error rate) is never
opened, so the "GPT-5-mini printed under a GPT-4o-mini label" worry is ruled out rather than merely
unproven; `slot_machine/gpt/` is opened under its true identity, GPT-4.1-mini. And neither
`slot_machine/gemma/` nor `slot_machine/llama/` appears anywhere in the loader, which is expected for
a four-API-model figure but worth stating.

What stays open is narrower and now nameable: the printed GPT-4o-mini column was drawn from an export
that was never released, so that one column cannot be reproduced from public data alone, only
approximated by its canonical successor.

**Reproduction status**
**REPRODUCES — top row.** Upgraded from UNREPRODUCIBLE. Recomputed this pass from the released
corpora under the subset above, with the canonical fixed-parsing file substituted for the unreleased
GPT-4o-mini export. The comparison target is the submitted artwork's own value labels, which
`scripts/figures/appendix_behavioral_panels.py` carries as its `A08` table.

Bet-increase rate, the figure's substantive claim, mean absolute error over the ten cells per model
(five win-streak lengths and five loss-streak lengths):

| Model | Bet-increase MAE | win streaks | loss streaks |
|---|---|---|---|
| GPT-4o-mini | 0.0067 | 0.0032 | 0.0102 |
| GPT-4.1-mini | 0.0067 | 0.0032 | 0.0101 |
| Gemini-2.5-Flash | 0.0144 | 0.0132 | 0.0156 |
| Claude-3.5-Haiku | 0.0129 | 0.0053 | 0.0204 |

**The subset was determined by measurement, not assumed.** All four combinations of arm and bet field
were run against the printed values, which is the only way to settle a question the paper does not
answer:

| Arm | Bet field | Bet-increase MAE |
|---|---|---|
| **variable only** | **`bet_amount`** | **0.0102** |
| variable only | `game_result.bet` | 0.0660 |
| pooled | `bet_amount` | 0.0957 |
| pooled | `game_result.bet` | 0.1433 |

Variable-only beats pooled by a factor of nine. This **corrects the recovered ancestor script's own
behaviour**: `create_individual_model_streak_analysis.py` applies no `bet_type` filter, and
`scripts/figures/recovered/README.md` records it accurately as pooling the arms. Pooling does not
reproduce the printed figure. The fixed arm is locked at \$10, so every fixed-arm streak contributes
a structural zero and pooling roughly halves the plotted rates. Either the printed art came from an
unrecovered revision that filtered the arm, or the four-model ancestor was run on a
variable-only export. Which of those is true is not determined; that the printed figure is
variable-only is.

**Bottom row: partially reproduces, and one cell does not.** Continuation rate MAE is 0.0126, 0.0044,
0.0132 and 0.0038 on win streaks, which is the same tolerance as the top row. On **loss** streaks it
is 0.0382, 0.0215, 0.1428 and 0.0125 — and the Gemini-2.5-Flash column is a real disagreement, not a
tolerance. The printed series falls from 0.95 to 0.59 across streak lengths 1 to 5 while the recompute
stays between 0.92 and 0.82. No arm or bet-field combination closes it. Recorded as open rather than
explained away.

The load invariant is passing on all six slot-machine corpora, four of which this figure reads. This
manifest does not claim VERIFIED in the cell-by-cell sense: it claims reproduction to a stated
tolerance on the top row, with a stated cause for the residual, and one unexplained column on the
bottom row.
