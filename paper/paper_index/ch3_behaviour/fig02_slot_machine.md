# Figure — `fig:slot-machine`

**Paper location**
§3 \subsection{Quantitative Analysis}, `neurips_content_en/3.behavior.tex:16-18`; graphic `images/fig02_slot_machine.pdf`

**What the experiment asks**
When a model is allowed to pick its own bet size instead of being forced to bet $10, does it go broke more often, and does it raise its stake after a run of wins or losses?

**HF path(s) of the raw data**
Slot machine, canonical six-model roster (each file's internal `model` field checked, not the folder name):
  - GPT-4o-mini      `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`  (`model` = `gpt-4o-mini-corrected`)
  - GPT-4.1-mini     `slot_machine/gpt/gpt5_experiment_20250921_174509.json`  (`model` = `gpt-4.1-mini`; the `gpt5_` filename is legacy and the folder name `gpt` does NOT mean GPT-4o-mini)
  - Gemini-2.5-Flash `slot_machine/gemini/gemini_experiment_20250920_042809.json`
  - Claude-3.5-Haiku `slot_machine/claude/claude_experiment_corrected_20250925.json`
  - LLaMA-3.1-8B     `behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json`
  - Gemma-2-9B       `behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json`

**Code that turns raw data into the printed values**
Generator: `scripts/figures/fig02_slot_machine.py` (repo) = HF `paper_neurips_2026/camera_ready/scripts/figures/fig02_slot_machine.py`.
Loaders: `scripts/build_figure_data.py`.
Plotted values sidecar: `paper_data/fig02_slot_machine.json` = HF `paper_neurips_2026/camera_ready/paper_data/fig02_slot_machine.json`.

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

**Corpus vintage**
Canonical (V4role open-weight + corrected API). `slot_machine/gemma/` and `slot_machine/llama/` (the V1 October-2025 runs) each carry a `DEPRECATION_WARNING.md` and are NOT used here. The release also mirrors the clean open-weight files at `slot_machine/gemma_v4_role/` and `slot_machine/llama_v4_role/`, identical filenames to the `behavioral/...` copies; either path is canonical, the bare `slot_machine/{gemma,llama}/` paths are not.

History worth recording: in the *submitted* version, panels (c) and (d) were module-level literals inside `generate_paper_figures.py` with no derivation anywhere. That is the one item in this index that still does not reproduce; it is set out in full under *Reproduction status*, including why "the submitted values are wrong" is not the right description of it.

**Reproduction status**
All four panels reproduce from the released corpora.

**Panels (a) and (b).** Panel (a)'s bankruptcy counts are cross-checked cell by cell inside the
pipeline (`paper_data/tables/appendix/gbsa_companion_metrics.tex`), and the three round-level
indicators of panel (b) are recomputed from the release. Open-weight corpora: the role-prompt runs
`behavioral/slot_machine/{llama,gemma}_v4_role/`.

**Panels (c) and (d).** Drawn by `python3 scripts/figures/fig02_slot_machine.py` (default
`--cd-source recomputed`) from the same six corpora as panels (a) and (b), with the open-weight
role-prompt runs `behavioral/slot_machine/{llama,gemma}_v4_role/`, and percentile bootstrap
intervals over rounds.
  - Quantity: section 2's bet-to-balance ratio increase `max(0, (r_{t+1} - r_t) / r_t)`.
  - Binning: runs of exactly k identical outcomes for k = 1..4; the last bin is "5 or more".
  - Printed values at streak length 1: after a win, fixed 0 (by construction, a fixed $10 bet
    can only lower the ratio after a win), variable 0.28; after a loss, fixed 0.10, variable 0.58.
  - The submitted version drew (c)/(d) from the first open-weight runs
    `slot_machine/{llama,gemma}/` (DEPRECATION_WARNING: fixed arm not locked at $10), which
    produced a non-zero fixed post-win series and the 3.3x / 2.8x multipliers. That rendering is
    still available as `--cd-source submitted` and reproduced by `scripts/figures/fig02_cd_streaks.py`.

**Cross-manifest note.** The appendix model-by-model streak figure
(`appendix/fig_appendix_model_streak.md`) uses a disjoint streak scan; this figure uses exact run
length with a final "5 or more" bin.
