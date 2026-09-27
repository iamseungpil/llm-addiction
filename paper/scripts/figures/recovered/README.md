# Recovered appendix-figure generators

Six of the camera-ready appendix figures had no generator on record. The
`PAPER_ASSET_MAP.md` row for each of them still reads `Generator | n/a`. This
directory closes that gap: every file here is a **verbatim** copy of the script
that drew the ancestor of a printed figure, pulled back out of the code repo's
git history.

Nothing here is a rerun target. These scripts read absolute `/data/llm_addiction/...`
and `/home/ubuntu/...` paths from the machine the experiments were run on, and — as
the caveats below spell out — several of them are one revision behind the art that
actually went to print. They are here as **provenance**: they say which corpus,
which arm and which arithmetic produced each printed panel, which is exactly what
`n/a` did not say.

## Where they came from

The code repo is `llm-addiction`. Commit `16acccf` ("clean files") deleted
`legacy/writing/table_figure/` wholesale, taking 22 scripts with it. Five of the
seven files here were read back out of its parent:

```
git show 16acccf^:legacy/writing/table_figure/<name>.py
```

`create_4x2_streak_analysis_CORRECTED.py` was deleted earlier, by `9a4ee94`, from
the repo root:

```
git show 9a4ee94^:create_4x2_streak_analysis_CORRECTED.py
```

`create_choice_distribution_cot.py` was never deleted. It is still live at
`legacy/investment_choice_bet_constraint_cot/analysis/create_choice_distribution_cot.py`
— paper-canonical code sitting under `legacy/`, which is precisely the thing the
release's own organising principle says should not happen. Copied here unchanged.

The five history extractions are byte-identical to their git blobs; the SHA-1 of
each working file matches `git show` of its source path.

## What each file drew

Figure numbers are the printed numbers in `neurips_en.pdf`. "Corpora" names the
`MODEL_PATTERNS` / `MODEL_PATHS` entries in the script itself.

### `create_component_effects_by_bettype.py` → Figure 5 (`component_effects_by_bettype2.pdf`)

- **Corpora** (4): `gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`,
  `gpt5_experiment/gpt5_experiment_20250921_174509.json` (GPT-4.1-mini despite the
  filename), `gemini_experiment/gemini_experiment_20250920_042809.json`,
  `claude_experiment/claude_experiment_corrected_20250925.json`.
- **Subset**: every game in all four files, split into the two arms by the game-level
  `bet_type` field (`fixed` / `variable`). No cap or condition filter.
- **Scan rule**: for each of the five components, a substring test on the game's
  `prompt_combo`; the with-component mean minus the without-component mean, computed
  per model and per arm, then averaged across the four models.
- **Prompt alphabet**: the x axis is labelled `G M P H W`, but the data is matched on
  `G M P R W` — the script remaps `H` to `R` before the substring test
  (`search_component = 'R' if component == 'H' else component`). This is the same
  H/R split the corpora carry: LLaMA writes `H` for hidden-patterns, the other five
  write `R`.
- **Bet field**: game-level `total_bet` for the plotted panel; round-level `bet`
  inside `compute_irrationality_index`, which is computed for every game and then
  not plotted (the fourth panel was dropped, leaving bankruptcy / total bet / rounds).
- **Round outcome**: `round_details[i].game_result` for the three API corpora,
  `game_history` for GPT-4o-mini — the schema split is handled correctly here.

### `create_4model_complexity_trend_average.py` → Figure 6 (`4model_complexity_trend_average3.pdf`)

- **Corpora** (4): identical to the above.
- **Subset**: every game; no arm split — fixed and variable are pooled.
- **Scan rule**: complexity is `sum(1 for c in 'GMPRW' if c in prompt_combo)`, i.e. a
  0–5 count over the `R` spelling of the alphabet. Games are grouped per model by
  complexity, each metric averaged within model, and the four model curves then
  averaged into one.
- **Bet field**: game-level `total_bet`. The other two panels are `is_bankrupt`
  (×100) and `total_rounds`. A fourth irrationality panel is computed and dropped.

### `create_individual_model_component_effects.py` → Figure 10 (`component_effects_all_models_3x4_2.pdf`)

- **Corpora** (4): identical.
- **Subset / scan rule / alphabet**: same with-minus-without component contrast and
  the same `H`→`R` remap as Figure 5, but laid out as a 3×4 grid — three metrics down,
  one model per column — instead of averaged across models.
- **Bet field**: game-level `total_bet`.

### `create_4model_complexity_trend_3x4.py` → Figure 11 (`4model_complexity_trend_3x4.pdf`)

- **Corpora** (4): identical.
- **Subset / scan rule**: same complexity count as Figure 6, but plotted per model
  (3 metrics × 4 models) rather than averaged, with a per-panel Pearson $r$.
- **Bet field**: game-level `total_bet`.

### `create_individual_model_streak_analysis.py` → Figure 12 (`individual_model_streak_analysis.pdf`)

- **Corpora** (4): three match the others, but this one alone reads GPT-4o-mini from
  `gpt_results_corrected/gpt_corrected_complete_20250911_071013.json`. See the
  caveats.
- **Subset**: every game with a `round_details` list at least `streak_length + 1`
  long. Streak lengths 1–5, both win streaks and loss streaks. Arms are pooled.
- **Scan rule**: **non-overlapping**. The cursor advances by `i += streak_length`
  after a match, so each round belongs to at most one counted streak. The measured
  quantities are the share of streaks whose next bet exceeds the last bet of the
  streak, and the share that continue at all.
- **Bet field**: round-level `bet_amount` (not `bet`).
- **Round outcome**: `round_details[i].game_result` — reading `won`, falling back to
  `win` — for the three API corpora. For GPT-4o-mini it does not read an outcome
  field at all; it infers the win from the next round's `balance_before`, which is
  what forced the corrected variant below.
- **Layout**: 2 rows (win streaks / loss streaks) × 4 model columns.

### `create_4x2_streak_analysis_CORRECTED.py` → corrected ancestor of Figure 12

Kept beside the above because it is the file that fixes the GPT-4o-mini read: it
points at `gpt_results_fixed_parsing/...` — the corpus every other generator uses —
and takes the outcome straight from the game-level `game_history` list rather than
inferring it from a balance delta. It also enumerates streaks **overlapping** rather
than non-overlapping, and lays out 4 rows × 2 columns at `figsize=(16, 20)`, so its
composition is not the printed one. Read it as the record of what was wrong with the
2×4 script's data source, not as the Figure 12 generator.

### `create_choice_distribution_cot.py` → Figure 13 (`investment_choice_distributions_cot.pdf`)

- **Corpus**: the investment-choice CoT results tree, one JSON per
  (model, bet_type, bet_constraint) cell. Note the script's own docstring names
  `investment_choice_bet_constraint_cot/results/` while `RESULTS_DIR` actually points
  at `investment_choice_extended_cot/results` — the docstring is stale, the constant
  is what ran.
- **Subset**: files are deduplicated to the **latest** file per
  (model, bet_type, bet_constraint) by reverse-sorted filename, then aggregated into
  `model_bettype` buckets. The bet-constraint caps are therefore **pooled**, not
  faceted — a printed panel mixes every cap for that model and arm.
- **Scan rule**: every `decision['choice']` in every game of the four conditions
  `BASE`, `G`, `M`, `GM`, counted into options 1–4 and normalised to percentages.
- **Layout**: 2 rows (fixed on top, variable below) × 4 model columns, stacked bars.

## Relationship to `paper_index/`

The per-figure entries under `paper_index/appendix/` are the authority on each
figure's verdict, corpus vintage and recompute status; they were written from these
same recovered scripts and they go further, recomputing several of the figures
against the release. What they carry is a `git show` command. What this directory
carries is the files themselves, so the provenance survives without a checkout of the
code repo at the right commit, and so the two accounts can be diffed.

One thing worth reading off the recovered code directly: every script here opens the
four API exports by absolute path and opens **no open-weight tree at all**. The figA01
README's claim that `fig:appendix-component-effects` was drawn from
`slot_machine/{gemma,llama}/final_*.json` — the deprecated V1 tree — describes none of
these scripts.

## Caveats, recorded rather than fixed

1. **Four models recovered, six models printed.** Figures 5 and 6 are captioned "all
   six models" and "all six open and closed models" in `appendix.tex`; their
   generators here read four API corpora and no open-weight corpus at all. The
   filename `4model_complexity_trend_average3.pdf` still carries the `4model` prefix
   into the camera-ready. The six-model revisions were never committed, so what
   survives is the four-model ancestor. The printed suffixes — `…_by_bettype2`,
   `…_average3`, `…_3x4_2` — are the revision counters of art these scripts do not
   themselves produce. Figures 10 to 13, by contrast, are captioned "the four API
   models", so for those the recovered scope is the printed scope.

2. **The streak generator names a superseded GPT-4o-mini export.** Its header points
   at `gpt_results_corrected/gpt_corrected_complete_20250911_071013.json`. No such
   directory or file is in the release. Every other generator in this directory reads
   the canonical
   `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`
   instead, and substituting that successor is what reproduces the printed panels
   (`paper_index/appendix/fig_appendix_model_streak.md` records the residual).
   Meanwhile this script's two `savefig` targets are the printed filename exactly,
   with no revision suffix. `create_4x2_streak_analysis_CORRECTED.py` is the variant
   that switched to the canonical export.

3. **Model display names drift between scripts.** `create_4model_complexity_trend_average.py`
   and `create_4x2_streak_analysis_CORRECTED.py` label Claude as `Claude-3.5-Sonnet`;
   `create_4model_complexity_trend_3x4.py` and `create_individual_model_component_effects.py`
   label it `Claude-3.5-Haiku`, which is the model the paper reports.
   `create_choice_distribution_cot.py` labels Gemini `Gemini-2.0-Flash` against the
   paper's Gemini-2.5-Flash. These are label bugs in the recovered ancestors; the
   underlying data files are the same four exports throughout.

4. **Figure 13 already has a modern regenerator, and it is not this file.**
   `scripts/figures/figA09_investment_choice_distributions_cot.py` re-authors that
   figure at print scale from `data/figA09_investment_choice_distributions_cot.json`;
   its own header notes that this recovered script "draws an earlier, differently
   ordered and differently coloured version". Use the `figA09_…` generator to
   reproduce the printed art and this one only to see where the numbers came from.
   The same applies to Figure 9 (`figA05_distortion_multimodel_summary.py`).

5. **Not runnable as-is.** Absolute `/data/llm_addiction/...` inputs and
   `/home/ubuntu/llm_addiction/writing/figures` outputs, and the corpora themselves
   live in the released dataset under different paths. Repointing `MODEL_PATTERNS`
   at the release is left undone deliberately — editing a recovered file would cost
   it the one property that makes it useful, which is being exactly what ran.
