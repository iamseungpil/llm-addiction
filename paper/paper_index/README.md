# Paper index — every float in `neurips_content_en/`, and where its data lives

One manifest per figure and table in the paper. Each manifest carries six fields:
paper location, what the experiment asks in one plain sentence, HF path(s) of the raw data,
the code that turns raw data into the printed values, **corpus vintage**, and reproduction status.

Manifests whose printed values depend on round-level outcomes carry **two more**: a **schema map**
and a **load invariant**. Those two fields exist because the first version of this index recorded
*where* the data is without recording *what shape* it has, and a defect walked through that gap into
print. See *The gap that let a defect through* below.

This directory is **additive**. Nothing on HuggingFace or elsewhere in this repo was moved,
renamed or deleted to build it, and no `.tex` file was touched. The `legacy/` do-not-cite
markers stay exactly where they are.

Floats are enumerated from the paper itself (`grep -n '\label{fig:\|\label{tab:' neurips_content_en/`):
**52 labels on 51 floats — 17 figures and 34 tables.** One table float carries two labels (`tab:rq2-sharing` = `tab:sharing-transfer`), and its manifest
names both. The body condition-modulation float used to carry a second label,
`tab:appendix-condition-full`, which put the word "appendix" on a body table and was referenced
by nothing; it has been removed, so `tab:condition-modulation` is now that float's only label.

**In-body table numbers, checked against `neurips_en.aux` (the compiled ground truth, since table
numbering runs continuously across body and appendix with no counter reset).** The body carries
exactly three numbered tables: **Table 1** `tab:neurips-sae-results` (SAE readout $R^2$), **Table 2**
`tab:rq2-sharing`/`tab:sharing-transfer` (cross-task sharing audit), **Table 3**
`tab:condition-modulation`. Two manifests in `ch4_neural/` previously cited stale numbers from a
draft where the sharing table still lived in the appendix (calling it "body Table 3" and the
condition-modulation table "body Table 4"); both are corrected in this pass, and
`tab:rq2_sharing.md`'s paper-location line — which pointed at an `appendix.tex` line the float no
longer occupies — is corrected alongside it.

**Third pass.** The index was first built on 2026-08-25 and re-enumerated on 2026-08-27. This pass
added the schema-map and load-invariant fields, ran the invariant against the public release, and
recovered six generators that earlier passes had concluded did not exist. **The UNREPRODUCIBLE
column is now empty**, and one float is recorded as not reproducing for a reason that is not a data
defect. See *What this pass changed*.

---

## Why the corpus-vintage field is here

The release contains several corpora that look interchangeable and are not.

| Trap | What it looks like | What it is |
|---|---|---|
| `slot_machine/gpt/` | a GPT folder | **GPT-4.1-mini**, not GPT-4o-mini. The canonical GPT-4o-mini file is `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json` (`model` = `gpt-4o-mini-corrected`) |
| `slot_machine/gemma/`, `slot_machine/llama/` | the obvious open-weight paths | V1 October-2025 runs. Each carries a `DEPRECATION_WARNING.md`: Gemma **CORRUPTED, must not be used**; LLaMA **MILD CORRUPTION**. Canonical replacements: `behavioral/slot_machine/{gemma,llama}_v4_role/` (also mirrored at `slot_machine/{gemma,llama}_v4_role/`) |
| `sae_patching/` | an SAE analysis folder | carries a `DEPRECATION_WARNING.md`; built on the corrupted V1 data. Replacement: `sae_features_v3/` |
| `legacy/v17_leaky_pipeline/`, `legacy/pre_groupkfold_sweep/` | ordinary result files | label-leaking pipelines, do-not-cite per `NEURIPS_CANONICAL_INDEX.md` §5. **These carry no `DEPRECATION_WARNING.md` file** — grepping for that filename will not find them |
| `rq2_audit_consistent_layer.json` | a results file | an error stub (`layer 23 not in...`), not values |
| `*_SUPERSEDED_oldscript.json` under `nested_baseline_and_audits_e2/` | live results | withdrawn artefacts sitting beside the live ones |
| `figures/appendix/figA0N/code/` | the figure's generator | for five of the nine appendix figures it is **one script copied five times** that draws none of them. See *A `code/` directory is not a generator* below |
| a figure README's "Source data" line | the corpus that was read | boilerplate in at least two cases, and wrong in one: figA03's README names the deprecated V1 tree while its own generator opens the canonical V4role file |

**Rules applied in every manifest.** The `model` field inside a file decides which model it is,
never the directory name. Where a README and an executable generator disagree, the generator
decides. Only three `DEPRECATION_WARNING.md` files exist in the whole release —
`sae_patching/`, `slot_machine/gemma/`, `slot_machine/llama/`, re-confirmed 2026-08-27 against the
full 9,449-file HF listing — so the absence of one is *not* evidence a path is canonical.

### A `code/` directory is not a generator

Every one of the nine appendix figures figA01–figA09 ships a `code/*.py` on HF, so a file-level
check finds a generator for all nine. Reading them changes the picture:

| Figures | What is published | Draws the printed artwork? |
|---|---|---|
| A03 escalation, A05 distortion, A09 CoT distributions | a real, figure-specific script | **yes** — A09 even writes the exact printed filename |
| A04 temperature | a real script, but it parses an unpublished `.log` and draws grouped **bars** where the paper prints a **line plot with a shaded band** | no |
| A01, A02, A06, A07, A08 | `generate_behavioral_figures.py`, **byte-identical under all five**, which saves only `behavioral_gemma_vs_llama.png`, `bet_type_asymmetry.png` and `prompt_component_effects.png` | no |

The reverse trap also exists. `fig:sign-transfer` and `fig:cross-context-write` have no `figA0N`
bundle at all, which reads as no generator — but one script writes all three of their PDFs
(`fig_axis_alignment.pdf`, `fig_xctx_signmap.pdf`, `fig_xctx_ladders.pdf`), and it is published:
`scripts/gen_fig_cross_context_write.py`, mirrored on HF at
`paper_neurips_2026/camera_ready/scripts/gen_fig_cross_context_write.py`. Confirmed 2026-08-27 by
reading its three `savefig` targets and locating it in the dataset listing.
**Since superseded for `fig:cross-context-write`:** the float dropped its heatmap panel (now
`tab:causal-transfer-matrix`) and prints only the ladders, redrawn as
`images/fig_xctx_ladders_solo.pdf` by the newer `scripts/figures/fig_cross_context_write.py`. The
`fig_xctx_signmap.pdf` / non-solo `fig_xctx_ladders.pdf` that the script above writes are now orphan
outputs — see `ch4_causal/fig_cross_context_write.md`.

### A deleted script is not a missing script

The trap above has a third form, and it is the one that produced this index's longest-standing wrong
verdict. Five behavioural appendix figures were recorded across two passes as UNREPRODUCIBLE, "no
generator exists in the repo or in the HF upload". That statement was true and the conclusion drawn
from it was false. **There is a third place to look: the code repo's git history.** The scripts lived
in `iamseungpil/llm-addiction` under `legacy/writing/table_figure/`, and commit `16acccf`
("clean files") deleted the whole directory. Git still has every one of them:

```sh
git -C <llm-addiction> show 16acccf^:legacy/writing/table_figure/
```

Six generators came back that way, and each one writes either the printed filename outright or its
immediate predecessor under this project's trailing-digit revision marker
(`component_effects_by_bettype.pdf` to `...bettype2.pdf`, and so on):

| Float | Recovered generator | Writes |
|---|---|---|
| `fig:appendix-component-effects` | `create_component_effects_by_bettype.py` | `component_effects_by_bettype.pdf` (printed: `...bettype2.pdf`) |
| `fig:appendix-complexity` | `create_4model_complexity_trend_average.py` | `4model_complexity_trend_average.pdf` (printed: `...average3.pdf`) |
| `fig:appendix-model-components` | `create_individual_model_component_effects.py` | `component_effects_all_models_3x4.pdf` (printed: `..._3x4_2.pdf`) |
| `fig:appendix-model-complexity` | `create_4model_complexity_trend_3x4.py` | `4model_complexity_trend_3x4.pdf` — **the printed filename** |
| `fig:appendix-model-streak` | `create_individual_model_streak_analysis.py` | `individual_model_streak_analysis.pdf` — **the printed filename** |
| `fig:appendix-choice-distribution` | `legacy/investment_choice_bet_constraint_cot/analysis/create_choice_distribution_cot.py` (not deleted; in the working tree) | `investment_choice_distributions_cot.pdf` — **the printed filename** |

Recovering them settles more than the reproduction status. Each script's loader is a hard-coded
dictionary of absolute paths, so the roster is **read rather than inferred**, and that closes two
vintage questions this index had listed as open and unclosable: the abandoned GPT-5-mini run is
opened by none of them, and neither `slot_machine/gemma/` nor `slot_machine/llama/` appears in any
loader. The figA01 README's claim to the contrary is overruled under the rule already stated above —
where a README and an executable generator disagree, the generator decides.

### The gap that let a defect through

This index recorded, for every float, the HF path its data comes from. Every one of those paths was
correct. A defect reached print anyway, because knowing *where* a corpus is says nothing about *what
shape* it has, and the six slot-machine corpora do not agree on shape:

| Corpus | Round outcome lives at | Type |
|---|---|---|
| Gemma-2-9B, LLaMA-3.1-8B (`*_v4_role`) | `history[i].win`, equivalently `decisions[i].result` | `bool` / `"W"`-`"L"` `str` |
| GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku | `round_details[i].game_result.result` | **`dict`** |
| GPT-4o-mini (`gpt_results_fixed_parsing`) | `game_history[i].result` — no `game_result` on the round at all | `"W"`-`"L"` `str` |

`str(game_result).startswith("W")` on a dict returns `False` rather than raising, so a loader written
against one schema reports **0.000 wins** on the three corpora written against another. That is the
defect that reached print: three corpora at 0.00 and a pooled 15.4%.

The fix is a **load invariant** — an assertion with an expected value and a runnable command,
carried by every manifest whose printed values depend on round-level outcomes. For the slot-machine
corpora it is: *each corpus's round-level win rate, over outcome-bearing rounds, lies in
[0.25, 0.35]; task spec 0.30.* Per corpus, never pooled. Run against the public release this pass,
it passes on all six, every rate within 0.004 of 0.30. The full statement, the one-line command and
the expected table are in `ch3_behaviour/fig02_slot_machine.md` and repeated in the four other
round-outcome manifests.

**A manifest may not claim VERIFIED while its invariant is unrun.** Cell-by-cell agreement is not a
substitute: the defect that reached print produced correct-looking cells on the corpora it read
correctly, and a plausible-looking pooled number from the ones it did not.

### The subset is the other half, and it is measurable

A schema map says how to read a corpus. It does not say *which rows to read*, and for the behavioural
figures that second question decides the numbers as completely as the first. Every recovered generator
now has its subset recorded in its manifest: which betting arm, which streak-scanning rule, which bet
field, which averaging order, which prompt alphabet.

The subset does not have to be guessed. `fig:appendix-model-streak` shows the method: all four
combinations of betting arm and bet field were run against the submitted artwork's own value labels,
and one combination beat the others by a factor of nine. That measurement **overturned the recovered
generator's own behaviour** — the ancestor script pools the arms; the printed figure is variable-only.
A subset read off a script is a hypothesis; a subset that reproduces the printed values is a finding.

---

## Summary

| Reproduction status | Floats |
|---|---|
| VERIFIED in full | 22 |
| VERIFIED in part (rest unreproducible, undefined, or defective) | 6 |
| **WRONG** (printed values come from a non-canonical corpus) | **0** |
| **UNREPRODUCIBLE** (no generator draws the figure) | **0** |
| NOT CHECKED | 21 |
| **Total** | **49** |

The UNREPRODUCIBLE column emptied because the generators were found, not because the standard moved.
Four of the five floats that held it are now recomputed or recomputable from the release; the fifth
pair of panels that genuinely does not reproduce is Figure 2(c)/(d), and it fails for a reason that
is not a data defect — see below.

### What this pass changed

| Float | Was | Is | Why |
|---|---|---|---|
| `fig:appendix-component-effects` | UNREPRODUCIBLE, "highest vintage risk" | **VERIFIED (0.000)**, vintage **canonical** | generator recovered from a deleted commit; reproduces six-model, exact on the printed values; its hard-coded loader reads the roster instead of inferring it, which also clears the vintage |
| `fig:appendix-complexity` | UNREPRODUCIBLE, vintage at risk | **VERIFIED (0.000)**, vintage **canonical** | same recovery; all three printed r values reproduce exactly. The `4model` stem versus six-model caption is explained: the stem is inherited from the four-model ancestor, the printed asset is the six-model revision |
| `fig:appendix-model-streak` | UNREPRODUCIBLE, vintage undetermined | **PART VERIFIED** — top row reproduces (MAE 0.007 / 0.007 / 0.014 / 0.013), one bottom-row column open | generator recovered and it writes the printed filename. The subset was settled by measuring all four arm-by-bet-field combinations against the submitted artwork's own value labels rather than assumed, which is how the arm filter was found: the recovered ancestor pools the arms and the printed figure does not |
| `fig:appendix-model-complexity`, `fig:appendix-model-components` | UNREPRODUCIBLE | **NOT CHECKED — reproducible**, vintage **canonical** | generators recovered, all named corpora present in the release; no recompute run, so the verdict is NOT CHECKED and not better |
| `fig:appendix-choice-distribution` | NOT CHECKED, generator on HF only | NOT CHECKED, generator in **two** places | the same generator is in the code repo working tree; its de-duplication rule and pooled-cap aggregation are now recorded as the subset |
| `fig:slot-machine` | NOT CHECKED | **PART VERIFIED** — (a)(b) reproduce, (c)(d) do not | the one item in this index that still does not reproduce, recorded in full and without overstating it |
| `tab:appendix-streak-length` | NOT CHECKED | NOT CHECKED, **blocked on three named definitions** | not a data problem: escalation definition, streak scan rule and exact-versus-at-least-k are all unpinned |
| `tab:information-not-bottleneck` | **WRONG, 41.7 pp** | VERIFIED (0.0 pp) | the paper now prints the canonical four-corpus values, `\rev{}`-marked; the corrected fragment is wired in |
| `fig:escalation` | UNREPRODUCIBLE | VERIFIED (0.0004) | the generator exists on HF and reads the **canonical** V4role LLaMA corpus; both printed means come out of `escalation_results.json` exactly. The earlier miss was a caption defect, not a corpus one — 0.443 is a mean Spearman **ρ**, not the mean ratio the caption calls it |
| `fig:temperature-robustness` | UNREPRODUCIBLE, raw data "not locatable" | caption VERIFIED, artwork UNREPRODUCIBLE | the 16-cell sweep **is** in the release; the published generator draws a different figure |
| `fig:appendix-distortion-summary` | UNREPRODUCIBLE, vintage at risk | NOT CHECKED, vintage **canonical** | the generator hard-codes the canonical six corpora and touches neither deprecated path |
| `fig:appendix-choice-distribution` | UNREPRODUCIBLE, "four API models" warning | NOT CHECKED | a real generator and both CoT corpora are in the release; the vintage warning was copied from three sibling floats and did not apply |
| `fig:appendix-component-effects` and 4 siblings | UNREPRODUCIBLE, "no generator" | UNREPRODUCIBLE, narrowed | a `code/` directory does exist; what it holds does not draw these figures |
| `tab:worked-example-intervals`, `tab:e7-per-model`, `tab:companion-metrics` | not in the index | VERIFIED (0.0 pp) | three fragments listed here as "wired into nothing" have since been wired in |

Every verdict reversal recorded above, across all three passes, is the same mistake made in a new
place: concluding *this does not exist* from a search that did not go far enough. The second pass
learned to check the full HF listing rather than the repo-side asset map. This pass learned that a
script absent from both the paper repo and the HF upload may simply have been **deleted from the
code repo**, and that `git show <commit>^:<path>` is part of the search. Every "no source exists"
line in this index has now been checked against the HF listing *and* the code repo's history.

### Floats whose corpus vintage looks wrong

1. **`tab:condition-modulation`** (formerly also labelled `tab:appendix-condition-full`) — *wrong by provenance, not by corpus.*
   The six variable-arm rows come from the GroupKFold pipeline; the printed **Fixed** row silently comes
   from a different one (continuous-I_LC, plain CV). Same block, two pipelines, no note in the paper.
   → `ch4_neural/tab_appendix_condition_full.md`
2. ~~**`fig:appendix-component-effects`** — the strongest remaining vintage risk.~~ **Resolved this
   pass, canonical.** The figA01 README does name the deprecated V1 tree, but a generator now exists
   to overrule it, and it opens none of those paths. → `appendix/fig_appendix_component_effects.md`
3. ~~**Four more figures with no generator** — vintage undeterminable.~~ **Resolved this pass,
   canonical.** All four generators are recovered, and each loader is a hard-coded path dictionary,
   so the roster is read rather than inferred: the abandoned GPT-5-mini run is opened by none of them
   and `slot_machine/gpt/` is opened under its true identity, GPT-4.1-mini. The
   `4model_complexity_trend_average3.pdf` filename-versus-caption mismatch is likewise explained
   rather than merely noted: the `4model` stem is inherited from the four-model ancestor script and
   the printed asset is its six-model revision.
   One narrower question replaces the four broad ones: `create_individual_model_streak_analysis.py`
   names a GPT-4o-mini export, `gpt_results_corrected/gpt_corrected_complete_*.json`, that is **not
   in the public release**, so that one column of `fig:appendix-model-streak` can be approximated
   from public data but not reproduced from it. → `appendix/fig_appendix_model_streak.md`
4. **`fig:temperature-robustness`** — *provenance, not vintage.* The corpus is identified and canonical,
   but the printed artwork is not the output of the published generator, so nothing in the release
   accounts for the figure as drawn. → `appendix/fig_temperature_robustness.md`
5. **`tab:appendix-band-readout`** — *band-definition mismatch.* Evaluated at L14–19 for LLaMA IC/MW while
   the paper's own causal protocol names L12–17 and L16–21 for those tasks. Numerically small (≤0.0087),
   and the caption now discloses it in a `\rev{}` sentence, but it is still the wrong slice.
   → `ch4_neural/tab_appendix_band_readout.md`
6. **`tab:appendix-sweep-verification`, `tab:appendix-sweep-peak-mismatch`** — deliberately built on the
   deprecated `legacy/` pipelines. Legitimate as a transparency comparison, but those `legacy/` paths carry
   no `DEPRECATION_WARNING.md`, so the do-not-cite status is invisible to a file-level check.
7. **`tab:rq2-sharing`, row (i)** — the printed cosines are faithful to the released audit JSON, but two
   independent recomputes from the raw hidden states disagree with it and with each other. Three
   definitions of "the bankruptcy direction" coexist and do not agree.
8. **`tab:appendix-verbatim-quotes`** — not a vintage error, but a citation error: 4 of 12 rows fail their
   own provenance check (one has the wrong bet-type coordinate, two quote sentences that recur 16 and 69
   times, one is not verbatim).

**Resolved since the first pass:** `tab:information-not-bottleneck`, the float this index was built
to catch, and `fig:escalation`, which was suspected on the strength of a failed recompute that had
been computing the wrong quantity.

---

## Contents

### `ch3_behaviour/` — body §1 and §3

| Label | Manifest | Status |
|---|---|---|
| `fig:experimental-overview` | [fig01_experimental_overview.md](ch3_behaviour/fig01_experimental_overview.md) | NOT CHECKED (hand-drawn) |
| `fig:slot-machine` | [fig02_slot_machine.md](ch3_behaviour/fig02_slot_machine.md) | PART VERIFIED — (a)(b) reproduce / (c)(d) unattributed in the submitted version |
| `fig:investment-choice` | [fig03_investment_choice.md](ch3_behaviour/fig03_investment_choice.md) | NOT CHECKED |

### `ch4_neural/` — §4.1 readout and its appendix supplements

| Label | Manifest | Status |
|---|---|---|
| `tab:neurips-sae-results` (body Table 1) | [tab01_neurips_sae_results.md](ch4_neural/tab01_neurips_sae_results.md) | VERIFIED (0.000) |
| `tab:appendix-groupkfold-sweep` | [tab_appendix_groupkfold_sweep.md](ch4_neural/tab_appendix_groupkfold_sweep.md) | VERIFIED (0.000) |
| `tab:appendix-band-readout` | [tab_appendix_band_readout.md](ch4_neural/tab_appendix_band_readout.md) | VERIFIED (0.000) — band mismatch flagged |
| `tab:appendix-sweep-verification` | [tab_appendix_sweep_verification.md](ch4_neural/tab_appendix_sweep_verification.md) | NOT CHECKED |
| `tab:appendix-sweep-peak-mismatch` | [tab_appendix_sweep_peak_mismatch.md](ch4_neural/tab_appendix_sweep_peak_mismatch.md) | NOT CHECKED |
| `tab:neurips-selectivity-l2` | [tab_neurips_selectivity_l2.md](ch4_neural/tab_neurips_selectivity_l2.md) | VERIFIED (0.000) |
| `tab:appendix-selectivity-controls` | [tab_appendix_selectivity_controls.md](ch4_neural/tab_appendix_selectivity_controls.md) | VERIFIED (0.000) |
| `tab:hidden-subspace-audit` | [tab_hidden_subspace_audit.md](ch4_neural/tab_hidden_subspace_audit.md) | NOT CHECKED |
| `tab:appendix-behavior-convergence` | [tab_appendix_behavior_convergence.md](ch4_neural/tab_appendix_behavior_convergence.md) | PART VERIFIED / I_LC undefined — schema map + invariant carried |
| `tab:appendix-feature-overlap` | [tab_appendix_feature_overlap.md](ch4_neural/tab_appendix_feature_overlap.md) | NOT CHECKED |
| `fig:sharing` | [fig_sharing_loto_pca.md](ch4_neural/fig_sharing_loto_pca.md) | NOT CHECKED |
| `tab:rq2-sharing` **=** `tab:sharing-transfer` (body Table 2 — moved from appendix; see manifest) | [tab_rq2_sharing.md](ch4_neural/tab_rq2_sharing.md) | VERIFIED (0.000) — row (i) caveat |
| `tab:appendix-condition-multi` | [tab_appendix_condition_multi.md](ch4_neural/tab_appendix_condition_multi.md) | VERIFIED (0.000) |
| `tab:condition-modulation` (body Table 3; former `tab:appendix-condition-full` alias removed) | [tab_appendix_condition_full.md](ch4_neural/tab_appendix_condition_full.md) | PART VERIFIED / Fixed row wrong-by-provenance |

### `ch4_causal/` — §4.2–4.3 causal battery and its appendix

| Label | Manifest | Status |
|---|---|---|
| `fig:causal-battery` | [fig_causal_battery.md](ch4_causal/fig_causal_battery.md) | NOT CHECKED |
| `fig:causal-removal` | [fig_causal_removal.md](ch4_causal/fig_causal_removal.md) | NOT CHECKED (figure as drawn) — panels VERIFIED in the tables below |
| `tab:causal-battery-suffnec` (Table 23, appendix) | [tab_causal_battery_suffnec.md](ch4_causal/tab_causal_battery_suffnec.md) | VERIFIED (0.000) |
| `tab:causal-transfer-matrix` | [tab_causal_transfer_matrix.md](ch4_causal/tab_causal_transfer_matrix.md) | VERIFIED (0.000) |
| `fig:sign-transfer` | [fig_sign_transfer.md](ch4_causal/fig_sign_transfer.md) | NOT CHECKED |
| `tab:causal-condition-writability` | [tab_causal_condition_writability.md](ch4_causal/tab_causal_condition_writability.md) | VERIFIED (0.000) |
| `fig:cross-context-write` | [fig_cross_context_write.md](ch4_causal/fig_cross_context_write.md) | NOT CHECKED — now a single ladders panel; the heatmap this float used to carry is `tab:causal-transfer-matrix` (VERIFIED 0.000) |

### `appendix/` — behavioural appendix, prompt design, language instrument, model-by-model

| Label | Manifest | Status |
|---|---|---|
| `tab:appendix-slot-components` | [tab_appendix_slot_components.md](appendix/tab_appendix_slot_components.md) | NOT CHECKED (design key) |
| `tab:appendix-investment-payoff` | [tab_appendix_investment_payoff.md](appendix/tab_appendix_investment_payoff.md) | VERIFIED (0.000) |
| `tab:appendix-slot-comprehensive` | [tab_appendix_slot_comprehensive.md](appendix/tab_appendix_slot_comprehensive.md) | VERIFIED (0.000) |
| `tab:appendix-investment-comprehensive` | [tab_appendix_investment_comprehensive.md](appendix/tab_appendix_investment_comprehensive.md) | VERIFIED (0.000) |
| `fig:appendix-component-effects` | [fig_appendix_component_effects.md](appendix/fig_appendix_component_effects.md) | VERIFIED (0.000) — generator recovered, vintage canonical |
| `fig:appendix-complexity` | [fig_appendix_complexity.md](appendix/fig_appendix_complexity.md) | VERIFIED (0.000) — generator recovered, vintage canonical |
| `tab:appendix-streak-length` | [tab_appendix_streak_length.md](appendix/tab_appendix_streak_length.md) | NOT CHECKED — blocked on three unpinned definitions |
| `fig:escalation` | [fig_escalation_trajectory.md](appendix/fig_escalation_trajectory.md) | VERIFIED (0.0004) — caption calls a ρ an r |
| `fig:temperature-robustness` | [fig_temperature_robustness.md](appendix/fig_temperature_robustness.md) | caption VERIFIED / artwork UNREPRODUCIBLE |
| `tab:distortion-lexicon` | [tab_distortion_lexicon.md](appendix/tab_distortion_lexicon.md) | NOT CHECKED (instrument) |
| `tab:convergent-codebook` | [tab_convergent_codebook.md](appendix/tab_convergent_codebook.md) | NOT CHECKED (crosswalk) |
| `tab:instrument-robustness` | [tab_instrument_robustness.md](appendix/tab_instrument_robustness.md) | NOT CHECKED |
| `fig:appendix-distortion-summary` | [fig_appendix_distortion_summary.md](appendix/fig_appendix_distortion_summary.md) | NOT CHECKED — vintage canonical, schema map + invariant carried |
| `tab:appendix-verbatim-quotes` | [tab_appendix_verbatim_quotes.md](appendix/tab_appendix_verbatim_quotes.md) | PART VERIFIED — 4 of 12 rows defective |
| `fig:appendix-model-components` | [fig_appendix_model_components.md](appendix/fig_appendix_model_components.md) | NOT CHECKED — generator recovered, reproducible |
| `fig:appendix-model-complexity` | [fig_appendix_model_complexity.md](appendix/fig_appendix_model_complexity.md) | NOT CHECKED — generator recovered, reproducible |
| `fig:appendix-model-streak` | [fig_appendix_model_streak.md](appendix/fig_appendix_model_streak.md) | PART VERIFIED — top row reproduces / one column open |
| `fig:appendix-choice-distribution` | [fig_appendix_choice_distribution.md](appendix/fig_appendix_choice_distribution.md) | NOT CHECKED — generator in two places, subset recorded |
| `tab:companion-metrics` | [tab_companion_metrics.md](appendix/tab_companion_metrics.md) | VERIFIED (0.0 pp) |

### `rebuttal/` — floats resting on the `rebuttal_neurips_2026/` corpora

| Label | Manifest | Status |
|---|---|---|
| `tab:matched-cap-intervals` | [tab_matched_cap_intervals.md](rebuttal/tab_matched_cap_intervals.md) | NOT CHECKED — open public promise |
| `tab:matched-cap-companion` | [tab_matched_cap_companion.md](rebuttal/tab_matched_cap_companion.md) | NOT CHECKED — no generator found |
| `tab:choice-ladder` | [tab_choice_ladder.md](rebuttal/tab_choice_ladder.md) | NOT CHECKED |
| `tab:added-controls` | [tab_added_controls.md](rebuttal/tab_added_controls.md) | NOT CHECKED |
| `tab:worked-example-intervals` | [tab_worked_example_intervals.md](rebuttal/tab_worked_example_intervals.md) | VERIFIED (0.0 pp) |
| `tab:e7-per-model` | [tab_e7_per_model.md](rebuttal/tab_e7_per_model.md) | VERIFIED (0.0 pp) |
| `tab:exposure-matched` | [tab_exposure_matched.md](rebuttal/tab_exposure_matched.md) | VERIFIED (0.0 pp) |
| `tab:information-not-bottleneck` | [tab_information_not_bottleneck.md](rebuttal/tab_information_not_bottleneck.md) | VERIFIED (0.0 pp) — corrected |

---

## Fragments that exist but are wired into nothing

These live in `paper_data/tables/` and are `PROPOSED`; no `.tex` `\input`s them. They are listed so
nobody mistakes them for the printed floats or, conversely, overlooks work already done.

| Fragment | For | Note |
|---|---|---|
| `body/table1_PROPOSED_r2_pm_se.tex` | `tab:neurips-sae-results` | drops the p column (14 of 18 cells sit at the .005 floor); two variants |
| `body/table1_perm_null_N200.tex` | `tab:neurips-sae-results` caption | the 200-draw game-block permutation null; repo-only, not on HF |
| `appendix/A01_sae_full.tex` | `tab:appendix-sae-full` | **that label no longer exists in `neurips_content_en/`** — the float it was written for is gone |
| `appendix/A03b_band_readout_protocol_layers.tex` | `tab:appendix-band-readout` | the L12–17 / L16–21 protocol-layer rows |
| `appendix/verbatim_quotes_provenance.tex` | `tab:appendix-verbatim-quotes` | the provenance audit, not a replacement table |

Four fragments that were on this list in the first pass are no longer PROPOSED — they are now the
printed floats: `appendix/info_not_bottleneck.tex` (`tab:information-not-bottleneck`),
`appendix/worked_example_intervals.tex`, `appendix/gbsa_q3_e7_per_model.tex` and
`appendix/gbsa_companion_metrics.tex`.

---

## Sources this index was built from

- The paper: `neurips_content_en/*.tex` (52 labels; enumerated, not assumed)
- `NEURIPS_CANONICAL_INDEX.md` — the canonical-vs-deprecated ruling this index applies per float
- `PAPER_ASSET_MAP.md` — float → HF asset paths. **Its `n/a` generator entries are a repo-side map, not
  an HF listing**; treating them as the latter is what produced the first pass's nine "no generator"
  verdicts. They are now checked against the dataset itself.
- Provenance headers in `paper_data/tables/*/*.tex`, their `code/` generators, and the sidecars `paper_data/*.json`
- `scripts/figures/` (including `README_appendix_A03_A04_A05_A09.md` and `appendix_behavioral_panels.py`),
  `scripts/tables/`, `scripts/build_figure_data.py`
- `archive/rebuttal_20260731/VERIFIED_FACTS.md` and `archive/rebuttal_20260731/posted_openreview/`
- HF dataset `llm-addiction-research/llm-addiction` (gated): the complete 9,449-file listing,
  the figure READMEs and generators under `paper_neurips_2026/figures/`, plus targeted reads
- **The code repo's git history** (`iamseungpil/llm-addiction`), which earlier passes did not search:
  `legacy/writing/table_figure/` at `16acccf^`, six generators for the behavioural appendix figures,
  deleted by commit `16acccf` ("clean files") and still recoverable with `git show`
- The public release read directly, for the load invariant: six slot-machine corpora, round-level
  win rate per corpus

Every "VERIFIED" verdict carries its own date. A deviation is the maximum absolute difference between
a value printed in the paper's tabular body and the corresponding value in the independently
regenerated fragment or released JSON, taken over all compared cells. "NOT CHECKED" means exactly
that — no comparison was run — and is never a substitute for a guess.

Where a manifest carries a **load invariant**, that invariant gates its verdict: a manifest may not
claim VERIFIED while its invariant is unrun, however well the cells agree.

All 289 repo-local paths cited across these manifests were confirmed to exist on 2026-08-27.
Line numbers in the *Paper location* fields were current on that date; the labels are the stable
handle, since `appendix.tex` is under active edit.
