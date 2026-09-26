# `R` vs `H` Key — Canonical Convention

**Decision (2026-05-10, codex AC persona Option D verdict, Round 1 CONVERGED)**: keep `R` as the paper-canonical key for the *hidden-patterns* prompt module in the slot-machine experiment. The `H` letter appears only in the paper body / figures / captions, where a one-way display map `R → H` is applied at presentation time.

## Rule

| Surface | Use | Why |
|---|---|---|
| Source code (this directory, `sm_cap_ablation/`, `legacy/`) | `R` | matches `gpt_fixed_bet_size_experiment.py` (Figure 3d source) byte-parity contract |
| HF data labels (`prompt_combo` field, 1{,}600 corrected SM games) | `R` | already written; never relabel |
| Analysis pipelines / figure-generation scripts | `R` internally, `H` only at display | provenance preserved, presentation cleaned |
| pytest parity tests (22/22 currently) | `R` | rename would break parity |
| Paper body (`neurips_content_en/`, `neurips_content/`, `content/`) | `H` (prose, table, caption, figure label) | reader-facing mnemonic only |

## Why not rename `R` -> `H` in source

Codex AC persona verdict (objective NeurIPS reviewer simulation, Round 1, 2026-05-10): every alternative is either lossy (Option A — high byte-parity risk, HF provenance drift, high engineering, negative rebuttal impact) or inferior to keeping the canonical (Option B alias / Option C analysis-only display). Option D (keep `R`, document) wins on all four axes:

1. byte-parity risk — lowest (no test rewrite, no fixture churn)
2. HF backward-compat risk — lowest (1{,}600 games stay readable)
3. engineering hours — lowest (zero)
4. NeurIPS rebuttal impact — best (signals reproducibility discipline over cosmetic cleanup)

## How to apply

- Writing new pipeline code: use `'R'` for the hidden-patterns module key. Never introduce an `'H'` key in the slot-machine cap-ablation lineage.
- Reading HF data: filter combos by `'R' in combo_label`.
- Producing a paper figure or table: build a one-way map `{'R': 'H'}` at the *display* layer (figure caption, axis tick, table header). Do not write the renamed labels back to disk.
- Cross-referencing the `slot_machine_6models/` runners (which use `H` natively for the unrelated *modality* experiment): treat those as a separate experiment cohort. They are not part of the cap-ablation track and their `H` is consistent within their own cohort.

## Mirrored notes

- Claude auto-memory: `~/.claude/projects/-home-v-seungplee/memory/feedback_r_h_key_canonical.md` (persists across conversations)
- Paper repo: `LLM_Addiction_NMT_KOR/R_KEY_CANONICAL.md` (same content, mirrored for paper-side reference)
