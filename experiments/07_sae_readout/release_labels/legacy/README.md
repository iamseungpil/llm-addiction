# `legacy/` — the convention

Nothing in this release is ever deleted. When a file turns out to be wrong,
superseded, or built on a corrupted input, it is **moved here and labelled**, so that
the record of what was run stays complete and the reason it is not cited stays
attached to the files themselves.

## The two markers, and why there are two

| Marker | What it means | Who reads it |
|---|---|---|
| `README.md` | prose: what this directory holds, why it was retired, what replaced it | a human who has already opened the directory |
| `DEPRECATION_WARNING.md` | the machine-checkable flag: **do not cite anything beside this file** | anyone grepping the release by filename |

Both belong in every retired directory. A `README.md` alone is not enough, because
the release's own audit trail checks for the *filename* `DEPRECATION_WARNING.md`, and
a directory carrying only a README reads as clean to that check. This was a real gap:
`v17_leaky_pipeline/` and `pre_groupkfold_sweep/` carried do-not-cite READMEs and no
warning file, so `paper_index/README.md` had to record in prose that "these carry no
`DEPRECATION_WARNING.md` file — grepping for that filename will not find them". Both
now carry one.

**The absence of a `DEPRECATION_WARNING.md` is not evidence that a path is canonical.**
The authority on what the paper cites is `NEURIPS_CANONICAL_INDEX.md` §5 (the
do-not-cite table) together with `PAPER_CANONICAL_CODE.md` (the figure-to-code map).
The warning files mirror those documents; they do not replace them.

## What is here

| Directory | Retired because |
|---|---|
| `v12_steering_invalidated/` | steering prompts did not match the §3 behavioural prompt distribution (no ROLE_INSTRUCTION, no G/M variants) |
| `v14_steering/` | SAE-feature-level steering follow-up; mostly null at corrected scale; paper does not cite |
| `v16_steering/` | multilayer steering; collapses to direction steering at scale; paper does not cite |
| `v17_leaky_pipeline/` | RandomForest deconfound fit before the CV split → label leakage, R² inflated (LLaMA/MW 0.293 → 0.779) |
| `pre_groupkfold_sweep/` | random-KFold layer sweeps → leakage across rounds of the same game; source of the abandoned L24/L16 peak-layer route |

All steering claims were removed from §4 of the paper, which is why the three steering
directories are retired as a group rather than individually corrected.

## A caution about the word "legacy"

`legacy/` **in this released dataset** means retired and do-not-cite, as above.

`legacy/` **in the `llm-addiction` code repository** does not. Several directories
under that name still hold paper-canonical code — see `PAPER_CANONICAL_CODE.md`,
which exists specifically to say so. Do not carry the meaning of one across to the
other.

## Retired paths that live outside `legacy/`

Three more do-not-cite paths are listed in `NEURIPS_CANONICAL_INDEX.md` §5 but were
never relocated here, so this file is not a complete index of what not to cite:

- `slot_machine/gpt/archived_gpt5mini_20250921/` — GPT-5-mini run abandoned at 9.4%
  after a 60% API error rate; replaced by GPT-4.1-mini
- `rq2_audit_consistent_layer.json` — an error stub, not values; replaced by
  `rq2_aligned_hidden_transfer_*L22_r1*.json`
- `*_SUPERSEDED_oldscript.json` under `nested_baseline_and_audits_e2/` — withdrawn
  artefacts sitting beside the live ones

`slot_machine/gemma/`, `slot_machine/llama/` and `sae_patching/` are also retired and
were likewise not relocated, but each already carries its own
`DEPRECATION_WARNING.md` in place.
