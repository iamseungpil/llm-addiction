# Table — `tab:hidden-subspace-audit`

**Paper location**
Appendix \subsection{Cross-task sharing at the hidden-state and sparse-feature levels}, `neurips_content_en/appendix.tex:649`

**What the experiment asks**
Do the three games share one internal 'about to go broke' direction, or does each game have its own?

**HF path(s) of the raw data**
`sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json`; hidden states `sae_features_v3/{task}/{gemma,llama}/hidden_states_dp.npz` at L22.

**Code that turns raw data into the printed values**
NOT CHECKED — no regeneration fragment was produced for this float (`scripts/tables/appendix_neural_tables.py` covers the sibling `tab:rq2-sharing` but not this one).

**Corpus vintage**
Canonical (`sae_features_v3/`, L22). The DEPRECATED sibling is `sae_patching/`, which carries a `DEPRECATION_WARNING.md` because it was built on the corrupted V1 slot-machine data; it is not read here. Related trap recorded in `NEURIPS_CANONICAL_INDEX.md`: the HF manifest once pointed at `rq2_audit_consistent_layer.json`, which is an **error stub** (`layer 23 not in...`), not real values.

**Reproduction status**
NOT CHECKED — no generator and no fragment. Note that the sibling table's regeneration found the released `shared_subspace_hidden_audit_20260410.json` cosines do not reproduce from `hidden_states_dp.npz` under two natural definitions of 'the bankruptcy direction'; see `ch4_neural/tab_rq2_sharing.md`.
