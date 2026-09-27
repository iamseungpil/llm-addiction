# Table — `tab:appendix-feature-overlap`

**Paper location**
Appendix \subsection{Cross-task sharing at the hidden-state and sparse-feature levels}, `neurips_content_en/appendix.tex:692`

**What the experiment asks**
Do the two hundred internal features chosen for one game overlap with the two hundred chosen for another game more than chance would give?

**HF path(s) of the raw data**
`sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/{gemma,llama}/sae_features_L22.npz` (top-200 selection by |Spearman| against I_BA).

**Code that turns raw data into the printed values**
NOT CHECKED — no regeneration fragment for this float.

**Corpus vintage**
Canonical `sae_features_v3/` at L22. The deprecated `sae_patching/` tree (which carries a `DEPRECATION_WARNING.md`) is not read. The cross-model rows are mechanically uninformative because Gemma and LLaMA use different dictionaries (GemmaScope 131K vs LlamaScope 32K), which the caption states.

**Reproduction status**
NOT CHECKED — no generator and no fragment.
