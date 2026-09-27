# Table — `tab:appendix-slot-comprehensive`

**Paper location**
Appendix \subsection{Comprehensive numbers across models}, `neurips_content_en/appendix.tex:149`

**What the experiment asks**
The full slot-machine scoreboard: for each model and each betting mode, how often it went broke, how long it played, how much it staked in total and how much it lost.

**HF path(s) of the raw data**
Slot machine, canonical six-model roster (each file's internal `model` field checked, not the folder name):
  - GPT-4o-mini      `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`  (`model` = `gpt-4o-mini-corrected`)
  - GPT-4.1-mini     `slot_machine/gpt/gpt5_experiment_20250921_174509.json`  (`model` = `gpt-4.1-mini`; the `gpt5_` filename is legacy and the folder name `gpt` does NOT mean GPT-4o-mini)
  - Gemini-2.5-Flash `slot_machine/gemini/gemini_experiment_20250920_042809.json`
  - Claude-3.5-Haiku `slot_machine/claude/claude_experiment_corrected_20250925.json`
  - LLaMA-3.1-8B     `behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json`
  - Gemma-2-9B       `behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json`

**Code that turns raw data into the printed values**
`scripts/tables/appendix_behavioural_tables.py` -> `paper_data/tables/appendix/slot_comprehensive.tex` (= HF `paper_neurips_2026/camera_ready/paper_data/tables/appendix/slot_comprehensive.tex`).

**Corpus vintage**
Canonical. `slot_machine/gemma/` and `slot_machine/llama/` (the V1 October-2025 runs) each carry a `DEPRECATION_WARNING.md` and are NOT used here. The release also mirrors the clean open-weight files at `slot_machine/gemma_v4_role/` and `slot_machine/llama_v4_role/`, identical filenames to the `behavioral/...` copies; either path is canonical, the bare `slot_machine/{gemma,llama}/` paths are not.

This is the reference table for detecting the mis-vintage failure elsewhere: it records LLaMA variable bankruptcy 72.31% and Gemma variable 5.44%. Any appendix float that prints a *low* LLaMA rate beside a *high* Gemma rate is reading the deprecated V1 corpora — which is exactly what `tab:information-not-bottleneck` turns out to do.

**Reproduction status**
VERIFIED, 2026-08-25. All 52 printed values reproduce; max deviation 0.000. Net P&L equals final_balance - 100 in every cell, i.e. the ledger identity holds.
