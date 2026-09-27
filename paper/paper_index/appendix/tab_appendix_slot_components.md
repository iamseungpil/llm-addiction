# Table — `tab:appendix-slot-components`

**Paper location**
Appendix \subsection{Prompt modules of the slot-machine experiment}, `neurips_content_en/appendix.tex:24`

**What the experiment asks**
A key to the five optional sentences that were switched on and off in the slot-machine prompts, and what each one was meant to do.

**HF path(s) of the raw data**
No measured quantity. The module texts themselves are in the experiment code, HF `paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py`, and are recoverable from the stored prompts inside every slot-machine corpus.

**Code that turns raw data into the printed values**
None — the table is written inline in `appendix.tex`.

**Corpus vintage**
Not applicable (design description, no corpus statistic). One naming trap belongs here: the fifth module is spelled **R** in the GPT / Claude / Gemma runs and **H** in the LLaMA V4role run. Any script that counts modules across models has to know that, and `scripts/tables/rebuttal_info_bottleneck.py` does.

**Reproduction status**
NOT CHECKED — nothing numeric to reproduce.
