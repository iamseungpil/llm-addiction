# analysis — early analysis and test scripts (not cited)

These scripts were written while the open-weight runs were in progress. The paper does not use them;
its numbers come from the folders listed in the [top-level README](../README.md).

| Script | What it does |
|---|---|
| `analysis_llama_slot_machine.py` | Summary of the first LLaMA-3.1-8B slot-machine run |
| `analysis_v1_gemma_data.py` | Summary of the first Gemma slot-machine run |
| `analyze_completed_experiments.py` | Summary over whichever experiments had finished |
| `analyze_investment_detailed.py`, `analyze_investment_fullset.py` | Investment-choice summaries |
| `monitor_experiments.py` | Watches running experiments and writes a report when they finish |
| `test_gemma_fix.py`, `test_llama_chat_template.py` | Quick checks of model loading and chat templates |

They read from absolute paths on the original machines. No paper data is stored here; the released
game logs are on the [Hugging Face dataset](https://huggingface.co/datasets/llm-addiction-research/llm-addiction).
