# Slot Machine Irrationality Evaluation

| Experiment | Total Games | EV Calc Rate | Bankrupt Rate | GPI | Data Path |
|-----------|-------------|--------------|---------------|-----|-----------|
| Claude | 3200 | 60.0% | 10.2% | 13.5% | `/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json` |
| Gemini | 3200 | 73.3% | 25.6% | 25.1% | `/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json` |
| GPT-5 | 3200 | 85.2% | 3.2% | 3.7% | `/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json` |
| GPT-Corrected | 3200 | 50.3% | 5.7% | 8.1% | `/data/llm_addiction/ARCHIVE_NON_ESSENTIAL/gpt_results_corrected/gpt_corrected_complete_20250911_071013.json` |

![Slot Machine Metrics](slot_machine_ev_metrics.png)

## Prompt Highlights
### Claude
- Worst prompt `GMW` → GPI 59.0% (EV-calculated rounds: 39)
- Best prompt `MP` → GPI 0.0% (EV-calculated rounds: 99)

### Gemini
- Worst prompt `GMW` → GPI 58.9% (EV-calculated rounds: 73)
- Best prompt `P` → GPI 0.0% (EV-calculated rounds: 97)

### GPT-5
- Worst prompt `MPW` → GPI 32.0% (EV-calculated rounds: 100)
- Best prompt `BASE` → GPI 0.0% (EV-calculated rounds: 86)

### GPT-Corrected
- Worst prompt `MPRW` → GPI 22.2% (EV-calculated rounds: 99)
- Best prompt `M` → GPI 0.0% (EV-calculated rounds: 28)

