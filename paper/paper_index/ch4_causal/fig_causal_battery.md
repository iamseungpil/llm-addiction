# Figure — `fig:causal-battery`

**Paper location**
§4, `neurips_content_en/4.neural.tex:47-49`; graphic `images/fig04_causal_battery.pdf`

**What the experiment asks**
If you reach into the model and push the 'bet bigger' direction harder or delete it entirely, does the amount it actually bets move accordingly?

**HF path(s) of the raw data**
Gemma dose ladder: `experiments/sec4_causal/checkpoints/sec4_p0/sec4_{behavioural,readout,confound}_a*.jsonl`
Gemma random-direction nulls: `experiments/sec4_causal/checkpoints/sec4_w2/sec4_w2_null{1..20}_a{m3,p3}.jsonl`
LLaMA ladder and nulls: `experiments/sec4_causal/checkpoints/sec4_w10/`
Removal (necessity): `experiments/sec4_causal/checkpoints/sec4_w13/`
Direction vectors: `experiments/sec4_causal/assets/*.npz`

**Code that turns raw data into the printed values**
`scripts/figures/fig04_causal_battery.py` (= HF `paper_neurips_2026/camera_ready/scripts/figures/fig04_causal_battery.py`); sidecar `paper_data/fig04_causal_battery.json`.
Wave-level narrative and adjudication log: GitHub `iamseungpil/llm-addiction` `multilayer_causal/experiments/sec4_causal/INDEX.md` (W1-W14) — canonical per `NEURIPS_CANONICAL_INDEX.md`.

**Corpus vintage**
Canonical §4 causal waves. The sidecar records that the Gemma halves were read from a machine-local `multilayer_causal/results/` tree while the LLaMA and removal halves were read from HF snapshot `21bdaa32904abf9f54c257ccdd455be51ab7d1f7` — same data, two access paths. No `DEPRECATION_WARNING.md` exists anywhere under `experiments/`.

Four things the shipped figure got wrong and this generator fixes, all recorded in its docstring: the random-direction band was drawn as a continuous ribbon although nulls were only *run* at some doses; dose means are over parse-ok subsets, not over the 200 games run (Gemma balance at alpha=-3 drops to 0.790 parse rate, below the pre-registered 0.80 gate); removal is seed-paired; and on Gemma removing the balance control *raises* betting.

**Reproduction status**
NOT CHECKED (figure as drawn). Its numbers are the same ones VERIFIED in `ch4_causal/tab_causal_battery_suffnec.md`.
