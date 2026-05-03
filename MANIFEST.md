# MANIFEST — Paper claim → HF file mapping

This dataset accompanies the NeurIPS 2026 / EMNLP 2026 submission:

> **Can Large Language Models Develop Gambling Addiction?**
> A two-phase study of behavioural and representational risk under autonomy
> conditions in six closed and open LLMs.

This file maps every quantitative claim in the paper to a single
authoritative file in this repository. If a number appears in the paper that
is not traceable through this manifest, treat it as non-canonical until
re-audited.

---

## 1. Paper sections to canonical files

| Paper element | Claim summary | Canonical file | Producing script |
|---|---|---|---|
| Abstract + §1 Intro | Two-level finding (behavioural + representational) | (qualitative) | — |
| §2 Setup | $I_\text{BA}$, $I_\text{EC}$, $I_\text{LC}$ definitions | (definitional, no JSON) | — |
| §3 Behavior Table | 6-model bankruptcy + cognitive-distortion grid | `paper_experiments/slot_machine_6models/` (raw game JSON) | `paper_experiments/slot_machine_6models/src/run_*.py` |
| §3 Behavior Fig 4 | Round-level $I_\text{BA}$/$I_\text{EC}$/$I_\text{LC}$ | (computed from game JSON) | `paper_experiments/slot_machine_6models/src/analysis/*.py` |
| §4.1 Table 1 | SAE→indicator $R^2$ at L22 (13 reportable cells) | `sae_v3_analysis/results/table1_groupkfold_L22.json` | `sae_v3_analysis/src/run_groupkfold_recompute.py` |
| §4.1 prose "$p<0.0001$" | Game-block permutation null on headline cells | `sae_v3_analysis/results/table1_perm_null.json` | `sae_v3_analysis/src/run_table1_perm_null.py` |
| §4.2 Fig 5 PCA | LOTO PCA scatter at Gemma L22 | `sae_v3_analysis/results/figures/fig5b_pca*.pdf` | `scripts/gen_fig5b_pca.py` (in paper repo) |
| §4.2 Table (rq2-sharing) | Cosine + sparse-feature transfer + LOTO PCA AUC | `sae_v3_analysis/results/iba_cross_task_transfer.json` + `sae_v3_analysis/results/rq2_audit_consistent_layer.json` | `sae_v3_analysis/src/run_rq2_aligned_hidden_transfer_sweep.py` |
| §4.3 Table (condition modulation) | $R^2$ by ±G, ±M, fixed | `sae_v3_analysis/results/condition_modulation_groupkfold_L22.json` | `sae_v3_analysis/src/run_groupkfold_recompute.py` (§4.3 part) |
| §4.3 prose continuous I_LC | I_LC magnitude under conditions | `sae_v3_analysis/results/condition_modulation_continuous_ilc_L22.json` | `sae_v3_analysis/src/run_condition_modulation_continuous_ilc.py` |
| §4.4 Summary "+143% / +38%" | Computed from §4.3 Table 2 | (derived from cond_modulation file above) | — |
| Appendix C.1 Table 1 (V17 vs strict) | Reproducibility of leak-free CV | `sae_v3_analysis/results/robustness/permutation_null_ilc.json` | `sae_v3_analysis/src/run_perm_null_ilc.py` |
| Appendix C.2 GroupKFold sweep | $R^2$ at 5 layers × 13 cells | `sae_v3_analysis/results/table1_groupkfold_L{8,12,22,25,30}.json` | `sae_v3_analysis/src/run_groupkfold_layer_sweep.py` |
| Appendix C.3+ feature transfer | $I_\text{BA}$ feature overlap | `sae_v3_analysis/results/headline_robustness.json` | `sae_v3_analysis/src/run_headline_robustness.py` |

---

## 2. Pipeline canonical (§4)

For every §4 cell:

```
hidden state at L22  →  Gemma-Scope / Llama-Scope SAE features
                        → Top-K=200 features by |Spearman ρ| with target
                        → Within-fold RF deconfound on
                          [bal, rn, bal², log1p(bal), bal·rn]
                        → StandardScaler + Ridge(α=100)
                        → 5-fold CV, GroupKFold by game_id
```

Hyperparameters:

| | Value |
|---|---|
| Top-K SAE features | 200 |
| Spearman ρ | features ranked by `|ρ|` with deconfounded target |
| Ridge α | 100 |
| CV folds | 5 (`sklearn.model_selection.GroupKFold`) |
| Group key | `game_id` (1 game = ~30 rounds) |
| RF deconfound trees | 50 |
| RF deconfound depth | 8 |
| RF deconfound parallelism | `n_jobs=-1` |
| Layer reported (body) | L22 (Gemma + LLaMA) |
| Layers reported (appendix) | L8, L12, L22, L25, L30 |
| Permutation null iters | 50 (headline) / 20 (other) |

Reproduction:

```bash
conda activate llm-addiction

# §4.1 Table 1 + §4.3 Table 2 at L22
python sae_v3_analysis/src/run_groupkfold_recompute.py

# Appendix C.2 layer sweep
for L in 8 12 25 30; do
  python sae_v3_analysis/src/run_groupkfold_layer_sweep.py --layer $L
done

# §4.1 prose p<0.0001 (50-iter game-block null)
python sae_v3_analysis/src/run_table1_perm_null.py
```

---

## 3. Repository layout (paper-relevant)

```
sae_v3_analysis/
├── README.md                    # §4 paper-canonical pipeline section + lookup
├── src/
│   ├── run_groupkfold_recompute.py        # §4.1 + §4.3 (L22)
│   ├── run_groupkfold_layer_sweep.py      # appendix C.2 (per-layer)
│   ├── run_perm_null_ilc.py               # strict-CV pipeline (paper-canonical)
│   ├── run_table1_perm_null.py            # game-block permutation null
│   ├── run_rq2_aligned_hidden_transfer_sweep.py  # §4.2 transfer audit
│   ├── run_condition_modulation_continuous_ilc.py  # §4.3 continuous I_LC
│   └── run_comprehensive_robustness.py    # I_BA helper
├── scripts/
│   ├── build_appendix_layer_sweep_table.py  # appendix C.2 LaTeX builder
│   ├── push_groupkfold_to_hf.py             # this dataset uploader
│   └── relocate_legacy_to_hf.py             # legacy/ relocation script
└── results/
    ├── README.md                # results-level guide + paper-canonical section
    ├── table1_groupkfold_L22.json    # §4.1 Table 1 (13 cells)
    ├── table1_groupkfold_L{8,12,25,30}.json  # appendix C.2 sweep
    ├── condition_modulation_groupkfold_L22.json  # §4.3 Table 2
    ├── condition_modulation_continuous_ilc_L22.json
    ├── table1_perm_null.json    # §4.1 p<0.0001 source
    ├── iba_cross_task_transfer.json    # §4.2 sparse-feature transfer
    ├── rq2_audit_consistent_layer.json # §4.2 cosine alignment
    ├── headline_robustness.json
    ├── figures/                 # paper figures (kept, mostly fig5b_pca*)
    ├── json/                    # raw experiment outputs (V13 + RQ data)
    └── robustness/              # selectivity controls + null distributions

paper_experiments/
├── slot_machine_6models/        # §3 behavioural experiment (6 LLMs × 8 conditions)
├── investment_choice_experiment/ # §3 IC ablation + paper Fig 3
├── llama_sae_analysis/          # §4 SAE pipeline early version (use sae_v3_analysis)
└── pathway_token_analysis/      # §3 cognitive-distortion text analysis

legacy/                          # ARCHIVED — DO NOT CITE
├── v12_steering_invalidated/    # invalidated 2026-04-14 (prompts mismatch)
├── v14_steering/                # paper does not cite
├── v16_steering/                # paper does not cite
├── v17_leaky_pipeline/          # leaky RF deconfound (paper_neural_audit.json + v17 text)
└── pre_groupkfold_sweep/        # random-KFold sweeps; LLaMA cells diverge from body

sae_features_v3/                 # SAE feature activations + hidden states (raw inputs)
sae_patching/                    # ICLR-era SAE feature patching (separate study)
slot_machine/, investment_choice/, mystery_wheel/  # raw game outputs
behavioral/                      # canonical behavioral catalogues
```

---

## 4. Spot-check: §4.4 summary "+143%" trace

Paper §4.4 EN:
> "goal-setting most clearly sharpens the slot-machine $I_\text{BA}$ readout
> ($+143\%$ on Gemma, $+38\%$ on LLaMA)."

Trace:

```
+143% (Gemma)  →  Δ = (R²(+G) − R²(−G)) / R²(−G) × 100
              →  R²(+G) = 0.153  R²(−G) = 0.063
              →  source: condition_modulation_groupkfold_L22.json
                         ["gemma_sm_i_ba_L22"]["subsets"]["plus_G"]["r2_mean"] = 0.1525
                         ["gemma_sm_i_ba_L22"]["subsets"]["minus_G"]["r2_mean"] = 0.0628
              →  computed +143.2% (matches +143% within rounding)
              →  produced by: src/run_groupkfold_recompute.py (§4.3 part)
```

Spot-check verifier:
```bash
python scripts/verify_section4_numbers.py     # paper repo
```
expected: `Summary: 0 FAIL/MISMATCH out of total checks`.

---

## 5. Versioning

| Date | Event |
|------|-------|
| 2026-04-14 | V12 steering invalidated (prompts mismatch) |
| 2026-05-02 | V17 leaky RF deconfound discovered → strict-CV pipeline adopted |
| 2026-05-03 | GroupKFold by game_id transition (paper §4.1 + §4.3 recomputed) |
| 2026-05-03 | Pre-GroupKFold sweep_3metrics quarantined |
| 2026-05-03 | Appendix C.2 GroupKFold layer sweep added |
| 2026-05-03 | This MANIFEST created; legacy/ structure finalized |

---

## 6. Contact

- Repository: `llm-addiction-research/llm-addiction` (HuggingFace dataset)
- Paper code: `https://github.com/iamseungpil/llm-addiction` (GitHub)
- Paper revisions: see `paper_experiments/` for §3 + `sae_v3_analysis/` for §4

For reproducibility issues, open an issue on GitHub or contact the
corresponding author listed in the paper.
