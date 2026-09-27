# Table (body Table 1) — `tab:neurips-sae-results`

**Paper location**
§4 \subsection{Can behavioural indicators be recovered from internal representations?}, `neurips_content_en/4.neural.tex:23`

**What the experiment asks**
Reading only the model's internal activity at the moment it decides, how much of its risky-betting behaviour can a simple predictor recover?

**HF path(s) of the raw data**
`sae_v3_analysis/results/table1_groupkfold_L22.json` (R^2 and n for all 18 cells).
Underlying features: `sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/{gemma,llama}/sae_features_L22.npz`; where no SAE npz exists (LLaMA SM and MW) `hidden_states_dp.npz` at the same path.
Labels come from the canonical behavioural corpora (`behavioral/{slot_machine,investment_choice,mystery_wheel}/...`).
Permutation p quoted in the caption: `paper_data/table1_perm_null_N200.json` (repo), 200 game-block draws.

**Code that turns raw data into the printed values**
Original pipeline: HF `sae_v3_analysis/src/run_groupkfold_recompute.py`.
Independent regeneration: `scripts/tables/body_tables.py` -> `paper_data/tables/body/table1_sae_readout_r2.tex`.
Permutation null: `scripts/tables/table1_perm_null.py` -> `paper_data/tables/body/table1_perm_null_N200.tex` (repo-only; not in the HF camera_ready upload).
A proposed +-SE variant sits at `paper_data/tables/body/table1_PROPOSED_r2_pm_se.tex` and is NOT applied to any .tex.

**Corpus vintage**
Canonical: strict GroupKFold-by-game-id at the fixed representative layer L22. The DEPRECATED alternatives are `legacy/v17_leaky_pipeline/paper_neural_audit.json` (RandomForest fitted before the CV split -> label leakage, R^2 inflated e.g. LLaMA/MW 0.293 -> 0.779) and `legacy/pre_groupkfold_sweep/` (random-KFold leakage, source of the abandoned L24/L16 peak-layer route). Neither is read here. No `DEPRECATION_WARNING.md` file sits beside any path used.

**Reproduction status**
VERIFIED, 2026-08-25. All 18 printed R^2 cells match the independently regenerated fragment exactly; max deviation 0.000. `NEURIPS_CANONICAL_INDEX.md` separately records a 13/13-cell exact match against `table1_groupkfold_L22.json` at HF snapshot `21bdaa32904abf9f54c257ccdd455be51ab7d1f7`.

Two caveats the regeneration surfaces and the paper does not: (i) no per-cell L22 permutation p exists anywhere in the *released* corpus, so the paper's cell-suppression decision is not reproducible from the release alone — the repo-only `table1_perm_null_N200.json` was computed to fill that hole; (ii) with 200 draws the p floor is 1/201 = .005 and 14 of 18 cells sit exactly at it.
