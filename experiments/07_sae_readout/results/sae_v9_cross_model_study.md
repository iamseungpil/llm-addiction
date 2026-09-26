# V9: Cross-Model, Cross-Domain Neural Basis of Risky Decision-Making in LLMs

**Authors**: Seungpil Lee, Donghyeon Shin, Yunjeong Lee, Sundong Kim (GIST)
**Models**: Gemma-2-9B-IT (42L, 3584-dim, GemmaScope 131K features) | LLaMA-3.1-8B-Instruct (32L, 4096-dim, LlamaScope 32K features)
**Paradigms**: Investment Choice (IC, 1600 games), Slot Machine (SM, 3200 games), Mystery Wheel (MW, 3200 games)
**Data**: Gemma: IC+SM+MW; LLaMA: IC+SM (MW pending extraction)
**Verified data sources**: `b1_b2_results_20260317_125620.json`, `llama_ic_results_20260317_130655.json`, `gap_filling_20260317_194117.json`, `final_verification_20260317_201024.json`, `llama_symmetric_20260318.json`, `selfcritique_20260318_114430.json`, `llama_hidden_analyses_20260318_172609.json`, `hidden_state_v8style_20260319_133042.json`
**Date**: 2026-03-19 (v9.5 — V8 hidden-state analyses reproduced + SAE analyses integrated)

> **Note on data integrity**: All numerical claims in this report are directly traceable to computed JSON outputs. Unverifiable numbers from previous drafts have been removed or replaced with verified results.

---

## Central Question

**Do different models, across different gambling tasks, under different prompt conditions, share a common neural basis for bankruptcy — the most extreme risk outcome?**

Three sub-questions:
- **RQ1**: Do Gemma and LLaMA share common activation/feature patterns that predict bankruptcy?
- **RQ2**: Do activation patterns remain invariant when the domain (IC, SM, MW) changes?
- **RQ3**: Are neural representations of risky behavior consistent across prompt conditions (Fixed/Variable, G/M/W/P/R)?

---

## Executive Summary

**RQ1 — Universal BK Features Exist Across Architectures.** 600 of 3,584 Gemma L22 neurons (16.7%) are sign-consistent universal BK predictors across three paradigms (FDR p<0.01). 744 SAE features across 7 layers show cross-domain consistency (L18+ binomial p<5e-3). LLaMA achieves near-identical BK classification (AUC 0.974 vs Gemma 0.976 in SM). Factor decomposition confirms 65–71% of SAE features encode BK independently of bet-type and paradigm (permutation p=0.000, null ~1%).

**RQ2 — Strong Domain Invariance (Gemma), Selective Invariance (LLaMA).** Gemma cross-domain transfer reaches AUC 0.87–0.93 (IC→MW, SM→MW; all perm p=0.000). A 3D shared BK subspace achieves AUC 0.86–0.97 despite near-orthogonal LR weight vectors. LLaMA IC→SM transfer succeeds only at L25 (AUC 0.783), while SM→IC is more stable (AUC 0.685 at L30). Hidden state transfer exceeds SAE transfer by 0.10–0.25 in most directions, indicating BK signals are distributed rather than sparse.

**RQ3 — BK Representation Is Condition-Invariant, but Condition Effects Are Model-Dependent.** LLaMA IC shows cos>0.81 between Variable and Fixed BK directions across all layers (n≥65 per group). Both models have ~33–37% shared BK neurons (bet-type-independent). G(Goal) prompt dominates BK behaviorally in Gemma (20.75x) but not LLaMA (1.26x), yet both models share the mechanism: G drives BK independently without modulating existing BK representations (amplifies_bk≈0). Warning effects are shallow-only in both models.

---

## 1. Setup

### 1.1 Data & Representations

**Table 1. Dataset Overview**

| Paradigm | Games | BK | BK% | Bet Types | Constraints | Prompt Conditions |
|----------|:-----:|:---:|:----:|-----------|-------------|-------------------|
| IC (Gemma) | 1,600 | 172 | 10.8% | Fixed/Variable | c10/c30/c50/c70 | BASE/G/M/GM |
| SM (Gemma) | 3,200 | 87 | 2.7% | Fixed/Variable | $10 fixed | 32 prompt combos (G/M/W/P/R) |
| MW (Gemma) | 3,200 | 54 | 1.7% | Fixed/Variable | $30 fixed | 32 prompt combos |
| IC (LLaMA) | 1,600 | 142 | 8.9% | Fixed/Variable | c10/c30/c50/c70 | BASE/G/M/GM |
| SM (LLaMA) | 3,200 | 1,164 | 36.4% | Fixed/Variable | $10 fixed | 32 prompt combos (G/M/H/W/P) |

**Hidden states**: Residual stream at each layer (Gemma: 3,584-dim × 42 layers; LLaMA: 4,096-dim × 32 layers).
**Decision Point (DP)**: Last decision time point. **Round 1 (R1)**: First round ($100 equal balance).
**SAE**: GemmaScope 131K features/layer; LlamaScope 32K features/layer.
**Pipeline**: StandardScaler → PCA(50) → LogReg(C=1.0, balanced) | 5-fold StratifiedKFold.

### 1.2 V8 Limitations Addressed

V8 (Gemma-only) left three weaknesses that V9 directly addresses:

1. **SAE cross-domain failure**: Only 1 of 131K features (#101036) was sign-consistent across 3 paradigms at L22 top-50. V9 expands to 7 layers × all active features → **744 sign-consistent features** (§2.2).
2. **Variable/Fixed "dissociation"**: Variable games were behaviorally riskier yet showed lower BK-projection. V9 shows this was an artifact of comparing all games rather than BK-only games (§4.1).
3. **Single model**: No LLaMA data. V9 adds LLaMA IC+SM with symmetric analyses (§2.3, §3.2, §4.4).

### 1.3 Cross-Model BK Rate Asymmetry

**Table 2. BK Rate Asymmetry Between Models**

| | Gemma IC | LLaMA IC | Gemma SM | LLaMA SM |
|--|:--------:|:--------:|:--------:|:--------:|
| Fixed BK rate | **19.75%** | 8.1% | — | — |
| Variable BK rate | 1.75% | **9.6%** | — | — |
| Overall BK rate | 10.8% | 8.9% | 2.7% | **36.4%** |
| Autonomy effect | -18.0pp (Fixed→Var: risk ↓) | +1.5pp (mild ↑) | — | — |

Gemma IC: Fixed betting is 11× riskier than Variable. LLaMA IC: nearly equal. Gemma SM BK rate is 13× lower than LLaMA SM. These asymmetries complicate cross-model comparisons and must be considered when interpreting all subsequent analyses.

---

## 2. RQ1: Universal BK Features

### 2.1 600 Universal BK Neurons (Gemma L22, Hidden States)

At L22, each of 3,584 neurons was tested for point-biserial correlation with BK outcome in each paradigm, with FDR correction (BH, p<0.01). Neurons significant in all three paradigms with consistent sign direction are "Universal BK Neurons."

**Table 3. Universal BK Neuron Summary (Gemma L22)**

| | IC | SM | MW | Cross-paradigm |
|--|:--:|:--:|:--:|:-:|
| FDR-significant neurons (p<0.01) | 2,528 (70.5%) | 2,725 (76.0%) | 2,314 (64.6%) | — |
| All-3-significant | — | — | — | 1,238 (34.5%) |
| **Sign-consistent (Universal)** | — | — | — | **600 (16.7%)** |
| BK-promoting | — | — | — | 302 |
| BK-inhibiting | — | — | — | 298 |

**Table 4. Top-5 Universal BK Neurons**

| Rank | Neuron | min|r| | IC r | SM r | MW r | Direction |
|:----:|:------:|:--------:|:----:|:----:|:----:|:---------:|
| 1 | **1763** | 0.217 | +0.305 | +0.248 | +0.217 | BK-promoting |
| 2 | 371 | 0.203 | -0.223 | -0.203 | -0.209 | BK-inhibiting |
| 3 | 2951 | 0.198 | -0.198 | -0.207 | -0.266 | BK-inhibiting |
| 4 | 1755 | 0.190 | -0.210 | -0.198 | -0.190 | BK-inhibiting |
| 5 | 864 | 0.182 | -0.267 | -0.182 | -0.182 | BK-inhibiting |

Neuron #1763 is the strongest universal BK-promoting neuron (min|r|=0.217). Promoting (302) ≈ Inhibiting (298) at L22, indicating a balanced representation. This exactly reproduces V8's finding (600 neurons, #1763 top).

### 2.2 744 SAE Features with Cross-Domain Consistency (Gemma, 7 Layers)

V8 examined only L22 top-50 features (0.038% of 131K). V9 tests 7 informative layers (L10, 12, 18, 22, 26, 30, 33) using all active features. A feature is "sign-consistent" if its Cohen's d has the same sign across IC, SM, and MW, with geometric mean |d| ≥ 0.2 for "strong" features.

**Table 5. Gemma Multi-Layer SAE Cross-Domain Consistency**

| Layer | Active | Sign-consistent | Strong (geo_mean d≥0.2) | % consistent |
|-------|:------:|:---------------:|:-----------------------:|:------------:|
| L10   | 92     | 30              | 23                      | 32.6%        |
| L12   | 112    | 35              | 31                      | 31.2%        |
| L18   | 170    | 58              | 52                      | 34.1%        |
| L22   | 281    | 109             | 95                      | 38.8%        |
| L26   | 303    | 133             | 119                     | 43.9%        |
| L30   | 415    | 188             | 173                     | 45.3%        |
| L33   | 456    | 191             | 172                     | 41.9%        |
| **Total** | —  | **744**         | **665**                 | —            |

V8 baseline: 1 feature → V9: 744 features (665 strong).

**Table 6. Binomial Test Against Chance (3-paradigm chance = 25%)**

| Layer | Observed % | Binomial p (vs 25%) | Significant? |
|-------|:----------:|:-------------------:|:------------:|
| L10 | 32.6% | 6.2e-02 | NS |
| L12 | 31.2% | 8.0e-02 | NS |
| L18 | 34.1% | 4.9e-03 | ** |
| L22 | 38.8% | 2.4e-07 | *** |
| L26 | 43.9% | 6.5e-13 | *** |
| L30 | 45.3% | 2.5e-19 | *** |
| L33 | 41.9% | 2.5e-15 | *** |

![Fig. 1: Gemma 3-paradigm sign-consistency with binomial test and strong feature counts](figures/fig1_crossdomain_consistency.png)

**Fig. 1 interpretation**: L10–L12 do not significantly exceed chance (25%, red dashed). From L18 onward, consistency far exceeds chance (***). Strong features peak at L30 (173). Deep-layer cross-domain signal is robust; shallow-layer apparent consistency is noise.

### 2.3 Cross-Model Comparison (Gemma vs LLaMA)

**Table 7. BK Classification AUC (DP, SAE features → PCA → LogReg)**

| Paradigm | Gemma-2-9B | LLaMA-3.1-8B |
|---------|:----------:|:------------:|
| SM best AUC | **0.976** (L20) | **0.974** (L8) |
| IC best AUC | **0.960** (L30) | **0.954** (L12) |

**Table 8. LLaMA SM AUC Across Layers**

| Layer | AUC | n_features | n_BK |
|-------|:---:|:----------:|:----:|
| L0  | 0.970 | 157 | 1,164 |
| L4  | 0.971 | 750 | 1,164 |
| **L8** | **0.974** | 818 | 1,164 |
| L12 | 0.973 | 854 | 1,164 |
| L15 | 0.970 | 962 | 1,164 |
| L20 | 0.963 | 1,028 | 1,164 |
| L25 | 0.961 | 1,182 | 1,164 |
| L30 | 0.959 | 1,368 | 1,164 |

**Table 9. SAE BK-Differential Features (SM, d≥0.3)**

| Layer | Gemma SM | Gemma BK+/BK- | LLaMA SM | LLaMA BK+/BK- |
|-------|:--------:|:-------------:|:--------:|:-------------:|
| L10 | 71 | 42 / 29 | 322 | 160 / 162 |
| L22 | 286 | 125 / 161 | 251 | 135 / 116 |
| L30 | 452 | 165 / 287 | 381 | 160 / 221 |

**Table 10. R1 Within-Bet-Type Classification**

| Paradigm/Bet type | AUC (Gemma IC, L22) | AUC (LLaMA IC, L10) |
|------------------|:-------------------:|:-------------------:|
| Fixed | 0.755 (n=800, BK=158) | 0.786 (n=800, BK=65) |
| Variable | 0.665 (n=800, BK=14) | 0.798 (n=800, BK=77) |
| SM Variable | 0.801 (n=1600, BK=87) | — |

![Fig. 6: BK classification AUC across layers — cross-model comparison for SM and IC](figures/fig6_classification_auc.png)

**Fig. 6 interpretation**: (a) Both models achieve AUC>0.95 in SM (Gemma peaks L18–20, LLaMA peaks L8). (b) In IC, Gemma peaks at L30 (0.960), LLaMA at L12 (0.954). Encoding depth differs but predictive performance is near-identical.

### 2.4 RQ1 Synthesis

Cross-model commonalities:

1. **Near-identical BK prediction**: Gemma 0.976 vs LLaMA 0.974 (SM). BK information content is architecture-invariant.
2. **BK-inhibiting dominance at L30**: Both models show BK- > BK+ at deepest layers (Gemma SM L30: p=0.000; LLaMA SM L30: p=0.001; LLaMA IC L30: p=0.000). However, **at L22, LLaMA SM shows BK+ > BK- (p=0.897, NS)** — the "risk suppression" interpretation applies only to deepest layers.
3. **Encoding depth differs**: Gemma peaks mid-deep (L20), LLaMA peaks early (L8). Where BK is encoded is architecture-dependent; that it exists is not.
4. **R1 early prediction**: BK is predictable from Round 1 (AUC 0.665–0.801), meaning bankruptcy signals exist from game start.
5. **Factor decomposition**: 65.2% (Gemma) and 69.5% (LLaMA) of SAE features encode BK independently of bet-type and paradigm (permutation p=0.000, null ~1%). BK representation has an independent axis in both architectures.

---

## 3. RQ2: Domain Invariance

### 3.1 Hidden State Cross-Domain Transfer (Gemma L22)

Gemma L22 DP hidden states: PCA(50) → LogReg(balanced) → cross-paradigm AUC. 200-permutation test for significance.

**Table 11. Hidden State vs SAE Transfer (Gemma L22)**

| Transfer | Hidden AUC | SAE AUC | Δ (HS - SAE) | HS perm_p |
|----------|:---------:|:-------:|:------------:|:---------:|
| IC → SM  | **0.746** | 0.499 (NS at L22) | +0.247 | 0.000 |
| IC → MW  | **0.826** | 0.876 | -0.050 | 0.000 |
| SM → MW  | **0.920** | 0.819 | +0.101 | 0.000 |

Hidden state transfer exceeds SAE transfer in IC→SM (+0.247) and SM→MW (+0.101). In IC→MW, SAE slightly outperforms hidden states. V8's "hidden > SAE" claim is confirmed for 2 of 3 directions. BK signal is distributed — SAE sparsification loses coherence in IC→SM.

### 3.2 SAE Cross-Domain Transfer with Permutation Test (Gemma)

For each (train, test) paradigm pair at layers [18, 22, 26, 30]: common active features (≥1% rate in both) → StandardScaler → PCA(50) → LogReg → AUC. 200-permutation null.

**Table 12. Best SAE Transfer AUC Per Direction (Gemma)**

| Transfer | Best Layer | AUC | perm_p | Null mean |
|----------|:----------:|:---:|:------:|:---------:|
| IC → SM  | L26        | **0.913** | 0.000 | 0.498 |
| IC → MW  | L18        | **0.932** | 0.000 | 0.504 |
| SM → MW  | L30        | **0.867** | 0.000 | 0.504 |
| SM → IC  | L30        | 0.646 | 0.000 | 0.500 |
| MW → IC  | L22        | 0.853 | 0.000 | 0.500 |

**Table 13. Layer-by-Layer SAE Transfer (Gemma, Key Directions)**

| Layer | IC→SM AUC | p | IC→MW AUC | p | SM→MW AUC | p |
|-------|:----------:|:-:|:-----------:|:-:|:----------:|:-:|
| L18 | 0.719 | 0.000 | 0.932 | 0.000 | 0.789 | 0.000 |
| L22 | 0.499 | 0.485 (NS) | 0.876 | 0.000 | 0.819 | 0.000 |
| L26 | **0.913** | 0.000 | 0.883 | 0.000 | 0.866 | 0.000 |
| L30 | 0.529 | 0.175 (NS) | 0.767 | 0.000 | 0.867 | 0.000 |

IC→MW and SM→MW transfer work consistently across all layers. IC→SM transfer is layer-dependent: L26 succeeds (0.913) but L22 fails (0.499 NS). This indicates IC and SM representations align only at specific layers.

![Fig. 4: Gemma cross-domain transfer AUC heatmap with significance](figures/fig4_transfer_heatmap.png)

**Fig. 4 interpretation**: IC→MW (0.77–0.93) and SM→MW (0.79–0.87) are significant at all layers. IC→SM is layer-dependent (L26: 0.913***, L22: 0.499 NS). Red cells indicate transfer failure.

### 3.3 3D Shared BK Subspace (Gemma L22)

PCA of 3 paradigm-specific LR weight vectors yields a 3D BK subspace:

**Table 14. 3D Shared Subspace Performance**

| Paradigm | 3D Subspace AUC | Full-dim AUC (50-PC) |
|----------|:--------------:|:--------------------:|
| IC | 0.862 ± 0.029 | ~0.95+ |
| SM | 0.899 ± 0.034 | ~0.97+ |
| MW | 0.970 ± 0.016 | ~0.97+ |

LR weight vector cosines: IC-SM=0.042, IC-MW=-0.026, SM-MW=-0.026 (near-orthogonal). Despite orthogonal weight vectors, the 3D subspace achieves AUC 0.86–0.97 — BK signal is distributed across a subspace, not a single direction. Consistent with V8 (0.91–0.94).

### 3.4 Factor Decomposition (Gemma 3-paradigm, LLaMA 2-paradigm)

Per-feature OLS regression: `feature ~ outcome + bet_type + paradigm`. Features where outcome is significant (p<0.01) after controlling bet_type and paradigm encode BK independently.

**Table 15. Factor Decomposition: Outcome-Significant Features**

| Factor | Gemma (581 features, IC+SM+MW) | LLaMA (1,418 features, IC+SM) |
|--------|:------:|:------:|
| **Outcome** (BK), after controlling others | **65.2%** (379) | **69.5%** (985) |
| Bet-type, after controlling others | 92.3% | 81.4% |
| Paradigm, after controlling others | 99.8% | 96.2% |

Permutation validation (selfcritique rerun): null outcome-significant ≈1% → observed 65–71% yields perm_p=0.000. BK representation's independent existence is statistically established in both architectures.

![Fig. 3: Factor decomposition — BK signal independence validated by permutation test](figures/fig3_factor_decomposition.png)

**Fig. 3 interpretation**: Both models show 65–71% outcome-significant features (blue/orange bars), exceeding the permutation null (~1%, red dashed) by 60×+. BK representation independence is architecture-invariant.

### 3.5 LLaMA Cross-Domain Transfer (IC ↔ SM)

LLaMA IC (n=1,600, BK=142) ↔ SM (n=3,200, BK=1,164). SAE features → PCA(50) → LogReg → AUC. 200-permutation test.

**Table 16. LLaMA IC↔SM Transfer**

| Transfer | Layer | AUC | perm_p | Null mean |
|---------|:-----:|:---:|:------:|:---------:|
| IC → SM | L25 | **0.783** | 0.000 | 0.500 |
| IC → SM | L15 | 0.522 | 0.020 | 0.501 |
| IC → SM | L5  | 0.269 | 1.000 | 0.499 |
| SM → IC | L30 | **0.685** | 0.000 | 0.498 |
| SM → IC | L12 | 0.662 | 0.000 | 0.498 |
| SM → IC | L8  | 0.624 | 0.000 | 0.499 |

Complete IC→SM profile: L5=0.269(p=1.0), L8=0.417(p=1.0), L10=0.373(p=1.0), L12=0.280(p=1.0), L15=0.522(p=0.02), L20=0.493(p=0.795), **L25=0.783(p=0.000)**, L30=0.418(p=1.0).

Transfer is asymmetric: SM→IC works at multiple layers, IC→SM succeeds only at L25. This mirrors Gemma's pattern (IC→SM layer-dependent at L26) — a cross-model phenomenon where IC and SM align only at specific abstract layers.

### 3.6 LLaMA IC-SM SAE Sign-Consistency

**Table 17. LLaMA 2-Paradigm SAE Sign-Consistency**

| Layer | Active | Sign-consistent | % | Strong (d≥0.2) |
|-------|:------:|:---------------:|:-:|:--------------:|
| L0  | 140 | 75 | 53.6% | 52 |
| L4  | 634 | 362 | 57.1% | 261 |
| L8  | 656 | 353 | 53.8% | 240 |
| L12 | 660 | 342 | 51.8% | 251 |
| L16 | 705 | 370 | 52.5% | 253 |
| L20 | 645 | 343 | 53.2% | 230 |
| L26 | 791 | 412 | 52.1% | 258 |
| L30 | 881 | 436 | 49.5% | 281 |
| **Total (32L)** | — | **5,374** | ~51–57% | **3,633** |

**Critical assessment**: 2-paradigm chance level is 50% (vs 25% for 3-paradigm). Binomial test: only 3 of 16 layers significantly exceed chance (L4: p=0.0002, L6: p=0.0001, L8: p=0.028). **Raw sign-consistency is mostly at chance level.** However, transfer AUC (§3.5) provides stronger evidence for cross-domain signal, as it leverages joint feature distributions rather than individual feature comparisons.

### 3.7 Within-Bet-Type Cross-Domain Consistency

**Table 18. Gemma Within-Bet-Type SAE Consistency (L22)**

| Bet-type | Paradigms | Active | Sign-consistent | % |
|----------|:---------:|:------:|:---------------:|:-:|
| Fixed | IC + MW | 311 | **178** | 57.2% |
| Variable | IC + SM | 279 | **172** | 61.6% |

**Table 19. LLaMA Within-Bet-Type Cross-Domain Consistency**

| Bet-type | Paradigms | Active | Sign-consistent | % | d_correlation | p |
|----------|:---------:|:------:|:---------------:|:-:|:-------------:|:-:|
| Fixed | IC + SM | 630 | 351 | 55.7% | **0.209** | 1.2e-7 |
| Variable | IC + SM | 719 | 309 | 43.0% | **-0.097** | 0.009 |

**Critical finding — Variable negative d_correlation**: LLaMA Variable IC-SM d_correlation is negative (r=-0.097). BK rate matching bootstrap (50×, subsampled to IC=9.6%): mean=-0.075, 95% CI=[-0.179, +0.032], only 18% positive. This is not a BK rate artifact — IC and SM Variable BK representations are genuinely different. Fixed cross-domain consistency is confirmed (r=0.209, p=1.2e-7).

### 3.8 RQ2 Synthesis

| Evidence | Strength | Source |
|------|:----:|------|
| Gemma IC→MW transfer | **Strong** | AUC 0.88–0.93, all perm p=0.000 (§3.2) |
| Gemma SM→MW transfer | **Strong** | AUC 0.79–0.87, all p=0.000 (§3.2) |
| Gemma 3-paradigm SAE consistency (L18+) | **Strong** | 33–45% vs 25% chance, p<5e-3 (§2.2) |
| Factor decomposition (both models) | **Strong** | 65–71% outcome-significant, perm p=0.000 (§3.4) |
| 3D Shared subspace | **Strong** | AUC 0.86–0.97 despite orthogonal weights (§3.3) |
| Hidden > SAE transfer | **Strong** | +0.10–0.25 in 2/3 directions (§3.1) |
| Gemma IC→SM transfer | **Moderate** | Layer-dependent: L26=0.913 but L22=0.499 NS (§3.2) |
| LLaMA SM→IC transfer | **Moderate** | L30=0.685, L12=0.662, all p=0.000 (§3.5) |
| LLaMA IC→SM transfer | **Moderate** | Only L25=0.783 (§3.5) |
| LLaMA Fixed cross-domain | **Moderate** | 55.7%, d_corr=0.209, p=1.2e-7 (§3.7) |
| LLaMA Variable cross-domain | **Counter** | d_corr=-0.075 post-matching — genuinely different (§3.7) |
| LLaMA raw sign-consistency | **Weak** | 13/16 layers at chance (§3.6) |

---

## 4. RQ3: Prompt Condition Effects

### 4.1 Fixed vs Variable BK Direction (Hidden States)

**Definition**: BK direction = BK_mean − Safe_mean, computed separately for Fixed and Variable BK games. Cosine similarity measures whether both bet types converge to the same BK representation.

**Table 20. Gemma IC BK Direction Comparison (3,584 neurons)**

| Layer | Var BK | Fix BK | cos(dir_Var, dir_Fix) | Sign-consistent common neurons |
|-------|:------:|:------:|:---------------------:|:------------------------------:|
| L10   | 14     | 158    | **-0.195**            | 178                            |
| L18   | 14     | 158    | **-0.082**            | 197                            |
| L22   | 14     | 158    | **+0.330**            | 555                            |
| L26   | 14     | 158    | **+0.443**            | 1,053                          |
| L30   | 14     | 158    | **+0.401**            | 1,073                          |

**Table 21. LLaMA IC BK Direction Comparison (4,096 neurons)**

| Layer | cos(dir_Var, dir_Fix) | Sign-consistent common neurons |
|-------|:---------------------:|:------------------------------:|
| L8  | **0.882** | 2,513 (61.4%) |
| L12 | **0.814** | 2,424 (59.2%) |
| L22 | **0.835** | 2,587 (63.2%) |
| L25 | **0.837** | 2,545 (62.1%) |
| L30 | **0.819** | 2,488 (60.7%) |

![Fig. 2: BK direction cosine similarity across layers — Gemma vs LLaMA IC](figures/fig2_bk_direction_crossmodel.png)

**Fig. 2 interpretation**: Gemma IC (blue) shows negative cosine at shallow layers, converging positive at deep layers. LLaMA IC (orange) maintains cos>0.81 at all layers. Gemma's shallow negative is likely a sample-size artifact (Variable BK=14 vs LLaMA's balanced 77 vs 65). When sample sizes are balanced (LLaMA), BK directions converge at all depths.

**Caveat**: Gemma IC Variable BK=14 is extremely small. LLaMA's balanced sample (77 vs 65) provides more reliable estimates.

### 4.2 Shared BK Neurons (Interaction Regression)

Per-neuron regression: `activation ~ β1·outcome + β2·bet_type + β3·(outcome × bet_type)`. Shared BK neurons: β1 significant (p<0.01) + β3 not significant — BK effect is independent of bet-type.

**Table 22. Shared BK Neurons Across Models and Paradigms**

| | Gemma IC L22 | **LLaMA IC L22** | **LLaMA SM L22** |
|--|:------------:|:----------------:|:----------------:|
| n_neurons | 3,584 | 4,096 | 4,096 |
| n_BK | 172 | 142 | 1,164 |
| Outcome-sig (p<0.01) | 1,639 (45.7%) | **3,150 (76.9%)** | 1,052 (25.7%) |
| Bet-type-sig (p<0.01) | 3,225 (90.0%) | 3,742 (91.4%) | 3,027 (73.9%) |
| Interaction-sig (p<0.01) | 448 (12.5%) | **1,605 (39.2%)** | 1,495 (36.5%) |
| **Shared BK neurons** | **1,182 (33.0%)** | **1,501 (36.6%)** | **72 (1.8%)** |

Gemma IC 33.0% and LLaMA IC 36.6% — similar proportions of bet-type-independent BK neurons across architectures. LLaMA SM shows only 1.8% shared neurons because SM BK concentrates in Variable games (72.3% vs 0.4%), making "bet-type-independent" BK neurons nearly impossible.

![Fig. 7: Shared vs Interaction BK neurons across models and paradigms](figures/fig7_shared_bk_neurons.png)

**Fig. 7 interpretation**: Gemma IC (33.0%) and LLaMA IC (36.6%) show similar shared BK neuron proportions. LLaMA SM (1.8%) reflects the Variable-dominated BK structure in SM.

### 4.3 Balance Confound Control

**Table 23. Balance Partial Correlation (L22)**

| | Gemma IC | LLaMA IC |
|--|:--------:|:--------:|
| Raw BK-sig neurons | 2,578 | 3,238 |
| Balance-controlled | 2,569 | 3,269 |
| Retained | **99.7%** | **101.0%** |

Balance removal preserves 99.7–101% of BK signal. BK representation encodes strategic/behavioral patterns, not simply balance information. LLaMA's 101% indicates balance acts as a suppressor variable.

### 4.4 Prompt Component Analysis (G/M/W/P/R)

**Gemma SM** uses hidden state neurons (3,584-dim, 6 layers). **LLaMA SM** uses SAE features (32K, 5 layers). Direct absolute counts are not comparable; percentages are used.

**Table 24. Prompt Component BK Rate Effects (Cross-Model)**

| Component | Gemma SM ratio | LLaMA SM ratio | Cross-model |
|-----------|:-------------:|:--------------:|:-----------:|
| **G** (Goal) | **20.75x** | **1.26x** | Gemma >>>>> LLaMA |
| M (Mood/Money) | 2.35x | 1.31x | Gemma > LLaMA |
| W (Warning) | 2.35x | 1.15x | Gemma > LLaMA |
| H (Hint) | — | 1.05x | LLaMA only |
| R (Reset) | 1.72x | — | Gemma only |
| P (Persona) | 1.17x | 1.06x | Both negligible |

**Table 25. Gemma SM Neural Analysis (comp × BK interaction neurons)**

| Component | L10 | L18 | L22 | L26 | L30 | L33 | Pattern |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:-------:|
| G | 109 | 168 | 168 | 308 | **552** | **592** | Increases with depth |
| M | 41 | 23 | 25 | 10 | 8 | 15 | Negligible |
| R | 205 | 314 | 235 | 102 | 95 | 97 | Mid → decreasing |
| W | 183 | 70 | 23 | 1 | 0 | 0 | Vanishes with depth |
| P | 53 | 39 | 38 | 8 | 5 | 3 | Negligible |

**Table 26. BK-Amplifying Interaction Neurons (Gemma SM)**

| Component | L10 | L22 | L30 | Interpretation |
|-----------|:---:|:---:|:---:|:--------------:|
| G | 0 | 0 | 1 | G drives BK independently; does not modulate BK signal |
| R | 18 | 63 | 15 | R amplifies BK at mid-layers |
| W | 42 | 3 | 0 | W has shallow-only BK amplification |
| M, P | 2–12 | 3–6 | 0 | Negligible |

**Table 27. LLaMA SM Neural Analysis (% of active SAE features)**

| Component | L8 int% | L12 int% | L30 int% | L30 amp% | Pattern |
|-----------|:-------:|:--------:|:--------:|:--------:|:-------:|
| G | 45.5% | 61.5% | **63.0%** | 8.2% | Interaction increases with depth; low amplification |
| M | **55.4%** | 48.2% | **51.2%** | **14.5%** | Highest amplification across all layers |
| H | 25.9% | 31.2% | 33.3% | 7.4% | Weak |
| W | 36.2% | 37.2% | 19.2% | 5.8% | Shallow-strong, deep-weak |
| P | 26.0% | 27.1% | 31.2% | 4.8% | Weak |

![Fig. 5: Prompt component BK effect — dramatic cross-model difference for G (Goal)](figures/fig5_prompt_components.png)

**Fig. 5 interpretation**: G(Goal) produces 20.8× BK increase in Gemma (dominant) but only 1.26× in LLaMA (mild). The same prompt produces 16× different behavioral effects across models. M, W, P are weak in both.

### 4.5 G-Prompt BK Direction Alignment (Gemma L22)

G-prompt direction (G_mean − non-G_mean) vs BK direction (BK_mean − Safe_mean) cosine:

**Table 28. G-Prompt Alignment**

| Paradigm | cos(G_dir, BK_dir) | BK with G | BK without G |
|----------|:-------------------:|:---------:|:------------:|
| IC | **-0.143** | 10.9% | 10.6% |
| SM | **+0.850** | 5.2% | 0.3% |
| MW | **+0.367** | 2.5% | 0.9% |

In SM, G-prompt direction aligns strongly with BK direction (cos=+0.85). G pushes the activation space directly toward BK. This is consistent with G driving BK independently — G moves the hidden state toward BK without modulating existing BK representations.

### 4.6 Bet Constraint Linear Mapping (Gemma IC L22)

**Table 29. Bet Constraint → BK Projection**

| Constraint | n | Mean BK prob | Actual BK rate |
|:----------:|:---:|:------------:|:--------------:|
| c10 ($1–$10) | 400 | 0.000 | 0.0% |
| c30 ($1–$30) | 400 | 0.056 | 5.2% |
| c50 ($1–$50) | 400 | 0.212 | 16.8% |
| c70 ($1–$70) | 400 | 0.270 | 21.0% |

Linear correlation: r=0.979, p=0.021. (V8: r=0.98, nearly identical.)

### 4.7 RQ3 Synthesis

| Evidence | Strength | Source |
|------|:----:|------|
| LLaMA IC Var/Fix cos>0.81 (all layers) | **Strong** | Balanced sample (77 vs 65), §4.1 |
| Shared BK neurons ~33–37% (both IC) | **Strong** | Cross-model consistent, §4.2 |
| Balance confound control (both models) | **Strong** | 99.7–101% retained, §4.3 |
| G independent BK driver (both models) | **Strong** | amplifies_bk≈0 (Gemma), 8.2% (LLaMA), §4.4 |
| W shallow pattern (both models) | **Moderate** | Vanishes deep in both, §4.4 |
| Gemma IC Var/Fix cos convergence | **Moderate** | Deep cos>0.33 but Var BK=14, §4.1 |
| G behavioral effect size difference | **Caution** | 20.75× vs 1.26× — same prompt, 16× different effect, §4.4 |
| M amplification difference | **Moderate** | LLaMA M=14.5% (highest), Gemma M≈0, §4.4 |

---

## 5. Limitations & Critical Assessment

### 5.1 Permutation-Validated Claims

| Claim | Test | Result |
|-------|------|--------|
| Factor decomposition 67–71% | 100-perm shuffle | p=0.000 (null 1%) ✅ |
| BK-inhibiting dominance L30 | Binomial test | Gemma p=0.000, LLaMA SM p=0.001, LLaMA IC p=0.000 ✅ |
| BK-inhibiting dominance L22 | Binomial test | LLaMA SM p=0.897 ❌ — not established at mid-layer |
| Variable cross-domain d_corr | BK rate matching + bootstrap | mean=-0.075, CI[-0.179,+0.032] ❌ — genuine difference |
| Classification AUC | 100-perm shuffle | p=0.000 ✅ |
| Gemma 3-paradigm consistency (L18+) | Binomial vs 25% | p<5e-3 ✅ |

### 5.2 Cross-Model Effect Size Confound

| | Gemma SM | LLaMA SM |
|--|:--------:|:--------:|
| n_BK | 87 | 1,164 |
| mean |d| | 0.571 | 0.209 |
| % d≥0.3 | 68.8% | 23.2% |

Gemma's higher |d| likely reflects small-sample inflation (N=87 → noisy, inflated estimates). LLaMA's lower |d| may converge to true effects with larger N. **Cross-model effect size comparison is unreliable.** Classification AUC is a more robust comparison metric.

### 5.3 Statistical Power Issues

| Analysis | Weakness | Impact |
|------|------|------|
| Var BK direction (Gemma IC) | n_var_bk=14 | Cosine is a single estimate with unknown CI; t-test power very low |
| LLaMA sign-consistency | 2-paradigm chance=50%; 13/16 layers NS | Raw sign-consistency is weak evidence for cross-domain signal |
| Gemma L10, L12 consistency | 3-paradigm binomial NS (p>0.05) | Shallow-layer cross-domain signal may be noise |

### 5.4 Multiple Comparison Issues

FDR correction is not systematically applied across all analyses. Expected false positives: Analysis 3 (3,584 neurons × t-tests → ~36 FP at p<0.01); Analysis 4 (581 features → ~6 FP); Analysis 9 (3,584 × 5 components × 6 layers → many tests). Analyses 1, 5, 7 use permutation tests, partially addressing this. For other analyses, effect size (Cohen's d) should be treated as primary, p-values as secondary.

### 5.5 Data Constraints

1. **LLaMA MW not extracted**: 3-paradigm cross-model comparison not possible.
2. **Gemma IC Variable BK=14**: Very limited statistical power for bet-type comparisons.
3. **MW Variable BK=4**: Below threshold (n=5), MW bet-type comparison not possible.
4. **LLaMA hidden states**: Only 5 layers extracted (L8,12,22,25,30), not all 32.
5. **No causal validation**: All analyses are correlational. Activation patching needed.

### 5.6 BK Rate Heterogeneity

Gemma IC Fixed 19.75% vs Variable 1.75%; LLaMA IC Fixed 8.1% vs Variable 9.6% (reversed); Gemma SM 2.7% vs LLaMA SM 36.4% (13×). Cross-model and cross-paradigm comparisons must account for base rate confounds.

### 5.7 Prompt Component Model-Dependence

G component: Gemma 20.75×, LLaMA 1.26×. "Prompt condition invariance" applies to neural structure (both models show G drives BK independently) but not to behavioral effect size.

---

## 6. Next Steps

1. ~~**LLaMA Hidden State Extraction**~~ → **Completed (v9.4)**. Analysis 2, 3 symmetric analyses done.
2. **Activation patching (causal validation)**: Test top candidates from 665 cross-domain strong features.
3. **LLaMA MW extraction**: Complete 3-paradigm cross-model comparison.
4. **IC→SM Transfer mechanism**: Investigate why L26 succeeds / L22 fails.
5. **FDR correction**: Apply to all per-feature/per-neuron tests.
6. **G component cross-model deep dive**: Compare G processing pathways to explain 20.75× vs 1.26× divergence.
7. **BK rate confound control**: Balanced subsampling or SMOTE for IC-SM transfer re-analysis.

---

## Appendix: Supplementary Tables

### A.1 Comparison: Hidden State (600 neurons) vs SAE (744 features)

| Level | Cross-domain consistent | FDR? | Binomial significant? |
|-------|:----------------------:|:----:|:---------------------:|
| Hidden state (3584-dim, L22) | 600 (16.7%) | Yes (p<0.01) | — |
| SAE features (131K, 7 layers) | 744 (total), L18+ sig | No FDR | L18+ p<5e-3 |

Hidden state (16.7% universal, FDR-corrected) and SAE (L18+ significant vs chance) consistently support cross-domain BK signal existence.

### A.2 Gemma SM Prompt Component Detail

| Component | BK with | BK without | Ratio |
|-----------|:-------:|:----------:|:-----:|
| **G** (Goal) | 5.2% | 0.2% | **20.75x** |
| M (Mood) | 3.8% | 1.6% | 2.35x |
| W (Warning) | 3.8% | 1.6% | 2.35x |
| R (Reset) | 3.4% | 2.0% | 1.72x |
| P (Persona) | 2.9% | 2.5% | 1.17x |

### A.3 LLaMA IC→SM Complete Transfer Profile

L5=0.269(p=1.0), L8=0.417(p=1.0), L10=0.373(p=1.0), L12=0.280(p=1.0), L15=0.522(p=0.02), L20=0.493(p=0.795), **L25=0.783(p=0.000)**, L30=0.418(p=1.0).
