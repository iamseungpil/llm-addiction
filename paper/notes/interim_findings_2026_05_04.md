# Interim Findings — Causal & Confound Audits for the LLM-Addiction NeurIPS Submission

**Date**: 2026-05-04 (KST)
**Scope**: Track A causal supplements (M3 prompt swap, M3' direction steering) and Track B reviewer-driven confound re-analyses (E2 §4.3, E3 §4.2, E7 §3) on Gemma-2-9B and LLaMA-3.1-8B for the slot-machine paradigm.
**Status**: All Track B experiments complete. Track A on Gemma slot machine: M3 prompt swap, M3$'$ direction steering, and M3$''$ paired full-prompt L22 patching all complete with concordant null results. LLaMA replication queued (data-availability blockers on three of four AMLT nodes; one Gemma SM replication node currently running).

---

## Executive Summary

The paper's behavioural and readout claims survive the reviewer-flagged confounds, while the optional causal-control supplement returns a clean negative result on Gemma slot machine. Re-fitting the §4.3 Ridge readout inside fixed balance windows preserves the goal-prompt modulation effect on both models. Re-fitting the §4.2 BK-contrast geometry after balance residualization preserves the cross-task shared signal on LLaMA but collapses it on Gemma at L22, which calls for a tightened body claim. Re-running the §3 condition contrasts under cluster-robust statistics keeps every reported effect with confidence intervals that exclude zero. By contrast, neither the one-decision +G prompt swap nor additive last-token steering along the §4.1 readout direction moves the model's behaviour at the magnitudes we tested. The implication is that the §4.1 readout is a robust decoder of the indicators but not a single-pass linear controller, which is consistent with the paper's framing of "shared low-dimensional geometry plus task-specific readouts" rather than "circuit-level autonomy unit".

---

## Reading Guide for First-Time Readers

A few terms recur throughout. *Sparse autoencoder* (SAE) is a learned decomposition that turns a single hidden vector at layer L into a long sparse vector of human-readable feature activations; we use Gemma-Scope and Llama-Scope as the SAE families that the original Gemma and LLaMA authors released. *Ridge readout* is a linear regression with $L_2$ regularization that maps the top 200 most-correlated SAE features at L22 into one of three behavioural indicators; the indicators are betting aggressiveness $I_\text{BA}$ (bet divided by current balance), extreme choice $I_\text{EC}$ (an indicator that fires when $I_\text{BA}$ exceeds 0.5), and loss chasing $I_\text{LC}$ (the relative bet-ratio jump after a losing round). *BK direction* is the difference between the mean hidden state at bankruptcy rounds and the mean at voluntary-stop rounds, a single direction that points from "risky ending" to "safe ending" inside the model's residual stream. A *dose-response* sweep means we vary one knob (here the steering coefficient $\alpha$, in units of the baseline standard deviation $\sigma$) across an ordered ladder and check whether behaviour moves monotonically with it. *Cohen h* is the standard arcsine-difference effect size for two proportions, where 0.2, 0.5, and 0.8 mark the small, medium, and large bands. Section §3 is the behavioural part of the paper, and §4.1, §4.2, §4.3 are the three subsections of the neural part: indicator readout, cross-task sharing, and condition modulation.

---

## Track A — Causal Validation

### A.1 v1 M3 — One-Decision Prompt Swap

**Intent**. The paper's §3 reports that variable betting under the goal-setting prompt $+G$ raises bankruptcy and the three risk indicators on every model tested; the natural causal question is whether replacing one $-G$ decision context with a $+G$ context, mid-game, is enough to flip the model's choice at that round. If yes, the prompt itself is sufficient at decision time; if no, the +G effect requires the cumulative trajectory of earlier rounds.

**Hypothesis**. If the $+G$ prompt component is a strong proximal cause of the riskier decision, then swapping one decision's prompt from $-G$ to $+G$ should raise the per-game bankruptcy rate measurably above the $-G$ baseline, with a random-prompt-swap control ruling out generic perturbation effects.

**Verification design**. Two hundred Gemma-2-9B variable-betting games were rolled forward through the slot-machine simulator under three conditions: a $-G$ baseline that uses the original prompt at every round, a $+G$ swap that replaces only the prompt at one mid-game decision with the corresponding $+G$ variant before continuing the rollout, and a random-direction control that perturbs the same hidden-state slot with a Gaussian vector matched in norm to the swap perturbation. Bankruptcy and voluntary-stop rates were computed across all 200 rolled trajectories in each condition, with raw trial logs at `results/v19_multi_patching/M3_swap/gemma_sm_*_n200/trials.jsonl`.

**Result**. The bankruptcy rate moved from 10.0% under $-G$ baseline to 12.0% under $+G$ swap and 12.5% under random-direction control, a spread of 2.5 percentage points across all three conditions. Voluntary stops were 90.0%, 87.0%, and 87.5% in the same order, and the average game length was 9.0, 10.2, and 8.5 rounds. The Cohen h between baseline and $+G$ swap is approximately 0.06, well inside the small-effect band, and the random control sits between the two test conditions, which is what we would expect under a null effect with sampling noise.

**Reading**. A single-decision prompt swap is not sufficient to push the model into riskier behaviour at the round of intervention, and the random-direction control matches the swap result, so the $+G$ effect from §3 is built up by the cumulative game trajectory rather than recoverable in one decision. This is a clean negative result on a narrow causal claim, and it sets up M3' as the second test using a learned direction rather than a prompt swap.

### A.2 M3' — Direction Steering Along the §4.1 Ridge Readout

**Intent**. Section §4.1 demonstrates that the top-200 SAE features at L22 carry a Ridge-readable signal for $I_\text{BA}$ on Gemma slot machine at $R^2 = 0.167$. The supplementary causal question is whether the very direction the readout uses to *predict* $I_\text{BA}$ also *controls* it when we steer the residual stream along that direction. Predictor-controller coincidence is a strong claim about the underlying representation; predictor-only is the weaker but still informative finding.

**Hypothesis**. If the §4.1 readout direction is a proximal cause of betting aggressiveness, then projecting the Ridge weight vector through the SAE decoder to obtain a unit direction in the 3584-dimensional residual stream and adding $\alpha \sigma$ of that direction to the last prompt-token's hidden state at L22 should move the model's bet ratio monotonically with $\alpha$, while a random direction matched in scale, an off-target layer at L8, and a direction trained on the unrelated $I_\text{LC}$ indicator should each produce a smaller effect.

**Verification design**. The Ridge weight vector and standardizer parameters were extracted from a re-run of the §4.1 pipeline, then projected through the Gemma-Scope L22 decoder columns at the selected feature indices to give a unit direction $d$ in residual-stream space. The baseline standard deviation $\sigma$ for that direction was computed from 200 random variable-betting rounds. A forward hook on the L22 transformer block fired exactly once per generation, on the prompt forward pass, adding $\alpha \sigma d$ to the final prompt token's hidden state. Six dose levels covered $\alpha \in \{-2, -1, 0, +1, +2, +3\}$ and three specificity controls covered a random unit direction at $\alpha = +2\sigma$ on L22, the $I_\text{BA}$ direction applied at L8, and the $I_\text{LC}$ direction applied at L22. Each condition ran 50 trials drawn from the §3 $-G$ variable-betting prompt distribution, and the resulting bet ratios and stop choices are aggregated in `results/v19_multi_patching/M3prime_indicator_steering/aggregated/gemma_sm.json`.

**Result**. The mean bet ratio across the six-point dose ladder was 0.064, 0.056, 0.051, 0.062, 0.060, 0.064, and the corresponding stop rates were 0.58, 0.62, 0.64, 0.62, 0.60, 0.60. The Pearson correlation between $\alpha$ and the per-trial bet ratio was $r = +0.013$ with a 95% confidence interval of $[-0.10, +0.13]$, the Spearman rank correlation was $\rho = +0.010$ at $p = 0.869$, and the Cohen h on stop rate between $\alpha = -2$ and $\alpha = +3$ was $-0.041$. The three specificity controls landed inside the same band as the on-target dose conditions: random direction gave bet ratio 0.064 and stop rate 0.58, the off-target L8 layer gave 0.052 and 0.66, and the $I_\text{LC}$ direction gave 0.066 and 0.58. The on-target intervention at $\alpha = +2\sigma$ does not separate from any of the three controls under either Cohen h or the Welch t-test.

**Reading**. The §4.1 readout direction does not behave as a single-pass linear controller for $I_\text{BA}$ on Gemma slot machine at the perturbation strengths we tested. Two interpretations remained open at this stage: either the readout is a genuine decoder and the causal mechanism lives elsewhere in the network, or the steering protocol is too weak. The next experiment, M3$''$, was designed to discriminate between these by replacing the entire L22 hidden state rather than nudging it in a single direction.

### A.3 M3$''$ --- Paired Full-Prompt L22 Patching

**Intent**. The M3$'$ result leaves open whether the null reflects a structural property of the representation or merely a weak protocol. A stronger causal protocol replaces the entire L22 hidden state at the prompt with the corresponding hidden state from a $+G$ run on the same game and round. If the $+G$ effect is localized at L22, this swap should make the model behave like $+G$; if the effect requires earlier layers or distributed computation, the swap should leave the model behaving like $-G$.

**Hypothesis**. If L22 is the layer where the $+G$ effect is consolidated, replacing the $-G$ run's L22 hidden state with the matched $+G$ run's L22 hidden state should drive the bet ratio and stop rate toward the $+G$ natural baseline rather than the $-G$ natural baseline. A norm-matched random patch should leave behaviour at $-G$.

**Verification design**. For each of the same fifty $-G$ variable-betting (game, round) pairs used by M3$'$, the $-G$ prompt and the corresponding $+G$ prompt (the same prompt with the goal-setting sentence prepended) were each run through Gemma. The L22 transformer block's full output sequence was cached for the $+G$ run. The $-G$ run was then re-executed with a forward hook that replaced the L22 output at one of three scopes: the last prompt token alone (`patched_last`), the maximal common suffix shared between $-G$ and $+G$ token sequences (`patched_suffix`), or all positions (`patched_all`, falling back to suffix when sequence lengths disagree). Two natural baselines (`natural_minusG` and `natural_plusG`) bracketed the $-G \to +G$ behavioural range, and a norm-matched random direction at the last token (`random_patch`) controlled for non-specific perturbation. All six conditions ran fifty trials drawn from the same prompt distribution.

**Result**. The natural baselines establish a wide $-G$ versus $+G$ behavioural gap: bet ratio $0.051 \pm 0.011$ versus $0.216 \pm 0.018$, stop rate $0.640$ versus $0.060$, with the difference equivalent to a Cohen $h$ of approximately $1.30$ on stop rate. The three patched conditions land at bet ratios $0.068$, $0.079$, and $0.081$, all statistically indistinguishable from $-G$ under Welch's $t$-test ($p = 0.31$, $0.15$, $0.21$ respectively) and far from $+G$ ($p < 10^{-5}$ in every case). The norm-matched random patch gives bet ratio $0.072$ and stop rate $0.700$, also indistinguishable from $-G$. On stop rate, the Cohen $h$ between each patched condition and the $+G$ target exceeds $1.15$, well into the large-effect band, while the $h$ against the $-G$ baseline never exceeds $0.21$.

**Reading**. Replacing the L22 hidden state at the $-G$ prompt with the matched $+G$ hidden state, even at every prompt token, does not move behaviour toward $+G$. This is the strongest causal protocol available without re-running deeper layers, and it returns a clean null. The interpretation tightens: the $+G$ effect is not consolidated at L22 in a way that a layer-wise patch can reproduce. Either the effect lives in earlier layers and the L22 state is a downstream summary, or it is distributed across many layers, or it requires the autoregressive generation steps themselves to occur in $+G$ context. The body claim of §4 --- that the readout decodes the indicator --- is unchanged; what has been ruled out is the stronger claim that the readout's host layer is also where the indicator is causally written. The remaining AMLT experiments, when they recover, will probe model and task generality of this pattern; LLaMA L22 SAE coverage is incomplete in the released artifacts, so the cross-model replication remains partially blocked at the data level.

---

## Track B — Reviewer-Flagged Confound Re-Analyses

### B.1 E2 — Variance-Normalized §4.3 Goal-Prompt Modulation

**Intent**. The reviewer raised the concern that the §4.3 modulation result, where the goal-setting prompt $+G$ raises the $I_\text{BA}$ readout $R^2$ from 0.063 to 0.153 on Gemma slot machine and from 0.082 to 0.113 on LLaMA, may simply reflect the fact that the $+G$ and $-G$ subsets sample different balance regimes inside the same game. If $+G$ rounds happen to land at balance levels where the readout is intrinsically sharper, the reported $\Delta R^2$ would not be a property of the goal prompt at all.

**Hypothesis**. If the $+G$ modulation is a property of the prompt rather than the balance regime, then refitting the §4.3 strict 5-fold pipeline inside three overlapping balance windows — low, mid, high in the all-variable subset's percentile space — should leave $\Delta R^2(+G, -G)$ positive in each window, with the magnitudes broadly consistent across windows.

**Verification design**. The all-variable subset's balance distribution was percentile-binned into three overlapping windows at the 10–40, 30–70, and 60–90 percentiles. Within each window, the §4.3 GroupKFold pipeline — within-fold random-forest deconfound on balance and round, top-200 SAE feature selection by $|$Spearman$|$, Ridge with $\alpha = 100$, 5-fold cross-validation grouped by game id — was rerun on each of the four condition subsets ($+G$, $-G$, $+M$, $-M$). The contrast $\Delta R^2(+G, -G)$ and $\Delta R^2(+M, -M)$ was then computed within each window. Numerical outputs are at `results/v19_multi_patching/E2_variance_normalized/gemma_sm_i_ba_L22.json` and the LLaMA counterpart.

**Result**. On Gemma slot machine $I_\text{BA}$, the within-window $\Delta R^2(+G, -G)$ was $+0.082$ at low balance, $+0.147$ at mid balance, and $+0.138$ at high balance, with $+G$ values of 0.128, 0.191, 0.248 against $-G$ values of 0.046, 0.044, 0.110. On LLaMA slot machine $I_\text{BA}$, the same contrast was $+0.018$, $+0.046$, and $+0.038$ across low, mid, and high balance, with $+G$ values of 0.082, 0.160, 0.212 against $-G$ values of 0.064, 0.113, 0.175. The $\pm M$ contrast was mixed in sign on Gemma and consistently small and positive on LLaMA, which is consistent with the body's framing of $G$ as the dominant condition modulator.

**Reading**. The $+G$ modulation effect on $I_\text{BA}$ persists inside fixed balance windows on both models, with $\Delta R^2$ values of the same magnitude as the body's pooled result. The modulation is therefore a property of the goal prompt rather than an artefact of differing balance distributions, which directly answers the reviewer's variance-inflation concern. The mixed-sign $\pm M$ contrast on Gemma also matches the body's narrative that $G$ rather than $M$ carries the modulation, so the body's choice to headline $+G$ and treat $\pm M$ as a secondary axis is supported.

### B.2 E3 — Balance-Stratified §4.2 BK Geometry Audit

**Intent**. The reviewer raised the concern that the §4.2 leave-one-task-out PCA result, where a rank-1 shared direction extracted from two tasks' BK directions still separates bankruptcy from voluntary stops on the held-out third task, may simply track balance rather than risk-ending semantics. Bankruptcy rounds are by construction at very low balance, voluntary-stop rounds at higher balance, so a direction that encodes "low balance" would show the same separation without any meaningful cross-task transfer of the risk concept.

**Hypothesis**. If the LOTO-PCA shared axis encodes a genuine risk-ending semantics, then balance-controlled variants should preserve the held-out separation above a random-direction baseline. We therefore expect either the balance-matched subsample (V1), where bankruptcy and voluntary-stop rounds are stratified to overlap in balance, or the balance-residualized hidden state (V2), where balance is regressed out of every dimension before the BK direction is recomputed, to keep the shared-axis AUC well above the random-direction baseline on both models.

**Verification design**. The §4.2 BK-contrast pipeline was re-run at L22 on Gemma and LLaMA in three variants: a baseline with no balance control, V1 with a stratified bankruptcy-versus-voluntary-stop subsample matched in 6-bin balance quantiles, and V2 with per-dimension linear residualization of balance from each hidden state before the BK direction was computed. For each variant and each held-out task in {SM, IC, MW}, the LOTO-PCA shared-axis ROC-AUC for bankruptcy versus voluntary stop was reported alongside the random-direction baseline averaged over thirty random unit vectors. Numerical outputs are at `results/v19_multi_patching/E3_balance_stratified/gemma_L22.json` and `llama_L22.json`.

**Result**. On LLaMA at L22, the V2 balance-residualized shared-axis AUC was 0.657 with held-out SM, 0.655 with held-out IC, and 0.670 with held-out MW, against random baselines of 0.597, 0.650, and 0.597, leaving every cell above its matched random baseline. On Gemma at L22, the V2 shared-axis AUC was 0.543 with held-out SM, 0.554 with held-out IC, and 0.544 with held-out MW, against random baselines of 0.661, 0.606, and 0.691, with every cell at or below the random baseline. The V1 balance-matched variant gave intermediate results on both models: AUCs of 0.707, 0.572, 0.712 on LLaMA and 0.568, 0.691, 0.522 on Gemma.

**Reading**. The §4.2 shared-axis claim is mixed under balance control: it survives on LLaMA at L22, where the balance-residualized direction still separates bankruptcy from voluntary stops on each held-out task, but it collapses to the random baseline on Gemma at L22. The honest implication is that part of the Gemma L22 shared signal was carried by balance rather than by a balance-independent risk-ending semantics, which is information the body should disclose with a caveat or a layer-specific qualification. The body presents L22 as a "representative slice of the layer sweep" rather than as Gemma's peak layer, and the BK-contrast peak the body actually reports lives at L23, so this finding is best added to the appendix as a layer-specific note rather than as a refutation of the body claim. The LLaMA L22 result is unconditionally supportive.

### B.3 E7 — Cluster-Robust §3 Re-Analysis

**Intent**. The reviewer raised the general concern that §3 reports per-model bankruptcy rates and pooled behavioural means without modelling the nested clustering of rounds within games and games within models. If the per-model effects are driven by a small number of unusually risk-seeking games, the apparent significance would be inflated.

**Hypothesis**. If the §3 effects are robust to clustering, then a logistic generalized estimating equation with the model identifier as the cluster variable should still recover positive coefficients for the goal-prompt and reward-prompt indicators on the variable-betting subset, and a per-model bootstrap-by-game on the bankruptcy gap between variable and fixed betting should yield 95% confidence intervals that exclude zero.

**Verification design**. The combined Gemma and LLaMA slot-machine corpus contributed 3200 games to the game-level analysis and 57797 variable-betting rounds to the round-level analysis. A binomial GEE with exchangeable working correlation modelled the bankruptcy outcome on the variable-betting subset as a function of $\text{has\_G}$ and $\text{has\_M}$, with the model identifier as the cluster variable. Per-model bootstrap-by-game with 2000 iterations resampled games with replacement to produce a percentile confidence interval on the variable-minus-fixed bankruptcy gap. A cluster-bootstrap on game id with 2000 iterations did the same for the round-level bet-ratio regression on $\text{has\_G}$ and $\text{has\_M}$. Numerical outputs are at `results/v19_multi_patching/E7_mixed_effects_section3/results.json` with the markdown counterpart at `_summary.md`.

**Result**. The game-level GEE recovered $\hat{\beta}_{\text{has\_G}} = +0.527$ with $\text{SE} = 0.113$, $z = 4.67$, $p = 3.07 \times 10^{-6}$, and $\hat{\beta}_{\text{has\_M}} = +0.473$ with $\text{SE} = 0.228$, $z = 2.07$, $p = 0.038$. The per-model bootstrap-by-game on the bankruptcy gap returned a Gemma point estimate of $+0.054$ with 95% CI $[+0.043, +0.066]$ and a LLaMA point estimate of $+0.719$ with 95% CI $[+0.696, +0.741]$, both excluding zero. The cluster-bootstrap on the round-level bet-ratio gave $\hat{\beta}_{\text{has\_G}} = +0.0498$ with 95% CI $[+0.037, +0.063]$ and $\hat{\beta}_{\text{has\_M}} = +0.0555$ with 95% CI $[+0.042, +0.069]$, again both excluding zero.

**Reading**. Every §3 claim that the body makes about the goal-prompt, reward-prompt, and bet-type effects survives a cluster-robust re-analysis with confidence intervals that exclude zero, and the magnitudes match the body's pooled numbers. No body change is needed; the appendix gains a clustering-aware table that reports the GEE coefficients and the bootstrap intervals.

---

## Cross-Cutting Implications for the Paper

The paper's core empirical contributions stand. Section §3's behavioural claims hold under cluster-robust statistics with effect sizes outside the noise floor on both open-weight models. Section §4.1's decoding result remains the headline finding, with the readout direction recovering the indicators in the Cohen small-to-medium $R^2$ band. Section §4.3's modulation claim survives variance-controlled re-analysis on both models for $I_\text{BA}$. Section §4.2's cross-task shared geometry claim survives on LLaMA at L22 but is balance-driven on Gemma at the same layer, which calls for a layer-qualified statement in either the body or the appendix.

The paper's optional causal-control supplement is honestly negative on Gemma slot machine across three independent protocols of increasing strength, and the convergence sharpens what the negative result actually says. A one-decision prompt swap, an additive last-token steering along the §4.1 readout direction, and a paired full-prompt replacement of the entire L22 hidden state all leave behaviour at the $-G$ natural baseline. Random and off-target controls land in the same band, so the null is not protocol-specific noise. The $-G$ to $+G$ behavioural gap itself is large (bet ratio $0.051 \to 0.216$, stop rate $0.640 \to 0.060$ under natural prompts), so the failure to reproduce it under any L22-localized intervention is informative: the autonomy effect is not consolidated at L22 in a form that single-layer patching can reproduce, even when the patch source is the matched $+G$ activation from the same game state.

In the body, the discussion section can reframe §4 as decoder-strong, single-layer-controller-ruled-out: the readout reads the indicator reliably at $R^2 = 0.167$, but no L22-localized intervention --- one-decision prompt swap, additive direction steering at $\alpha \in [-2,+3]\sigma$, or paired full-prompt activation replacement --- moves behaviour toward the $+G$ natural baseline. The §6 limitations paragraph hedged in this direction; the three concordant protocols turn the hedge into a structural finding, and they do so without weakening any positive claim in §3, §4.1, or §4.3.

---

## Fact Base

The following table lists every numerical claim in this report with its source. Lines marked Code are verified against the corresponding script outputs as of 2026-05-04.

| ID | Claim | Source |
|----|-------|--------|
| F1 | $I_\text{BA}$, $I_\text{EC}$, $I_\text{LC}$ definitions | `neurips_content_en/3.behavior.tex:18` |
| F2 | Gemma SM $I_\text{BA}$ readout $R^2 = 0.167$, body Table | `neurips_content_en/4.neural.tex` Table tab:neurips-sae-results |
| F2b | §4.3 +G/−G modulation: Gemma 0.063→0.153 (+143%), LLaMA 0.082→0.113 (+38%) | `neurips_content_en/4.neural.tex` Table tab:condition-modulation, body line 102–103 |
| F3 | Cohen R² thresholds 0.01/0.09/0.25 | Cohen 1988, in body §4.1 |
| F4 | v1 M3 swap baseline 10.0% / +G swap 12.0% / random 12.5% (n=200 each) | `M3_swap/gemma_sm_*_n200/trials.jsonl` (verified by aggregator) |
| F5 | M3' Gemma SM per-condition bet ratio and stop rate | `M3prime_indicator_steering/aggregated/gemma_sm.json` |
| F6 | Pearson r=+0.013, Spearman ρ=+0.010 (p=0.869), Cohen h −0.041, n=300 | `M3prime_indicator_steering/aggregated/gemma_sm.json` |
| F7 | E2 Gemma SM $I_\text{BA}$ Δ R²(+G,−G) = +0.082 / +0.147 / +0.138 | `E2_variance_normalized/gemma_sm_i_ba_L22.json` |
| F8 | E2 LLaMA SM $I_\text{BA}$ Δ R²(+G,−G) = +0.018 / +0.046 / +0.038 | `E2_variance_normalized/llama_sm_i_ba_L22.json` |
| F9 | E3 Gemma L22 V2 shared-axis AUC = 0.543 / 0.554 / 0.544; random 0.661 / 0.606 / 0.691 | `E3_balance_stratified/gemma_L22.json` |
| F10 | E3 LLaMA L22 V2 shared-axis AUC = 0.657 / 0.655 / 0.670; random 0.597 / 0.650 / 0.597 | `E3_balance_stratified/llama_L22.json` |
| F11 | E7 GEE has_G β=+0.527 (p=3.07e−06), has_M β=+0.473 (p=0.038) | `E7_mixed_effects_section3/results.json` |
| F12 | E7 bootstrap BK gap Gemma +0.054 [+0.043, +0.066]; LLaMA +0.719 [+0.696, +0.741] | `E7_mixed_effects_section3/results.json` |
| F13 | E7 cluster-bootstrap has_G β=+0.0498 [+0.037, +0.063]; has_M β=+0.0555 [+0.042, +0.069]; n_rounds=57797, n_games=3195 | `E7_mixed_effects_section3/results.json` |
| F14 | Gemma fixed BK 0.000 / variable 0.054; LLaMA fixed 0.004 / variable 0.723 (n=1600 each) | `E7_mixed_effects_section3/results.json` pooled_means |
| F15 | Steering hook: layer L22 forward hook, last-token additive, fires once per generation, α∈{−2..+3}σ | Code: `src/run_m3prime_indicator_steering.py` |
| F16 | SAE: gemma-scope-9b-pt-res-canonical L22 width-131k; Llama-Scope LXR-8x | Code: `src/compute_section4_steering_directions.py` |
| F17 | M3$''$ natural -G bet ratio 0.051±0.011 / stop 0.640; +G 0.216±0.018 / stop 0.060 | `M3pp_strong_patching/aggregated_gemma_sm.json` |
| F18 | M3$''$ patched_last 0.068, patched_suffix 0.079, patched_all 0.081, random_patch 0.072 (Welch p vs -G all > 0.15; vs +G all $< 10^{-5}$) | `M3pp_strong_patching/aggregated_gemma_sm.json` |
| F19 | M3$''$ Cohen $h$ on stop rate vs +G: patched_last +1.156, patched_suffix +1.237, patched_all +1.444, random +1.487 | `M3pp_strong_patching/aggregated_gemma_sm.json` |

All paths are relative to `results/v19_multi_patching/` under the SAE v3 analysis tree, except where explicitly noted as code.
