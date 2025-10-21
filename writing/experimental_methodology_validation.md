# Experimental Methodology Validation — Goals, Risks, and Related Methods
Last updated: 2025-09-04

## 1) Current Analysis Goals (Exp1 vs. Exp2)
- Goal alignment: Quantify whether specific internal features (from SAEs at L25/L30) causally influence risk-seeking vs. risk-averse behavior in a slot-machine setting.
- Exp1 (Multi‑round baseline): Establish behavioral priors and prompts that elicit distinct end states (bankruptcy vs. voluntary stop), with multi‑round dynamics and history effects. Output: rates of bankruptcy/stop, rounds, and qualitative prompt efficacy.
- Exp2 (Activation patching with population means): Test causal influence by clamping single SAE features toward safe vs. risky population means at controlled scales, measuring monotonic changes in betting/stop rate across trials and prompts.

## 2) Ultra‑Review of the Current Setup (Strengths, Risks, Mitigations)
- Strengths:
  - Direct, causal test: Single‑feature interventions via SAE encode/decode at specific layers; monotonicity checks across scales; two target prompts (risky/safe) for bidirectional evidence.
  - Behavioral metrics with face validity: avg bet, stop rate; Exp1 adds bankruptcy in multi‑round setting for stronger outcome signal.
  - Practical engineering: Mid‑run checkpoints; feature index partitioning across GPUs; stable SAE loader with verified norms.
- Methodological risks and mitigations:
  - Hook fragility: Forward hooks can be brittle if module outputs differ (tuple vs. tensor). Mitigation: keep the returned structure identical; pre‑flight small feature slices; unit test the hook on both layers with synthetic inputs.
  - Memory pressure: 8B model in float32 + SAE float32 can approach VRAM limits. Mitigation: bfloat16 model weights; clear CUDA cache between features; keep `use_cache=False` only if needed.
  - Parsing bias: Fallback `bet=10` on parse error can bias means. Mitigation: treat failed parses as invalid and exclude from aggregates; log invalid rate per condition.
  - Language variance: Stop detection currently English‑centric. Mitigation: include multilingual stop lexicon (e.g., Korean tokens already used in other scripts) and numeric “2” patterns.
  - Scale granularity/statistical power: Only 3 scales can yield unstable rank correlations. Mitigation: optionally add 1–2 scales (e.g., 0.3, 2.0) for robustness; bootstrap CIs across trials.
  - Negative controls: Need “sham” patches. Mitigation: patch random non‑selected features or patch in non‑decision layers; expect null effects.
  - Overfitting to two targets: Effects tied to exact prompts. Mitigation: small panel of semantically similar targets; check sign consistency.
  - Multiple comparisons: Many features tested. Mitigation: focus on effect size and sign consistency across targets/scales; use FDR only if interpreting p‑values.
  - Reproducibility: Unseeded sampling increases variance. Mitigation: fixed seeds per (feature, scale, prompt) when stress‑testing; keep stochasticity in final runs but report variance.

## 3) How This Maps to Established Methods (Validation by Analogy)
- Activation patching (Transformer Circuits literature): Replace or modify internal activations to test whether hypothesized sub‑computations drive outputs. Our SAE‑based clamping is a structured variant that edits a single latent feature and decodes back to residual space.
- Interchange interventions (counterfactual activation replacement): Swap hidden states between a base and a counterfactual input to probe causal pathways. Our “move toward safe/risky population means” is a continuous analogue of swapping with a counterfactual donor.
- Causal scrubbing (hypothesis checks): Validate that a proposed circuit/pathway explains behavior by systematically intervening while holding irrelevant paths constant. Our negative controls and monotonicity checks fill this role for SAE features.
- Representation editing / steering vectors: Control behavior by adding latent directions or attribute vectors at chosen layers (e.g., activation additions, PPLM‑style guidance). Our approach differs by using SAE features (sparse, interpretable) instead of dense neuron/activation directions, but shares the core principle of internal intervention.
- Sparse Autoencoders for interpretability (Anthropic and follow‑ups): SAEs recover sparse, interpretable features from residual stream activations. Using SAEs as the basis for feature‑level interventions is consistent with this line, enabling semantically targeted edits.

## 4) Reference Landscape (Method Families and Canonical Works)
Note: Grouped by method; intended as anchors for methodology, not an exhaustive bibliography.

- Sparse Autoencoders / Monosemantic Features
  - Anthropic “Sparse Autoencoders” and “Scaling Monosemanticity” (2023–2024): Train SAEs on residual stream; show interpretable, often localized features; discuss normalization and dataset shuffling to avoid spurious correlations.
  - SAE best practices: layer choice (mid/late), residual stream focus, dataset norm scaling; emphasize reconstruction vs. sparsity trade‑offs.

- Activation Patching / Causal Interventions in Transformers
  - Transformer Circuits (Olah et al., 2020–2022): Activation patching as core tool to test circuit hypotheses by replacing hidden activations from counterfactual inputs.
  - Interchange interventions (Geiger et al. line of work): Swap internal activations between base/counterfactual to test causal contribution of specific components/positions.
  - Redwood Research “Causal Scrubbing” (circa 2023): Systematic intervention framework to validate that a proposed mechanism actually explains outputs.

- Representation Editing / Steering
  - Plug‑and‑Play Language Models (PPLM, 2019/2020): Gradient‑based manipulation of activations to control attributes without fine‑tuning.
  - Activation additions / steering vectors (various 2022–2024): Edit hidden states along attribute directions to steer toxicity, sentiment, etc.
  - Knowledge editing (ROME/MEMIT/HyperNetworks): Weight‑level edits for factual associations; conceptually adjacent as targeted internal changes to alter behavior.

- Behavioral Evaluation & Terminal‑State Focus
  - CLS/last‑token precedent: Use of a single token vector (BERT [CLS]; GPT last token) as a sequence summary for classification.
  - Outcome‑centric RL framing: Emphasis on terminal states for credit assignment and evaluation; aligns with our use of end‑moment prompts and multi‑round bankruptcy signals.

## 5) Concrete Validation Checks to Strengthen Claims
- Monotonicity stress test: Add 1–2 additional scales; verify effect direction consistency and diminishing returns beyond some scale.
- Bidirectional symmetry: For a feature with risky>safe mean, verify risky‑ward clamp increases risk metrics and safe‑ward clamp decreases them (and vice versa).
- Layer specificity: Repeat for L25 vs. L30; show effect concentration at specific layers.
- Sham controls: Random feature clamp; non‑decision token positions; expect near‑zero effects.
- Cross‑prompt robustness: Replicate on a small panel of semantically similar targets (paraphrases) to avoid prompt overfitting.
- Trial variance reporting: Include per‑condition CIs (bootstrap over trials) to quantify stochasticity under sampling.
- Parse‑robust metrics: Track invalid parse rate; perform sensitivity analysis excluding vs. imputing.

## 6) How Exp1 Supports Exp2
- Exp1 establishes that the task framing yields separable behavioral regimes (high voluntary stop vs. bankruptcy tails) and that decision‑moment prompts are effective.
- Exp2 then tests whether single latent features mediate those regimes via controlled internal interventions, following established activation‑patching/causal‑intervention paradigms.

## 7) Suggested Paper Framing
“Building on activation‑patching and interchange‑intervention methodologies from the mechanistic interpretability literature, we perform single‑feature interventions using sparse autoencoders to test whether specific latent features causally modulate risk‑seeking behavior. We evaluate monotonic responses across manipulation scales and verify bidirectional effects across risky/safe targets. A multi‑round baseline experiment provides outcome‑level validation (bankruptcy vs. voluntary stop), grounding our intervention results in behavior.”

