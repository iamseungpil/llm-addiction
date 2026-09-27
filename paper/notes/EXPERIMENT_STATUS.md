# Experiment status — submission v9 (2026-05-06)

**Supersedes**: PATCHING_PLAN.md v4 (now historical)
**Scope**: which experiments are in the submission, which were planned but deferred, which post-submission directions are recommended.

---

## §1 Submission state

- EN body: 9 pages exactly (NeurIPS limit)
- KO mirror: 9 pages exactly
- 28,800 behavioural games + L22 SAE readout audit + LOTO PCA cross-task geometry
- Appendix A–F complete; M3/M3'/M3'' patching audits in §E reported as null
- All reviewer-driven calibrations applied: dual 6-model consistency framing, monitoring-relevant readout language, cognitive-distortion as scaling device, layer-sweep robustness pointer

---

## §2 Completed and reported in paper

### Behavioural (§3)
| ID | Experiment | Result | Where in paper |
|----|-----------|--------|----------------|
| F1 | Slot-machine variable vs fixed bankruptcy (BASE prompt, 6 models) | LLaMA 0.4→72.3%, Gemini 3.1→48.1%, all 6 ↑ | §3 Finding 1, Fig 2a |
| F2 | Streak escalation post-win/post-loss | 3.3× / 2.8× under variable | §3 Finding 2, Fig 2c–d |
| F3 | IC option distribution + goal escalation under G/M/GM | 6/6 models bankrupt more under G; high-variance share 26→38/42% | §3 Finding 3, Fig 3a–c |
| F4 | Matched-cap inversion ($10/$30/$50/$70) | Fixed +7pp above Variable across all caps; bet-size effect does not transfer | §3 Finding 4, Fig 3d |
| F5 | Cognitive-distortion keyword scan | Loss-chasing rises in 5/6 (variable), 6/6 (goal) at FDR p<0.05 | §3 Finding 5, Appx Table 8 |

### Internal readout (§4, two open-weight models)
| ID | Experiment | Result | Where in paper |
|----|-----------|--------|----------------|
| RQ1 | SAE features → IBA/ILC/IEC at L22, GroupKFold | R²∈[0.05,0.31]; task-specific binding indicator | §4.1, Table 1 |
| RQ2 | LOTO PCA shared axis, cosine alignment, sparse-feature transfer | rank-1 SM 0.69, IC 0.59, MW 0.53; rank-2 recovers MW 0.91 | §4.2, Fig 4, Table 2 |
| RQ3 | Condition modulation: ±G, ±M | Gemma SM IBA +143%, LLaMA +38%, MW IBA/ILC modulated, IC saturated | §4.3, Table 3, Appx Table 16 |

### Confound-removal supplements (Appendix E)
| ID | Experiment | Result | Where |
|----|-----------|--------|-------|
| E2 | Variance-normalised §4.3 (balance-window stratified) | Gemma +0.082/+0.147/+0.138; LLaMA +0.018/+0.046/+0.038 — modulation persists | §E.2 |
| E3 | Balance-stratified §4.2 BK shared axis (V1 matched + V2 residualised) | LLaMA L22 V2 above random; Gemma L22 V2 collapses to random | §E.3 |
| E7 | Cluster-aware §3 GEE + bootstrap | $\hat\beta_G=+0.527$, $p=3.07\times 10^{-6}$; per-model bootstrap CIs exclude zero | §E.4 |

### Causal-control patching (Appendix M3)
| Protocol | Method | Result | Where |
|----------|--------|--------|-------|
| M3 | Single-decision prompt swap −G→+G | bk 10.0% → 12.0% (Cohen h≈0.06, null) | §M3 |
| M3' | Ridge-weight × SAE-decoder direction steering, dose ladder α∈{−2..+3} | Pearson r=+0.013, p=0.869; controls null | §M3 |
| M3'' | Paired full-prompt L22 patching at last/suffix/all | bk 0.068/0.079/0.081, Welch t p>0.15 vs −G; far from +G | §M3 |

**Reading**: §4.1 Ridge readout decodes the indicator without acting as a single-layer controller on Gemma slot machine. §5 conclusion explicitly hedges to monitoring-relevant readout signals.

---

## §3 Planned but deferred from submission

### LLaMA at L22 for M3'/M3''
- **Status**: not run for submission
- **Reason**: released Llama-Scope SAE artifacts at L22 differ from the L25 / L31 sets used in §4.1; cross-protocol consistency would require re-extraction and re-validation
- **Effort**: ~6h on a single A100 once Llama-Scope L22 features are extracted

### Tier 2 reviewer-deferred experiments (from PATCHING_PLAN v4 §2)
| Reviewer Q | Experiment | Reason deferred |
|-----------|-----------|-----------------|
| Q2 | E1 — Discrete bet-grid ablation (autonomy vs output-format complexity) | Action-space dissociation already covered by F4 matched-cap |
| Q4 | E5 — LLM-judge cognitive-distortion validation | F5 keyword scan is now repositioned as scaling device, not diagnostic |
| Q5 | E8 — Decoding parameter sweep (top-p, repetition penalty) | Temperature robustness in Appx D already covers main attack surface |
| Q6 | E6 — DeLLMa-style mitigation prompt baseline | Out of scope for this paper's diagnostic framing |

---

## §4 Reviewer NeurIPS feedback (rev1 + rev2) — addressed status

| Reviewer concern | Resolution in v9 |
|------------------|------------------|
| "Within-slot-machine matched-cap ambiguity (autonomy vs action range)" | Finding 1 closing now names the effect as "broad bet-size-flexibility effect — risk rises when bet-size choice is delegated to the model in a wider action range — rather than as a pure freedom-to-choose effect"; F4 matched-cap dissociation explicit |
| "Abstract still defensive / 'internalize' too strong" | Abstract last 2 sentences: "decodable from decision-time hidden states; behaviour-only monitoring can miss a complementary readout-level signal" |
| "Cognitive-distortion keyword scan brittle" | F5 closing: "do not validate a clinical diagnosis or a causal reasoning mechanism; they scale the qualitative observation that high-risk regimes are accompanied by loss-recovery and control-like language" |
| "MW task surface under-described" | §4 opening: "a roulette-style task in which the model repeatedly bets on hidden colour–payout outcomes under a different surface structure from SM and IC" |
| "Layer-sweep cherry-picking risk" | §4.4 Summary: "task-binding pattern is preserved across a layer sweep {L8, L12, L22, L25, L30} rather than peak-selected at L22" + Appx layer-sweep verification |
| "Output parsing / closed-model refusal handling" | Appx F documents release path; not addressed in main text by design (out-of-scope confound) |
| "Inter-seed variability for §4.3 modulation deltas" | Per-fold CIs in Appx E.2 (5-fold within-condition); inter-seed not run |
| "Programmatic baseline (Kelly, stop-loss)" | Recognised as natural future work (post-submission §5 below) |

---

## §5 Post-submission experiment roadmap

The submission already covers Tier 1 (M3/M3'/M3''/E2/E3/E7). The following extensions would strengthen a journal version or rebuttal package.

### Tier P0 — within-1-week, low-cost
1. **LLaMA M3'/M3'' at L22**: re-extract Llama-Scope L22 features; reproduce M3'/M3'' protocols on LLaMA SM
   - **Output**: cross-model causal generalisation of the null
2. **Cognitive-distortion human validation (100-sample)**: stratified random sample (5×4×5 = 100 quotes), single annotator + blinded re-annotation, compute precision/recall against keyword scan
   - **Output**: appendix subsection reporting precision ≥ 0.7 / recall ≥ 0.5 expected
3. **Inter-seed stability for §4.3 modulation**: re-run §4.3 modulation analysis with 3 seeds (game-block shuffle)
   - **Output**: 95% CI on Gemma SM IBA Δ_G%

### Tier P1 — within-1-month
4. **Within-slot-machine matched-cap**: introduce a fixed-cap variant on slot machine (model can choose any integer ≤ cap; cap matched to fixed condition)
   - **Output**: full bet-size autonomy / action-range disambiguation that current matched-cap only does in IC
5. **Programmatic baselines**: run Kelly, fixed stop-loss, fixed stop-gain policies on the same SM/IC games
   - **Output**: rationality benchmark band; positions LLM behaviour relative to risk-adjusted optima
6. **Mitigation prompt sweep (E6 from v4)**: DeLLMa-style structured-uncertainty + stop-rule prompts
   - **Output**: prompt-engineering corollary for reducing bankruptcy

### Tier P2 — within-3-months
7. **Multi-layer M3' steering**: extend M3' to additive interventions across {L18, L22, L25, L28} simultaneously; learnable subspace (DAS-style) within the SAE basis
   - **Output**: stronger causal evidence (potentially flips the §4.5 null)
8. **Deployment monitoring testbed**: package §3 indicators + L22 readout into a runtime monitor; evaluate on third-party agentic benchmarks (e.g., trading agents)
   - **Output**: practical monitoring artifact + cross-domain transfer test
9. **Closed-model parsing & refusal-rate audit**: full disclosure of how we parsed each model's output, refusal/format-violation rates, post-processing rules
   - **Output**: appendix expansion + reproducibility supplement

---

## §6 Status of v4 plan items (cleanup)

All Code 1–7 from PATCHING_PLAN v4 §11 are complete. AMLT 4-node deployment (§10.5) was used for M3' parallelisation; results integrated into M3 appendix.

Items still open (from v4 §9):
1. ~~AMLT 4 paused nodes~~ — completed
2. ~~Phase 2 LLaMA replication~~ — moved to Tier P0 (post-submission)
3. ~~Tier 2 (E1/E5/E6/E8)~~ — closed; F5 reposition + temperature robustness sufficient
4. ~~v1 (M3 swap)~~ — completed and reported
