# §4 Causal-Strengthening — Rung Ledger

Pre-registered causal study (design: `docs/specs/2026-07-05-section4-causal-unified-design.md`;
plan: `docs/plans/2026-07-06-sec4-causal-phase1.md`). Each rung is one analysed
wave. `sec4_autoresearch.py` appends a row per completed wave with the verdict
against the §3 GO/NO-GO criteria (H1 behavioural writes + H2 readout inert →
GO). No auto-launch — expansion is a human gate.

| rung | phase | hypothesis | config | key numbers | verdict | status |
|------|-------|-----------|--------|-------------|---------|--------|
| P0 | sec4_p0 | H1 behavioural writes, H2 readout inert | arms_sec4_p0.yaml (Gemma SM, 3 axes × α∈[-3,+3] + nulls + baseline + cum) | pending | pending | NOT RUN |

<!-- sec4_p0 -->
### Wave-1 `sec4_p0` — Gemma SM, I_BA, L16-21 (mlc-sec4-p0-0706c)
- **Hypotheses:** H1 behavioural axis writes betting; H2 readout inert; H3 confound inert; H4 locality.
- **Config:** `configs/arms_sec4_p0.yaml` (30 arms, n=200, alpha -3..+3, held-out state_offset 300, addiction_role_gm, alpha-independent seeds).
- **Result:** behavioural Spearman(a,I_BA)=+0.96 (p=0.000), delta=0.242, z@+3 vs null=+11.4; I_EC co-moves (delta=0.010). readout weak (rho=+0.68, z@+3=+3.0); confound inert (z@+3=+0.4). Locality: cum_16_21=0.25585167758891375.
- **Caveat:** read!=write UNRESOLVED — Wave-1 null is thin single-dose; Wave-2 adds a thick multi-dose null-slope band.
- **Verdict:** WRITE_CONFIRMED (behavioural writes I_BA & co-moves I_EC; read>=weak, confound inert)
- **HF:** rollouts `experiments/sec4_causal/checkpoints/sec4_p0/`, axes `experiments/sec4_causal/assets/`, analysis `results/sec4_p0/sec4_p0_analysis.json`.

### Wave-3 rung Q1-1 `postloss_rung1` — FREE re-analysis of sec4_p0+w2 (no node)
- **Hypothesis (Q1):** the behavioural axis drives loss-chasing-like behaviour (post-loss-specific potency), i.e. indicator commonality beyond bet-size.
- **Method:** seed->pool join (4460/4460 rows validated, 0 unlabeled), (alpha,seed) dedup vs W1/W2 pseudoreplication (1378 dropped), OLS dose x postloss interaction.
- **Result:** behavioural interaction coef=+0.0129, t=3.98 (n=1378) — steering is MORE potent on post-loss states in both directions (a+2: 0.248 vs 0.184; a-2: 0.015 vs 0.031); baseline cells equal (0.059 vs 0.066) so not a baseline artifact. readout: t=-1.30 NO_INTERACTION (dissociation again). shared: t=2.75 but UNDERPOWERED (W2 incomplete at fetch).
- **Verdict:** LC_LIKE_INTERACTION (Q1 rung1 positive). Dedicated arms (rung2, state_filter n=200/cell) in mlc-sec4-w3-0706.
- **Repro:** `python -m multilayer_causal.src.postloss_analysis --from-hf`; results `experiments/sec4_causal/analysis/postloss_rung1.json`.

### Wave-2 `sec4_w2` — thick multi-dose null adjudication (mlc-sec4-w2-0706 + auto-resume r1)
- **Result (I_BA, 20-direction null SLOPE band mean 0.0007 sd 0.0101):** behav_iba slope +0.0457 z=+4.45 ABOVE; shared(iba,iec SVD) +0.0328 z=+3.17 ABOVE; confound +0.0072 z=+0.64 INSIDE; **Wave-1 readout slope +0.0083 -> z=+0.75 INSIDE = statistically indistinguishable from random. monitor != controller now FIRM.**
- **I_EC:** runner `extreme` field degenerate (baseline 0/200 -> floor); proxy bet>=0.5 re-analysis: behav_iba EC-slope +0.0058 vs null sd 0.0022 (z~+2.8) — the I_BA axis moves extreme betting too; SVD-shared is EC-inert (dilution). Formal NO_SHARED_AXIS verdict reflects the degenerate field, not the axis.
- **Q1 synthesis: POSITIVE — the common indicator axis is the I_BA behavioural axis itself** (drives I_BA z4.45 + EC-proxy z2.8 + post-loss amplification), not the SVD composite.

### Wave-3 `sec4_w3` — Q1 rung2 / Q2 rung1 / Q3 rung1 (mlc-sec4-w3-0706c + auto-resume r1)
- **Q1 rung2 (dedicated postloss/postwin arms): POSTLOSS_STEER_AMPLIFIED** — slope_postloss 0.0416 vs postwin 0.0296, diff +0.0120 CI95[0.0061,0.0181] (replicates free rung-1 t=3.98). Q1 ADJUDICATED POSITIVE.
- **Q3 rung1: CONDITION_MODULATES** — slope(+G twin) 0.0535 vs slope(-G, W1) 0.0457, diff +0.0078 CI95[0.0040,0.0123]. Autonomy increases causal WRITABILITY, not just readability (§4.3 causal upgrade). Single-comparison; 2x2 G/M = next rung.
- **Q2 rung1 (expression-matched sh3 on SM/IC; IC parse gate 0.45 sensitivity — IC baseline parse 0.65):** geometric recon cos(sm,ic)=+0.43, cos(sm,mw)=+0.56, cos(ic,mw)=+0.65 (behavioural axes SHARED, unlike near-orthogonal BK). BUT causal transfer is SIGN-INVERTED and one-sided on IC: -3 dose RAISES risky-choice rate 0.062->0.281 (sh3) / 0.284 (ic_own), z -5.4/-9.5 vs IC null band; +doses ~null. **Seed-matched PAIRED analysis kills the selection-bias explanation: same 125 seeds, Δrisky +0.240 (↑36/↓6, sign-p<1e-4); newly-parsed seeds same composition (0.294 vs 0.279). Parse IMPROVES toward -3 (0.93-0.95) and craters at +3 (0.14-0.34).**
- **Q2 interpretation (rung1): geometry is shared, coherent control is NOT — the SM-calming direction makes IC choices MORE risky (genuine paired causal inversion, echoing invalidated-era sign flips but now with paired rigor). Next rungs: IC expression re-check (invested-amount vs option-choice), MW arm, re-baseline.**
- **Repro:** analyses `experiments/sec4_causal/analysis/sec4_w2_analysis.json`, `sec4_w3_analysis.json`; rollouts `experiments/sec4_causal/checkpoints/sec4_{w2,w3}/`; paired script inline in INDEX history (git).

### Q2 microstructure (free): IC choice distribution by dose
- **Finding:** `risky` = option 3/4. Baseline concentrates on opt2 (0.705) with opt4(=riskiest tail) at 0.000. At -3 the tail appears: sh3 opt4 0.157/opt3 0.124; ic_own opt4 0.226 with SAFE opt1 dropping 0.233->0.084. The inversion is not de-concentration noise — it CREATES risky-tail choices the baseline never makes.
- **Wave-4 launched (mlc-sec4-w4):** Q2 rung-2 = does the inversion replicate on MW (shared3@MW-scale + mw_own + MW null band, mw replay-window now excluded from build); Q3 rung-2 = 2x2 (plusM twin ladder + same-wave minusG control).

### Wave-5 build reconnaissance (Q2 rung-3): the inversion EXPLAINED geometrically
- **cos(ic_rc, ic_iba) = -0.67** — within IC, the risky-CHOICE axis is anti-aligned with IC's own bet-AMOUNT axis. The Wave-3 "sign inversion" (-amount-dose raising risky choice) is exactly what this geometry predicts: -(amount axis) ~ +(choice-risk axis).
- **cos(ic_rc, sm_iba) = -0.48, cos(ic_rc, mw_iba) = -0.77** — the two behavioural EXPRESSIONS of risk (amount vs choice) sit on OPPOSITE ends of a shared axis family across tasks. Sharing exists, but behaviour maps onto it with expression-dependent SIGN.
- **Revised sharing hypothesis (tested by mlc-sec4-w5-0707b):** shared3b = SVD(sm_iba, ic_rc, mw_iba) transfers with predictable per-expression sign — +dose should raise SM/MW bet ratio AND lower IC risky choice (or vice versa, per loading signs).

### Wave-4+5 adjudication — SIGN-PREDICTABLE CAUSAL SHARING on SM<->IC; MW = next object fix
- **Q2 rung-3 (W5, expression-matched IC axis): INVERSION RESOLVED.** ic_rc own axis correctly signed: risky 0.062->0.221 over alpha -3..+3 (rho +0.75); paired +3 vs 0: Δ+0.162 (↑24/↓6, p=0.0014). The Wave-3 inversion was an AXIS-OBJECT artifact (amount axis ~ -(choice axis), cos=-0.67).
- **shared3b sign-prediction test: 2/3.** Loadings cos(shared3b, sm_iba)=+0.77 / ic_rc=-0.88 / mw_iba=+0.92. Observed slopes: SM +0.0246 ✓, IC -0.0162 ✓ (paired -3 vs 0: Δrisky +0.133 p=0.017, per loading), MW -0.0162 ✗. **First genuine cross-task causal transfer with geometry-predicted signs: ONE axis raises SM betting AND lowers IC risky choice as its loadings dictate.**
- **MW = systematic anomaly, same signature IC had pre-fix**: W4 sh3_mw inverted (rho -0.75; -3 raises bet 0.122->0.248, paired Δ+0.149 ↑100/↓21 p<1e-4; parse improves at -3, craters at +3); mw_own weak/incoherent (+3 parse 0.19); W5 sh3b_mw -3 rises (paired Δ+0.103 p=0.044) against the +0.92 loading. **Diagnosis: MW's amount axis is also the wrong object.** MW's second expression = CONTINUE-vs-STOP (catalog: choice2+bet=spin n=8948 == paper n; choice1=stop n=3146). Next rung (W6): mw_rc = spin-vs-stop contrast axis (requires keep-mask change: stop rows have no bet), shared3c = SVD(sm_iba, ic_rc, mw_rc).
- **Q3 rung-2 (W4, 2x2): G-SPECIFIC writability modulation.** slopes: +G 0.0535 > -G same-wave 0.0404 (~ W1 0.0457) > +M 0.0383. GOAL-SETTING raises causal writability; reward-max does NOT — mirrors §4.3's G-primary readability finding. Q3 refined-positive.
- **Repro:** rollouts experiments/sec4_causal/checkpoints/sec4_{w4,w5}/; paired scripts in git history; axes assets experiments/sec4_causal/assets/ (shared3b loadings verified from npz).

### Wave-6 adjudication + OBJECT FREEZE + W7 PRE-REGISTRATION (committed BEFORE W7 launch)
- **W6 verdict: MW_RC_CORRECTLY_SIGNED.** mw_rc (spin/stop) ladder: spin-rate rho +0.90, slope +0.0549, z=+4.87 vs MW null band; bet corroborates (z=+5.48); monotone 0.67@-1 -> 0.94@+3. The expression-matched object fix GENERALIZES (IC rung-3 prescription worked on MW). Parse asymmetry now mirrors the object: -3 craters (0.215) while + side coherent (0.62-0.66) — -3/-2 cells parse-gated; verdict from -1..+3.
- **OBJECT FREEZE (no further object changes permitted):** SM = sm_iba (bet-ratio amount axis); IC = ic_rc (risky-option choice axis); MW = mw_rc (spin/stop continue axis); shared = shared3c = SVD-top1(sm_iba, ic_rc, mw_rc).
- **Frozen geometry (from W6 build):** cos(mw_rc,mw_iba)=-0.29 (amount/choice anti-alignment generalizes); cos(mw_rc,sm_iba)=+0.10; cos(mw_rc,ic_rc)=+0.24; loadings cos(shared3c, sm_iba)=-0.80, ic_rc=+0.89, mw_rc=+0.26. NOTE shared3c is dominated by the sm/ic anti-aligned pair; mw_rc weakly loaded.
- **W7 PRE-REGISTERED SIGN TABLE** (source axis steered on target; predicted sign of d(binding outcome)/d(alpha) = sign(cos(source, target-own axis)); outcome: SM=bet_ratio, IC=risky rate, MW=spin rate):
  | source \ target | SM | IC | MW |
  | sm_iba   | + (self) | - (cos -0.48) | + LOW-CONF (cos +0.10) |
  | ic_rc    | - (cos -0.48) | + (self) | + (cos +0.24) |
  | mw_rc    | + LOW-CONF (cos +0.10) | + (cos +0.24) | + (self) |
  | shared3c | - (cos -0.80) | + (cos +0.89) | + (cos +0.26) |
- **Pre-registered adjudication rule:** PRIMARY = the 10 confident cells (|cos|>=0.15): success = >=8/10 sign hits each above its target null band. All 12 cells reported regardless; the 2 LOW-CONF cells (|cos|=0.10) are reported but excluded from the primary criterion (predicted +, null acceptable). New seed_base 3000042 (RNG-independent of all discovery waves). Diagonal 7 doses, off-diagonal {-3,0,+3}; per-target nulls 3 dirs x +/-3 re-run under the new seed_base; n=200/arm; parse gates: SM 0.8, IC 0.45, MW 0.45.

### Wave-7 ADJUDICATION — pre-registered 12-cell symmetric confirmatory matrix (mlc-sec4-w7-0707, seed_base 3000042)
- **PRIMARY (pre-registered >=8/10 confident cells, sign + above target null band): 7/10 — BELOW threshold.** Reported honestly per the pre-registration.
- **BUT the structure is clean: sign direction hits 11/12; all 3 misses are MW-TARGET cells, and the cause is a spin-rate CEILING** (baseline spin: SM-target 0.063, IC-target 0.077, MW-target **0.823**). MW as a target has no headroom to push spin UP, so icrc->mw (z+0.3), mwrc->mw (self, z+0.3), sh3c->mw (z-0.3) all sit in the null band. This is a measurement limit, not absent sharing.
- **SM/IC-target = FULLY CONFIRMED (8/8 cells, all |z|>2, correct sign):** smiba->sm +z6.0, smiba->ic -z2.2, icrc->sm -z3.0, icrc->ic +z3.2, mwrc->sm +z4.2, mwrc->ic +z3.0, sh3c->sm -z3.7, sh3c->ic +z3.5. The SHARED axis shared3c drives SM betting DOWN and IC risky-choice UP exactly as its loadings dictate (cos -0.80 / +0.89); cross-task sources transfer with geometry-predicted signs.
- **MW works as a SOURCE:** mwrc->sm paired Δ(+3−0)=+0.156 (↑126/↓25, p<1e-4), mwrc->ic +z3.0. Only MW-as-target is ceiling-blocked.
- **Paired robustness (seed-matched, +3 vs 0):** sh3c->sm ↑22/↓64 p<1e-4 (down, per pred -); sh3c->ic Δ+0.088 (↑17/↓8); smiba->mw Δ+0.132 p=0.008; mwrc->sm Δ+0.156 p<1e-4. (Mean can flip on outliers where the sign test agrees with slope — slope/sign are the primary read.)
- **VERDICT: sign-predictable cross-task causal sharing CONFIRMED on the SM<->IC axis pair (8/8 target cells + the shared axis); MW enters the shared axis as a source but its spin-rate ceiling blocks target-side confirmation. Pre-registered 8/10 bar not met due to that ceiling; a MW mid-range re-baseline is the fix (future rung), reported as a stated limitation.**
- **Next: W8 LLaMA model symmetry** (the paper is 2-model; this whole matrix is Gemma). Then §4.2 causal companion table + limitation.

### W8-extract — LLaMA SM/MW full 32-layer hidden re-extraction (mlc-sec4-w8extract-0707)
- **DONE:** SM (61,895 rounds) + MW (85,140 rounds) full 32-layer phase_a hidden, fp16, uploaded to sae_features_v3/{slot_machine,mystery_wheel}/llama/checkpoints/phase_a_hidden_states.npz (16.2GB / 22.3GB). Existence-guarded, zero overwrite; 'layers'=0..31 provenance written.
- **dp-parity verify: cos 0.9999 (PASS), relL2 med 1.5e-2 (gate FAIL) — DIAGNOSED as a pure ~1.5% SCALE offset, not a direction error.** Evidence: the dp dump is itself fp16 (fp16 round-trip relL2=0, so fp16 storage is not the cause); cos≈1 with relL2≈0.015 is exactly a=1.015·b; only 2-4/3200 rows exceed the cos gate (all still cos>0.998). Cause = forward-pass condition drift (transformers/attention version) vs the original dp extraction. HARMLESS for our use: the behavioural axis is a DIRECTION (mean-diff) and is per-layer re-normalised by scales_from_phase_a, so a global scale cancels. Re-extraction accepted as AXIS-EQUIVALENT (not bit-exact) — stated limitation.
- **Next:** repoint paper_axes LLAMA_SM_HIDDEN (5-layer dp -> full checkpoints/) + add LLAMA_MW_HIDDEN, then the LLaMA window scan (full symmetry, no 5-layer approximation).
