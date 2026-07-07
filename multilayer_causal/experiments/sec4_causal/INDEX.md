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
