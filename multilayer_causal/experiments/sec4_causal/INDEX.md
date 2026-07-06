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
