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
