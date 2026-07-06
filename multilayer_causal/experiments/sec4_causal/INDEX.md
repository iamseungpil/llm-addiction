# §4 Causal-Strengthening — Rung Ledger

Pre-registered causal study (design: `docs/specs/2026-07-05-section4-causal-unified-design.md`;
plan: `docs/plans/2026-07-06-sec4-causal-phase1.md`). Each rung is one analysed
wave. `sec4_autoresearch.py` appends a row per completed wave with the verdict
against the §3 GO/NO-GO criteria (H1 behavioural writes + H2 readout inert →
GO). No auto-launch — expansion is a human gate.

| rung | phase | hypothesis | config | key numbers | verdict | status |
|------|-------|-----------|--------|-------------|---------|--------|
| P0 | sec4_p0 | H1 behavioural writes, H2 readout inert | arms_sec4_p0.yaml (Gemma SM, 3 axes × α∈[-3,+3] + nulls + baseline + cum) | pending | pending | NOT RUN |
