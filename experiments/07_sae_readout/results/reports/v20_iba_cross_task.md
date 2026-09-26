# V20: I_BA Cross-Task Analysis Results

**Date**: 2026-04-07

---

## I_BA Within-Task Results (NL Deconfounded, 8/8 significant)

| Model | Task | Layer | n | R² | Random | p |
|-------|------|-------|---|-----|--------|---|
| Gemma | SM | L24 | 12,246 | **0.158** | 0.001 | <0.05 |
| Gemma | SM | L18 | 12,246 | **0.127** | 0.001 | <0.05 |
| Gemma | MW | L24 | 8,948 | **0.034** | -0.001 | <0.05 |
| Gemma | MW | L18 | 8,948 | **0.041** | -0.001 | <0.05 |
| LLaMA | SM | L16 | 45,551 | **0.120** | 0.023 | <0.05 |
| LLaMA | SM | L22 | 45,551 | **0.110** | 0.032 | <0.05 |
| LLaMA | MW | L16 | 57,220 | **0.067** | 0.026 | <0.05 |
| LLaMA | MW | L22 | 57,220 | **0.068** | 0.031 | <0.05 |

## I_BA Cross-Task Transfer (SM ↔ MW)

**All transfers fail** — R² negative for all directions.

| Model | Layer | SM→MW | MW→SM | Random | Within SM | Within MW |
|-------|-------|-------|-------|--------|-----------|-----------|
| Gemma | L24 | -2.01 | -0.06 | -3.01 | 0.011 | 0.059 |
| Gemma | L18 | -12.3 | -1.87 | -12.3 | 0.012 | 0.056 |
| LLaMA | L16 | -7.70 | -0.08 | -3.10 | 0.029 | 0.069 |
| LLaMA | L22 | -2.86 | -0.14 | -2.01 | 0.029 | 0.069 |

**Note**: Within-task R² with shared features only (0.01-0.07) is much lower than full-feature analysis (0.034-0.158), confirming that predictive features are task-specific.

## Interpretation

Same as I_LC: **convergent evolution**. Both I_BA and I_LC are encoded in multiple paradigms, but through task-specific feature sets. The behavioral patterns converge, the neural pathways diverge.

## Paper Impact

- **Table 1**: Corrected I_BA MW from "-- n.s." to actual values (Gemma 0.034, LLaMA 0.067)
- **Framing**: I_LC and I_BA presented as complementary (Codex recommendation)
  - I_LC = "deep" metric (high R², DSM-5 clinical relevance)
  - I_BA = "broad" metric (SM+MW coverage)
- **Discussion**: Updated to include cross-task transfer failure and convergent evolution interpretation
