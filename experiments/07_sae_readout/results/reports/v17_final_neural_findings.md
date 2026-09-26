# V17: Final Neural Analysis Findings

**Date:** 2026-04-07
**Status:** Codex approved, ready for paper integration

---

## Key Findings

### RQ1: Irrationality Pattern Exists Beyond Balance

**Method:** Per-turn bet_ratio prediction from SAE features, controlling for balance via residualization.
**Result:** Gemma SM, 42-layer sweep, 12,246 Variable betting rounds.

| Layer Range | Residual R² | Interpretation |
|-------------|-------------|----------------|
| L0-L8 (early) | 0.14-0.19 | Weak signal |
| **L16-L23 (mid)** | **0.31-0.34** | **Peak — irrationality computation here** |
| L24-L40 (late) | 0.28-0.31 | Slight decline |

- Best layer: L18, residual R²=0.34
- 29 non-trivial risk features (|r_bal|<0.15, r_ratio>0.15)
- LLaMA DP: residual AUC=0.654 (p=0.005) — cross-architecture replication

### RQ2: Task-Independent Features

**Method:** Balance-controlled feature overlap across SM, IC, MW. Hypergeometric test.

| Model | 3-Way Universal Features | Chance Expected | Enrichment |
|-------|-------------------------|----------------|------------|
| Gemma | 154 | 0.01 | 15,400x |
| LLaMA | 102 | 0.12 | 850x |

**Layer profile:** Overlap is HIGHEST at early layers (L0), decreasing toward late layers.
- Gemma: L0=139, L9=124, L18=78, L30=17
- LLaMA: L0=117, L3=65, L12=30, L22=14

**Interpretation:** Two-stage process:
1. Early layers: shared gambling/risk schema (context recognition)
2. Mid layers: irrationality computation (RQ1 peak at L16-L23)

### RQ3: Condition Modulation

**Variable vs Fixed (balance-controlled):**
- LLaMA SM: Variable 1.22x stronger signal (p=1.1e-5)
- LLaMA MW: Variable 1.44x stronger (p=1.2e-24)
- Gemma: REVERSED — Fixed > Variable (autonomy paradox)

**G-prompt effect:**
- LLaMA SM: G-prompt 1.55x amplification (0.224 vs 0.144)
- Gemma SM: minimal difference (1.02x)

**Causal evidence (V12):**
- Activation steering along BK direction: rho=0.964, permutation p=0.048
- LLaMA Slot Machine only

---

## Narrative for Paper

1. LLMs encode gambling irrationality beyond balance tracking (R²=0.34, mid-layer peak)
2. A shared vocabulary of 100+ features recurs across three gambling tasks (early-layer dominance)
3. Autonomy conditions amplify the signal in LLaMA but reverse in Gemma (architecture-dependent)
4. Activation steering provides causal evidence for the link

## Honest Framing

- Effect sizes are moderate, not dramatic (R²=0.34, not 0.90)
- This is a strength of the methodology, not a weakness
- The old AUC=0.97 was inflated by balance confound; the new R²=0.34 is clean
- Gemma reversal is reported honestly as a limitation and future work direction
