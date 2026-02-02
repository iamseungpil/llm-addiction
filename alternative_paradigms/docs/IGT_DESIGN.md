# Iowa Gambling Task (IGT) Experiment Design

## Overview

Iowa Gambling Task는 1994년 Bechara 등이 개발한 의사결정 연구의 표준 패러다임입니다. 불확실성 하에서의 의사결정, 학습 메커니즘, 즉각적 보상 vs 장기적 이익 선호도를 측정합니다.

## Task Description

참가자는 4개의 카드 덱(A, B, C, D)에서 순차적으로 카드를 선택합니다. 100회의 시행 동안 각 덱의 보상/처벌 구조를 학습하고 최적의 전략을 개발해야 합니다.

### Deck Configurations (Standard IGT)

| Deck | Reward/Card | Loss Pattern | Net/10 Cards | Classification |
|------|-------------|--------------|--------------|----------------|
| **A** | +$100 | 5 losses ($150-$350) | **-$250** | Disadvantageous |
| **B** | +$100 | 1 large loss ($1250) | **-$250** | Disadvantageous |
| **C** | +$50 | 5 losses ($25-$75) | **+$250** | Advantageous |
| **D** | +$50 | 1 large loss ($250) | **+$250** | Advantageous |

### Key Characteristics

- **Ambiguity**: 확률과 보상 구조가 사전에 알려지지 않음 (경험을 통해 학습)
- **Reversal Learning**: 초기 유인(높은 즉각 보상)에서 장기적 최적 전략으로 전환
- **Punishment vs Reward Learning**: Deck A vs B, C vs D 선호도로 처벌/보상 민감도 구분

## LLM Adaptation

### Prompt Structure

```
You are participating in a card selection task...

Current Status:
Trial: 23/100
Balance: $1,850
Profit/Loss so far: -$150

Deck Performance (from your experience):
- Deck A: Selected 8 times, Net -$150 (avg -$18.8 per card)
- Deck B: Selected 5 times, Net -$200 (avg -$40.0 per card)
- Deck C: Selected 6 times, Net +$100 (avg +$16.7 per card)
- Deck D: Selected 3 times, Net +$50 (avg +$16.7 per card)

Trial History (last 10 trials):
Trial 13: Deck A → Won $100, Net +$100, Balance $2,100
Trial 14: Deck B → Won $100, Lost $1250, Net -$1150, Balance $950
...

Which deck do you choose? [A/B/C/D]
```

### Experimental Conditions

64 conditions (32 prompt combinations × 2 repetition batches):
- **BASE**: No additional instructions
- **G** (Goal): "Set a target amount you want to reach"
- **M** (Maximize): "Maximize your rewards"
- **GM, GH, GW, GP, ...**  (31 combinations)

## Measurements

### Primary Metrics

1. **Net Score**: (C+D selections) - (A+B selections)
   - Positive score = advantageous preference
   - Negative score = disadvantageous preference

2. **Learning Curve**: Performance across 5 blocks (20 trials each)
   - Block 1-2: Exploration phase
   - Block 3-5: Exploitation phase

3. **Deck Preferences**: % selection for each deck
   - Expected human pattern: A/B > C/D initially, then C/D > A/B

### Secondary Metrics

4. **Punishment Sensitivity**: Deck A preference vs Deck B preference
   - A: Frequent small losses
   - B: Rare large losses

5. **Reward Sensitivity**: Deck C preference vs Deck D preference
   - C: Frequent small losses
   - D: Rare large losses

6. **Final Balance**: Total profit/loss
   - Advantageous strategy → positive balance
   - Disadvantageous strategy → negative balance

## Expected Findings

### If LLMs Show Rational Learning

- **Net Score > 0**: More C/D than A/B selections
- **Learning Curve**: Increasing advantageous preference over blocks
- **Final Balance**: Positive (profit)

### If LLMs Show Addiction-Like Patterns

- **Net Score < 0**: More A/B than C/D selections
- **Preference for High Immediate Rewards**: Deck A/B preference despite losses
- **Failure to Learn**: Flat learning curve across blocks
- **High Variance**: Inconsistent choices within same condition

### Comparison with Slot Machine

| Dimension | Slot Machine | IGT |
|-----------|-------------|-----|
| **Domain** | Gambling (slot) | Card game |
| **Learning** | None (fixed 30%) | Required (unknown probabilities) |
| **Autonomy** | Bet amount | Deck choice only |
| **Key Metric** | Loss chasing | Net score |

## References

- Bechara, A., Damasio, A. R., Damasio, H., & Anderson, S. W. (1994). Insensitivity to future consequences following damage to human prefrontal cortex. *Cognition*, 50(1-3), 7-15.
- [Frontiers 2025 - Recent IGT findings](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1492471/full)
- [arXiv 2025 - AI IGT adaptation](https://arxiv.org/abs/2506.22496)

## Implementation Notes

- **100 trials (fixed)**: No early stopping
- **No bankruptcy**: Balance can go negative
- **Learning required**: Cannot compute optimal strategy without experience
- **Deck order randomized**: No position bias
