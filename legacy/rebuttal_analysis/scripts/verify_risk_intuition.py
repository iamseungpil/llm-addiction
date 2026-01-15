#!/usr/bin/env python3
"""
Verify intuition: 베팅 금액 크고 + 승리 확률 작으면 → Risky?
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("Risk 요인 분석: 베팅 금액 vs 승리 확률")
print("="*80)

# 1. 베팅 금액의 영향
print("\n" + "="*80)
print("요인 1: 베팅 금액 (Bet Amount)")
print("="*80)

print("\nOption 2 (50% chance of 1.8x), 베팅 금액 변화:")
print(f"\n{'Bet':<10} {'Variance':<15} {'SD':<15} {'관계':<20}")
print("-"*60)

for bet in [1, 10, 50, 100]:
    var = bet**2 * 0.81  # Coefficient for Option 2
    sd = np.sqrt(var)

    if bet == 1:
        relation = "기준"
    else:
        ratio = var / 0.81
        relation = f"{ratio:.0f}배 증가"

    print(f"${bet:<9} {var:<15.2f} ${sd:<14.2f} {relation:<20}")

print("\n✓ 결론: Variance ∝ Bet²")
print("  → 베팅 금액이 2배면 변동폭은 4배!")
print("  → 베팅 금액 ↑ = Risk ↑")

# 2. 승리 확률의 영향
print("\n" + "="*80)
print("요인 2: 승리 확률 (Win Probability)")
print("="*80)

print("\n동일 기댓값(EV=0)을 유지하면서 승리 확률만 변경:")
print("\n조건: 베팅 $10, EV = $0 (공정한 게임)")
print(f"\n{'Win Prob':<12} {'Win Payout':<15} {'Variance':<15} {'SD':<15}")
print("-"*60)

bet = 10

for win_prob in [0.99, 0.9, 0.5, 0.25, 0.1, 0.01]:
    # EV = 0이 되도록 payout 계산
    # win_prob × payout + (1-win_prob) × 0 = bet
    # payout = bet / win_prob

    win_payout = bet / win_prob

    # Variance 계산
    ev = bet  # EV = bet for fair game
    var = win_prob * (win_payout - ev)**2 + (1 - win_prob) * (0 - ev)**2
    sd = np.sqrt(var)

    print(f"{win_prob:<12.2f} ${win_payout:<14.2f} {var:<15.2f} ${sd:<14.2f}")

print("\n✓ 결론: 승리 확률 ↓ = Variance ↑")
print("  → 승리 확률이 낮으면 승리 시 보상이 커야 EV 유지")
print("  → 결과 분포가 극단적 = 변동폭 큼")

# 3. 우리 실험의 실제 옵션들
print("\n" + "="*80)
print("요인 3: 실제 Investment Choice 옵션들 (Bet=$10)")
print("="*80)

options = {
    2: {'prob': 0.5, 'multiplier': 1.8, 'ev_mult': -0.1},
    3: {'prob': 0.25, 'multiplier': 3.2, 'ev_mult': -0.2},
    4: {'prob': 0.1, 'multiplier': 9.0, 'ev_mult': -0.1},
}

print(f"\n{'Option':<10} {'Win Prob':<12} {'Multiplier':<15} {'EV':<10} {'Variance':<15} {'SD':<15}")
print("-"*80)

bet = 10

for opt, data in options.items():
    prob = data['prob']
    mult = data['multiplier']

    win_payout = bet * mult
    ev = bet * (1 + data['ev_mult'])

    var = prob * (win_payout - ev)**2 + (1 - prob) * (0 - ev)**2
    sd = np.sqrt(var)

    print(f"Option {opt}  {prob:<12.2f} {mult:<15.1f}x ${ev:<9.2f} {var:<15.2f} ${sd:<14.2f}")

print("\n관찰:")
print("  Option 2: 50% 확률, 1.8배 보상 → Var = 81")
print("  Option 3: 25% 확률, 3.2배 보상 → Var = 192 (2.4배)")
print("  Option 4: 10% 확률, 9.0배 보상 → Var = 729 (9배!)")
print("\n✓ 승리 확률 ↓ & 보상 배율 ↑ = Variance ↑↑")

# 4. 두 요인의 조합
print("\n" + "="*80)
print("종합: 베팅 금액 × 확률 분포 = Risk")
print("="*80)

print("\n실제 예시 비교:")
print(f"\n{'시나리오':<40} {'Variance':<15} {'SD':<15}")
print("-"*70)

scenarios = [
    ("Option 2, $10 bet (50%, 1.8x)", 10, 2),
    ("Option 4, $10 bet (10%, 9.0x)", 10, 4),
    ("Option 2, $50 bet (50%, 1.8x)", 50, 2),
    ("Option 4, $1 bet (10%, 9.0x)", 1, 4),
]

var_values = {
    2: 0.81,
    4: 7.29,
}

for scenario, bet, opt in scenarios:
    var = bet**2 * var_values[opt]
    sd = np.sqrt(var)
    print(f"{scenario:<40} {var:<15.2f} ${sd:<14.2f}")

print("\n핵심 발견:")
print("  1. Option 2, $50 (Var=2,025) > Option 4, $10 (Var=729)")
print("     → 보수적 옵션 + 큰 베팅 > 위험한 옵션 + 작은 베팅")
print()
print("  2. Option 4, $10 (Var=729) > Option 2, $10 (Var=81)")
print("     → 동일 베팅 시, 낮은 확률 옵션이 더 위험")

# 5. 수학적 정리
print("\n" + "="*80)
print("수학적 정리")
print("="*80)

print("""
Variance = Bet² × Variance_Coefficient

where:
  Variance_Coefficient = f(win_prob, payout_multiplier)

Option의 Variance Coefficient:
  - Option 2 (50%, 1.8x): 8.1
  - Option 3 (25%, 3.2x): 19.2
  - Option 4 (10%, 9.0x): 72.9

최종 Variance:
  Variance = Bet² × Coefficient

결론:
  1. 베팅 금액 ↑ → Variance ↑ (제곱 비례)
  2. 승리 확률 ↓ (but 보상 ↑) → Coefficient ↑
  3. 실제 Risk = (베팅 금액)² × (확률 분포의 극단성)
""")

# 6. 정확한 표현
print("\n" + "="*80)
print("질문에 대한 정확한 답변")
print("="*80)

print("""
질문: "베팅 금액이 크고, 승리 확률이 작을수록 risky한가?"

답변: ✅ 거의 맞지만, 더 정확하게는:

  "베팅 금액이 크고, 결과 분포가 극단적일수록 risky하다"

여기서 "결과 분포가 극단적" = 승리 확률은 낮지만 승리 시 보상이 큼

예시:
  ✗ 1% 확률로 $2, 99% 확률로 $0 → Variance 작음
     (승리 확률 낮지만, 승리 보상도 작음)

  ✓ 1% 확률로 $100, 99% 확률로 $0 → Variance 큼
     (승리 확률 낮고, 승리 보상이 큼 = 극단적 분포)

우리 실험:
  - Option 2: 50% × 1.8x (중간 분포)
  - Option 4: 10% × 9.0x (극단적 분포)
  → Option 4가 9배 더 큰 variance!

핵심:
  Risk = (베팅 금액)² × (확률 분포의 극단성)

  베팅 금액 ↑ = Risk ↑ ✓
  승리 확률 ↓ (+ 보상 ↑) = Risk ↑ ✓
""")
