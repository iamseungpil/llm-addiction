#!/usr/bin/env python3
"""
Verify: Does our variance calculation correctly measure "자산 변동폭"?

사용자 질문: "자산에 가져올 수 있는 변동폭"을 측정하는가?
"""

import numpy as np

print("="*80)
print("자산 변동폭 계산 검증")
print("="*80)

print("\n" + "="*80)
print("현재 계산 방식 vs 실제 자산 변동폭 비교")
print("="*80)

# Option 2, $10 bet 예시
print("\n예시: Option 2 (50% chance of 1.8x), $10 베팅")
print("-"*80)

bet = 10

print("\n[방법 1] 현재 계산 (Payout의 variance):")
outcomes_payout = [18, 0]
probs = [0.5, 0.5]
ev_payout = sum(p * o for p, o in zip(probs, outcomes_payout))
var_payout = sum(p * (o - ev_payout)**2 for p, o in zip(probs, outcomes_payout))

print(f"  Outcomes (payout): {outcomes_payout}")
print(f"  EV(payout): ${ev_payout}")
print(f"  Var(payout): {var_payout}")
print(f"  SD(payout): ${np.sqrt(var_payout):.2f}")

print("\n[방법 2] 자산 변동폭 (Balance change = payout - bet):")
balance_changes = [payout - bet for payout in outcomes_payout]
ev_change = sum(p * c for p, c in zip(probs, balance_changes))
var_change = sum(p * (c - ev_change)**2 for p, c in zip(probs, balance_changes))

print(f"  Outcomes (balance change): {balance_changes}")
print(f"  EV(balance change): ${ev_change}")
print(f"  Var(balance change): {var_change}")
print(f"  SD(balance change): ${np.sqrt(var_change):.2f}")

print(f"\n✓ 결과 비교: Var(payout) = {var_payout}, Var(balance change) = {var_change}")
print(f"  → {'동일함!' if var_payout == var_change else '다름!'}")

# 수학적 설명
print("\n" + "="*80)
print("수학적 증명")
print("="*80)
print("""
Balance change = Payout - Bet (상수)

수학 정리:
  Var(X - c) = Var(X)  (상수 c를 빼도 분산은 불변)

따라서:
  Var(Balance change) = Var(Payout - Bet) = Var(Payout)

결론: 현재 계산이 맞음! ✓
""")

# 모든 옵션에 대해 검증
print("\n" + "="*80)
print("전체 옵션 검증 ($10 베팅 기준)")
print("="*80)

options = {
    1: {'outcomes': [10], 'probs': [1.0]},
    2: {'outcomes': [18, 0], 'probs': [0.5, 0.5]},
    3: {'outcomes': [32, 0], 'probs': [0.25, 0.75]},
    4: {'outcomes': [90, 0], 'probs': [0.1, 0.9]},
}

bet = 10

print(f"\n{'Option':<10} {'Var(Payout)':<15} {'Var(Balance Δ)':<15} {'Match?':<10}")
print("-"*60)

for opt_num, data in options.items():
    outcomes_payout = data['outcomes']
    probs = data['probs']

    # Payout variance
    ev_payout = sum(p * o for p, o in zip(probs, outcomes_payout))
    var_payout = sum(p * (o - ev_payout)**2 for p, o in zip(probs, outcomes_payout))

    # Balance change variance
    balance_changes = [o - bet for o in outcomes_payout]
    ev_change = sum(p * c for p, c in zip(probs, balance_changes))
    var_change = sum(p * (c - ev_change)**2 for p, c in zip(probs, balance_changes))

    match = "✓" if abs(var_payout - var_change) < 0.01 else "✗"

    print(f"Option {opt_num}  {var_payout:<15.2f} {var_change:<15.2f} {match:<10}")

# 실제 의미 설명
print("\n" + "="*80)
print("'자산 변동폭'의 실제 의미")
print("="*80)

print("\nOption 2, $10 베팅:")
print("  승리 (50%): 자산 +$8")
print("  패배 (50%): 자산 -$10")
print("  평균 변동: -$1 (손실)")
print("  변동 범위: -$10 ~ +$8")
print(f"  Variance: {var_change} = 변동폭의 제곱 평균")
print(f"  SD: {np.sqrt(var_change):.2f} = 대표적인 변동 크기")

print("\n해석:")
print("  → Variance가 크다 = 자산 변동폭이 크다")
print("  → 결과가 불확실하다 = 위험하다")

# 베팅 금액 영향
print("\n" + "="*80)
print("베팅 금액이 자산 변동폭에 미치는 영향")
print("="*80)

print("\nOption 2 (50% chance of 1.8x):")
print(f"\n{'Bet Amount':<15} {'Balance Change':<25} {'Variance':<15} {'SD':<10}")
print("-"*70)

for bet in [1, 10, 50, 100]:
    outcomes = [bet * 1.8, 0]
    changes = [o - bet for o in outcomes]
    probs = [0.5, 0.5]

    ev_change = sum(p * c for p, c in zip(probs, changes))
    var_change = sum(p * (c - ev_change)**2 for p, c in zip(probs, changes))
    sd_change = np.sqrt(var_change)

    change_str = f"[{changes[0]:+.0f}, {changes[1]:+.0f}]"

    print(f"${bet:<14} {change_str:<25} {var_change:<15.2f} ${sd_change:<9.2f}")

print("\n관찰:")
print("  → 베팅 금액이 2배 증가 → Variance는 4배 증가 (Var ∝ Bet²)")
print("  → 베팅 금액이 클수록 자산 변동폭이 기하급수적으로 증가")

# 최종 확인
print("\n" + "="*80)
print("최종 결론")
print("="*80)

print("""
질문: "자산에 가져올 수 있는 변동폭이 크면 클수록 위험하다"고 정의했을 때,
      현재 계산이 맞는가?

답변: ✅ 맞습니다!

근거:
1. 우리가 계산한 Variance = Var(Payout)
2. Var(Payout) = Var(Balance Change) (수학적으로 동일)
3. Balance Change = 자산 변동폭
4. Variance = 변동폭의 분산 (변동 크기의 제곱 평균)

해석:
- Variance가 크다 = 자산 변동폭이 크다 = 결과 불확실성 높음 = 위험함
- SD(표준편차) = 대표적인 변동 크기 (해석이 더 직관적)

예시:
  Option 2, $10: Var=81, SD=$9
    → 평균적으로 약 ±$9 정도 변동

  Option 4, $10: Var=729, SD=$27
    → 평균적으로 약 ±$27 정도 변동
    → Option 2보다 3배 큰 변동폭!

Variable Betting:
  Variance ∝ Bet²
  → 베팅 금액이 2배면 변동폭은 4배!
""")
