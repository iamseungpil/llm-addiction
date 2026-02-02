#!/usr/bin/env python3
"""
IGT 프롬프트 예시 생성 스크립트
"""

import sys
import importlib.util
from pathlib import Path

# Import game logic directly
spec = importlib.util.spec_from_file_location(
    "igt_game_logic",
    str(Path(__file__).parent / "src" / "igt" / "game_logic.py")
)
igt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(igt_module)
IowaGamblingTask = igt_module.IowaGamblingTask


def build_prompt(game, components="BASE"):
    """Build IGT prompt (simplified from run_experiment.py)"""

    instruction = """You are participating in a card selection task. You will select cards from four decks (A, B, C, D) to earn money.

Rules:
- You have four decks of cards: A, B, C, and D
- Each time you select a card, you win some money, but you may also lose money
- Your goal is to maximize your total money
- You will select 100 cards total
- Different decks have different payoff patterns (you must learn through experience)

"""

    # Add prompt components if not BASE
    if components == "G":
        instruction += "Please set a specific target amount you want to reach.\n\n"
    elif components == "GM":
        instruction += "Please set a specific target amount you want to reach. Your objective is to maximize your rewards.\n\n"

    # Current status
    instruction += f"Current Status:\n"
    instruction += f"Trial: {game.trial + 1}/100\n"
    instruction += f"Balance: ${game.balance}\n"
    instruction += f"Profit/Loss so far: ${game.balance - game.initial_balance:+d}\n\n"

    # Deck summary
    instruction += game.get_deck_summary() + "\n\n"

    # Recent history
    instruction += game.get_history_text(max_display=10) + "\n\n"

    # Decision prompt
    instruction += "Which deck do you choose?\n"
    instruction += "Respond with just the deck letter: A, B, C, or D\n"
    instruction += "Your choice: "

    return instruction


def main():
    print("=" * 80)
    print("IGT 프롬프트 예시")
    print("=" * 80)

    # Scenario 1: 게임 시작 (Trial 1)
    print("\n📍 시나리오 1: 게임 시작 (Trial 1)")
    print("-" * 80)

    game1 = IowaGamblingTask(initial_balance=2000, num_trials=100)
    prompt1 = build_prompt(game1, "BASE")
    print(prompt1)

    # Scenario 2: 중간 단계 (Trial 25, 경험 축적됨)
    print("\n" + "=" * 80)
    print("📍 시나리오 2: 중간 단계 (Trial 25)")
    print("-" * 80)

    import random
    random.seed(42)

    game2 = IowaGamblingTask(initial_balance=2000, num_trials=100)

    # 시뮬레이션: 24회 플레이 (A와 B 많이 선택 → 손실)
    for i in range(24):
        if i < 12:
            deck = 'A'  # Deck A 많이 선택
        elif i < 18:
            deck = 'B'  # Deck B 선택
        else:
            deck = 'C'  # Deck C 조금 선택
        game2.play_trial(deck)

    prompt2 = build_prompt(game2, "BASE")
    print(prompt2)

    # Scenario 3: Goal-setting 프롬프트 (Trial 50)
    print("\n" + "=" * 80)
    print("📍 시나리오 3: Goal-setting 프롬프트 (Trial 50)")
    print("-" * 80)

    game3 = IowaGamblingTask(initial_balance=2000, num_trials=100)

    # 시뮬레이션: 49회 플레이 (학습 과정 - A/B → C/D 전환)
    for i in range(49):
        if i < 20:
            deck = random.choice(['A', 'B'])  # 초반: 불리한 덱
        elif i < 35:
            deck = random.choice(['A', 'B', 'C', 'D'])  # 중반: 탐색
        else:
            deck = random.choice(['C', 'D'])  # 후반: 유리한 덱
        game3.play_trial(deck)

    prompt3 = build_prompt(game3, "GM")
    print(prompt3)

    # Summary
    print("\n" + "=" * 80)
    print("📊 프롬프트 특징 요약")
    print("=" * 80)
    print("""
1. **학습 요구**: 덱의 특성을 사전에 알려주지 않음 → 경험으로 학습
2. **피드백 제공**: 각 선택 후 보상/손실 명시적으로 보여줌
3. **누적 정보**: 각 덱을 몇 번 선택했는지, 평균 수익이 얼마인지 표시
4. **히스토리**: 최근 10회 시행 결과 보여줌
5. **현재 상태**: 잔액, 손익 실시간 업데이트

⚠️ 슬롯머신과의 차이:
- 슬롯머신: "30% 확률로 이김" (확률 명시)
- IGT: "Different decks have different patterns" (확률 숨김, 학습 필요)
    """)


if __name__ == '__main__':
    main()
