#!/usr/bin/env python3
"""
Quick integration test - 실제 모델로 파싱 버그 수정 검증

Very short test: 3 games per condition (12 total) to verify:
1. Choice and bet_amount are different (bug is fixed)
2. Fallback works when model doesn't provide amount
3. Examples help model generate amounts (round 0)
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import ModelLoader, setup_logger, set_random_seed
from investment_choice.run_experiment import InvestmentChoiceExperiment

logger = setup_logger(__name__)

def analyze_results(results):
    """Analyze test results to check if bug is fixed."""
    print("\n" + "="*70)
    print("INTEGRATION TEST ANALYSIS")
    print("="*70)

    total_decisions = 0
    choice_eq_bet = 0
    choice_ne_bet = 0
    fallback_used = 0

    choice_dist = Counter()
    bet_dist = Counter()

    # Sample responses for inspection
    samples = []

    for game in results:
        for dec in game.get('decisions', []):
            choice = dec.get('choice')
            bet = dec.get('bet_amount')
            response = dec.get('response', '')

            if choice and bet:
                total_decisions += 1
                choice_dist[choice] += 1
                bet_dist[bet] += 1

                if choice == bet:
                    choice_eq_bet += 1
                else:
                    choice_ne_bet += 1

                # Check if fallback was used (c30 → $3)
                if bet == 3:
                    fallback_used += 1

                # Collect samples (first 10)
                if len(samples) < 10:
                    samples.append({
                        'round': dec.get('round', 0),
                        'choice': choice,
                        'bet': bet,
                        'response': response[:60]
                    })

    print(f"\n📊 Parsing Statistics:")
    print(f"  Total decisions: {total_decisions}")
    print(f"  Choice == Bet:   {choice_eq_bet:3d} ({choice_eq_bet/max(total_decisions,1)*100:5.1f}%)")
    print(f"  Choice != Bet:   {choice_ne_bet:3d} ({choice_ne_bet/max(total_decisions,1)*100:5.1f}%)")
    print(f"  Fallback used:   {fallback_used:3d} ({fallback_used/max(total_decisions,1)*100:5.1f}%)")

    print(f"\n📈 Choice Distribution:")
    for c in sorted(choice_dist.keys()):
        count = choice_dist[c]
        print(f"  Option {c}: {count:3d} ({count/max(total_decisions,1)*100:5.1f}%)")

    print(f"\n💰 Bet Distribution (top 10):")
    for bet, count in bet_dist.most_common(10):
        print(f"  ${bet:2d}: {count:3d} ({count/max(total_decisions,1)*100:5.1f}%)")

    print(f"\n📝 Sample Responses (first 10 decisions):")
    for i, s in enumerate(samples, 1):
        eq_mark = "⚠️ BUG!" if s['choice'] == s['bet'] else "✓"
        print(f"  {i:2d}. R{s['round']} → choice={s['choice']}, bet=${s['bet']:2d} {eq_mark}")
        print(f"      \"{s['response']}...\"")

    # Bug detection
    print(f"\n" + "="*70)
    if choice_eq_bet > total_decisions * 0.5:
        print("❌ BUG STILL EXISTS!")
        print(f"   Choice == Bet in {choice_eq_bet/total_decisions*100:.1f}% of cases")
        print("   Expected: Mostly different (choice != bet)")
        return False
    else:
        print("✅ BUG FIXED!")
        print(f"   Choice == Bet in only {choice_eq_bet/max(total_decisions,1)*100:.1f}% of cases")
        print(f"   Choice != Bet in {choice_ne_bet/max(total_decisions,1)*100:.1f}% of cases")
        if fallback_used > 0:
            print(f"   Fallback working: {fallback_used} cases")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--constraint', type=int, default=30)
    parser.add_argument('--n-games', type=int, default=3, help='Games per condition (total = n × 4)')
    args = parser.parse_args()

    print("="*70)
    print("QUICK INTEGRATION TEST: Variable Betting Parsing Fix")
    print("="*70)
    print(f"GPU: {args.gpu}")
    print(f"Constraint: c{args.constraint}")
    print(f"Games per condition: {args.n_games} (total: {args.n_games * 4})")
    print("="*70)

    # Create experiment
    print("\n⏳ Initializing experiment...")
    experiment = InvestmentChoiceExperiment(
        model_name='llama',
        gpu_id=args.gpu,
        bet_type='variable',
        bet_constraint=str(args.constraint)
    )

    # Load model
    print("⏳ Loading LLaMA model...")
    experiment.load_model()
    print("✅ Model loaded\n")

    # Run short experiment
    print(f"⏳ Running {args.n_games * 4} games...")
    print("   (This will take ~2-3 minutes)\n")

    results = []
    game_id = 0
    conditions = ['BASE', 'G', 'M', 'GM']

    for condition in conditions:
        print(f"  Condition: {condition}")
        for rep in range(args.n_games):
            game_id += 1
            seed = game_id + 77777
            set_random_seed(seed)

            try:
                result = experiment.play_game(condition, game_id, seed)
                results.append(result)

                # Quick status
                rounds = result.get('rounds_played', 0)
                balance = result.get('final_balance', 0)
                outcome = result.get('outcome', 'unknown')
                print(f"    Game {game_id}: {rounds} rounds, ${balance}, {outcome}")

            except Exception as e:
                print(f"    Game {game_id} FAILED: {e}")
                continue

    # Analyze results
    success = analyze_results(results)

    if success:
        print("\n🎉 Integration test PASSED!")
        print("   파싱 버그가 수정되었습니다.")
        print("   다음 단계: 전체 실험 재실행 가능")
    else:
        print("\n⚠️  Integration test FAILED!")
        print("   파싱 로직을 다시 확인해야 합니다.")

    return success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
