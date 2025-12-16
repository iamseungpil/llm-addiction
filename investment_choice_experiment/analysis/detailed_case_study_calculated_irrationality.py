"""
Detailed Case Study: Calculated Irrationality Paradox
분석 대상: GPT-4o-mini Fixed G/GM 조건에서 EV 계산 후 Option 4 선택 사례
"""

import json
import re

DATA_DIR = "/data/llm_addiction/investment_choice_experiment/results/"
FILE = f"{DATA_DIR}gpt4o_mini_fixed_20251119_042406.json"

def detect_ev_calculation(response):
    """Detect if response contains EV calculation"""
    response_lower = response.lower()

    # Explicit EV mentions
    explicit_ev = bool(re.search(r'expected\s+(value|return|outcome)', response_lower))

    # Mathematical operations
    math_calc = bool(re.search(r'\d+\.?\d*\s*[×*x]\s*\d+\.?\d*', response_lower))

    # Probability mentions
    prob_mention = bool(re.search(r'(probability|chance|likely|unlikely|\d+%)', response_lower))

    return {
        'explicit_ev': explicit_ev,
        'math_calc': math_calc,
        'prob_mention': prob_mention,
        'has_calculation': explicit_ev or math_calc
    }

def extract_reasoning_for_option4(response):
    """Extract reasoning for choosing Option 4"""
    response_lower = response.lower()

    reasons = []

    # Goal-related reasoning
    if re.search(r'(goal|target|\$200|double)', response_lower):
        reasons.append("GOAL_FIXATION")

    # High reward mention
    if re.search(r'(\$90|highest|maximum)', response_lower):
        reasons.append("HIGH_REWARD_FOCUS")

    # Risk-taking language
    if re.search(r'(risk|aggressive|bold)', response_lower):
        reasons.append("EXPLICIT_RISK_TAKING")

    # Despite/although (acknowledging irrationality)
    if re.search(r'(despite|although|even though)', response_lower):
        reasons.append("ACKNOWLEDGED_CONFLICT")

    return reasons

def analyze_case_study(game, decision_idx=0):
    """Analyze single case in detail"""
    decision = game['decisions'][decision_idx]
    response = decision['response']

    ev_detection = detect_ev_calculation(response)
    reasoning = extract_reasoning_for_option4(response)

    print("="*100)
    print(f"CASE STUDY: Game {game['game_id']}, Condition: {game['prompt_condition']}, Round: {decision['round']}")
    print("="*100)

    print(f"\n📊 Choice: Option {decision['choice']}")
    print(f"💰 Balance: ${decision['balance_before']} → ${decision['balance_after']}")
    print(f"🎲 Outcome: {decision['outcome']}")

    print(f"\n🧮 EV Calculation Detection:")
    print(f"  - Explicit EV mention: {'✅' if ev_detection['explicit_ev'] else '❌'}")
    print(f"  - Mathematical calculation: {'✅' if ev_detection['math_calc'] else '❌'}")
    print(f"  - Probability mention: {'✅' if ev_detection['prob_mention'] else '❌'}")
    print(f"  - Has calculation: {'✅' if ev_detection['has_calculation'] else '❌'}")

    print(f"\n🎯 Reasoning Patterns:")
    for reason in reasoning:
        print(f"  - {reason}")

    print(f"\n📝 Full Response:")
    print("-"*100)
    print(response)
    print("-"*100)

    return {
        'game_id': game['game_id'],
        'condition': game['prompt_condition'],
        'choice': decision['choice'],
        'ev_detection': ev_detection,
        'reasoning': reasoning,
        'response': response
    }

def find_paradox_cases(data, condition, min_cases=5):
    """Find cases where model calculated EV but chose Option 4"""
    games = [g for g in data['results'] if g['prompt_condition'] == condition]

    paradox_cases = []

    for game in games:
        for decision in game['decisions']:
            if decision['choice'] == 4:
                ev_detection = detect_ev_calculation(decision['response'])
                if ev_detection['has_calculation']:
                    paradox_cases.append({
                        'game': game,
                        'decision_idx': game['decisions'].index(decision)
                    })
                    break  # Only analyze first Option 4 choice per game

    return paradox_cases[:min_cases]

def compare_g_vs_m():
    """Compare G condition (100% irrational) vs M condition (6% irrational)"""
    with open(FILE) as f:
        data = json.load(f)

    print("\n" + "="*100)
    print("COMPARATIVE ANALYSIS: G vs M Conditions")
    print("="*100)

    # G condition paradox cases
    print("\n\n### PART 1: G Condition (Goal: $200) - Calculated Irrationality ###")
    g_cases = find_paradox_cases(data, 'G', min_cases=3)

    g_analyses = []
    for i, case in enumerate(g_cases, 1):
        print(f"\n--- G Condition Example {i} ---")
        analysis = analyze_case_study(case['game'], case['decision_idx'])
        g_analyses.append(analysis)

    # M condition for comparison
    print("\n\n### PART 2: M Condition (Maximize) - Rational Behavior ###")
    m_games = [g for g in data['results'] if g['prompt_condition'] == 'M']

    # Find Option 1 or 2 choices with EV calculation in M condition
    m_rational_cases = []
    for game in m_games:
        for decision in game['decisions']:
            if decision['choice'] in [1, 2]:
                ev_detection = detect_ev_calculation(decision['response'])
                if ev_detection['has_calculation']:
                    m_rational_cases.append({
                        'game': game,
                        'decision_idx': game['decisions'].index(decision)
                    })
                    break

    m_analyses = []
    for i, case in enumerate(m_rational_cases[:3], 1):
        print(f"\n--- M Condition Example {i} ---")
        analysis = analyze_case_study(case['game'], case['decision_idx'])
        m_analyses.append(analysis)

    # Summary comparison
    print("\n\n" + "="*100)
    print("SUMMARY: Reasoning Pattern Comparison")
    print("="*100)

    print(f"\n🔴 G Condition (Irrational):")
    g_goal_fixation = sum(1 for a in g_analyses if 'GOAL_FIXATION' in a['reasoning'])
    g_high_reward = sum(1 for a in g_analyses if 'HIGH_REWARD_FOCUS' in a['reasoning'])
    print(f"  - Goal fixation: {g_goal_fixation}/{len(g_analyses)} ({g_goal_fixation/len(g_analyses)*100:.1f}%)")
    print(f"  - High reward focus: {g_high_reward}/{len(g_analyses)} ({g_high_reward/len(g_analyses)*100:.1f}%)")
    print(f"  - All calculated EV: {len(g_analyses)}/{len(g_analyses)} (100%)")
    print(f"  - But chose Option 4: {len(g_analyses)}/{len(g_analyses)} (100%)")

    print(f"\n🟢 M Condition (Rational):")
    m_goal_fixation = sum(1 for a in m_analyses if 'GOAL_FIXATION' in a['reasoning'])
    m_high_reward = sum(1 for a in m_analyses if 'HIGH_REWARD_FOCUS' in a['reasoning'])
    print(f"  - Goal fixation: {m_goal_fixation}/{len(m_analyses)} ({m_goal_fixation/len(m_analyses)*100:.1f}%)")
    print(f"  - High reward focus: {m_high_reward}/{len(m_analyses)} ({m_high_reward/len(m_analyses)*100:.1f}%)")
    print(f"  - All calculated EV: {len(m_analyses)}/{len(m_analyses)} (100%)")
    print(f"  - Chose safer options: {len(m_analyses)}/{len(m_analyses)} (100%)")

def analyze_gm_worst_condition():
    """Analyze GM condition - worst combination"""
    with open(FILE) as f:
        data = json.load(f)

    print("\n\n" + "="*100)
    print("SPECIAL ANALYSIS: GM Condition (Worst Combination)")
    print("="*100)

    gm_cases = find_paradox_cases(data, 'GM', min_cases=5)

    gm_analyses = []
    for i, case in enumerate(gm_cases, 1):
        print(f"\n--- GM Condition Example {i} ---")
        analysis = analyze_case_study(case['game'], case['decision_idx'])
        gm_analyses.append(analysis)

    print("\n\n" + "="*100)
    print("GM SUMMARY: Goal + Maximize = Maximum Irrationality")
    print("="*100)

    gm_goal = sum(1 for a in gm_analyses if 'GOAL_FIXATION' in a['reasoning'])
    gm_high_reward = sum(1 for a in gm_analyses if 'HIGH_REWARD_FOCUS' in a['reasoning'])
    gm_conflict = sum(1 for a in gm_analyses if 'ACKNOWLEDGED_CONFLICT' in a['reasoning'])

    print(f"\n📊 Reasoning Patterns:")
    print(f"  - Goal fixation: {gm_goal}/{len(gm_analyses)} ({gm_goal/len(gm_analyses)*100:.1f}%)")
    print(f"  - High reward focus: {gm_high_reward}/{len(gm_analyses)} ({gm_high_reward/len(gm_analyses)*100:.1f}%)")
    print(f"  - Acknowledged conflict: {gm_conflict}/{len(gm_analyses)} ({gm_conflict/len(gm_analyses)*100:.1f}%)")
    print(f"\n🔥 Paradox: {len(gm_analyses)}/{len(gm_analyses)} calculated EV but chose Option 4 (100%)")

if __name__ == "__main__":
    print("\n" + "="*100)
    print("DETAILED CASE STUDY: Calculated Irrationality Paradox")
    print("GPT-4o-mini Fixed Betting Experiment")
    print("="*100)

    # Main analysis: G vs M comparison
    compare_g_vs_m()

    # Special analysis: GM condition
    analyze_gm_worst_condition()

    print("\n\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
