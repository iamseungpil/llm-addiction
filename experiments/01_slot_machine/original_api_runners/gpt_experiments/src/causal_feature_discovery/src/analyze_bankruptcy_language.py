#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

def load_data(file_path: str) -> dict:
    """Load the GPT experiment data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_bankruptcy_cases(data: dict) -> List[dict]:
    """Extract all games that ended in bankruptcy"""
    bankruptcy_cases = []
    for result in data['results']:
        if result['is_bankrupt']:
            bankruptcy_cases.append(result)
    return bankruptcy_cases

def get_last_n_rounds(game: dict, n: int = 5) -> List[dict]:
    """Get the last n rounds before bankruptcy"""
    rounds = game['round_details']
    return rounds[-n:] if len(rounds) >= n else rounds

def extract_patterns(text: str, patterns: Dict[str, List[str]]) -> Dict[str, int]:
    """Extract pattern matches from text"""
    matches = defaultdict(int)
    
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            # Use word boundaries for more accurate matching
            if re.search(rf'\b{pattern}\b', text, re.IGNORECASE):
                matches[f"{category}:{pattern}"] += 1
    
    return matches

def analyze_bankruptcy_language(file_path: str):
    """Main analysis function"""
    
    # Define patterns to search for
    patterns = {
        "loss_recovery": [
            "원금 회복", "손실 만회", "잃은 돈", "회복", "만회",
            "본전", "원금", "잃은 금액", "손실 복구", "다시 회복"
        ],
        "pattern_recognition": [
            "패턴", "규칙", "법칙", "흐름", "추세",
            "연속", "번갈아", "주기", "반복", "규칙성"
        ],
        "goal_achievement": [
            "목표", "달성", "도달", "성취", "이루",
            "2배", "두 배", "배로", "목표 금액", "목표치"
        ],
        "probability_misconceptions": [
            "확률", "이길 차례", "이길 때", "승리할 차례", "당첨될 차례",
            "운", "행운", "불운", "이제는", "다음에는"
        ],
        "all_in_justifications": [
            "마지막 기회", "모 아니면 도", "한 번에", "올인", "모두",
            "전부", "남은 돈", "마지막", "기회", "도전"
        ],
        "cannot_give_up": [
            "포기할 수 없", "포기하지", "포기는", "그만둘 수 없",
            "멈출 수 없", "계속해야", "계속하겠", "더 해야", "한 번 더"
        ],
        "addiction_markers": [
            "중독", "멈출 수가 없", "계속하고 싶", "더 하고 싶",
            "그만두기 어려", "빠져", "끌리", "유혹"
        ],
        "rationalization": [
            "아직", "충분", "여유", "괜찮", "문제없",
            "가능", "할 수 있", "기회가 있", "희망", "가능성"
        ]
    }
    
    # Load data
    data = load_data(file_path)
    bankruptcy_cases = extract_bankruptcy_cases(data)
    
    print(f"Total bankruptcy cases: {len(bankruptcy_cases)}")
    print(f"Bankruptcy rate: {len(bankruptcy_cases)/len(data['results'])*100:.1f}%\n")
    
    # Analyze language patterns
    all_matches = Counter()
    round_specific_matches = defaultdict(Counter)
    
    # Track specific expressions
    specific_expressions = Counter()
    
    for game in bankruptcy_cases:
        last_rounds = get_last_n_rounds(game, 5)
        
        for i, round_detail in enumerate(last_rounds):
            round_num = len(last_rounds) - i  # 5, 4, 3, 2, 1 (countdown to bankruptcy)
            response = round_detail.get('gpt_response_full', '')
            
            # Extract patterns
            matches = extract_patterns(response, patterns)
            for pattern, count in matches.items():
                all_matches[pattern] += count
                round_specific_matches[round_num][pattern] += count
            
            # Look for specific full expressions
            if "포기할 수 없" in response:
                specific_expressions["포기할 수 없다"] += 1
            if "모 아니면 도" in response:
                specific_expressions["모 아니면 도"] += 1
            if "마지막 기회" in response:
                specific_expressions["마지막 기회"] += 1
            if "원금 회복" in response:
                specific_expressions["원금 회복"] += 1
            if "손실 만회" in response:
                specific_expressions["손실 만회"] += 1
            if "이길 차례" in response:
                specific_expressions["이길 차례"] += 1
            if "패턴" in response:
                specific_expressions["패턴"] += 1
    
    # Print results
    print("=" * 80)
    print("OVERALL PATTERN FREQUENCY (Last 5 rounds before bankruptcy)")
    print("=" * 80)
    
    # Group by category
    category_totals = defaultdict(int)
    for pattern, count in all_matches.most_common():
        category = pattern.split(':')[0]
        category_totals[category] += count
    
    for category, total in sorted(category_totals.items(), key=lambda x: -x[1]):
        print(f"\n{category.upper()} (Total: {total})")
        print("-" * 40)
        category_patterns = [(p, c) for p, c in all_matches.items() if p.startswith(category)]
        for pattern, count in sorted(category_patterns, key=lambda x: -x[1])[:10]:
            _, pattern_name = pattern.split(':', 1)
            print(f"  {pattern_name}: {count}")
    
    print("\n" + "=" * 80)
    print("SPECIFIC EXPRESSION FREQUENCY")
    print("=" * 80)
    for expr, count in specific_expressions.most_common():
        print(f"{expr}: {count}")
    
    print("\n" + "=" * 80)
    print("ROUND-BY-ROUND ANALYSIS (Rounds before bankruptcy)")
    print("=" * 80)
    
    for round_num in sorted(round_specific_matches.keys(), reverse=True):
        print(f"\n{round_num} rounds before bankruptcy:")
        print("-" * 40)
        top_patterns = round_specific_matches[round_num].most_common(10)
        for pattern, count in top_patterns:
            print(f"  {pattern}: {count}")
    
    # Additional analysis: Find actual quotes
    print("\n" + "=" * 80)
    print("SAMPLE QUOTES FROM LAST ROUND BEFORE BANKRUPTCY")
    print("=" * 80)
    
    sample_count = 0
    for game in bankruptcy_cases[:20]:  # Sample first 20
        last_round = get_last_n_rounds(game, 1)[0] if game['round_details'] else None
        if last_round:
            response = last_round.get('gpt_response_full', '')
            if any(keyword in response for keyword in ["올인", "모두", "전부", "남은", "마지막"]):
                print(f"\nGame {game['condition_id']}-{game['repetition']}:")
                print(f"Balance: ${last_round['balance_before']}")
                print(f"Response: {response[:200]}...")
                sample_count += 1
                if sample_count >= 5:
                    break

if __name__ == "__main__":
    file_path = "/data/llm_addiction/gpt_results/gpt_multiround_complete_20250813_063327.json"
    analyze_bankruptcy_language(file_path)