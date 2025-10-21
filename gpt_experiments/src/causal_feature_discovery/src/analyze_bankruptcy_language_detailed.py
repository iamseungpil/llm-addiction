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

def analyze_detailed_expressions(file_path: str):
    """Detailed analysis of language patterns"""
    
    # Load data
    data = load_data(file_path)
    bankruptcy_cases = extract_bankruptcy_cases(data)
    
    print(f"Total bankruptcy cases: {len(bankruptcy_cases)}")
    print(f"Total games: {len(data['results'])}")
    print(f"Bankruptcy rate: {len(bankruptcy_cases)/len(data['results'])*100:.1f}%\n")
    
    # Detailed expression tracking
    exact_expressions = {
        # Loss recovery expressions
        "원금 회복": 0,
        "손실 만회": 0,
        "본전": 0,
        "잃은 돈": 0,
        "잃은 금액": 0,
        "손실을 만회": 0,
        "손실을 회복": 0,
        "회복하": 0,  # 회복하다, 회복하겠다, etc.
        "만회하": 0,  # 만회하다, 만회하겠다, etc.
        
        # Pattern recognition
        "패턴": 0,
        "규칙": 0,
        "흐름": 0,
        "연속": 0,
        "번갈아": 0,
        "연패": 0,
        "연승": 0,
        
        # Goal-related
        "목표": 0,
        "2배": 0,
        "두 배": 0,
        "200달러": 0,
        "$200": 0,
        "목표 금액": 0,
        "초기 자금을 2배": 0,
        
        # Probability misconceptions
        "이길 차례": 0,
        "이길 때": 0,
        "승리할 차례": 0,
        "당첨될 차례": 0,
        "이제는": 0,
        "이번에는": 0,
        "다음에는": 0,
        "확률": 0,
        "30%": 0,
        
        # All-in related
        "마지막 기회": 0,
        "마지막 한 번": 0,
        "마지막으로": 0,
        "모 아니면 도": 0,
        "올인": 0,
        "모두 베팅": 0,
        "전부 베팅": 0,
        "남은 돈 전부": 0,
        "남은 금액 전부": 0,
        
        # Cannot give up
        "포기할 수 없": 0,
        "포기하지": 0,
        "포기는": 0,
        "그만둘 수 없": 0,
        "멈출 수 없": 0,
        "계속해야": 0,
        "계속하겠": 0,
        "한 번 더": 0,
        "한 번만 더": 0,
        
        # Risk justification
        "위험하지만": 0,
        "위험을 감수": 0,
        "리스크를 감수": 0,
        "도전": 0,
        "시도": 0,
        "기회": 0,
        
        # Balance rationalization
        "아직": 0,
        "여전히": 0,
        "충분": 0,
        "여유": 0,
        "가능": 0,
        "할 수 있": 0,
    }
    
    # Track by round position
    round_expressions = defaultdict(lambda: defaultdict(int))
    
    # Analyze each bankruptcy case
    for game in bankruptcy_cases:
        last_rounds = get_last_n_rounds(game, 5)
        
        for i, round_detail in enumerate(last_rounds):
            round_num = len(last_rounds) - i  # 5, 4, 3, 2, 1 (countdown to bankruptcy)
            response = round_detail.get('gpt_response_full', '')
            
            # Count exact expressions
            for expr in exact_expressions:
                if expr in response:
                    exact_expressions[expr] += 1
                    round_expressions[round_num][expr] += 1
    
    # Print results
    print("=" * 80)
    print("EXACT EXPRESSION COUNTS (All occurrences in last 5 rounds)")
    print("=" * 80)
    
    # Group expressions by category
    categories = {
        "Loss Recovery": ["원금 회복", "손실 만회", "본전", "잃은 돈", "잃은 금액", 
                         "손실을 만회", "손실을 회복", "회복하", "만회하"],
        "Pattern Recognition": ["패턴", "규칙", "흐름", "연속", "번갈아", "연패", "연승"],
        "Goal Achievement": ["목표", "2배", "두 배", "200달러", "$200", "목표 금액", "초기 자금을 2배"],
        "Probability": ["이길 차례", "이길 때", "승리할 차례", "당첨될 차례", "이제는", 
                       "이번에는", "다음에는", "확률", "30%"],
        "All-in": ["마지막 기회", "마지막 한 번", "마지막으로", "모 아니면 도", "올인", 
                   "모두 베팅", "전부 베팅", "남은 돈 전부", "남은 금액 전부"],
        "Cannot Give Up": ["포기할 수 없", "포기하지", "포기는", "그만둘 수 없", "멈출 수 없", 
                          "계속해야", "계속하겠", "한 번 더", "한 번만 더"],
        "Risk Justification": ["위험하지만", "위험을 감수", "리스크를 감수", "도전", "시도", "기회"],
        "Rationalization": ["아직", "여전히", "충분", "여유", "가능", "할 수 있"]
    }
    
    for category, expressions in categories.items():
        category_total = sum(exact_expressions[expr] for expr in expressions if expr in exact_expressions)
        if category_total > 0:
            print(f"\n{category} (Total: {category_total}):")
            print("-" * 50)
            for expr in expressions:
                if exact_expressions[expr] > 0:
                    print(f"  '{expr}': {exact_expressions[expr]}")
    
    # Show expressions by round
    print("\n" + "=" * 80)
    print("TOP EXPRESSIONS BY ROUND (Before bankruptcy)")
    print("=" * 80)
    
    for round_num in sorted(round_expressions.keys(), reverse=True):
        print(f"\n{round_num} rounds before bankruptcy:")
        print("-" * 50)
        top_expr = sorted(round_expressions[round_num].items(), key=lambda x: -x[1])[:10]
        for expr, count in top_expr:
            if count > 0:
                print(f"  '{expr}': {count}")
    
    # Look for specific full sentences
    print("\n" + "=" * 80)
    print("SEARCHING FOR NOTABLE FULL EXPRESSIONS")
    print("=" * 80)
    
    notable_patterns = [
        (r"포기할 수 없\w*", "포기할 수 없다"),
        (r"모 아니면 도", "모 아니면 도"),
        (r"마지막 기회\w*", "마지막 기회"),
        (r"원금\s*회복", "원금 회복"),
        (r"손실\s*만회", "손실 만회"),
        (r"이길\s*차례", "이길 차례"),
        (r"한\s*번\s*더", "한 번 더"),
        (r"계속\s*해야", "계속해야"),
        (r"그만둘\s*수\s*없", "그만둘 수 없다"),
        (r"멈출\s*수\s*없", "멈출 수 없다"),
    ]
    
    pattern_counts = Counter()
    examples = defaultdict(list)
    
    for game in bankruptcy_cases:
        last_rounds = get_last_n_rounds(game, 5)
        for round_detail in last_rounds:
            response = round_detail.get('gpt_response_full', '')
            for pattern, name in notable_patterns:
                if re.search(pattern, response):
                    pattern_counts[name] += 1
                    if len(examples[name]) < 2:  # Keep 2 examples
                        examples[name].append({
                            'game': f"{game['condition_id']}-{game['repetition']}",
                            'balance': round_detail['balance_before'],
                            'context': response[:150] + "..."
                        })
    
    print("\nNotable expression occurrences:")
    for expr, count in pattern_counts.most_common():
        print(f"\n'{expr}': {count} occurrences")
        if expr in examples:
            for ex in examples[expr]:
                print(f"  Example: Game {ex['game']}, Balance ${ex['balance']}")
                print(f"  Context: {ex['context']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total bankruptcy cases analyzed: {len(bankruptcy_cases)}")
    print(f"Most frequent expressions:")
    
    top_expressions = sorted(exact_expressions.items(), key=lambda x: -x[1])[:15]
    for expr, count in top_expressions:
        if count > 0:
            avg_per_game = count / len(bankruptcy_cases)
            print(f"  '{expr}': {count} times ({avg_per_game:.2f} per bankruptcy)")

if __name__ == "__main__":
    file_path = "/data/llm_addiction/gpt_results/gpt_multiround_complete_20250813_063327.json"
    analyze_detailed_expressions(file_path)