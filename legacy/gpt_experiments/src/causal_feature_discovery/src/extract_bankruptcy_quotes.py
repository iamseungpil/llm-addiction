#!/usr/bin/env python3
import json
import random
from typing import List, Dict

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

def get_last_round_quotes(file_path: str):
    """Extract and display quotes from the last round before bankruptcy"""
    
    data = load_data(file_path)
    bankruptcy_cases = extract_bankruptcy_cases(data)
    
    print(f"Total bankruptcy cases: {len(bankruptcy_cases)}")
    print("=" * 80)
    print("SAMPLE QUOTES FROM LAST ROUND BEFORE BANKRUPTCY")
    print("=" * 80)
    
    # Group by prompt combo
    by_prompt = {}
    for game in bankruptcy_cases:
        prompt_combo = game['prompt_combo']
        if prompt_combo not in by_prompt:
            by_prompt[prompt_combo] = []
        by_prompt[prompt_combo].append(game)
    
    # Sample from each prompt combo
    for prompt_combo in sorted(by_prompt.keys()):
        games = by_prompt[prompt_combo]
        print(f"\n\nPROMPT COMBO: {prompt_combo} ({len(games)} bankruptcies)")
        print("-" * 80)
        
        # Sample up to 3 games
        sample_games = random.sample(games, min(3, len(games)))
        
        for game in sample_games:
            if game['round_details']:
                last_round = game['round_details'][-1]
                balance = last_round['balance_before']
                response = last_round['gpt_response_full']
                
                print(f"\nGame {game['condition_id']}-{game['repetition']} (Balance: ${balance})")
                print(f"Bet type: {game['bet_type']}")
                print(f"Response:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
                # Highlight key phrases if present
                key_phrases = ["마지막", "목표", "패턴", "회복", "위험을 감수", "할 수 있", 
                              "기회", "포기", "계속", "모두", "전부", "올인"]
                found_phrases = [phrase for phrase in key_phrases if phrase in response]
                if found_phrases:
                    print(f"Key phrases found: {', '.join(found_phrases)}")

def extract_specific_pattern_examples(file_path: str):
    """Extract examples of specific language patterns"""
    
    data = load_data(file_path)
    bankruptcy_cases = extract_bankruptcy_cases(data)
    
    print("\n\n" + "=" * 80)
    print("SPECIFIC PATTERN EXAMPLES")
    print("=" * 80)
    
    patterns = {
        "Loss Recovery": ["손실을 만회", "회복하", "만회하", "잃은 금액"],
        "Pattern Recognition": ["패턴", "연속", "번갈아"],
        "Risk Taking": ["위험을 감수", "리스크를 감수", "도전"],
        "Last Chance": ["마지막", "기회"],
        "All-in": ["모두", "전부", "올인", "남은"]
    }
    
    for category, keywords in patterns.items():
        print(f"\n\n{category.upper()}")
        print("-" * 80)
        found_count = 0
        
        for game in bankruptcy_cases:
            if found_count >= 3:  # Limit examples per category
                break
                
            for round_detail in game['round_details'][-3:]:  # Check last 3 rounds
                response = round_detail['gpt_response_full']
                
                # Check if any keyword is in response
                matching_keywords = [kw for kw in keywords if kw in response]
                if matching_keywords:
                    print(f"\nGame {game['condition_id']}-{game['repetition']}")
                    print(f"Balance: ${round_detail['balance_before']}")
                    print(f"Matching keywords: {', '.join(matching_keywords)}")
                    print(f"Response excerpt:")
                    
                    # Extract relevant sentence
                    sentences = response.split('.')
                    for sent in sentences:
                        if any(kw in sent for kw in matching_keywords):
                            print(f"  -> {sent.strip()}.")
                    
                    found_count += 1
                    break

if __name__ == "__main__":
    file_path = "/data/llm_addiction/gpt_results/gpt_multiround_complete_20250813_063327.json"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    get_last_round_quotes(file_path)
    extract_specific_pattern_examples(file_path)