#!/usr/bin/env python3
"""
GPT-4o-mini 실험 결과 엄밀한 재분석
첫 게임 후 상태를 기준으로 정확한 통계 계산
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Any

class GPTRigorousAnalysis:
    def __init__(self, data_path: str):
        """Initialize with GPT experiment data"""
        self.data_path = Path(data_path)
        self.data = self.load_data()
        self.results = []
        
        print(f"📊 Loaded {len(self.data['results'])} experiments")
        print(f"📊 Experiment config: {self.data['experiment_config']}")
        
    def load_data(self) -> Dict:
        """Load GPT experiment results"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_single_experiment(self, exp: Dict) -> Dict:
        """Analyze a single experiment with rigorous parsing"""
        
        # Basic info
        condition_id = exp['condition_id']
        repetition = exp['repetition']
        bet_type = exp['bet_type']
        first_result = exp['first_result']
        prompt_combo = exp['prompt_combo']
        
        # Game state after first game (our baseline)
        first_game_end_balance = 130 if first_result == 'W' else 90
        
        # Total rounds played AFTER first game
        additional_rounds = exp['total_rounds']
        
        # Final state
        final_balance = exp['final_balance']
        is_bankrupt = exp['is_bankrupt']
        voluntary_stop = exp['voluntary_stop']
        
        # Calculate net profit/loss from first-game-end baseline
        net_profit_from_baseline = final_balance - first_game_end_balance
        
        # Detailed round analysis
        round_details = exp.get('round_details', [])
        parsing_log = exp.get('parsing_log', [])
        game_history = exp.get('game_history', [])
        
        # Extract betting pattern
        betting_amounts = []
        decisions = []
        parsing_info = []
        
        for i, round_detail in enumerate(round_details):
            decisions.append(round_detail['decision'])
            bet_amount = round_detail.get('bet_amount')
            if bet_amount:
                betting_amounts.append(bet_amount)
            
            # Store parsing information for validation
            parsing_info.append({
                'round': round_detail['round'],
                'parsing_info': round_detail['parsing_info'],
                'response_length': len(round_detail.get('gpt_response_full', '')),
                'bet_amount': bet_amount,
                'decision': round_detail['decision']
            })
        
        # Calculate statistics
        avg_bet = np.mean(betting_amounts) if betting_amounts else 0
        total_bet = sum(betting_amounts)
        
        # Immediate stop indicator (stopped right after first game)
        immediate_stop = (additional_rounds == 0)
        
        # Validate data consistency
        validation_errors = self.validate_experiment_data(exp, parsing_info)
        
        return {
            'experiment_id': exp.get('experiment_id', f"{condition_id}_{repetition}"),
            'condition_id': condition_id,
            'repetition': repetition,
            'bet_type': bet_type,
            'first_result': first_result,
            'prompt_combo': prompt_combo,
            
            # Baseline calculations (first game end = baseline)
            'first_game_end_balance': first_game_end_balance,
            'final_balance': final_balance,
            'net_profit_from_baseline': net_profit_from_baseline,
            
            # Round calculations
            'additional_rounds': additional_rounds,  # Rounds after first game
            'total_rounds_including_first': additional_rounds + 1,  # Including first game
            'immediate_stop': immediate_stop,
            
            # Betting analysis
            'betting_amounts': betting_amounts,
            'avg_bet_amount': avg_bet,
            'total_bet_after_first': total_bet,
            
            # Outcomes
            'is_bankrupt': is_bankrupt,
            'voluntary_stop': voluntary_stop,
            
            # Parsing validation
            'parsing_info': parsing_info,
            'validation_errors': validation_errors,
            
            # Raw data for debugging
            'raw_round_details': round_details[:2],  # First 2 rounds for inspection
        }
    
    def validate_experiment_data(self, exp: Dict, parsing_info: List[Dict]) -> List[str]:
        """Validate experiment data consistency"""
        errors = []
        
        # Check if total_rounds matches round_details length
        expected_rounds = len(exp.get('round_details', []))
        actual_rounds = exp['total_rounds']
        if expected_rounds != actual_rounds:
            errors.append(f"Round count mismatch: details={expected_rounds}, total={actual_rounds}")
        
        # Check balance progression
        first_result = exp['first_result']
        expected_start = 130 if first_result == 'W' else 90
        
        if exp.get('round_details'):
            first_round_balance_before = exp['round_details'][0].get('balance_before')
            if first_round_balance_before != expected_start:
                errors.append(f"Balance mismatch: expected={expected_start}, actual={first_round_balance_before}")
        
        # Check parsing quality
        for info in parsing_info:
            if info['decision'] not in ['continue', 'stop']:
                errors.append(f"Invalid decision: {info['decision']}")
            
            if info['decision'] == 'continue' and not info['bet_amount']:
                errors.append(f"Continue decision without bet amount in round {info['round']}")
        
        return errors
    
    def analyze_all_experiments(self) -> pd.DataFrame:
        """Analyze all experiments and return DataFrame"""
        print("🔍 Analyzing all experiments...")
        
        analyzed_results = []
        validation_errors_count = 0
        
        for i, exp in enumerate(self.data['results']):
            if i % 100 == 0:
                print(f"   Processing experiment {i+1}/{len(self.data['results'])}")
            
            result = self.analyze_single_experiment(exp)
            analyzed_results.append(result)
            
            if result['validation_errors']:
                validation_errors_count += 1
                if validation_errors_count <= 5:  # Show first 5 errors
                    print(f"⚠️  Validation errors in exp {result['experiment_id']}: {result['validation_errors']}")
        
        print(f"✅ Analysis complete. Found {validation_errors_count} experiments with validation issues.")
        
        # Convert to DataFrame
        df = pd.DataFrame(analyzed_results)
        
        # Add derived columns
        df['prompt_complexity'] = df['prompt_combo'].apply(self.get_prompt_complexity)
        df['has_goal'] = df['prompt_combo'].str.contains('G')
        df['has_maximize'] = df['prompt_combo'].str.contains('M')
        df['has_probability'] = df['prompt_combo'].str.contains('P')
        df['has_reward_info'] = df['prompt_combo'].str.contains('W')
        df['has_pattern'] = df['prompt_combo'].str.contains('R')
        
        return df
    
    def get_prompt_complexity(self, prompt_combo: str) -> int:
        """Get prompt complexity (number of components)"""
        if prompt_combo == 'BASE':
            return 0
        return len(prompt_combo)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics"""
        
        stats = {}
        
        # Overall statistics
        stats['total_experiments'] = len(df)
        stats['bankruptcy_rate'] = df['is_bankrupt'].mean()
        stats['immediate_stop_rate'] = df['immediate_stop'].mean()
        stats['avg_additional_rounds'] = df['additional_rounds'].mean()
        stats['avg_total_rounds'] = df['total_rounds_including_first'].mean()
        
        # By bet type
        bet_type_stats = {}
        for bet_type in ['fixed', 'variable']:
            mask = df['bet_type'] == bet_type
            bet_data = df[mask]
            
            bet_type_stats[bet_type] = {
                'n': len(bet_data),
                'bankruptcy_rate': bet_data['is_bankrupt'].mean(),
                'immediate_stop_rate': bet_data['immediate_stop'].mean(),
                'avg_additional_rounds': bet_data['additional_rounds'].mean(),
                'avg_total_rounds': bet_data['total_rounds_including_first'].mean(),
                'avg_net_profit': bet_data['net_profit_from_baseline'].mean(),
                'avg_bet_amount': bet_data['avg_bet_amount'].mean(),
            }
        
        stats['by_bet_type'] = bet_type_stats
        
        # By first result
        first_result_stats = {}
        for first_result in ['W', 'L']:
            mask = df['first_result'] == first_result
            first_data = df[mask]
            
            first_result_stats[first_result] = {
                'n': len(first_data),
                'bankruptcy_rate': first_data['is_bankrupt'].mean(),
                'immediate_stop_rate': first_data['immediate_stop'].mean(),
                'avg_additional_rounds': first_data['additional_rounds'].mean(),
                'avg_total_rounds': first_data['total_rounds_including_first'].mean(),
                'avg_net_profit': first_data['net_profit_from_baseline'].mean(),
                'avg_bet_amount': first_data['avg_bet_amount'].mean(),
            }
        
        stats['by_first_result'] = first_result_stats
        
        # By prompt complexity
        complexity_stats = {}
        for complexity in sorted(df['prompt_complexity'].unique()):
            mask = df['prompt_complexity'] == complexity
            comp_data = df[mask]
            
            complexity_stats[str(complexity)] = {
                'n': len(comp_data),
                'bankruptcy_rate': comp_data['is_bankrupt'].mean(),
                'immediate_stop_rate': comp_data['immediate_stop'].mean(),
                'avg_additional_rounds': comp_data['additional_rounds'].mean(),
                'avg_total_rounds': comp_data['total_rounds_including_first'].mean(),
                'avg_net_profit': comp_data['net_profit_from_baseline'].mean(),
                'avg_bet_amount': comp_data['avg_bet_amount'].mean(),
            }
        
        stats['by_prompt_complexity'] = complexity_stats
        
        # High-risk prompts (top 5 bankruptcy rates)
        prompt_stats = df.groupby('prompt_combo').agg({
            'is_bankrupt': ['count', 'mean'],
            'immediate_stop': 'mean',
            'additional_rounds': 'mean',
            'total_rounds_including_first': 'mean',
            'net_profit_from_baseline': 'mean',
            'avg_bet_amount': 'mean'
        }).round(4)
        
        prompt_stats.columns = ['n', 'bankruptcy_rate', 'immediate_stop_rate', 
                               'avg_additional_rounds', 'avg_total_rounds', 
                               'avg_net_profit', 'avg_bet_amount']
        
        # Get top 5 risky prompts
        top_risky = prompt_stats.sort_values('bankruptcy_rate', ascending=False).head(5)
        stats['high_risk_prompts'] = top_risky.to_dict('index')
        
        return stats
    
    def create_latex_tables(self, df: pd.DataFrame, stats: Dict) -> Dict[str, str]:
        """Create LaTeX tables for the paper"""
        
        tables = {}
        
        # Table 1: Comprehensive metrics (bet type and first result)
        table1_data = []
        
        # Bet type section
        table1_data.append("\\multicolumn{6}{c}{\\textbf{베팅 타입}} \\\\")
        for bet_type in ['fixed', 'variable']:
            bt_stats = stats['by_bet_type'][bet_type]
            korean_name = '고정 베팅' if bet_type == 'fixed' else '가변 베팅'
            table1_data.append(
                f"{korean_name} & {bt_stats['n']} & {bt_stats['bankruptcy_rate']*100:.1f} & "
                f"{bt_stats['avg_net_profit']:.2f} & {bt_stats['avg_total_rounds']:.1f} & "
                f"{bt_stats['avg_bet_amount']:.2f} \\\\"
            )
        
        table1_data.append("\\midrule")
        table1_data.append("\\multicolumn{6}{c}{\\textbf{첫 게임 결과}} \\\\")
        
        for first_result in ['W', 'L']:
            fr_stats = stats['by_first_result'][first_result]
            korean_name = '승리' if first_result == 'W' else '패배'
            table1_data.append(
                f"{korean_name} & {fr_stats['n']} & {fr_stats['bankruptcy_rate']*100:.1f} & "
                f"{fr_stats['avg_net_profit']:.2f} & {fr_stats['avg_total_rounds']:.1f} & "
                f"{fr_stats['avg_bet_amount']:.2f} \\\\"
            )
        
        table1_data.append("\\midrule")
        table1_data.append("\\multicolumn{6}{c}{\\textbf{고위험 프롬프트 조합 (상위 5개)}} \\\\")
        
        for prompt, p_stats in list(stats['high_risk_prompts'].items())[:5]:
            table1_data.append(
                f"{prompt} & {int(p_stats['n'])} & {p_stats['bankruptcy_rate']*100:.1f} & "
                f"{p_stats['avg_net_profit']:.2f} & {p_stats['avg_total_rounds']:.1f} & "
                f"{p_stats['avg_bet_amount']:.2f} \\\\"
            )
        
        tables['comprehensive_metrics'] = "\\n".join(table1_data)
        
        # Table 2: Prompt complexity
        table2_data = []
        for complexity in sorted(stats['by_prompt_complexity'].keys()):
            c_stats = stats['by_prompt_complexity'][complexity]
            complexity_label = f"{complexity} (BASE)" if complexity == 0 else str(complexity)
            table2_data.append(
                f"{complexity_label} & {c_stats['n']} & {c_stats['bankruptcy_rate']*100:.1f} & "
                f"{c_stats['avg_net_profit']:.2f} & {c_stats['avg_total_rounds']:.1f} & "
                f"{c_stats['avg_bet_amount']:.2f} \\\\"
            )
        
        tables['prompt_complexity'] = "\\n".join(table2_data)
        
        return tables
    
    def save_detailed_results(self, df: pd.DataFrame, stats: Dict, output_dir: str):
        """Save detailed analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed DataFrame
        df.to_csv(output_path / 'gpt_rigorous_analysis_detailed.csv', index=False)
        
        # Save summary statistics (convert numpy types)
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
            
        with open(output_path / 'gpt_rigorous_analysis_summary.json', 'w') as f:
            json.dump(stats, f, indent=2, default=convert_types)
        
        # Save LaTeX tables
        tables = self.create_latex_tables(df, stats)
        with open(output_path / 'gpt_latex_tables.txt', 'w') as f:
            for table_name, table_content in tables.items():
                f.write(f"=== {table_name.upper()} ===\n")
                f.write(table_content)
                f.write("\n\n")
        
        print(f"📁 Results saved to {output_path}")
        return tables

def main():
    # Initialize analyzer
    data_path = "/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json"
    analyzer = GPTRigorousAnalysis(data_path)
    
    # Analyze all experiments
    df = analyzer.analyze_all_experiments()
    
    # Generate statistics
    stats = analyzer.generate_summary_statistics(df)
    
    # Print key findings
    print("\n" + "="*60)
    print("📊 KEY FINDINGS")
    print("="*60)
    print(f"Total experiments: {stats['total_experiments']}")
    print(f"Overall bankruptcy rate: {stats['bankruptcy_rate']*100:.1f}%")
    print(f"Immediate stop rate: {stats['immediate_stop_rate']*100:.1f}%")
    print(f"Average additional rounds (after first): {stats['avg_additional_rounds']:.2f}")
    print(f"Average total rounds (including first): {stats['avg_total_rounds']:.2f}")
    
    print(f"\nBy bet type:")
    for bet_type, bt_stats in stats['by_bet_type'].items():
        print(f"  {bet_type}: bankruptcy={bt_stats['bankruptcy_rate']*100:.1f}%, "
              f"immediate_stop={bt_stats['immediate_stop_rate']*100:.1f}%, "
              f"avg_profit={bt_stats['avg_net_profit']:.2f}")
    
    print(f"\nBy first result:")
    for first_result, fr_stats in stats['by_first_result'].items():
        print(f"  {first_result}: bankruptcy={fr_stats['bankruptcy_rate']*100:.1f}%, "
              f"immediate_stop={fr_stats['immediate_stop_rate']*100:.1f}%, "
              f"avg_profit={fr_stats['avg_net_profit']:.2f}")
    
    # Save results
    output_dir = "/home/ubuntu/llm_addiction/analysis"
    tables = analyzer.save_detailed_results(df, stats, output_dir)
    
    print(f"\n✅ Rigorous analysis complete!")
    print(f"📄 LaTeX tables ready for paper update")
    
    return df, stats, tables

if __name__ == "__main__":
    df, stats, tables = main()