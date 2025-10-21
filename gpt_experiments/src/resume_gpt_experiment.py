#!/usr/bin/env python3
"""
Resume GPT experiment from where it stopped
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('/home/ubuntu/llm_addiction/gpt_experiments/src')
from gpt_multiround_experiment import GPTMultiRoundExperiment

class ResumeGPTExperiment(GPTMultiRoundExperiment):
    """Resume GPT experiment from specific point"""
    
    def __init__(self):
        super().__init__()
        self.existing_results = []
        self.resume_info = None
        
    def load_resume_info(self):
        """Load resume information"""
        resume_file = Path('/home/ubuntu/llm_addiction/gpt_experiments/src/gpt_resume_info.json')
        with open(resume_file, 'r') as f:
            self.resume_info = json.load(f)
        
        print(f"Resume from: {self.resume_info['total_completed']} experiments completed")
        print(f"Remaining: {self.resume_info['total_remaining']} experiments")
        
        # Load existing results
        results_dir = Path('/data/llm_addiction/gpt_results')
        latest_file = sorted(results_dir.glob('gpt_multiround_intermediate_*.json'))[-1]
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        self.existing_results = data['results']
        print(f"Loaded {len(self.existing_results)} existing results")
    
    def resume_experiment(self):
        """Resume the experiment from where it stopped"""
        print("\n" + "="*80)
        print("RESUMING GPT EXPERIMENT")
        print("="*80)
        
        self.load_resume_info()
        
        # Generate all conditions
        bet_types = ['fixed', 'variable']
        first_results = ['W', 'L']
        prompt_combos = self.generate_prompt_combinations()
        
        print(f"\nResuming from experiment {len(self.existing_results) + 1}/6400")
        
        # Create set of completed experiments for quick lookup
        completed = set()
        rep_counts = {}
        
        for exp in self.existing_results:
            key = (exp['bet_type'], exp['first_result'], exp['prompt_combo'])
            completed.add(key)
            rep_counts[key] = rep_counts.get(key, 0) + 1
        
        # Continue with remaining experiments
        all_results = self.existing_results.copy()
        experiment_id = len(self.existing_results)
        
        # Process conditions in order
        for bet_type in bet_types:
            for first_result in first_results:
                for prompt_combo in prompt_combos:
                    key = (bet_type, first_result, prompt_combo)
                    completed_reps = rep_counts.get(key, 0)
                    
                    # Skip if this condition is already complete
                    if completed_reps >= 50:
                        continue
                    
                    print(f"\n{'='*60}")
                    print(f"Condition: {bet_type}_{first_result}_{prompt_combo}")
                    print(f"Completed: {completed_reps}/50, Need: {50 - completed_reps}")
                    print(f"{'='*60}")
                    
                    # Run remaining repetitions for this condition
                    for rep in range(completed_reps, 50):
                        experiment_id += 1
                        
                        print(f"\n📊 Experiment {experiment_id}/6400: {prompt_combo}_{bet_type}_{first_result} (Rep {rep+1}/50)")
                        
                        # Run single game
                        try:
                            game_result = self.run_single_game(
                                bet_type, first_result, prompt_combo,
                                experiment_id, rep + 1
                            )
                            
                            # Add metadata
                            game_result['experiment_id'] = experiment_id
                            game_result['repetition'] = rep + 1
                            game_result['resumed'] = True
                            
                            all_results.append(game_result)
                            
                            # Save every 10 experiments
                            if experiment_id % 10 == 0:
                                self.save_intermediate_results(all_results, experiment_id)
                                
                        except KeyboardInterrupt:
                            print(f"\n⏸️ Interrupted at experiment {experiment_id}")
                            self.save_intermediate_results(all_results, experiment_id)
                            return all_results
                        
                        except Exception as e:
                            print(f"❌ Error in experiment {experiment_id}: {e}")
                            # Continue with next experiment
                            continue
                    
                    if experiment_id >= 6400:
                        break
                
                if experiment_id >= 6400:
                    break
            
            if experiment_id >= 6400:
                break
        
        # Save final results
        self.save_final_results(all_results)
        return all_results
    
    def save_intermediate_results(self, results, current_exp):
        """Save intermediate results"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(f'/data/llm_addiction/gpt_results/gpt_multiround_intermediate_{timestamp}.json')
        
        summary = {
            'timestamp': timestamp,
            'experiment_type': 'gpt_multiround_resumed',
            'model': 'gpt-4o-mini',
            'total_experiments': len(results),
            'current_experiment': current_exp,
            'resumed_from': self.resume_info['total_completed'] if self.resume_info else 0,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f)
        
        print(f"💾 Saved intermediate results: {current_exp}/6400")
    
    def save_final_results(self, results):
        """Save final results"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(f'/data/llm_addiction/gpt_results/gpt_multiround_complete_{timestamp}.json')
        
        summary = {
            'timestamp': timestamp,
            'experiment_type': 'gpt_multiround_COMPLETE',
            'model': 'gpt-4o-mini',
            'total_experiments': len(results),
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f)
        
        print(f"\n✅ Final results saved: {output_file}")
        print(f"Total experiments: {len(results)}")

def main():
    experiment = ResumeGPTExperiment()
    results = experiment.resume_experiment()
    
    print("\n" + "="*80)
    print("GPT EXPERIMENT RESUME COMPLETE!")
    print("="*80)
    print(f"Total experiments: {len(results)}")

if __name__ == "__main__":
    main()