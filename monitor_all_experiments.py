#!/usr/bin/env python3
"""
Monitor all running experiments for errors and progress
Checks logs and provides summary of all 5 experiments
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import re

def get_tmux_sessions():
    """Get list of running tmux sessions"""
    try:
        result = subprocess.run(['tmux', 'ls'], capture_output=True, text=True)
        if result.returncode == 0:
            sessions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    session_name = line.split(':')[0]
                    sessions.append(session_name)
            return sessions
        return []
    except:
        return []

def check_log_for_errors(log_file: Path, tail_lines: int = 100):
    """Check log file for errors"""
    if not log_file.exists():
        return {'exists': False}

    try:
        # Get file size
        file_size = log_file.stat().st_size / (1024 * 1024)  # MB

        # Read last N lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            tail = lines[-tail_lines:] if len(lines) > tail_lines else lines

        tail_text = ''.join(tail)

        # Check for errors
        errors = []
        if 'error' in tail_text.lower() or 'exception' in tail_text.lower():
            for line in tail:
                if 'error' in line.lower() or 'exception' in line.lower():
                    errors.append(line.strip())

        # Extract progress if available
        progress = None
        for line in reversed(tail):
            # Look for progress indicators
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                progress = f"{match.group(1)}/{match.group(2)}"
                break

            match = re.search(r'(\d+)%', line)
            if match:
                progress = f"{match.group(1)}%"
                break

        return {
            'exists': True,
            'size_mb': f"{file_size:.2f}",
            'errors': errors[:5],  # Show first 5 errors
            'progress': progress
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}

def monitor_experiment(name: str, sessions: list, log_paths: list):
    """Monitor single experiment"""
    print(f"\n{'='*80}")
    print(f"📊 {name}")
    print(f"{'='*80}")

    # Check tmux sessions
    active_sessions = [s for s in sessions if name.lower() in s.lower()]
    if active_sessions:
        print(f"✅ Running sessions: {', '.join(active_sessions)}")
    else:
        print(f"❌ No active sessions found")

    # Check logs
    for log_path in log_paths:
        log_file = Path(log_path)
        print(f"\n📄 Log: {log_file.name}")

        info = check_log_for_errors(log_file)

        if not info['exists']:
            print("   ⚠️  Log file not found")
            continue

        print(f"   Size: {info['size_mb']} MB")

        if info.get('progress'):
            print(f"   Progress: {info['progress']}")

        if info.get('errors'):
            print(f"   ⚠️  Errors found:")
            for err in info['errors']:
                print(f"      {err[:100]}")
        else:
            print("   ✅ No errors in recent output")

def main():
    print("="*80)
    print("🔍 EXPERIMENT MONITOR")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Get tmux sessions
    sessions = get_tmux_sessions()
    print(f"\n📌 Active tmux sessions: {len(sessions)}")
    if sessions:
        for s in sessions:
            print(f"   - {s}")

    # Monitor each experiment
    monitor_experiment(
        "Exp0: LLaMA/Gemma Restart",
        sessions,
        [
            "/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/logs/exp0_llama.log",
            "/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/logs/exp0_gemma.log"
        ]
    )

    monitor_experiment(
        "Exp1: Layer Pathway",
        sessions,
        ["/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/logs/exp1_pathway.log"]
    )

    monitor_experiment(
        "Exp2: Multilayer Patching L1-31",
        sessions,
        [
            "/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L1_8.log",
            "/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L9_16.log",
            "/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L17_24.log",
            "/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_L25_31.log"
        ]
    )

    monitor_experiment(
        "Exp3: Feature-Word Analysis",
        sessions,
        ["/home/ubuntu/llm_addiction/experiment_3_feature_word_6400/logs/exp3_word.log"]
    )

    monitor_experiment(
        "Exp5: Multi-Round Patching",
        sessions,
        ["/home/ubuntu/llm_addiction/experiment_5_multiround_patching/exp5_restart.log"]
    )

    print("\n" + "="*80)
    print("✅ MONITORING COMPLETE")
    print("="*80)
    print("\nTo attach to a session:")
    print("  tmux attach -t <session_name>")
    print("\nTo check GPU usage:")
    print("  nvidia-smi")

if __name__ == '__main__':
    main()
