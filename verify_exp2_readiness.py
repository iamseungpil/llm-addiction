#!/usr/bin/env python3
"""
Verify Experiment 2 readiness - check all dependencies and requirements

This script checks:
1. NPZ input file exists
2. SAE checkpoints accessible
3. GPU availability
4. Script compatibility
5. Dependencies importable
"""

import os
import sys
from pathlib import Path
import json

def check_file(path, description):
    """Check if file exists and report size"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        size_str = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
        print(f"  ✅ {description}: {size_str}")
        return True
    else:
        print(f"  ❌ {description}: NOT FOUND")
        return False

def check_dir(path, description):
    """Check if directory exists"""
    if os.path.exists(path):
        print(f"  ✅ {description}: EXISTS")
        return True
    else:
        print(f"  ❌ {description}: NOT FOUND")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        gpus = result.stdout.strip().split('\n')
        free_gpus = []
        for gpu_info in gpus:
            idx, free_mem = gpu_info.split(',')
            idx = int(idx.strip())
            free_mem = int(free_mem.strip())
            if idx in [4, 5, 6, 7]:  # Target GPUs
                if free_mem > 20000:  # > 20 GB free
                    free_gpus.append(idx)
                    print(f"  ✅ GPU {idx}: {free_mem} MB free (AVAILABLE)")
                else:
                    print(f"  ⚠️  GPU {idx}: {free_mem} MB free (LOW MEMORY)")
        return len(free_gpus) >= 4
    except Exception as e:
        print(f"  ❌ GPU check failed: {e}")
        return False

def check_imports():
    """Check if required Python packages can be imported"""
    required = [
        'torch',
        'numpy',
        'transformers',
        'scipy',
        'tqdm'
    ]

    all_ok = True
    for module in required:
        try:
            __import__(module)
            print(f"  ✅ {module}: importable")
        except ImportError:
            print(f"  ❌ {module}: NOT importable")
            all_ok = False

    return all_ok

def check_sae_loader():
    """Check if SAE loader works"""
    try:
        sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
        from llama_scope_working import LlamaScopeWorking
        print(f"  ✅ LlamaScopeWorking: importable")
        return True
    except Exception as e:
        print(f"  ❌ LlamaScopeWorking: {e}")
        return False

def main():
    print("=" * 70)
    print("EXPERIMENT 2 READINESS CHECK")
    print("=" * 70)

    checks = []

    # Check 1: Input files
    print("\n1. Input Files:")
    checks.append(check_file(
        '/data/llm_addiction/results/L1_31_GLOBAL_FDR_features_20251110_214621.npz',
        'NPZ input file'
    ))

    # Check 2: Experiment script
    print("\n2. Experiment Scripts:")
    checks.append(check_file(
        '/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py',
        'Experiment 2 main script'
    ))
    checks.append(check_file(
        '/home/ubuntu/llm_addiction/convert_npz_to_json.py',
        'NPZ to JSON converter'
    ))

    # Check 3: SAE checkpoints
    print("\n3. SAE Checkpoints:")
    checks.append(check_dir(
        '/data/.cache/huggingface/hub/models--fnlp--Llama3_1-8B-Base-LXR-8x',
        'SAE checkpoint directory'
    ))

    # Check 4: Output directories
    print("\n4. Output Directories:")
    os.makedirs('/data/llm_addiction/experiment_2_multilayer_patching', exist_ok=True)
    os.makedirs('/data/llm_addiction/experiment_1_L1_31_extraction', exist_ok=True)
    checks.append(check_dir(
        '/data/llm_addiction/experiment_2_multilayer_patching',
        'Experiment 2 output directory'
    ))
    checks.append(check_dir(
        '/data/llm_addiction/experiment_1_L1_31_extraction',
        'Experiment 1 output directory'
    ))

    # Check 5: GPUs
    print("\n5. GPU Availability:")
    checks.append(check_gpu())

    # Check 6: Python dependencies
    print("\n6. Python Dependencies:")
    checks.append(check_imports())

    # Check 7: SAE loader
    print("\n7. SAE Loader:")
    checks.append(check_sae_loader())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_checks = len(checks)
    passed_checks = sum(checks)

    print(f"Checks passed: {passed_checks}/{total_checks}")

    if passed_checks == total_checks:
        print("\n✅ ALL CHECKS PASSED - READY TO RUN EXPERIMENT 2")
        print("\nNext steps:")
        print("  1. Run: python3 /home/ubuntu/llm_addiction/convert_npz_to_json.py")
        print("  2. Edit: /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py")
        print("     Line 139: Update features_file path")
        print("  3. Launch: cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31")
        print("     ./launch_corrected_single.sh")
    else:
        print("\n⚠️  SOME CHECKS FAILED - REVIEW ISSUES ABOVE")
        print(f"   {total_checks - passed_checks} issue(s) need to be resolved")

    print("=" * 70)

    return passed_checks == total_checks

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
