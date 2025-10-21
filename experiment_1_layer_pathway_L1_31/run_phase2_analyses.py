#!/usr/bin/env python3
"""
Run all Phase 2 analyses for Experiment 1 Layer Pathway
Phase 2 focuses on understanding HOW information flows, not just WHICH layers are important
"""

import subprocess
import sys
import time

def run_analysis(script_name, description):
    """Run a single analysis script"""
    print("\n" + "="*100)
    print(f"🚀 Starting: {description}")
    print("="*100)

    start_time = time.time()

    try:
        result = subprocess.run(
            ['python3', script_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(result.stdout)
            print(f"✅ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ Failed with error:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*100)
    print("PHASE 2: ADVANCED PATHWAY ANALYSIS")
    print("="*100)
    print("\nGoal: Understand HOW information flows through layers")
    print("  - Logit Lens: When does the model 'decide'?")
    print("  - Feature Pathway: How do features propagate across layers?")
    print("  - Decision Signature: What is the multi-layer pattern of decisions?")
    print()

    analyses = [
        # Skip Logit Lens for now (requires model loading, slow)
        # ('analysis_phase2_logit_lens.py', 'Logit Lens Analysis'),

        ('analysis_phase2_feature_pathway.py', 'Feature Pathway Tracing'),
        ('analysis_phase2_decision_signature.py', 'Decision Signature Analysis'),
    ]

    results = {}

    for script, description in analyses:
        success = run_analysis(script, description)
        results[script] = success

    # Summary
    print("\n" + "="*100)
    print("PHASE 2 ANALYSIS SUMMARY")
    print("="*100)

    for script, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {script}")

    all_success = all(results.values())

    if all_success:
        print("\n🎉 All Phase 2 analyses completed successfully!")
        print("\n📁 Generated Files:")
        print("  - feature_network.png: Feature correlation network")
        print("  - correlation_heatmap.png: L8 -> L31 correlation matrix")
        print("  - decision_space.png: PCA/t-SNE visualization of decisions")
        print("  - layer_contributions.png: Layer activation by decision type")
        print("  - feature_pathway_results.json: Correlation data")
        print("  - decision_signature_results.json: Signature data")

        print("\n💡 Next Steps:")
        print("  1. Review feature pathways to identify L8 -> L31 connections")
        print("  2. Examine decision signatures to understand multi-layer patterns")
        print("  3. (Optional) Run Logit Lens to see when model 'decides'")
        print("  4. Wait for Exp3 completion to get semantic word mappings")
    else:
        print("\n⚠️ Some analyses failed. Check logs above.")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
