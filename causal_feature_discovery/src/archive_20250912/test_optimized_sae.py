#!/usr/bin/env python3
"""
Quick test of optimized SAE loading
"""

import os
import sys
import torch

# Set environment variables before any imports
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from llama_scope_working import LlamaScopeWorking

def test_sae_loading():
    print("🧪 Testing optimized SAE loading...")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name()}")
        
        # Check GPU memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total_mem:.1f}GB total")
    
    try:
        # Test Layer 25 loading
        print("\n🔧 Testing Layer 25 SAE loading...")
        sae_25 = LlamaScopeWorking(layer=25, device="cuda")
        print("✅ Layer 25 SAE loaded successfully!")
        
        # Test Layer 30 loading
        print("\n🔧 Testing Layer 30 SAE loading...")
        sae_30 = LlamaScopeWorking(layer=30, device="cuda")
        print("✅ Layer 30 SAE loaded successfully!")
        
        print("\n✅ All SAE tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ SAE loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sae_loading()
    sys.exit(0 if success else 1)