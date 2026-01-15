#!/usr/bin/env python3
"""
Check the actual keys in safetensors file
"""

from safetensors.torch import load_file

checkpoint_path = "/data/.cache/huggingface/hub/models--fnlp--Llama3_1-8B-Base-LXR-8x/snapshots/8dbc1d85edfced43081c03c38b05514dbab1368b/Llama3_1-8B-Base-L25R-8x/checkpoints/final.safetensors"

print("Loading safetensors file...")
checkpoint = load_file(checkpoint_path, device='cpu')

print(f"Keys in safetensors file: {list(checkpoint.keys())}")

for key, tensor in checkpoint.items():
    print(f"  {key}: {tensor.shape} ({tensor.dtype})")