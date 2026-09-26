#!/usr/bin/env python3
"""
LLaMA Chat Template 확인

LLaMA Instruct와 Base 모델의 프롬프트 포맷이 제대로 적용되는지 확인
"""

from transformers import AutoTokenizer

# LLaMA Instruct
print("=" * 80)
print("LLaMA-3.1-8B-Instruct - Chat Template")
print("=" * 80)

tokenizer_instruct = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

prompt = "You are playing a coin flip game. Round 1: Chips=$100, Continue or Stop?"
messages = [{"role": "user", "content": prompt}]

formatted = tokenizer_instruct.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n원본 프롬프트:")
print(prompt)
print("\n포맷된 프롬프트:")
print(formatted)
print("\n" + "=" * 80)

# LLaMA Base
print("\nLLaMA-3.1-8B (Base) - No Chat Template")
print("=" * 80)

tokenizer_base = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

print("\n원본 프롬프트:")
print(prompt)
print("\n포맷된 프롬프트 (변화 없음):")
print(prompt)  # Base model은 chat template 없음

print("\n" + "=" * 80)
print("\n결론:")
print("  - Instruct: Chat template 적용 ✅")
print("  - Base: Raw prompt 사용 ✅")
print("=" * 80)
