"""
Model Loader for LLaMA, Gemma, and Qwen models

Handles loading and inference for local open-weight models.
"""

import os
import torch
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import clear_gpu_memory, setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    """
    Unified model loader for LLaMA, Gemma, and Qwen models.

    Follows the same pattern as slot machine experiments:
    - bf16 precision
    - eager attention (no flash attention)
    - device_map for single GPU
    """

    MODEL_CONFIGS = {
        "llama": {
            "model_id": "meta-llama/Llama-3.1-8B",
            "memory_gb": 19,
            "chat_template": False  # Use base model
        },
        "gemma": {
            "model_id": "google/gemma-2-9b-it",
            "memory_gb": 22,
            "chat_template": True
        },
        "qwen": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "memory_gb": 16,
            "chat_template": True
        }
    }

    def __init__(self, model_name: str, gpu_id: int = 0):
        """
        Initialize model loader.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID (after CUDA_VISIBLE_DEVICES)
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'

        self.config = self.MODEL_CONFIGS[model_name]
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer"""
        logger.info(f"🚀 Loading {self.model_name.upper()} model on GPU {self.gpu_id}")
        logger.info(f"   Model ID: {self.config['model_id']}")
        logger.info(f"   Expected memory: ~{self.config['memory_gb']}GB")

        # Clear GPU memory
        clear_gpu_memory()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Disable torch.compile (for Gemma-2 sliding window attention)
        os.environ['TORCH_COMPILE'] = '0'

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_id'],
            torch_dtype=torch.bfloat16,
            device_map={'': self.gpu_id},
            low_cpu_mem_usage=True,
            use_cache=False,
            attn_implementation="eager",
            _attn_implementation_internal="eager"
        )
        self.model.eval()

        # Set default dtype
        torch.set_default_dtype(torch.bfloat16)

        logger.info(f"✅ {self.model_name.upper()} loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Apply chat template if available
        if self.config['chat_template']:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def unload(self):
        """Unload model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        clear_gpu_memory()
        logger.info(f"✅ {self.model_name.upper()} unloaded")
