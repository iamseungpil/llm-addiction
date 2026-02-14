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
            "chat_template": False  # Use base model for SAE compatibility
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

        # Disable torch.compile completely (for Gemma-2 sliding window attention)
        os.environ['TORCH_COMPILE'] = '0'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'

        # Disable torch.compile and dynamo at Python level
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.reset()
        except Exception:
            pass

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

    # Invalid response patterns for Base models
    INVALID_PATTERNS = [
        'do not include',
        'do not use',
        'do not type',
        'do not end',
        'no other information',
        'no other text',
        'no other formatting',
        'this will be used',
        'this will help',
        'this line will',
        'you will be graded',
        'you will receive',
        'you will not receive',
        'you must do this',
        'your response will',
        'your response must',
        'your response should',
        'if you do not',
        'if you have any questions',
        '(optional)',
        '(e.g.,',
        'where x is',
    ]

    def _is_invalid_response(self, text: str) -> bool:
        """Check if response matches known invalid patterns (Base model artifacts)."""
        text_lower = text.lower().strip()

        # Empty or too short
        if len(text_lower) < 3:
            return True

        # Just punctuation
        if text_lower in ['.', '..', '...', ',', '!', '?']:
            return True

        # Starts with invalid patterns (prompt continuation)
        for pattern in self.INVALID_PATTERNS:
            if text_lower.startswith(pattern):
                return True

        return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        max_retries: int = 3
    ) -> str:
        """
        Generate text from prompt with automatic retry for invalid responses.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            max_retries: Maximum retries for invalid responses (Base model)

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

        # Generate with retry for invalid responses
        for attempt in range(max_retries):
            with torch.no_grad():
                # Increase temperature slightly on retries for diversity
                current_temp = temperature + (attempt * 0.1)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=min(current_temp, 1.0),
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Check if valid response (skip check for Instruct models)
            if self.config['chat_template'] or not self._is_invalid_response(generated_text):
                return generated_text

            # Log retry for Base models
            if attempt < max_retries - 1:
                logger.debug(f"Invalid response (attempt {attempt + 1}), retrying: {generated_text[:30]}...")

        # Return last attempt even if invalid
        return generated_text

    def generate_with_hidden_states(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate text and extract hidden states (for SAE analysis).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Tuple of (generated_text, hidden_states)
            hidden_states: Tensor of shape (1, hidden_dim) from last token of last layer
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

        # Generate with hidden states
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # Extract generated text
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Extract hidden states from last token of last generation step
        # outputs.hidden_states is a tuple of tuples: (step1, step2, ..., stepN)
        # Each step contains (layer0, layer1, ..., layerN)
        # We want the last step, last layer, last token
        last_step_hidden = outputs.hidden_states[-1]  # Last generation step
        last_layer_hidden = last_step_hidden[-1]  # Last layer
        last_token_hidden = last_layer_hidden[:, -1, :]  # Last token: (batch_size, hidden_dim)

        return generated_text.strip(), last_token_hidden.cpu()

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
