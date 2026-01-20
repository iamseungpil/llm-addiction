#!/usr/bin/env python3
"""
Hidden State Extractor

Extracts hidden states from LLM for given prompts.
Forward pass only (no generation) for efficiency.
"""

import os
import gc
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from pathlib import Path


class HiddenStateExtractor:
    """
    Extracts hidden states from Gemma or LLaMA models.
    Uses forward pass only (no generation) for efficiency.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Args:
            model_name: 'gemma' or 'llama'
            device: CUDA device
            torch_dtype: Model precision
        """
        self.model_name = model_name.lower()
        self.device = device
        self.torch_dtype = torch_dtype

        self.model = None
        self.tokenizer = None
        self.n_layers = None
        self.d_model = None

        # Model IDs
        self.model_ids = {
            'gemma': 'google/gemma-2-9b',
            'llama': 'meta-llama/Llama-3.1-8B'
        }

    def load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.model_ids[self.model_name]
        print(f"Loading {self.model_name.upper()} ({model_id})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            attn_implementation="eager"
        )
        self.model.eval()

        # Get model dimensions
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

        print(f"Loaded: {self.n_layers} layers, d_model={self.d_model}")

    def extract_single(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
        position: str = 'last'
    ) -> Dict[int, np.ndarray]:
        """
        Extract hidden states for a single prompt.

        Args:
            prompt: Input text
            layers: Layer indices to extract (None = all layers)
            position: 'last' (last token) or 'mean' (mean over all tokens)

        Returns:
            Dict: layer_idx -> hidden state array [d_model]
        """
        if layers is None:
            layers = list(range(self.n_layers))

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        hidden_states = {}
        for layer in layers:
            if layer < len(outputs.hidden_states):
                h = outputs.hidden_states[layer]  # [1, seq_len, d_model]
                if position == 'last':
                    h = h[:, -1, :]  # [1, d_model]
                elif position == 'mean':
                    h = h.mean(dim=1)  # [1, d_model]
                hidden_states[layer] = h.float().cpu().numpy().squeeze()

        return hidden_states

    def extract_batch(
        self,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        position: str = 'last',
        batch_size: int = 8,
        show_progress: bool = True
    ) -> Dict[int, np.ndarray]:
        """
        Extract hidden states for multiple prompts.

        Args:
            prompts: List of input texts
            layers: Layer indices (None = all)
            position: 'last' or 'mean'
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            Dict: layer_idx -> array [n_samples, d_model]
        """
        if layers is None:
            layers = list(range(self.n_layers))

        # Initialize storage
        all_hiddens = {layer: [] for layer in layers}

        # Process in batches
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting hidden states")

        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )

            # Extract each sample in batch
            for layer in layers:
                h = outputs.hidden_states[layer]  # [batch, seq_len, d_model]

                if position == 'last':
                    # Get last non-padding token for each sample
                    attention_mask = inputs['attention_mask']
                    seq_lengths = attention_mask.sum(dim=1) - 1  # Last token index
                    batch_hiddens = []
                    for b_idx in range(h.size(0)):
                        last_idx = seq_lengths[b_idx].item()
                        batch_hiddens.append(h[b_idx, last_idx, :])
                    h = torch.stack(batch_hiddens)  # [batch, d_model]
                elif position == 'mean':
                    # Mean over non-padding tokens
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    h = (h * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

                all_hiddens[layer].append(h.float().cpu().numpy())

            # Clear cache periodically
            if (i // batch_size) % 10 == 0:
                self._clear_cache()

        # Concatenate batches
        for layer in layers:
            all_hiddens[layer] = np.vstack(all_hiddens[layer])

        return all_hiddens

    def _clear_cache(self):
        """Clear GPU memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._clear_cache()
        print("Model unloaded")


def save_hidden_states(
    hidden_states: Dict[int, np.ndarray],
    labels: List[int],
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save extracted hidden states to npz file.

    Args:
        hidden_states: Dict of layer -> array
        labels: List of labels
        output_path: Path to save
        metadata: Optional metadata dict
    """
    save_dict = {
        f'layer_{k}': v for k, v in hidden_states.items()
    }
    save_dict['labels'] = np.array(labels)

    if metadata:
        # Save metadata as JSON string in the npz
        import json
        save_dict['metadata'] = np.array([json.dumps(metadata)])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved: {output_path}")


def load_hidden_states(path: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, Optional[Dict]]:
    """
    Load hidden states from npz file.

    Returns:
        (hidden_states dict, labels array, metadata dict or None)
    """
    import json

    data = np.load(path, allow_pickle=True)

    hidden_states = {}
    for key in data.files:
        if key.startswith('layer_'):
            layer_idx = int(key.split('_')[1])
            hidden_states[layer_idx] = data[key]

    labels = data['labels']

    metadata = None
    if 'metadata' in data.files:
        metadata = json.loads(str(data['metadata'][0]))

    return hidden_states, labels, metadata


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Quick test
    print("Testing HiddenStateExtractor...")

    extractor = HiddenStateExtractor('gemma', device='cuda:0')
    extractor.load_model()

    # Test single extraction
    test_prompt = "You are playing a slot machine game at a casino.\nInitial funds: $100\nCurrent balance: $50"
    layers = [15, 20, 25, 30]

    print(f"\nExtracting from layers {layers}...")
    hiddens = extractor.extract_single(test_prompt, layers=layers)

    for layer, h in hiddens.items():
        print(f"  Layer {layer}: shape={h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")

    # Test batch extraction
    test_prompts = [test_prompt] * 4
    print(f"\nBatch extraction ({len(test_prompts)} samples)...")
    batch_hiddens = extractor.extract_batch(test_prompts, layers=layers, batch_size=2)

    for layer, h in batch_hiddens.items():
        print(f"  Layer {layer}: shape={h.shape}")

    extractor.unload_model()
    print("\nTest complete!")
