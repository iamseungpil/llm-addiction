#!/usr/bin/env python3
"""
Phase 1: Steering Vector Extraction

Extracts steering vectors from behavioral experiment data by:
1. Loading experiment data and reconstructing prompts
2. Running model forward pass to get hidden states at target layers
3. Grouping by outcome (bankruptcy vs voluntary_stop)
4. Computing steering vector: mean(bankrupt) - mean(safe) for each layer
5. Saving steering vectors to disk

Usage:
    python extract_steering_vectors.py --model llama --gpu 0
    python extract_steering_vectors.py --model gemma --gpu 1

Design: Modular architecture with checkpoint support for resumable extraction.
"""

# CRITICAL: Set CUDA_VISIBLE_DEVICES before torch import
# This ensures the --gpu argument maps to the correct physical GPU
import os
import sys
import argparse

def _set_gpu_before_torch():
    """Parse --gpu argument and set CUDA_VISIBLE_DEVICES before torch import."""
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            return gpu_id
    return None

_gpu_set = _set_gpu_before_torch()

import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import yaml
import gc

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logging,
    load_experiment_data,
    group_by_outcome,
    PromptBuilder,
    ModelRegistry,
    CheckpointManager,
    load_model_and_tokenizer,
    get_gpu_memory_info,
    clear_gpu_memory,
    get_default_config_path
)


class HiddenStateExtractor:
    """
    Extract hidden states from model forward passes.

    This class handles the extraction of hidden states at specified layers
    during model inference, which are then used to compute steering vectors.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        target_layers: List[int],
        device: str = 'cuda:0',
        logger=None
    ):
        """
        Initialize the hidden state extractor.

        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            model_name: Name of the model ('llama' or 'gemma')
            target_layers: List of layer indices to extract hidden states from
            device: Device for computation
            logger: Optional logger
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.target_layers = target_layers
        self.device = device
        self.logger = logger
        self.model_config = ModelRegistry.get(model_name)

        # Validate layers
        max_layer = self.model_config.n_layers - 1
        for layer in target_layers:
            if layer < 0 or layer > max_layer:
                raise ValueError(f"Layer {layer} out of range [0, {max_layer}] for {model_name}")

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def extract_hidden_states(
        self,
        prompt: str,
        token_position: str = 'last'
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states at target layers for a prompt.

        Args:
            prompt: Input prompt text
            token_position: Which token's hidden state to extract
                           'last' for last token, 'all' for all tokens

        Returns:
            Dict mapping layer index to hidden state tensor
        """
        # Format prompt for chat models
        if self.model_config.use_chat_template:
            chat = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Forward pass with hidden states output
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract hidden states at target layers
        # hidden_states is a tuple: (embedding, layer1, layer2, ..., layerN)
        hidden_states = outputs.hidden_states

        result = {}
        for layer in self.target_layers:
            # Layer 0 is embedding, layer 1 is first transformer layer, etc.
            # So to get layer L's output, we access hidden_states[L+1]
            layer_hidden = hidden_states[layer + 1]

            if token_position == 'last':
                # Get last token's hidden state
                result[layer] = layer_hidden[0, -1, :].cpu().float()
            else:
                # Get all tokens
                result[layer] = layer_hidden[0, :, :].cpu().float()

        # Clean up
        del outputs, hidden_states, inputs
        clear_gpu_memory(self.device)

        return result

    def extract_batch(
        self,
        prompts: List[str],
        token_position: str = 'last'
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Extract hidden states for a batch of prompts.

        Args:
            prompts: List of input prompts
            token_position: Token position to extract

        Returns:
            Dict mapping layer to list of hidden state tensors
        """
        result = {layer: [] for layer in self.target_layers}

        for prompt in prompts:
            hidden_states = self.extract_hidden_states(prompt, token_position)
            for layer, hs in hidden_states.items():
                result[layer].append(hs)

        return result


class SteeringVectorComputer:
    """
    Compute steering vectors from grouped hidden states.

    Steering vector = mean(bankruptcy_hidden_states) - mean(safe_hidden_states)
    """

    def __init__(self, target_layers: List[int], logger=None):
        """
        Initialize the steering vector computer.

        Args:
            target_layers: List of layer indices
            logger: Optional logger
        """
        self.target_layers = target_layers
        self.logger = logger

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def compute(
        self,
        bankrupt_states: Dict[int, List[torch.Tensor]],
        safe_states: Dict[int, List[torch.Tensor]]
    ) -> Dict[int, Dict]:
        """
        Compute steering vectors for each layer.

        Args:
            bankrupt_states: Dict mapping layer to list of bankrupt hidden states
            safe_states: Dict mapping layer to list of safe hidden states

        Returns:
            Dict mapping layer to steering vector info dict containing:
                - 'vector': The steering vector tensor
                - 'bankrupt_mean': Mean of bankrupt states
                - 'safe_mean': Mean of safe states
                - 'n_bankrupt': Number of bankrupt samples
                - 'n_safe': Number of safe samples
                - 'magnitude': L2 norm of steering vector
        """
        result = {}

        for layer in self.target_layers:
            b_states = bankrupt_states.get(layer, [])
            s_states = safe_states.get(layer, [])

            if not b_states or not s_states:
                self._log(f"Warning: No states for layer {layer}, skipping")
                continue

            # Stack tensors
            b_stack = torch.stack(b_states)  # [n_bankrupt, d_model]
            s_stack = torch.stack(s_states)  # [n_safe, d_model]

            # Compute means
            b_mean = b_stack.mean(dim=0)
            s_mean = s_stack.mean(dim=0)

            # Compute steering vector: bankrupt - safe
            # Positive direction = toward risky/bankruptcy behavior
            # Negative direction = toward safe behavior
            steering_vector = b_mean - s_mean
            magnitude = torch.norm(steering_vector).item()

            result[layer] = {
                'vector': steering_vector,
                'bankrupt_mean': b_mean,
                'safe_mean': s_mean,
                'n_bankrupt': len(b_states),
                'n_safe': len(s_states),
                'magnitude': magnitude
            }

            self._log(f"Layer {layer}: |steering| = {magnitude:.4f}, "
                     f"n_bankrupt={len(b_states)}, n_safe={len(s_states)}")

        return result


def save_steering_vectors(
    steering_data: Dict[int, Dict],
    output_path: Path,
    metadata: Dict
) -> None:
    """
    Save steering vectors to disk.

    Args:
        steering_data: Dict mapping layer to steering vector info
        output_path: Path to save .npz file
        metadata: Metadata dict to include
    """
    save_dict = {
        'layers': np.array(list(steering_data.keys())),
        'metadata': metadata
    }

    for layer, data in steering_data.items():
        save_dict[f'layer_{layer}_vector'] = data['vector'].numpy()
        save_dict[f'layer_{layer}_bankrupt_mean'] = data['bankrupt_mean'].numpy()
        save_dict[f'layer_{layer}_safe_mean'] = data['safe_mean'].numpy()
        save_dict[f'layer_{layer}_n_bankrupt'] = np.array(data['n_bankrupt'])
        save_dict[f'layer_{layer}_n_safe'] = np.array(data['n_safe'])
        save_dict[f'layer_{layer}_magnitude'] = np.array(data['magnitude'])

    np.savez(output_path, **save_dict)


def load_steering_vectors(npz_path: Path) -> Dict[int, Dict]:
    """
    Load steering vectors from disk.

    Args:
        npz_path: Path to .npz file

    Returns:
        Dict mapping layer to steering vector info
    """
    data = np.load(npz_path, allow_pickle=True)
    layers = data['layers']

    result = {}
    for layer in layers:
        layer = int(layer)
        result[layer] = {
            'vector': torch.from_numpy(data[f'layer_{layer}_vector']),
            'bankrupt_mean': torch.from_numpy(data[f'layer_{layer}_bankrupt_mean']),
            'safe_mean': torch.from_numpy(data[f'layer_{layer}_safe_mean']),
            'n_bankrupt': int(data[f'layer_{layer}_n_bankrupt']),
            'n_safe': int(data[f'layer_{layer}_n_safe']),
            'magnitude': float(data[f'layer_{layer}_magnitude'])
        }

    return result


def main():
    parser = argparse.ArgumentParser(description='Extract steering vectors from experiment data')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'gemma_base'],
                       help='Model to use')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to config file (default: auto-detect)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per group (overrides config)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated list or range of layers (e.g., "0,1,2" or "0-15")')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load configuration
    config_path = args.config or str(get_default_config_path())
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'

    # Setup logging
    logger = setup_logging(f'extract_{args.model}', log_dir)
    logger.info("=" * 80)
    logger.info(f"Starting steering vector extraction for {args.model.upper()}")
    logger.info(f"GPU: {args.gpu}")
    logger.info("=" * 80)

    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_dir, f'extraction_{args.model}')

    # Load experiment data
    data_path = config[f'{args.model}_data_path']
    exp_data = load_experiment_data(data_path, logger)
    results = exp_data['results']

    # Group by outcome
    grouped = group_by_outcome(results)
    bankrupt_games = grouped['bankruptcy']
    safe_games_all = grouped['voluntary_stop']

    logger.info(f"Bankruptcy games: {len(bankrupt_games)}")
    logger.info(f"Voluntary stop games: {len(safe_games_all)}")

    # CONDITION MATCHING: Match safe games to bankruptcy games by condition
    # This ensures the steering vector captures behavioral difference, not confounds
    # (e.g., balance, history length, prompt complexity)
    logger.info("Matching safe games to bankruptcy games by condition (bet_type, prompt_combo)...")

    # Group safe games by condition for efficient matching
    safe_by_condition = {}
    for game in safe_games_all:
        key = (game['bet_type'], game['prompt_combo'])
        if key not in safe_by_condition:
            safe_by_condition[key] = []
        safe_by_condition[key].append(game)

    # Match each bankruptcy game with a safe game from the same condition
    # Using sampling WITHOUT replacement to avoid reusing same safe games
    matched_safe_games = []
    unmatched_count = 0

    # Create mutable copies for sampling without replacement
    safe_by_condition_remaining = {k: list(v) for k, v in safe_by_condition.items()}
    safe_games_all_remaining = list(safe_games_all)

    for bankrupt_game in bankrupt_games:
        key = (bankrupt_game['bet_type'], bankrupt_game['prompt_combo'])
        if key in safe_by_condition_remaining and safe_by_condition_remaining[key]:
            # Randomly sample and REMOVE from pool (without replacement)
            idx = random.randrange(len(safe_by_condition_remaining[key]))
            safe_game = safe_by_condition_remaining[key].pop(idx)
            matched_safe_games.append(safe_game)
        else:
            # Fallback: use random safe game from remaining pool
            unmatched_count += 1
            if safe_games_all_remaining:
                idx = random.randrange(len(safe_games_all_remaining))
                safe_game = safe_games_all_remaining.pop(idx)
                matched_safe_games.append(safe_game)
            else:
                # Exhausted all safe games - this shouldn't happen normally
                logger.warning("Exhausted all safe games, reusing with replacement")
                matched_safe_games.append(random.choice(safe_games_all))

    if unmatched_count > 0:
        logger.warning(f"Could not match {unmatched_count} bankruptcy games by condition (used random fallback)")

    logger.info(f"Matched {len(matched_safe_games)} safe games to {len(bankrupt_games)} bankruptcy games")

    # Get steering config section (used for multiple settings below)
    steering_config = config.get('steering', {})

    # Get token position from config (default: 'last')
    token_position = steering_config.get('token_position', 'last')
    logger.info(f"Token position for extraction: {token_position}")

    # Limit samples if specified
    max_samples = args.max_samples or steering_config.get('max_samples_per_group', 500)
    if len(bankrupt_games) > max_samples:
        logger.info(f"Limiting samples to {max_samples}")
        bankrupt_games = bankrupt_games[:max_samples]
        matched_safe_games = matched_safe_games[:max_samples]

    # Reconstruct prompts (using matched pairs)
    logger.info("Reconstructing decision prompts...")
    bankrupt_prompts = [PromptBuilder.reconstruct_decision_prompt(g) for g in tqdm(bankrupt_games, desc="Bankrupt prompts")]
    safe_prompts = [PromptBuilder.reconstruct_decision_prompt(g) for g in tqdm(matched_safe_games, desc="Safe prompts")]

    logger.info(f"Reconstructed {len(bankrupt_prompts)} bankrupt prompts")
    logger.info(f"Reconstructed {len(safe_prompts)} matched safe prompts")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, 'cuda:0', torch.bfloat16, logger)

    # Get target layers from new config structure (steering_config already defined above)
    # Priority: CLI --layers > config extract_all_layers > config target_layers

    if args.layers:
        # Parse CLI --layers argument
        if '-' in args.layers and ',' not in args.layers:
            # Range format: "0-15"
            start, end = map(int, args.layers.split('-'))
            target_layers = list(range(start, end + 1))
        else:
            # Comma-separated: "0,1,2,3"
            target_layers = [int(l.strip()) for l in args.layers.split(',')]
        logger.info(f"Using CLI-specified layers: {target_layers}")
    elif steering_config.get('extract_all_layers', False):
        # Extract ALL layers
        model_config = ModelRegistry.get(args.model)
        n_layers = steering_config.get(
            f'{args.model}_n_layers',
            model_config.n_layers - 1  # Exclude output layer
        )
        target_layers = list(range(n_layers))
        logger.info(f"Extracting ALL {n_layers} layers")
    else:
        # Use specified target layers (from steering config only)
        target_layers = steering_config.get('target_layers', [10, 15, 20, 25, 30])

    logger.info(f"Target layers: {target_layers}")

    # Initialize extractor
    extractor = HiddenStateExtractor(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        target_layers=target_layers,
        device='cuda:0',
        logger=logger
    )

    # Check for checkpoint (use torch.load for .pt files)
    checkpoint_data = None
    if args.resume:
        # Find latest .pt checkpoint
        pt_checkpoints = sorted(checkpoint_dir.glob('*_checkpoint_*.pt'))
        if pt_checkpoints:
            latest_ckpt = pt_checkpoints[-1]
            logger.info(f"Loading checkpoint from {latest_ckpt}")
            checkpoint_data = torch.load(latest_ckpt, weights_only=False)
            logger.info(f"Resumed from checkpoint: bankrupt={checkpoint_data.get('bankrupt_processed', 0)}, safe={checkpoint_data.get('safe_processed', 0)}")

    # Extract hidden states for bankrupt games
    logger.info("Extracting hidden states for bankruptcy games...")
    bankrupt_states = {layer: [] for layer in target_layers}

    start_idx = 0
    if checkpoint_data and checkpoint_data.get('bankrupt_processed', 0) > 0:
        # Load existing states from checkpoint (tensors are already stacked)
        for layer in target_layers:
            if f'bankrupt_{layer}' in checkpoint_data:
                stacked = checkpoint_data[f'bankrupt_{layer}']
                if stacked.numel() > 0:  # Check not empty
                    bankrupt_states[layer] = [stacked[i] for i in range(stacked.shape[0])]
        start_idx = checkpoint_data.get('bankrupt_processed', 0)
        logger.info(f"Resuming bankrupt extraction from index {start_idx}")

    for i, prompt in enumerate(tqdm(bankrupt_prompts[start_idx:], desc="Bankrupt hidden states", initial=start_idx)):
        try:
            hidden = extractor.extract_hidden_states(prompt, token_position)
            for layer, hs in hidden.items():
                bankrupt_states[layer].append(hs)

            # Checkpoint periodically (use torch.save to avoid memory leak)
            if (i + start_idx + 1) % steering_config.get('checkpoint_frequency', 100) == 0:
                ckpt_path = checkpoint_dir / f'bankrupt_checkpoint_{i + start_idx + 1}.pt'
                ckpt_data = {
                    'bankrupt_processed': i + start_idx + 1,
                    'safe_processed': 0
                }
                for layer in target_layers:
                    ckpt_data[f'bankrupt_{layer}'] = torch.stack(bankrupt_states[layer]) if bankrupt_states[layer] else torch.tensor([])
                torch.save(ckpt_data, ckpt_path)
                logger.info(f"Checkpoint saved at {i + start_idx + 1} bankrupt samples")

        except Exception as e:
            logger.warning(f"Error processing bankrupt sample {i + start_idx}: {e}")
            continue

    # Extract hidden states for safe games
    logger.info("Extracting hidden states for safe games...")
    safe_states = {layer: [] for layer in target_layers}

    start_idx = 0
    if checkpoint_data and checkpoint_data.get('safe_processed', 0) > 0:
        # Load existing states from checkpoint (tensors are already stacked)
        for layer in target_layers:
            if f'safe_{layer}' in checkpoint_data:
                stacked = checkpoint_data[f'safe_{layer}']
                if stacked.numel() > 0:  # Check not empty
                    safe_states[layer] = [stacked[i] for i in range(stacked.shape[0])]
        start_idx = checkpoint_data.get('safe_processed', 0)
        logger.info(f"Resuming safe extraction from index {start_idx}")

    for i, prompt in enumerate(tqdm(safe_prompts[start_idx:], desc="Safe hidden states", initial=start_idx)):
        try:
            hidden = extractor.extract_hidden_states(prompt, token_position)
            for layer, hs in hidden.items():
                safe_states[layer].append(hs)

            # Checkpoint periodically (use torch.save to avoid memory leak)
            if (i + start_idx + 1) % steering_config.get('checkpoint_frequency', 100) == 0:
                ckpt_path = checkpoint_dir / f'full_checkpoint_{i + start_idx + 1}.pt'
                ckpt_data = {
                    'bankrupt_processed': len(bankrupt_prompts),
                    'safe_processed': i + start_idx + 1
                }
                for layer in target_layers:
                    ckpt_data[f'bankrupt_{layer}'] = torch.stack(bankrupt_states[layer]) if bankrupt_states[layer] else torch.tensor([])
                    ckpt_data[f'safe_{layer}'] = torch.stack(safe_states[layer]) if safe_states[layer] else torch.tensor([])
                torch.save(ckpt_data, ckpt_path)
                logger.info(f"Checkpoint saved at {i + start_idx + 1} safe samples")

        except Exception as e:
            logger.warning(f"Error processing safe sample {i + start_idx}: {e}")
            continue

    # Free model memory
    logger.info("Unloading model to free memory...")
    del model, tokenizer, extractor
    clear_gpu_memory('cuda:0')

    # Compute steering vectors
    logger.info("Computing steering vectors...")
    computer = SteeringVectorComputer(target_layers, logger)
    steering_data = computer.compute(bankrupt_states, safe_states)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Include layer range in filename for parallel runs
    if args.layers:
        layer_suffix = f"_L{min(target_layers)}-{max(target_layers)}"
    else:
        layer_suffix = ""
    output_path = output_dir / f'steering_vectors_{args.model}{layer_suffix}_{timestamp}.npz'

    metadata = {
        'model': args.model,
        'target_layers': target_layers,
        'n_bankrupt_samples': len(bankrupt_prompts),
        'n_safe_samples': len(safe_prompts),
        'timestamp': timestamp,
        'data_source': data_path
    }

    save_steering_vectors(steering_data, output_path, metadata)
    logger.info(f"Steering vectors saved to {output_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("STEERING VECTOR SUMMARY")
    logger.info("=" * 80)
    for layer, data in steering_data.items():
        logger.info(f"Layer {layer}:")
        logger.info(f"  Magnitude: {data['magnitude']:.4f}")
        logger.info(f"  Samples: {data['n_bankrupt']} bankrupt, {data['n_safe']} safe")

    # Also save a human-readable summary
    summary_path = output_dir / f'steering_vectors_{args.model}{layer_suffix}_{timestamp}_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Steering Vector Extraction Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Data source: {data_path}\n")
        f.write(f"Target layers: {target_layers}\n")
        f.write(f"Bankrupt samples: {len(bankrupt_prompts)}\n")
        f.write(f"Safe samples: {len(safe_prompts)}\n")
        f.write(f"\nPer-layer statistics:\n")
        for layer, data in steering_data.items():
            f.write(f"  Layer {layer}: |v| = {data['magnitude']:.4f}\n")

    logger.info(f"Summary saved to {summary_path}")
    logger.info("Extraction complete!")


if __name__ == '__main__':
    main()
