"""
Phase 1: Feature Extraction

Extract hidden states and SAE features from investment choice decision prompts.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.prompt_utils import (
    load_investment_choice_data,
    filter_decisions,
    print_dataset_summary,
    get_prompt_text
)
from src.analysis_utils import clear_gpu_memory


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract SAE features from investment choice decisions."""

    def __init__(self, config: Dict[str, Any], model_name: str, gpu_id: int = 0):
        """
        Initialize feature extractor.

        Args:
            config: Experiment configuration
            model_name: 'gemma' or 'llama'
            gpu_id: GPU device ID (-1 for CPU)
        """
        self.config = config
        self.model_name = model_name
        self.model_config = config['models'][model_name]
        self.phase1_config = config['phase1']

        # Setup device
        if gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Using GPU {gpu_id}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")

        # Paths
        self.output_dir = Path(config['data']['output_dir']) / model_name / 'features'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model and SAE
        self.model = None
        self.tokenizer = None
        self.saes = {}  # Dictionary of SAEs by layer

    def load_model(self):
        """Load the base language model."""
        model_id = self.model_config['model_id']

        logger.info(f"Loading model: {model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            low_cpu_mem_usage=True
        )

        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"Model dtype: {self.model.dtype}")
        logger.info(f"Model device: {self.model.device}")

    def load_sae(self, layer: int):
        """
        Load SAE for a specific layer.

        Args:
            layer: Layer number
        """
        if layer in self.saes:
            return  # Already loaded

        logger.info(f"Loading SAE for layer {layer}")

        sae_release = self.model_config['sae_release']
        model_id = self.model_config['model_id']

        # Format SAE ID based on model
        if self.model_name == 'gemma':
            # Gemma: google/gemma-scope-9b-pt-res-canonical
            sae_id = f"layer_{layer}/width_{self.model_config['sae_width']}/canonical"
        elif self.model_name == 'llama':
            # LLaMA: fnlp/Llama3_1-8B-Base-LXR-8x
            sae_id = f"layer_{layer}"
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Load SAE from HuggingFace
        sae, _, _ = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=str(self.device)
        )

        self.saes[layer] = sae

        logger.info(f"SAE loaded for layer {layer}")
        logger.info(f"SAE shape: {sae.W_enc.shape}")

    def extract_features_for_decision(
        self,
        prompt: str,
        layers: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Extract SAE features for a single decision prompt.

        Args:
            prompt: Decision prompt text
            layers: List of layer numbers to extract

        Returns:
            Dictionary mapping layer -> SAE features [d_sae]
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, d_model]

        # Extract last token hidden state for each layer
        layer_features = {}

        for layer in layers:
            # Get hidden state at this layer
            # hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
            layer_hidden = hidden_states[layer + 1]  # +1 because index 0 is embeddings

            # Get last token
            last_token_hidden = layer_hidden[0, -1, :]  # [d_model]

            # Load SAE if not already loaded
            if layer not in self.saes:
                self.load_sae(layer)

            # Encode through SAE
            sae = self.saes[layer]
            sae_features = sae.encode(last_token_hidden)  # [d_sae]

            # Convert to numpy
            layer_features[layer] = sae_features.cpu().numpy()

        return layer_features

    def process_decisions(
        self,
        decisions: List[Dict[str, Any]],
        target_layers: List[int],
        checkpoint_freq: int = 100
    ):
        """
        Process all decisions and extract features.

        Args:
            decisions: List of decision records
            target_layers: Layers to extract features from
            checkpoint_freq: Save checkpoint every N decisions
        """
        n_decisions = len(decisions)
        n_layers = len(target_layers)
        d_sae = self.model_config['d_sae']

        logger.info(f"Processing {n_decisions} decisions across {n_layers} layers")

        # Initialize storage for each layer
        layer_data = {}
        for layer in target_layers:
            layer_data[layer] = {
                'features': np.zeros((n_decisions, d_sae), dtype=np.float32),
                'choices': np.zeros(n_decisions, dtype=np.int32),
                'game_ids': np.zeros(n_decisions, dtype=np.int32),
                'rounds': np.zeros(n_decisions, dtype=np.int32),
                'prompt_conditions': np.empty(n_decisions, dtype=object),
                'bet_types': np.empty(n_decisions, dtype=object),
                'models': np.empty(n_decisions, dtype=object)
            }

        # Process each decision
        for i, decision in enumerate(tqdm(decisions, desc="Extracting features")):
            try:
                # Get prompt
                prompt = get_prompt_text(decision)

                if not prompt:
                    logger.warning(f"Empty prompt for decision {i}, skipping")
                    continue

                # Extract features for all layers
                features = self.extract_features_for_decision(prompt, target_layers)

                # Store features and metadata
                for layer in target_layers:
                    layer_data[layer]['features'][i] = features[layer]
                    layer_data[layer]['choices'][i] = decision['choice']
                    layer_data[layer]['game_ids'][i] = decision['game_id']
                    layer_data[layer]['rounds'][i] = decision['round']
                    layer_data[layer]['prompt_conditions'][i] = decision['prompt_condition']
                    layer_data[layer]['bet_types'][i] = decision['bet_type']
                    layer_data[layer]['models'][i] = decision['model']

                # Checkpoint
                if (i + 1) % checkpoint_freq == 0:
                    logger.info(f"Processed {i + 1}/{n_decisions} decisions")
                    self.save_checkpoint(layer_data, target_layers, i + 1)

            except Exception as e:
                logger.error(f"Error processing decision {i}: {e}")
                continue

        # Final save
        logger.info(f"Saving final results for {n_decisions} decisions")
        self.save_features(layer_data, target_layers)

    def save_checkpoint(
        self,
        layer_data: Dict[int, Dict[str, np.ndarray]],
        layers: List[int],
        n_processed: int
    ):
        """Save checkpoint."""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        for layer in layers:
            checkpoint_file = checkpoint_dir / f'layer_{layer}_checkpoint_{n_processed}.npz'

            # Save only processed data
            np.savez_compressed(
                checkpoint_file,
                features=layer_data[layer]['features'][:n_processed],
                choices=layer_data[layer]['choices'][:n_processed],
                game_ids=layer_data[layer]['game_ids'][:n_processed],
                rounds=layer_data[layer]['rounds'][:n_processed],
                prompt_conditions=layer_data[layer]['prompt_conditions'][:n_processed],
                bet_types=layer_data[layer]['bet_types'][:n_processed],
                models=layer_data[layer]['models'][:n_processed],
                n_processed=n_processed
            )

        logger.info(f"Checkpoint saved: {n_processed} decisions")

    def save_features(
        self,
        layer_data: Dict[int, Dict[str, np.ndarray]],
        layers: List[int]
    ):
        """Save final features to NPZ files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for layer in layers:
            output_file = self.output_dir / f'layer_{layer}_features.npz'

            np.savez_compressed(
                output_file,
                features=layer_data[layer]['features'],
                choices=layer_data[layer]['choices'],
                game_ids=layer_data[layer]['game_ids'],
                rounds=layer_data[layer]['rounds'],
                prompt_conditions=layer_data[layer]['prompt_conditions'],
                bet_types=layer_data[layer]['bet_types'],
                models=layer_data[layer]['models'],
                layer=layer,
                timestamp=timestamp,
                model_name=self.model_name
            )

            logger.info(f"Saved layer {layer} features to {output_file}")
            logger.info(f"  Shape: {layer_data[layer]['features'].shape}")

        logger.info("All features saved successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Phase 1: Feature Extraction')
    parser.add_argument('--model', type=str, required=True, choices=['gemma', 'llama'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--config', type=str,
                        default='configs/experiment_config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")

    # Load data
    data_dir = config['data']['data_dir']
    experiment_variant = config['phase1']['experiment_variant']

    logger.info(f"Loading data from {data_dir}/{experiment_variant}")

    decisions = load_investment_choice_data(data_dir, experiment_variant)

    # Filter decisions
    include_models = config['phase1']['include_models']
    include_bet_types = config['phase1']['include_bet_types']
    include_conditions = config['phase1']['include_conditions']

    decisions = filter_decisions(
        decisions,
        include_models=include_models if include_models != 'all' else None,
        include_bet_types=include_bet_types if include_bet_types != 'both' else None,
        include_conditions=include_conditions if include_conditions != 'all' else None
    )

    # Print summary
    print_dataset_summary(decisions)

    if len(decisions) == 0:
        logger.error("No decisions to process after filtering")
        return

    # Initialize extractor
    extractor = FeatureExtractor(config, args.model, args.gpu)

    # Load model
    extractor.load_model()

    # Get target layers
    target_layers = config['models'][args.model]['target_layers']
    checkpoint_freq = config['phase1']['checkpoint_frequency']

    logger.info(f"Target layers: {target_layers}")
    logger.info(f"Checkpoint frequency: {checkpoint_freq}")

    # Process decisions
    extractor.process_decisions(decisions, target_layers, checkpoint_freq)

    # Clean up
    clear_gpu_memory()

    logger.info("Phase 1 complete!")


if __name__ == '__main__':
    main()
