#!/usr/bin/env python3
"""
Phase 0: SAE Boost (Residual SAE Training)

Train residual SAE BEFORE feature extraction to reduce reconstruction error.
This must run before Phase 1 so that all downstream analysis uses accurate features.

Method: "Teach Old SAEs New Domain Tricks with Boosting" (arXiv:2507.12990)
1. Extract hidden states from gambling data
2. Compute residual: e = x - x̂ (where x̂ = BaseSAE.decode(BaseSAE.encode(x)))
3. Train Residual SAE to predict: ê = ResidualSAE(x)
4. Save trained Residual SAE for Phase 1

Input:
    - Experiment data JSON (gambling games)

Output:
    - phase0_sae_boost/residual_sae_layer_{L}.pt (trained models)
    - phase0_sae_boost/hidden_states/layer_{L}.npy (extracted hidden states)
    - phase0_sae_boost/boost_metrics.json (training metrics)

Usage:
    python phase0_sae_boost.py --gpu 0
    python phase0_sae_boost.py --gpu 0 --layers 25,30,35 --epochs 10
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_phase_output_dir
from utils import (
    GemmaBaseModel, GemmaSAE,
    load_experiment_data, reconstruct_decision_prompt,
    clear_gpu_memory, logger
)


class ResidualSAE(nn.Module):
    """
    Residual SAE for reducing Base SAE reconstruction error.

    Architecture: JumpReLU-style with learnable threshold
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        init_threshold: float = 0.01
    ):
        super().__init__()

        self.d_model = d_model
        self.d_sae = d_sae

        # Encoder: x -> features
        self.encoder = nn.Linear(d_model, d_sae, bias=True)

        # Decoder: features -> x_hat
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Learnable threshold for JumpReLU
        self.threshold = nn.Parameter(torch.ones(d_sae) * init_threshold)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming."""
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)

        # Decoder initialized to transpose of encoder
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse features using JumpReLU.

        JumpReLU(z) = z * H(z - θ) where H is Heaviside step function
        """
        pre_acts = self.encoder(x)
        # JumpReLU: zero out values below threshold
        features = pre_acts * (pre_acts > self.threshold).float()
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to residual space."""
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with reconstruction and sparsity."""
        pre_acts = self.encoder(x)
        features = pre_acts * (pre_acts > self.threshold).float()
        reconstruction = self.decoder(features)

        return {
            'reconstruction': reconstruction,
            'features': features,
            'pre_acts': pre_acts
        }


class SAEBoostTrainer:
    """Trainer for Residual SAE using SAE Boost method."""

    def __init__(
        self,
        base_sae: GemmaSAE,
        layer: int,
        d_model: int,
        residual_sae_features: int,
        device: str = 'cuda:0',
        learning_rate: float = 1e-4,
        sparsity_lambda: float = 1e-3
    ):
        self.base_sae = base_sae
        self.layer = layer
        self.device = device
        self.sparsity_lambda = sparsity_lambda

        # Create Residual SAE
        self.residual_sae = ResidualSAE(
            d_model=d_model,
            d_sae=residual_sae_features
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.residual_sae.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def compute_base_residual(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute residual from Base SAE: e = x - x̂
        """
        with torch.no_grad():
            features = self.base_sae.encode(hidden_states, self.layer)
            if features is None:
                return None

            reconstruction = self.base_sae.decode(features, self.layer)
            if reconstruction is None:
                return None

            residual = hidden_states - reconstruction

        return residual

    def train_step(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.residual_sae.train()

        residual = self.compute_base_residual(hidden_states)
        if residual is None:
            return None

        output = self.residual_sae(hidden_states)

        # Reconstruction loss: ||e - ê||²
        recon_loss = F.mse_loss(output['reconstruction'], residual)

        # Sparsity loss: L1 on features
        sparsity_loss = torch.mean(torch.abs(output['features']))

        # Total loss
        loss = recon_loss + self.sparsity_lambda * sparsity_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.residual_sae.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            sparsity = (output['features'] > 0).float().mean().item()
            l0 = (output['features'] > 0).float().sum(dim=-1).mean().item()

        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sparsity': sparsity,
            'l0': l0
        }

    def evaluate(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """Evaluate reconstruction improvement."""
        self.residual_sae.eval()

        with torch.no_grad():
            base_features = self.base_sae.encode(hidden_states, self.layer)
            base_recon = self.base_sae.decode(base_features, self.layer)

            if base_recon is None:
                return None

            base_error = F.mse_loss(base_recon, hidden_states).item()
            base_l2 = torch.norm(hidden_states - base_recon, dim=-1).mean().item()

            residual_pred = self.residual_sae(hidden_states)['reconstruction']
            boosted_recon = base_recon + residual_pred

            boosted_error = F.mse_loss(boosted_recon, hidden_states).item()
            boosted_l2 = torch.norm(hidden_states - boosted_recon, dim=-1).mean().item()

            error_reduction = (base_error - boosted_error) / base_error * 100
            l2_reduction = (base_l2 - boosted_l2) / base_l2 * 100

        return {
            'base_mse': base_error,
            'boosted_mse': boosted_error,
            'mse_reduction_pct': error_reduction,
            'base_l2': base_l2,
            'boosted_l2': boosted_l2,
            'l2_reduction_pct': l2_reduction
        }

    def train(
        self,
        hidden_states: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 32,
        eval_frequency: int = 1
    ) -> Dict:
        """Train Residual SAE."""
        tensor_data = torch.from_numpy(hidden_states).float().to(self.device)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {
            'train_loss': [],
            'recon_loss': [],
            'sparsity_loss': [],
            'eval_metrics': []
        }

        for epoch in range(n_epochs):
            epoch_losses = []
            epoch_recon = []
            epoch_sparsity = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch in pbar:
                hidden = batch[0]
                metrics = self.train_step(hidden)
                if metrics is None:
                    continue

                epoch_losses.append(metrics['loss'])
                epoch_recon.append(metrics['recon_loss'])
                epoch_sparsity.append(metrics['sparsity_loss'])

                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'L0': f"{metrics['l0']:.1f}"
                })

            avg_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_loss)
            history['recon_loss'].append(np.mean(epoch_recon))
            history['sparsity_loss'].append(np.mean(epoch_sparsity))

            if (epoch + 1) % eval_frequency == 0:
                eval_metrics = self.evaluate(tensor_data)
                if eval_metrics:
                    history['eval_metrics'].append({
                        'epoch': epoch + 1,
                        **eval_metrics
                    })
                    logger.info(
                        f"  Epoch {epoch+1}: "
                        f"MSE reduction: {eval_metrics['mse_reduction_pct']:.1f}%"
                    )

            self.scheduler.step()
            clear_gpu_memory()

        return history

    def save(self, path: Path):
        """Save trained Residual SAE."""
        torch.save({
            'model_state_dict': self.residual_sae.state_dict(),
            'layer': self.layer,
            'd_model': self.residual_sae.d_model,
            'd_sae': self.residual_sae.d_sae
        }, path)

    @classmethod
    def load_residual_sae(cls, path: Path, device: str = 'cuda:0') -> ResidualSAE:
        """Load trained Residual SAE from file."""
        checkpoint = torch.load(path, map_location=device)
        model = ResidualSAE(
            d_model=checkpoint['d_model'],
            d_sae=checkpoint['d_sae']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


class Phase0SAEBoost:
    """
    Phase 0: Extract hidden states and train Residual SAE.

    This MUST run before Phase 1 to ensure all downstream analysis
    uses Boosted SAE with reduced reconstruction error.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        gpu_id: int = 0,
        target_layers: Optional[List[int]] = None
    ):
        self.config = load_config(config_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        if target_layers:
            self.target_layers = target_layers
        else:
            self.target_layers = self.config.get('target_layers', list(range(20, 42)))

        # Output directory
        self.output_dir = get_phase_output_dir(self.config, 0)

        # Phase 0 config (use phase6 config for backward compatibility)
        phase0_config = self.config.get('phase0', self.config.get('phase6', {}))
        self.residual_features = phase0_config.get('residual_sae_features', 4096)
        self.sparsity_lambda = phase0_config.get('sparsity_lambda', 0.001)
        self.learning_rate = phase0_config.get('learning_rate', 0.0001)
        self.n_epochs = phase0_config.get('n_epochs', 10)
        self.batch_size = phase0_config.get('batch_size', 32)
        self.max_samples = phase0_config.get('max_samples', 1000)

        # Models
        self.model = None
        self.base_sae = None
        self.d_model = self.config.get('model', {}).get('d_model', 3584)

    def extract_hidden_states(self) -> Dict[int, np.ndarray]:
        """
        Extract hidden states from gambling data for all target layers.

        Returns:
            Dict mapping layer -> hidden_states array [n_samples, d_model]
        """
        logger.info("Extracting hidden states from gambling data...")

        # Load experiment data
        data_path = self.config.get('data', {}).get('experiment_data')
        exp_data = load_experiment_data(data_path)
        games = exp_data['results']

        # Limit samples
        if len(games) > self.max_samples:
            np.random.seed(self.config.get('random_seed', 42))
            indices = np.random.choice(len(games), self.max_samples, replace=False)
            games = [games[i] for i in indices]

        logger.info(f"Processing {len(games)} games...")

        # Initialize storage
        layer_states = {layer: [] for layer in self.target_layers}

        # Extract hidden states
        for idx, game in enumerate(tqdm(games, desc="Extracting")):
            try:
                prompt = reconstruct_decision_prompt(game)
                hidden = self.model.get_hidden_states(
                    prompt, self.target_layers, position='last'
                )

                for layer in self.target_layers:
                    if layer in hidden:
                        layer_states[layer].append(
                            hidden[layer].squeeze(0).numpy()
                        )

            except Exception as e:
                logger.warning(f"Error extracting game {idx}: {e}")
                continue

            if (idx + 1) % 100 == 0:
                clear_gpu_memory()

        # Convert to arrays and save
        hidden_dir = self.output_dir / "hidden_states"
        hidden_dir.mkdir(exist_ok=True)

        result = {}
        for layer in self.target_layers:
            if layer_states[layer]:
                arr = np.array(layer_states[layer])
                result[layer] = arr

                # Save hidden states for potential reuse
                np.save(hidden_dir / f"layer_{layer}.npy", arr)
                logger.info(f"  Layer {layer}: {arr.shape}")

        return result

    def run(self):
        """Run Phase 0: SAE Boost training."""
        logger.info("=" * 70)
        logger.info("PHASE 0: SAE BOOST (RESIDUAL SAE TRAINING)")
        logger.info("=" * 70)
        logger.info(f"Target layers: {self.target_layers}")
        logger.info(f"Residual SAE features: {self.residual_features}")
        logger.info(f"Sparsity lambda: {self.sparsity_lambda}")
        logger.info(f"Epochs: {self.n_epochs}")
        logger.info(f"Max samples: {self.max_samples}")

        # Load models
        logger.info("\nLoading Gemma model...")
        self.model = GemmaBaseModel(device=self.device)
        self.model.load()

        logger.info("Loading Base SAE...")
        sae_config = self.config.get('sae', {})
        self.base_sae = GemmaSAE(
            device=self.device,
            width=sae_config.get('width', '16k')
        )

        # Extract hidden states
        hidden_states = self.extract_hidden_states()

        if not hidden_states:
            logger.error("No hidden states extracted!")
            return

        # Train Residual SAE for each layer
        all_metrics = {}

        for layer in self.target_layers:
            if layer not in hidden_states:
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"Training Residual SAE for Layer {layer}")
            logger.info("=" * 50)

            h_states = hidden_states[layer]
            logger.info(f"  Samples: {len(h_states)}")

            # Create trainer
            trainer = SAEBoostTrainer(
                base_sae=self.base_sae,
                layer=layer,
                d_model=self.d_model,
                residual_sae_features=self.residual_features,
                device=self.device,
                learning_rate=self.learning_rate,
                sparsity_lambda=self.sparsity_lambda
            )

            # Evaluate before training
            h_tensor = torch.from_numpy(h_states).float().to(self.device)
            before_metrics = trainer.evaluate(h_tensor)
            if before_metrics:
                logger.info(f"  Before: MSE={before_metrics['base_mse']:.6f}")

            # Train
            history = trainer.train(
                h_states,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size
            )

            # Evaluate after training
            after_metrics = trainer.evaluate(h_tensor)
            if after_metrics:
                logger.info(f"  After:  MSE={after_metrics['boosted_mse']:.6f}")
                logger.info(f"  Reduction: {after_metrics['mse_reduction_pct']:.1f}%")

            # Save model
            model_path = self.output_dir / f"residual_sae_layer_{layer}.pt"
            trainer.save(model_path)
            logger.info(f"  Saved: {model_path}")

            # Store metrics
            all_metrics[layer] = {
                'before': before_metrics,
                'after': after_metrics,
                'n_samples': len(h_states),
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None
            }

            del trainer, h_tensor
            clear_gpu_memory()

        # Save metrics
        self._save_metrics(all_metrics)

        logger.info("\n" + "=" * 70)
        logger.info("PHASE 0 COMPLETE")
        logger.info("=" * 70)
        logger.info("Residual SAEs are ready. Run Phase 1 with --use-boost flag.")

    def _save_metrics(self, all_metrics: Dict):
        """Save boost metrics."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        output = {
            'timestamp': timestamp,
            'config': {
                'residual_features': self.residual_features,
                'sparsity_lambda': self.sparsity_lambda,
                'n_epochs': self.n_epochs,
                'max_samples': self.max_samples
            },
            'layers': {}
        }

        total_reduction = []

        for layer, metrics in all_metrics.items():
            layer_data = {
                'n_samples': metrics['n_samples'],
                'base_mse': metrics['before']['base_mse'] if metrics['before'] else None,
                'boosted_mse': metrics['after']['boosted_mse'] if metrics['after'] else None,
                'mse_reduction_pct': metrics['after']['mse_reduction_pct'] if metrics['after'] else None
            }
            output['layers'][str(layer)] = layer_data

            if metrics['after']:
                total_reduction.append(metrics['after']['mse_reduction_pct'])

        output['summary'] = {
            'n_layers': len(all_metrics),
            'avg_mse_reduction_pct': float(np.mean(total_reduction)) if total_reduction else 0,
            'max_mse_reduction_pct': float(np.max(total_reduction)) if total_reduction else 0,
            'min_mse_reduction_pct': float(np.min(total_reduction)) if total_reduction else 0
        }

        metrics_file = self.output_dir / "boost_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved: {metrics_file}")

        logger.info(f"\nAverage MSE reduction: {output['summary']['avg_mse_reduction_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Phase 0: SAE Boost')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers')
    parser.add_argument('--epochs', type=int, default=None, help='Training epochs')
    parser.add_argument('--samples', type=int, default=None, help='Max samples')

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    phase0 = Phase0SAEBoost(
        config_path=args.config,
        gpu_id=args.gpu,
        target_layers=target_layers
    )

    if args.epochs:
        phase0.n_epochs = args.epochs
    if args.samples:
        phase0.max_samples = args.samples

    phase0.run()


if __name__ == '__main__':
    main()
