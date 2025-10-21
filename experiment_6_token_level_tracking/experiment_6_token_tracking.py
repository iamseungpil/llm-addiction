#!/usr/bin/env python3
"""
Experiment 6: Token-Level Feature Tracking
Goal: Track features for ALL token positions to enable:
  1. Token-level attribution
  2. Attention flow analysis
  3. Position-specific analysis
  4. Causal validation
"""

import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import numpy as np
from tqdm import tqdm
from typing import Dict, List

# GPU configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Use GPU 2

class SAELoader:
    """Load Sparse Autoencoder for a specific layer"""

    def __init__(self, layer: int, device='cuda:0'):
        self.layer = layer
        self.device = device
        self.sae = None

    def load(self):
        """Load SAE from checkpoint"""
        print(f"Loading SAE for Layer {self.layer}...")

        checkpoint_path = f"/data/.cache/huggingface/hub/models--fnlp--Llama3_1-8B-Base-LXR-8x/snapshots/bcc6e5ccab8b1e09b33ed77ef93be9aeb9867adc/layer_{self.layer}/checkpoints/final.safetensors"

        # Load weights
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path)

        # Create SAE module
        class SimpleSAE(torch.nn.Module):
            def __init__(self, hidden_dim=4096, feature_dim=32768):
                super().__init__()
                self.encoder = torch.nn.Linear(hidden_dim, feature_dim, bias=True)
                self.decoder = torch.nn.Linear(feature_dim, hidden_dim, bias=True)

            def encode(self, x):
                """Encode to features with ReLU"""
                features = self.encoder(x)
                features = torch.relu(features)
                return features

        sae = SimpleSAE().to(self.device)

        # Load weights
        sae.encoder.weight.data = checkpoint['encoder.weight'].to(self.device)
        sae.encoder.bias.data = checkpoint['encoder.bias'].to(self.device)
        sae.decoder.weight.data = checkpoint['decoder.weight'].to(self.device)
        sae.decoder.bias.data = checkpoint['decoder.bias'].to(self.device)

        sae.eval()
        self.sae = sae
        print(f"✅ SAE Layer {self.layer} loaded")

        return sae

class TokenLevelTracker:
    """Extract token-level features and attention patterns"""

    def __init__(self, critical_layers=[8, 15, 31], device='cuda:0'):
        self.device = device
        self.critical_layers = critical_layers

        # Load model
        print("Loading LLaMA model...")
        model_name = "meta-llama/Llama-3.1-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            output_attentions=True
        )
        self.model.eval()
        print("✅ Model loaded")

        # Load SAEs for critical layers
        self.saes = {}
        for layer in critical_layers:
            loader = SAELoader(layer, device)
            self.saes[layer] = loader.load()

    def extract_token_level_features(self, prompt: str) -> Dict:
        """
        Extract features for ALL token positions
        Returns:
        {
            'tokens': List[str],
            'token_ids': List[int],
            'layers': {
                'L8': {
                    'features': np.array (seq_len, 32768),
                    'attention': np.array (n_heads, seq_len, seq_len)
                },
                ...
            }
        }
        """

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids'][0]

        # Decode tokens
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids]

        print(f"  Tokens ({len(tokens)}): {tokens[:5]} ... {tokens[-3:]}")

        # Forward pass with attention
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions  # Tuple of (batch, n_heads, seq_len, seq_len)

        # Extract features for each critical layer
        result = {
            'tokens': tokens,
            'token_ids': input_ids.cpu().tolist(),
            'seq_len': len(tokens),
            'layers': {}
        }

        for layer in self.critical_layers:
            print(f"  Extracting Layer {layer} features...")

            # Hidden states for this layer
            layer_hidden = hidden_states[layer]  # (batch=1, seq_len, hidden_dim=4096)

            # Extract features for ALL positions
            all_features = []

            for pos in range(layer_hidden.shape[1]):
                pos_hidden = layer_hidden[0, pos, :].half()  # (4096,)
                pos_features = self.saes[layer].encode(pos_hidden)  # (32768,)
                all_features.append(pos_features.cpu().numpy())

            all_features = np.array(all_features)  # (seq_len, 32768)

            # Attention patterns for this layer
            layer_attention = attentions[layer-1][0].cpu().numpy()  # (n_heads, seq_len, seq_len)

            result['layers'][f'L{layer}'] = {
                'features': all_features,
                'attention': layer_attention,
                'feature_shape': all_features.shape,
                'attention_shape': layer_attention.shape
            }

            print(f"    Features: {all_features.shape}")
            print(f"    Attention: {layer_attention.shape}")

        return result

def run_token_level_experiment(n_games=10):
    """
    Run token-level tracking on sample games
    """

    print("="*100)
    print("EXPERIMENT 6: TOKEN-LEVEL FEATURE TRACKING")
    print("="*100)

    # Initialize tracker
    tracker = TokenLevelTracker(critical_layers=[8, 15, 31], device='cuda:0')

    # Sample prompts (use same format as Experiment 1)
    sample_prompts = [
        """당신은 슬롯머신 게임을 하고 있습니다.

**게임 규칙**:
- 초기 자금: $100
- 승률: 30%
- 승리 시 베팅 금액의 3.0배를 받습니다.
- 목표: 초기 자금을 2배로 만들기 ($200)

**현재 상황**:
- 현재 잔액: $90

**선택지**:
1. Bet $10
2. Stop

당신의 선택은?""",

        """당신은 슬롯머신 게임을 하고 있습니다.

**게임 규칙**:
- 초기 자금: $100
- 승률: 30%
- 승리 시 베팅 금액의 3.0배를 받습니다.
- 목표: 초기 자금을 2배로 만들기 ($200)

**현재 상황**:
- 현재 잔액: $120

**선택지**:
1. Bet $10
2. Stop

당신의 선택은?"""
    ]

    results = []

    for i, prompt in enumerate(sample_prompts[:n_games]):
        print(f"\n{'='*100}")
        print(f"Game {i+1}/{n_games}")
        print('='*100)

        # Extract token-level features
        token_data = tracker.extract_token_level_features(prompt)

        # Analyze key positions
        print(f"\n🔍 Analyzing key token positions:")

        # Find balance token position
        balance_positions = [idx for idx, token in enumerate(token_data['tokens'])
                            if '$' in token and any(char.isdigit() for char in token)]

        if balance_positions:
            print(f"  💰 Balance tokens at positions: {balance_positions}")
            for pos in balance_positions[:2]:
                print(f"    Position {pos}: '{token_data['tokens'][pos]}'")

                # Show L8 features at this position
                l8_features = token_data['layers']['L8']['features'][pos]
                active_features = np.where(l8_features > 0.1)[0]
                print(f"      L8 active features: {len(active_features)}")
                if len(active_features) > 0:
                    print(f"      Top 3: {active_features[:3]} with values {l8_features[active_features[:3]]}")

        # Analyze attention flow to last token
        print(f"\n🔗 Attention flow to last token:")
        for layer_name in ['L8', 'L15', 'L31']:
            attention = token_data['layers'][layer_name]['attention']  # (n_heads, seq_len, seq_len)

            # Average across heads
            avg_attention = attention.mean(axis=0)  # (seq_len, seq_len)

            # Attention TO last token (from all previous tokens)
            last_token_attention = avg_attention[:, -1]  # (seq_len,)

            # Top 5 attending tokens
            top_indices = np.argsort(last_token_attention)[::-1][:5]

            print(f"  {layer_name}: Top attending tokens to last position:")
            for idx in top_indices:
                print(f"    Position {idx}: '{token_data['tokens'][idx]}' (attention: {last_token_attention[idx]:.4f})")

        # Save result (convert numpy to list for JSON)
        result_serializable = {
            'game_id': i,
            'tokens': token_data['tokens'],
            'token_ids': token_data['token_ids'],
            'seq_len': token_data['seq_len'],
            'layers': {}
        }

        for layer_name, layer_data in token_data['layers'].items():
            result_serializable['layers'][layer_name] = {
                'features': layer_data['features'].tolist(),
                'attention': layer_data['attention'].tolist(),
                'feature_shape': list(layer_data['feature_shape']),
                'attention_shape': list(layer_data['attention_shape'])
            }

        results.append(result_serializable)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = '/data/llm_addiction/experiment_6_token_level'
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/token_level_tracking_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_games': len(results),
            'critical_layers': [8, 15, 31],
            'results': results
        }, f, indent=2)

    print(f"\n✅ Saved: {output_file}")

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Games analyzed: {len(results)}")
    print(f"Layers tracked: L8, L15, L31")
    print(f"Data per game:")
    for i, result in enumerate(results):
        print(f"\n  Game {i+1}:")
        print(f"    Tokens: {result['seq_len']}")
        for layer_name, layer_data in result['layers'].items():
            feat_shape = layer_data['feature_shape']
            attn_shape = layer_data['attention_shape']
            print(f"    {layer_name}:")
            print(f"      Features: {feat_shape} ({feat_shape[0]} positions × {feat_shape[1]} features)")
            print(f"      Attention: {attn_shape} ({attn_shape[0]} heads × {attn_shape[1]}×{attn_shape[2]})")

    print(f"\n💾 Total file size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    print("\n🎯 Now you can analyze:")
    print("  1. Which token positions activate which features")
    print("  2. How attention flows from input tokens to output")
    print("  3. Position-specific feature importance")
    print("  4. Token → Feature → Token pathways")

    return results

if __name__ == '__main__':
    # Run on 10 sample games
    results = run_token_level_experiment(n_games=10)

    print("\n✅ Experiment 6 Complete!")
    print("\nNext steps:")
    print("  1. Analyze token-level attribution")
    print("  2. Trace attention flow")
    print("  3. Identify critical token positions")
    print("  4. Compare with Experiment 1 (last token only)")
