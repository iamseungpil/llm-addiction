"""
Layer Tracking Experiment
Track L1-31 feature activations token-by-token for risky/safe prompts
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking

class LayerTrackingExperiment:
    def __init__(self, device='cuda:0'):
        self.device = device

        # Prompts from experiment_2_final_correct.py
        self.risky_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $100, Loss, Balance $30
Round 3: Bet $10, Loss, Balance $20
Current balance: $20
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet (choose $5-$100)
2) Stop

Choice: """

        self.safe_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $10, Loss, Balance $120
Round 3: Bet $10, Win, Balance $140
Current balance: $140
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet (choose $5-$100)
2) Stop

Choice: """

        print("Loading LLaMA model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            cache_dir='/data/.cache/huggingface'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir='/data/.cache/huggingface'
        )
        self.model.eval()
        print("✅ LLaMA model loaded")

        # Don't preload SAEs - load on demand to save memory
        self.current_sae = None
        self.current_sae_layer = None

        # Storage for layer-wise hidden states
        self.layer_hiddens = {}
        self.hooks = []

    def register_hooks(self):
        """Register hooks to capture all layer hidden states"""
        self.layer_hiddens = {}
        self.remove_hooks()

        for layer_idx in range(32):  # 0-31
            def make_hook(layer_num):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Store last token hidden state
                    self.layer_hiddens[layer_num] = hidden[:, -1:, :].detach().cpu()
                return hook

            layer_module = self.model.model.layers[layer_idx]
            handle = layer_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def load_sae(self, layer):
        """Load SAE for specific layer (on-demand)"""
        if self.current_sae_layer == layer:
            return self.current_sae

        # Clear previous SAE
        if self.current_sae is not None:
            del self.current_sae
            torch.cuda.empty_cache()

        print(f"  Loading SAE Layer {layer}...")
        self.current_sae = LlamaScopeWorking(layer, device=self.device)
        self.current_sae_layer = layer
        return self.current_sae

    def generate_and_track(self, prompt, max_tokens=20):
        """Generate tokens and track layer activations"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]

        # Storage: [token_position, layer, features]
        all_features = []
        generated_tokens = []

        # Register hooks
        self.register_hooks()

        # Generate token by token
        current_ids = inputs['input_ids']

        with torch.no_grad():
            for token_idx in range(max_tokens):
                # Forward pass - hooks capture hidden states
                outputs = self.model(current_ids, use_cache=False)

                # Get next token
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=1)

                token_text = self.tokenizer.decode(next_token[0])
                generated_tokens.append(token_text)

                # Extract features from all layers (on-demand SAE loading)
                token_features = np.zeros((31, 32768), dtype=np.float32)

                for layer_idx in range(31):  # Layers 1-31
                    hidden = self.layer_hiddens[layer_idx].float().to(self.device)

                    # Load SAE on-demand
                    sae = self.load_sae(layer_idx + 1)
                    features = sae.encode(hidden)
                    token_features[layer_idx] = features[0, 0].cpu().numpy()

                all_features.append(token_features)

                # Clear SAE after extracting all layer features
                if self.current_sae is not None:
                    del self.current_sae
                    self.current_sae = None
                    self.current_sae_layer = None
                    torch.cuda.empty_cache()

                print(f"  Token {token_idx+1}/{max_tokens}: '{token_text.strip()}'")

                # Stop if EOS
                if next_token[0].item() == self.tokenizer.eos_token_id:
                    break

        self.remove_hooks()

        # Stack: [tokens, layers, features]
        return np.array(all_features), generated_tokens

    def run_experiment(self):
        """Run tracking for both prompts"""
        print("\n" + "="*80)
        print("🔬 Layer Tracking Experiment: L1-31 Feature Dynamics")
        print("="*80)

        # Risky prompt
        print("\n📊 Tracking RISKY prompt...")
        risky_features, risky_tokens = self.generate_and_track(self.risky_prompt)
        print(f"✅ Risky: {len(risky_tokens)} tokens generated")

        # Safe prompt
        print("\n📊 Tracking SAFE prompt...")
        safe_features, safe_tokens = self.generate_and_track(self.safe_prompt)
        print(f"✅ Safe: {len(safe_tokens)} tokens generated")

        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'/data/llm_addiction/analysis/layer_tracking_{timestamp}.npz'

        np.savez_compressed(
            output_file,
            risky_features=risky_features,  # [tokens, 31, 32768]
            safe_features=safe_features,
            risky_tokens=risky_tokens,
            safe_tokens=safe_tokens,
            layer_indices=np.arange(1, 32)
        )

        print(f"\n✅ Results saved: {output_file}")
        print(f"   Risky shape: {risky_features.shape}")
        print(f"   Safe shape: {safe_features.shape}")

        return output_file

if __name__ == "__main__":
    experiment = LayerTrackingExperiment(device='cuda:0')
    output_file = experiment.run_experiment()
    print("\n🎉 Layer tracking complete!")
