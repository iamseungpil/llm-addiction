"""Extract compliance probe directions in hidden-state space.

For each (model, direction) in the 3-direction compliance battery:
  1. Load the model in bf16.
  2. Run forward passes on 50 pos + 50 neg prompts.
  3. Take the residual-stream hidden state at layer L (default 22) at the
     last input token (the position that immediately precedes the model's
     would-be next-token prediction — no sampling required).
  4. Compute mean-difference direction:
        d = mean(h_pos) − mean(h_neg)             (∈ R^d_model)
  5. Save as NPZ to <output.directions_subdir>/direction_{model}_{name}.npz.

Notes / design constraints:
  - bf16 model loading per CLAUDE.md.
  - Operates entirely in the residual-stream hidden-state space (the same
    space the cached SAE features were extracted from). No SAE encoding here.
  - We never decode tokens; we only need a single forward pass to read out
    the last-token activation. max_new_tokens is therefore set to 0 (just
    the prompt forward pass).
  - Plan v4 §3.2 calls for 100 prompts per side; the YAML default is
    n_filler_per_side=50 (50 pos + 50 neg = 100 forward passes per direction
    pair, matching plan total of 600 generations across 6 dir×model combos).

Reuse: matches the hidden-state coordinate system of
sae_v3_analysis/.../hidden_states_dp.npz so projection there is in the
same basis as the cached features.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "m5_config.yaml"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5.extract_directions")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_prompt_file(path: Path) -> List[str]:
    """One prompt per line. Empty/whitespace-only lines are skipped."""
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _resolve_prompt_path(rel: str, root: Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / rel)


def _load_model_and_tokenizer(model_cfg: dict, dtype: str, device: str):
    """Load HF model in bf16. Imported lazily so pytest collect-only does not
    require torch/transformers to be present."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    name = model_cfg["name_hf"]
    log.info("loading model %s in %s on %s", name, dtype, device)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
        device_map=device,
    )
    mdl.eval()
    return mdl, tok


def _last_token_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Returns (n_prompts, d_model) array of layer-`layer` residual-stream
    activations at the last *input* token of each prompt.

    Uses left-padding so the last input token sits at position [-1] for
    every batched item. Hidden states are cast to float32 for stable
    downstream linear algebra.
    """
    import torch

    tokenizer.padding_side = "left"
    out_chunks: List[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            out = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
            )
        # hidden_states is a tuple of length n_layers+1: index 0=embeddings,
        # index L+1 = residual stream after transformer block L. Canonical
        # parity with sae_v3_analysis/src/extract_all_rounds.py:488, which
        # uses outputs.hidden_states[layer + 1] to read "after block L".
        h_layer = out.hidden_states[layer + 1]  # (B, T, d_model)
        last = h_layer[:, -1, :].to(torch.float32).cpu().numpy()
        out_chunks.append(last)
    return np.concatenate(out_chunks, axis=0)


def compute_direction(h_pos: np.ndarray, h_neg: np.ndarray) -> np.ndarray:
    """Mean-difference direction in R^d_model.

    d = mean(h_pos, axis=0) − mean(h_neg, axis=0).
    Persona Vectors 2025 (arXiv:2507.21509) calls this the contrastive
    activation direction; its magnitude encodes effect size, but we only
    care about its 1-d span for orthogonal projection so normalisation is
    applied at projection time.
    """
    if h_pos.ndim != 2 or h_neg.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got {h_pos.shape} and {h_neg.shape}")
    if h_pos.shape[1] != h_neg.shape[1]:
        raise ValueError(
            f"d_model mismatch: pos has {h_pos.shape[1]}, neg has {h_neg.shape[1]}"
        )
    return h_pos.mean(axis=0) - h_neg.mean(axis=0)


def save_direction(
    out_path: Path,
    direction: np.ndarray,
    h_pos: np.ndarray,
    h_neg: np.ndarray,
    metadata: Dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        direction=direction.astype(np.float32),
        mean_pos=h_pos.mean(axis=0).astype(np.float32),
        mean_neg=h_neg.mean(axis=0).astype(np.float32),
        h_pos=h_pos.astype(np.float32),
        h_neg=h_neg.astype(np.float32),
    )
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("saved direction %s (||d||=%.4f)", out_path, float(np.linalg.norm(direction)))


def run_for_one_model(cfg: dict, model_key: str, prompt_root: Path, out_dir: Path) -> Dict:
    """Iterate the three probe directions for one model. Returns summary dict."""
    model_cfg = cfg["models"][model_key]
    layer = int(cfg["layer"])
    gen = cfg.get("generation", {})
    device = gen.get("device", "cuda")
    dtype = gen.get("dtype", "bfloat16")
    batch_size = int(gen.get("batch_size", 8))

    model, tok = _load_model_and_tokenizer(model_cfg, dtype, device)

    summary: Dict = {"model": model_key, "layer": layer, "directions": {}}
    try:
        for dname, dspec in cfg["probe_directions"].items():
            pos_path = _resolve_prompt_path(dspec["pos_file"], ROOT)
            neg_path = _resolve_prompt_path(dspec["neg_file"], ROOT)
            pos_prompts = read_prompt_file(pos_path)
            neg_prompts = read_prompt_file(neg_path)
            log.info(
                "[%s] direction %s: %d pos / %d neg prompts",
                model_key, dname, len(pos_prompts), len(neg_prompts),
            )
            h_pos = _last_token_hidden_states(model, tok, pos_prompts, layer, device, batch_size)
            h_neg = _last_token_hidden_states(model, tok, neg_prompts, layer, device, batch_size)
            d = compute_direction(h_pos, h_neg)

            out_path = out_dir / f"direction_{model_key}_{dname}.npz"
            md = {
                "model": model_key,
                "direction_name": dname,
                "description": dspec.get("description", ""),
                "layer": layer,
                "n_pos": int(h_pos.shape[0]),
                "n_neg": int(h_neg.shape[0]),
                "d_model": int(h_pos.shape[1]),
                "norm": float(np.linalg.norm(d)),
                "pos_file": str(pos_path),
                "neg_file": str(neg_path),
            }
            save_direction(out_path, d, h_pos, h_neg, md)
            summary["directions"][dname] = md
    finally:
        # Free GPU memory regardless of success/error.
        try:
            import torch
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    return summary


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M5 — extract compliance probe directions")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--model", choices=["gemma", "llama", "all"], default="all",
        help="restrict to a single model or run all",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="override config.output.root",
    )
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    out_root = Path(args.output or cfg["output"]["root"])
    dir_subdir = cfg["output"].get("directions_subdir", "directions")
    out_dir = out_root / dir_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(cfg.get("seeds", {}).get("numpy", 42)))

    models = ["gemma", "llama"] if args.model == "all" else [args.model]
    summaries = {}
    for mk in models:
        summaries[mk] = run_for_one_model(cfg, mk, ROOT / "prompts", out_dir)
    summary_path = out_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    log.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
