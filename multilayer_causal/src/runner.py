"""Trial loop: one arm = N single-decision trials (M3'' protocol, multi-layer)."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime

import numpy as np

from .checkpoint import ArmCheckpoint, VectorStore
from .hooks import (MultiLayerPatcher, MultiLayerSteerer, SubspacePatcher,
                    cache_layer_outputs)
from .prompts import build_prompt, parse_response
from .states import ensure_sm_catalog, load_minusG_states

MODEL_PATH = "google/gemma-2-9b-it"
N_LAYERS, D_MODEL = 42, 3584
TEMPERATURE, MAX_NEW_TOKENS = 0.7, 200
SEED_BASE = 42


def load_model(gpu: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{gpu}"}, token=os.environ.get("HF_TOKEN"))
    model.eval()
    assert len(model.model.layers) == N_LAYERS, "unexpected layer count"
    assert model.config.hidden_size == D_MODEL, "unexpected hidden size"
    return model, tok, f"cuda:{gpu}"


def _generate(model, tok, device, prompt, hookset, seed):
    import torch
    torch.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(device)
    if hookset is not None:
        hookset.install(model)
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=TEMPERATURE, do_sample=True,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
    finally:
        if hookset is not None:
            hookset.remove()


def _load_steer_assets(arm):
    """directions npz: keys 'directions' (L,D), 'scales' (L,) — built by analyze.py.

    direction: random → isotropic unit vectors (control), same per-layer scales.
    """
    import torch
    z = np.load(arm["directions_npz"])
    scales = {li: float(z["scales"][li]) for li in arm["layers"]}
    if arm.get("direction") == "random":
        rng = np.random.Generator(np.random.PCG64(int(arm["dir_seed"])))
        dirs = {}
        for li in arm["layers"]:
            v = rng.standard_normal(D_MODEL)
            dirs[li] = torch.tensor(v / np.linalg.norm(v), dtype=torch.float32)
        return dirs, scales
    dirs = {li: torch.tensor(z["directions"][li], dtype=torch.float32)
            for li in arm["layers"]}
    return dirs, scales


def _load_bases(arm):
    import torch
    if arm.get("basis", "pca") == "random":
        rng = np.random.Generator(np.random.PCG64(int(arm["basis_seed"])))
        bases = {}
        for li in arm["layers"]:
            g = rng.standard_normal((D_MODEL, int(arm["r"])))
            q, _ = np.linalg.qr(g)
            bases[li] = torch.tensor(q, dtype=torch.float32)
        return bases
    z = np.load(arm["basis_npz"])
    return {li: torch.tensor(z[f"L{li}"][:, :int(arm["r"])], dtype=torch.float32)
            for li in arm["layers"]}


def run_arm(arm, out_dir, gpu=0, n=None, smoke=False):
    n = n or arm["n"]
    if smoke:
        n = min(n, 3)
    phase = arm["phase"]
    ck = ArmCheckpoint(phase, arm["id"], out_dir,
                       hf_enabled=False if smoke else None)
    vec = None
    if arm.get("log_vectors"):
        vec = VectorStore(ck.path.with_name(f"{arm['id']}_vectors.npz"),
                          N_LAYERS, D_MODEL)
    ensure_sm_catalog(arm["model"])
    states = load_minusG_states(arm["model"], n=n + 50)
    assert states, "empty state pool"
    model, tok, device = load_model(gpu)
    done = ck.done_seeds()
    print(f"[{arm['id']}] n={n} done={len(done)} pool={len(states)}", flush=True)

    for i in range(n):
        seed = SEED_BASE + i * 997
        if seed in done:
            continue
        t0 = time.time()
        game, round_idx = states[i % len(states)]
        base_combo = game.get("prompt_combo", "")
        plus_combo = base_combo + "G" if "G" not in base_combo else base_combo
        minus_p = build_prompt(game, round_idx, override_combo=base_combo)
        plus_p = build_prompt(game, round_idx, override_combo=plus_combo)
        if minus_p is None or plus_p is None:
            continue

        mode = arm["mode"]
        eval_p, hookset = minus_p, None
        if mode == "anchor_plus":
            eval_p = plus_p
        elif mode == "patch":
            cached, _ = cache_layer_outputs(model, tok, device, plus_p, arm["layers"])
            hookset = MultiLayerPatcher(cached)
            if vec is not None and seed not in vec.seeds:
                cached_m, _ = cache_layer_outputs(model, tok, device, minus_p,
                                                  list(range(N_LAYERS)))
                vec.append(seed,
                           np.stack([cached_m[l][-1].float().cpu().numpy()
                                     for l in range(N_LAYERS)]),
                           np.stack([cached[l][-1].float().cpu().numpy()
                                     for l in range(N_LAYERS)]))
        elif mode == "subspace":
            cached, _ = cache_layer_outputs(model, tok, device, plus_p, arm["layers"])
            hookset = SubspacePatcher(cached, _load_bases(arm))
        elif mode == "steer":
            dirs, scales = _load_steer_assets(arm)
            hookset = MultiLayerSteerer(dirs, scales, alpha=float(arm["alpha"]))
        elif mode != "anchor_minus":
            raise ValueError(f"unknown mode {mode}")

        try:
            text = _generate(model, tok, device, eval_p, hookset, seed)
        except Exception as e:
            print(f"[{arm['id']} trial {i}] ERROR {type(e).__name__}: {e}", flush=True)
            continue

        bal_m = re.search(r"Current balance:\s*\$(\d+)", eval_p)
        balance = int(bal_m.group(1)) if bal_m else 100
        max_bet = min(100, balance)
        action, amount = parse_response(text, max_bet)
        ck.record({
            "trial_id": i, "seed": seed, "arm": arm["id"], "phase": phase,
            "mode": mode, "layers": arm.get("layers"),
            "r": arm.get("r"), "alpha": arm.get("alpha"),
            "source_state": {"prompt_combo": base_combo,
                             "bet_type": game.get("bet_type"),
                             "round_idx": round_idx, "balance": balance},
            "action": action, "amount": amount,
            "bet_ratio": amount / max(balance, 1) if action == "bet" else 0.0,
            "extreme": bool(action == "bet" and amount >= max_bet),
            "parse_ok": bool(re.search(r"Final Decision:", text, re.IGNORECASE)),
            "response": text[:300], "elapsed_s": round(time.time() - t0, 1),
            "timestamp": datetime.now().isoformat(),
        })
        if vec is not None:
            vec.save()
        print(f"  [{arm['id']} {i + 1}/{n}] {action} ${amount} "
              f"({time.time() - t0:.1f}s)", flush=True)
    ck.final_sync()
    print(f"[{arm['id']}] DONE {len(ck.done_seeds())} trials", flush=True)


def full_layer_list():
    return list(range(N_LAYERS))
