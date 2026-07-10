"""GPU-free dry-run of run_arm with a LLaMA-shaped mock (32 layers, 4096 dim).

Mirrors tests/test_dryrun.py's fake-model pattern but with llama geometry to
prove the runner derives n_layers/d_model from the LOADED model config
instead of the frozen gemma constants (42/3584): the patch path's VectorStore
stacks must come out (n, 32, 4096), the anchor path must run off the
llama_v4_role catalog, and steer log_vectors must record the L22 projection
manipulation check.
"""
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from multilayer_causal.src import runner

NL, D = 32, 4096  # meta-llama/Llama-3.1-8B-Instruct geometry
PATCH_LAYERS = list(range(12, 16))  # depth-matched window candidate (PR-2)


class _Batch(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0

    def __init__(self, default="I will bet. Final Decision: Bet $20"):
        self.default = default
        self.queue = []

    def __call__(self, prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        ids = [(ord(c) % 97) + 3 for c in prompt[::4]] or [5, 6]
        t = torch.tensor([ids], dtype=torch.long)
        return _Batch(input_ids=t, attention_mask=torch.ones_like(t))

    def decode(self, ids, skip_special_tokens=True):
        return self.queue.pop(0) if self.queue else self.default


class _IdLayer(nn.Module):
    def forward(self, x):
        return (x,)


class FakeLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_IdLayer() for _ in range(NL))
        self.config = types.SimpleNamespace(hidden_size=D)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = torch.zeros(1, input_ids.shape[1], D)
        for layer in self.model.layers:
            x = layer(x)[0]
        return x

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.forward(input_ids=input_ids)
        new = torch.full((1, 3), 11, dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


def _write_llama_catalog(root: Path):
    for model in ("llama", "gemma"):
        sm = root / "slot_machine" / f"{model}_v4_role"
        sm.mkdir(parents=True)
        games = []
        for combo in ["BASE", "M", "MH"]:  # variable, no G → all enter the pool
            decs = [{"balance_before": 100 - 2 * r} for r in range(9)]
            hist = [{"round": r + 1, "bet": 10, "win": r % 2 == 0,
                     "result": "W" if r % 2 == 0 else "L",
                     "balance": 100 - 2 * (r + 1)} for r in range(8)]
            games.append({"bet_type": "variable", "prompt_combo": combo,
                          "decisions": decs, "history": hist})
        (sm / f"final_{model}_dry.json").write_text(
            json.dumps({"results": games}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    _write_llama_catalog(tmp_path / "behavioral")
    monkeypatch.setenv("LLM_ADDICTION_BEHAVIORAL_ROOT",
                       str(tmp_path / "behavioral"))
    tok, model, load_calls = FakeTok(), FakeLlama(), []

    def fake_load(gpu, model_name="gemma"):
        load_calls.append(model_name)
        return model, tok, "cpu"

    monkeypatch.setattr(runner, "load_model", fake_load)
    return types.SimpleNamespace(tok=tok, model=model, out=tmp_path / "out",
                                 mp=monkeypatch, load_calls=load_calls)


def _arm(**kw):
    a = {"model": "llama", "task": "sm", "n": 1, "phase": "dryl"}
    a.update(kw)
    return a


def _records(out, arm_id):
    p = Path(out) / f"{arm_id}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_llama_patch_arm_uses_config_dims(env):
    spied = []
    real = runner.MultiLayerPatcher

    class SpyPatcher(real):
        def __init__(self, cached):
            super().__init__(cached)
            spied.append(self)

    env.mp.setattr(runner, "MultiLayerPatcher", SpyPatcher)
    runner.run_arm(_arm(id="dryl_patch", mode="patch", layers=PATCH_LAYERS,
                        log_vectors=True), env.out, smoke=True)
    assert env.load_calls == ["llama"]  # model name reached load_model
    recs = _records(env.out, "dryl_patch")
    assert len(recs) == 1
    r = recs[0]
    assert r["mode"] == "patch" and r["layers"] == PATCH_LAYERS
    assert r["action"] == "bet" and r["amount"] == 20 and r["parse_ok"] is True
    assert sorted(spied[0]._fired) == PATCH_LAYERS
    assert all(c == 1 for c in spied[0]._fired.values())
    # VectorStore stacks sized from the model config, NOT gemma's 42/3584
    z = np.load(Path(env.out) / "dryl_patch_vectors.npz")
    assert z["minus"].shape == (1, NL, D)
    assert z["plus"].shape == (1, NL, D)


def test_llama_anchor_arms(env):
    runner.run_arm(_arm(id="dryl_minus", mode="anchor_minus", n=2),
                   env.out, smoke=True)
    runner.run_arm(_arm(id="dryl_plus", mode="anchor_plus", n=1),
                   env.out, smoke=True)
    minus = _records(env.out, "dryl_minus")
    assert len(minus) == 2 and {r["seed"] for r in minus} == {42, 42 + 997}
    plus = _records(env.out, "dryl_plus")[0]
    assert plus["mode"] == "anchor_plus" and plus["action"] == "bet"
    bal = plus["source_state"]["balance"]
    assert 20 < bal < 100 and plus["bet_ratio"] == pytest.approx(20 / bal)
    assert env.load_calls == ["llama", "llama"]


def test_llama_steer_log_vectors_projection(env, tmp_path):
    # unit e0 direction at every layer, scale 1 → after steering layers
    # [12,15] at alpha 2 the residual stream is 8*e0, so the L22 tap must
    # report proj = h_norm = 8 along the npz row (manipulation check).
    dirs = np.zeros((NL, D), np.float32)
    dirs[:, 0] = 1.0
    npz = tmp_path / "dirs_llama.npz"
    np.savez(npz, directions=dirs, scales=np.ones(NL, np.float32))
    runner.run_arm(_arm(id="dryl_steer", mode="steer", layers=PATCH_LAYERS,
                        alpha=2.0, directions_npz=str(npz), log_vectors=True),
                   env.out, smoke=True)
    r = _records(env.out, "dryl_steer")[0]
    assert r["alpha"] == 2.0 and r["mode"] == "steer"
    vl = r["vector_log"]
    assert vl["layer"] == 22
    assert vl["proj"] == pytest.approx(8.0)
    assert vl["h_norm"] == pytest.approx(8.0)
    # steer arms log projections, never the patch-style VectorStore npz
    assert not (Path(env.out) / "dryl_steer_vectors.npz").exists()


def test_gemma_arm_keeps_single_arg_load_shape(env):
    """Frozen contract: gemma arms still call load_model(gpu) with ONE
    positional arg, so the original test_dryrun monkeypatch signature
    (``lambda gpu: ...``) keeps working unchanged."""
    env.mp.setattr(runner, "load_model",
                   lambda gpu: (env.model, env.tok, "cpu"))
    runner.run_arm({"model": "gemma", "task": "sm", "n": 1, "phase": "dryl",
                    "id": "dryl_g", "mode": "anchor_minus"},
                   env.out, smoke=True)
    assert len(_records(env.out, "dryl_g")) == 1
