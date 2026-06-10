"""GPU-free end-to-end dry-run of run_arm covering every W1 execution path.

Monkeypatches runner.load_model with a CPU fake (42 identity layers, real
forward-hook plumbing) and LLM_ADDICTION_BEHAVIORAL_ROOT with synthetic SM/IC
catalogs, then drives run_arm and asserts on the produced jsonl records.
Paths covered: patch / steer(real npz) / steer(random dir) / probe(bet,
all-in, stage1-stop) / ic(anchor_minus, steer) / resume.
"""
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from multilayer_causal.src import runner

ASSETS = Path(__file__).resolve().parents[1] / "assets"
IBA_V2 = ASSETS / "directions_iba_v2.npz"
LAYERS = list(range(18, 24))  # [18, 23] inclusive, registry-expanded
D, NL = runner.D_MODEL, runner.N_LAYERS


# ---------------------------------------------------------------- fakes ----

class _Batch(dict):
    """Mapping with .to(device) — mimics transformers BatchEncoding."""

    def to(self, device):
        return self


class FakeTok:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0

    def __init__(self, default="I will bet. Final Decision: Bet $20"):
        self.default = default
        self.queue = []  # FIFO of canned decode() outputs; falls back to default

    def __call__(self, prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        ids = [(ord(c) % 97) + 3 for c in prompt[::4]] or [5, 6]
        t = torch.tensor([ids], dtype=torch.long)
        return _Batch(input_ids=t, attention_mask=torch.ones_like(t))

    def decode(self, ids, skip_special_tokens=True):
        return self.queue.pop(0) if self.queue else self.default


class _IdLayer(nn.Module):
    def forward(self, x):
        return (x,)  # HF decoder layers return tuples; element 0 = hidden


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_IdLayer() for _ in range(NL))
        self.config = types.SimpleNamespace(hidden_size=D)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = torch.zeros(1, input_ids.shape[1], D)
        for layer in self.model.layers:
            x = layer(x)[0]  # forward hooks fire here
        return x

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.forward(input_ids=input_ids)  # hooks must fire during generation
        new = torch.full((1, 3), 11, dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


# ------------------------------------------------------ synthetic catalogs --

def _write_catalogs(root: Path):
    sm = root / "slot_machine" / "gemma_v4_role"
    sm.mkdir(parents=True)
    games = []
    for combo in ["", "M", "MH"]:  # variable, no G → all enter the pool
        decs = [{"balance_before": 100 - 2 * r} for r in range(9)]
        hist = [{"round": r + 1, "bet": 10, "win": r % 2 == 0,
                 "balance": 100 - 2 * (r + 1)} for r in range(8)]
        games.append({"bet_type": "variable", "prompt_combo": combo,
                      "decisions": decs, "history": hist})
    # filtered out: fixed bet_type, G in combo
    games.append({"bet_type": "fixed", "prompt_combo": "",
                  "decisions": [{"balance_before": 100}] * 8, "history": []})
    games.append({"bet_type": "variable", "prompt_combo": "G",
                  "decisions": [{"balance_before": 100}] * 8, "history": []})
    (sm / "final_gemma_dry.json").write_text(json.dumps({"results": games}))

    ic = root / "investment_choice" / "v2_role_gemma"
    ic.mkdir(parents=True)
    ic_games = [{"bet_constraint": "fixed",
                 "decisions": [{"full_prompt":
                                f"You have a balance. Pick an option. (game {i})",
                                "balance_before": 100, "choice": 2}]}
                for i in range(6)]
    (ic / "ic_dry.json").write_text(json.dumps({"results": ic_games}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    _write_catalogs(tmp_path / "behavioral")
    monkeypatch.setenv("LLM_ADDICTION_BEHAVIORAL_ROOT",
                       str(tmp_path / "behavioral"))
    tok, model = FakeTok(), FakeModel()
    monkeypatch.setattr(runner, "load_model", lambda gpu: (model, tok, "cpu"))
    return types.SimpleNamespace(tok=tok, model=model, out=tmp_path / "out",
                                 mp=monkeypatch)


def _arm(**kw):
    a = {"model": "gemma", "task": "sm", "n": 1, "phase": "dry"}
    a.update(kw)
    return a


def _records(out, arm_id):
    p = Path(out) / f"{arm_id}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()]


# ----------------------------------------------------------------- tests ----

def test_patch_arm(env):
    spied = []
    real = runner.MultiLayerPatcher

    class SpyPatcher(real):
        def __init__(self, cached):
            super().__init__(cached)
            spied.append(self)

    env.mp.setattr(runner, "MultiLayerPatcher", SpyPatcher)
    runner.run_arm(_arm(id="dry_patch", mode="patch", layers=LAYERS),
                   env.out, smoke=True)
    recs = _records(env.out, "dry_patch")
    assert len(recs) == 1
    r = recs[0]
    assert r["mode"] == "patch" and r["layers"] == LAYERS
    assert r["action"] == "bet" and r["amount"] == 20
    assert r["parse_ok"] is True and r["extreme"] is False
    bal = r["source_state"]["balance"]
    assert 20 < bal < 100  # synthetic catalog guarantee
    assert r["bet_ratio"] == pytest.approx(20 / bal)
    # the patch hooks actually fired exactly once per layer during generation
    assert len(spied) == 1
    assert sorted(spied[0]._fired) == LAYERS
    assert all(c == 1 for c in spied[0]._fired.values())


def test_steer_arm_real_directions(env):
    seen = []
    real = runner.MultiLayerSteerer

    class SpySteerer(real):
        def __init__(self, directions, scales, alpha):
            seen.append((directions, scales, alpha))
            super().__init__(directions, scales, alpha)

    env.mp.setattr(runner, "MultiLayerSteerer", SpySteerer)
    runner.run_arm(_arm(id="dry_steer", mode="steer", layers=LAYERS,
                        alpha=4.0, directions_npz=str(IBA_V2)),
                   env.out, smoke=True)
    recs = _records(env.out, "dry_steer")
    assert len(recs) == 1
    assert recs[0]["alpha"] == 4.0 and recs[0]["mode"] == "steer"
    assert recs[0]["action"] == "bet" and recs[0]["parse_ok"] is True
    dirs, scales, alpha = seen[0]
    assert alpha == 4.0 and set(scales) == set(LAYERS)
    z = np.load(IBA_V2)
    for li in LAYERS:
        assert scales[li] == pytest.approx(float(z["scales"][li]))
        assert torch.allclose(
            dirs[li], torch.tensor(z["directions"][li], dtype=torch.float32))


def test_steer_arm_random_direction(env):
    arm = _arm(id="dry_rnd", mode="steer", layers=LAYERS, alpha=2.0,
               direction="random", dir_seed=2026069999,
               directions_npz=str(IBA_V2))
    runner.run_arm(arm, env.out, smoke=True)
    recs = _records(env.out, "dry_rnd")
    assert len(recs) == 1 and recs[0]["action"] == "bet"
    # random directions: deterministic per dir_seed, unit norm, ≠ npz rows
    d1, s1 = runner._load_steer_assets(arm)
    d2, s2 = runner._load_steer_assets(arm)
    z = np.load(IBA_V2)
    for li in LAYERS:
        assert torch.equal(d1[li], d2[li])
        assert torch.linalg.norm(d1[li]).item() == pytest.approx(1.0, abs=1e-5)
        assert not torch.allclose(
            d1[li], torch.tensor(z["directions"][li], dtype=torch.float32))
        assert s1[li] == s2[li] == pytest.approx(float(z["scales"][li]))


def _probe_arm(arm_id):
    return _arm(id=arm_id, mode="steer", probe=True, layers=LAYERS,
                alpha=2.0, directions_npz=str(IBA_V2))


def test_probe_bet_branches_lc_arithmetic(env):
    env.tok.queue = [
        "ok. Final Decision: Bet $20",       # stage 1
        "chase! Final Decision: Bet $30",    # LOSS branch
        "fine. Final Decision: Bet $10",     # WIN branch
    ]
    runner.run_arm(_probe_arm("dry_probe"), env.out, smoke=True)
    r = _records(env.out, "dry_probe")[0]
    assert env.tok.queue == []  # exactly 3 generations: stage1 + 2 branches
    bal = r["source_state"]["balance"]
    p, r_t = r["probe"], 20 / bal
    assert p["bet"] == 20 and p["r_t"] == pytest.approx(r_t)
    loss, win = p["loss"], p["win"]
    # LOSS branch: balance bal-20, bet 30 → r up → lc > 0
    r_loss = 30 / (bal - 20)
    assert loss["action"] == "bet" and loss["amount"] == 30
    assert loss["parse_ok"] is True
    assert loss["r"] == pytest.approx(r_loss)
    assert loss["lc"] == pytest.approx((r_loss - r_t) / r_t)
    assert loss["lc"] > 0
    # WIN branch: balance bal+40 (3.0x payout, net +2*bet), bet 10 → r down → lc 0
    r_win = 10 / (bal + 40)
    assert win["action"] == "bet" and win["amount"] == 10
    assert win["parse_ok"] is True
    assert win["r"] == pytest.approx(r_win)
    assert win["lc"] == 0.0


def test_probe_all_in(env):
    # $100 clamps to max_bet = balance (<100 in catalog) → amount == balance
    env.tok.queue = ["all of it. Final Decision: Bet $100"]
    runner.run_arm(_probe_arm("dry_allin"), env.out, smoke=True)
    r = _records(env.out, "dry_allin")[0]
    assert r["amount"] == r["source_state"]["balance"]
    assert r["extreme"] is True
    assert r["probe"] == {"all_in": True}  # no branches generated
    assert env.tok.queue == []


def test_probe_stage1_stop(env):
    env.tok.queue = ["enough. Final Decision: Stop"]
    runner.run_arm(_probe_arm("dry_stop"), env.out, smoke=True)
    r = _records(env.out, "dry_stop")[0]
    assert r["action"] == "stop" and r["amount"] == 0 and r["bet_ratio"] == 0.0
    assert r["probe"] == {"stage1_stop": True}
    assert env.tok.queue == []


def test_ic_anchor_and_steer(env):
    env.tok.default = "Let me weigh the risk. Final Decision: Option 2"
    runner.run_arm(_arm(id="dry_ic_anchor", phase="dryic", task="ic",
                        mode="anchor_minus"), env.out, smoke=True)
    r = _records(env.out, "dry_ic_anchor")[0]
    assert r["task"] == "ic" and r["mode"] == "anchor_minus"
    # frozen parser: prompt Option 2 → game choice 3 (PROMPT_TO_GAME), risky
    assert r["choice"] == 3 and r["risky"] is True and r["parse_ok"] is True
    assert isinstance(r["game_id"], int)

    runner.run_arm(_arm(id="dry_ic_steer", phase="dryic", task="ic",
                        mode="steer", layers=LAYERS, alpha=4.0,
                        directions_npz=str(IBA_V2)), env.out, smoke=True)
    r2 = _records(env.out, "dry_ic_steer")[0]
    assert r2["task"] == "ic" and r2["alpha"] == 4.0
    assert r2["choice"] == 3 and r2["parse_ok"] is True


def test_resume_skips_done_seeds(env):
    arm = _arm(id="dry_anchor", mode="anchor_minus", n=2)
    runner.run_arm(arm, env.out, smoke=True)
    recs1 = _records(env.out, "dry_anchor")
    assert len(recs1) == 2
    assert {r["seed"] for r in recs1} == {42, 42 + 997}
    # second run: every seed already done → no generation, file unchanged
    env.tok.default = "POISON — a rerun trial would record this"
    runner.run_arm(arm, env.out, smoke=True)
    recs2 = _records(env.out, "dry_anchor")
    assert recs2 == recs1
