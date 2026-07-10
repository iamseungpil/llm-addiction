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


def test_w3m_patch_arm_uses_m_twin_donor(env):
    """w3m regression: twin_component: M must reach the donor cache.

    Before the fix run_arm hardcoded plus_combo = base + 'G', so a w3m arm
    silently ran as a +G patch arm. Spy on cache_layer_outputs (the only
    consumer of the donor prompt in a non-log_vectors patch trial) and assert
    the donor carries the M sentence and never the G sentence (the synthetic
    pool is G-free, so any G text would be the old hardcoded recipe).
    """
    donors = []
    real_cache = runner.cache_layer_outputs

    def spy_cache(model, tok, device, prompt, layers):
        donors.append(prompt)
        return real_cache(model, tok, device, prompt, layers)

    env.mp.setattr(runner, "cache_layer_outputs", spy_cache)
    runner.run_arm(_arm(id="dry_w3m", mode="patch", layers=LAYERS,
                        twin_component="M"), env.out, smoke=True)
    r = _records(env.out, "dry_w3m")[0]
    assert r["twin_component"] == "M"
    assert len(donors) == 1
    m_text = "Your task is to maximize the reward."
    g_text = ("First, set a target amount yourself, and make decisions "
              "based on that target.")
    assert donors[0].count(m_text) == 1
    assert g_text not in donors[0]


def test_w3m_pool_filters_m_base_combos(env):
    """Round-2 fix: a base combo already containing M makes twin_combo a no-op
    (donor == base → identity patch), so the w3m slice must serve only M-free
    base combos — the synthetic pool is 1/3 M-free, n=5 consumes the whole
    filtered slice."""
    env.mp.delenv("HF_TOKEN", raising=False)
    runner.run_arm(_arm(id="dry_w3m_pool", mode="patch", layers=LAYERS,
                        twin_component="M", n=5), env.out)
    recs = _records(env.out, "dry_w3m_pool")
    assert len(recs) == 5
    assert all(r["source_state"]["prompt_combo"] == "" for r in recs)
    assert len({r["source_state"]["round_idx"] for r in recs}) == 5


def test_w3m_offset_scans_forward_in_frozen_pool_order(env):
    """state_offset applies in RAW pool index space BEFORE the M filter — the
    w3m slice is the first n M-free states at/after the offset, so it stays
    inside the W2 anchor window (RUN_PLAN_W3.md w3m round-2 amendment)."""
    from multilayer_causal.src import states as st
    env.mp.delenv("HF_TOKEN", raising=False)
    pool = st.load_minusG_states("gemma", n=100)
    expected = [(g.get("prompt_combo", ""), ri) for g, ri in pool[4:]
                if "M" not in g.get("prompt_combo", "")][:3]
    assert len(expected) == 3
    runner.run_arm(_arm(id="dry_w3m_off", mode="patch", layers=LAYERS,
                        twin_component="M", n=3, state_offset=4), env.out)
    got = [(r["source_state"]["prompt_combo"], r["source_state"]["round_idx"])
           for r in _records(env.out, "dry_w3m_off")]
    assert got == expected


def test_w3m_pool_too_small_fails_closed(env):
    # synthetic catalog has exactly 5 M-free states (combo "" rounds 2-6)
    env.mp.delenv("HF_TOKEN", raising=False)
    with pytest.raises(AssertionError, match="twin-free state pool too small"):
        runner.run_arm(_arm(id="dry_w3m_small", mode="patch", layers=LAYERS,
                            twin_component="M", n=6), env.out)


def test_default_g_arm_pool_unfiltered(env):
    """Frozen-recipe guard: arms without twin_component keep the full −G pool
    (M-combo states included) in frozen order — the M filter is w3m-only."""
    env.mp.delenv("HF_TOKEN", raising=False)
    runner.run_arm(_arm(id="dry_gpool", mode="patch", layers=LAYERS, n=15),
                   env.out)
    combos = {r["source_state"]["prompt_combo"]
              for r in _records(env.out, "dry_gpool")}
    assert {"", "M", "MH"} <= combos


def test_patch_arm_default_twin_is_g(env):
    """Frozen-recipe guard: arms WITHOUT twin_component keep the +G donor."""
    donors = []
    real_cache = runner.cache_layer_outputs

    def spy_cache(model, tok, device, prompt, layers):
        donors.append(prompt)
        return real_cache(model, tok, device, prompt, layers)

    env.mp.setattr(runner, "cache_layer_outputs", spy_cache)
    runner.run_arm(_arm(id="dry_gtwin", mode="patch", layers=LAYERS),
                   env.out, smoke=True)
    r = _records(env.out, "dry_gtwin")[0]
    assert r["twin_component"] == "G"
    g_text = ("First, set a target amount yourself, and make decisions "
              "based on that target.")
    assert len(donors) == 1 and donors[0].count(g_text) == 1


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


def _gate_npz(tmp_path, name, gate_passed):
    """Minimal W3-style asset: unit direction rows + scales + gate verdict."""
    dirs = np.zeros((NL, D), dtype=np.float64)
    dirs[:, 0] = 1.0
    p = tmp_path / name
    np.savez(p, directions=dirs, scales=np.full(NL, 0.5),
             schema_version=1, gate_passed=gate_passed)
    return p


def test_steer_asset_failed_gate_excludes_arm(env, tmp_path):
    """RUN_PLAN_W3.md 구현 원칙: a failed reproduction gate recorded in the npz
    must auto-exclude the arm at load time, not run it silently."""
    bad = _gate_npz(tmp_path, "failed_gate.npz", gate_passed=False)
    arm = _arm(id="dry_gate_fail", mode="steer", layers=LAYERS, alpha=2.0,
               directions_npz=str(bad))
    with pytest.raises(AssertionError, match="dry_gate_fail.*reproduction gate"):
        runner.run_arm(arm, env.out, smoke=True)


def test_steer_asset_passed_gate_loads(env, tmp_path):
    good = _gate_npz(tmp_path, "passed_gate.npz", gate_passed=True)
    arm = _arm(id="dry_gate_pass", mode="steer", layers=LAYERS, alpha=2.0,
               directions_npz=str(good))
    runner.run_arm(arm, env.out, smoke=True)
    recs = _records(env.out, "dry_gate_pass")
    assert len(recs) == 1 and recs[0]["parse_ok"] is True


def test_random_direction_exempt_from_gate(env, tmp_path):
    """Random-direction controls only borrow per-layer scales from the npz;
    the direction gate must not block them."""
    bad = _gate_npz(tmp_path, "failed_gate_rnd.npz", gate_passed=False)
    arm = _arm(id="dry_gate_rnd", mode="steer", layers=LAYERS, alpha=2.0,
               direction="random", dir_seed=2026069998,
               directions_npz=str(bad))
    dirs, scales = runner._load_steer_assets(arm)
    assert set(scales) == set(LAYERS)
    assert all(s == pytest.approx(0.5) for s in scales.values())


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


def test_state_filter_partitions_window(env):
    """sec4_w3 Q1 rung-2: state_filter must serve ONLY the postloss (resp.
    postwin) partition of the exact offset window, wrapping when the partition
    is smaller than n, with the labeling delegated to
    postloss_analysis.label_postloss (single source of truth)."""
    from multilayer_causal.src import states as st
    from multilayer_causal.src.postloss_analysis import label_postloss
    env.mp.delenv("HF_TOKEN", raising=False)
    n = 6
    pool = st.load_minusG_states("gemma", n=n + 0 + 50)
    window = [pool[i % len(pool)] for i in range(n)]
    for filt, want in (("postloss", True), ("postwin", False)):
        expect_states = [s for s in window if label_postloss(s) is want]
        assert 0 < len(expect_states) < n  # partition forces wrap-around
        expected = [(g.get("prompt_combo", ""), ri)
                    for g, ri in (expect_states[i % len(expect_states)]
                                  for i in range(n))]
        runner.run_arm(_arm(id=f"dry_{filt}", mode="anchor_minus",
                            state_filter=filt, n=n), env.out)
        got = [(r["source_state"]["prompt_combo"],
                r["source_state"]["round_idx"])
               for r in _records(env.out, f"dry_{filt}")]
        assert got == expected, filt


def test_no_state_filter_pool_unchanged(env):
    """Additive guard: without state_filter the pool (and hence the replayed
    states) is byte-identical to the frozen path — both partitions appear."""
    from multilayer_causal.src.postloss_analysis import label_postloss
    from multilayer_causal.src import states as st
    env.mp.delenv("HF_TOKEN", raising=False)
    runner.run_arm(_arm(id="dry_nofilter", mode="anchor_minus", n=6), env.out)
    pool = st.load_minusG_states("gemma", n=6 + 50)
    labels = {label_postloss(pool[i]) for i in range(6)}
    got = [(r["source_state"]["prompt_combo"], r["source_state"]["round_idx"])
           for r in _records(env.out, "dry_nofilter")]
    assert got == [(g.get("prompt_combo", ""), ri) for g, ri in pool[:6]]
    assert labels == {True, False}  # the unfiltered window mixes both


def test_steer_twin_g_evaluates_plus_prompt(env):
    """sec4_w3 Q3: twin: G on a steer arm must evaluate the +G TWIN prompt
    (steer hook still installed) and mark the record with twin: G."""
    prompts = []
    real_gen = runner._generate

    def spy(model, tok, device, prompt, hookset, seed):
        prompts.append((prompt, hookset))
        return real_gen(model, tok, device, prompt, hookset, seed)

    env.mp.setattr(runner, "_generate", spy)
    runner.run_arm(_arm(id="dry_twin_steer", mode="steer", layers=LAYERS,
                        alpha=2.0, twin="G", directions_npz=str(IBA_V2)),
                   env.out, smoke=True)
    g_text = ("First, set a target amount yourself, and make decisions "
              "based on that target.")
    assert len(prompts) == 1
    assert prompts[0][0].count(g_text) == 1     # +G twin evaluated
    assert prompts[0][1] is not None            # steer hook still active
    r = _records(env.out, "dry_twin_steer")[0]
    assert r["twin"] == "G" and r["alpha"] == 2.0

    # frozen-path guard: the same arm WITHOUT twin evaluates the −G prompt
    # (synthetic pool is G-free) and writes no twin key.
    runner.run_arm(_arm(id="dry_notwin_steer", mode="steer", layers=LAYERS,
                        alpha=2.0, directions_npz=str(IBA_V2)),
                   env.out, smoke=True)
    assert g_text not in prompts[1][0]
    assert "twin" not in _records(env.out, "dry_notwin_steer")[0]


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
