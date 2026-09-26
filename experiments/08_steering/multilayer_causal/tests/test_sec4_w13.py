"""W13 — §4 NECESSITY via PROJECT-OUT: project-out math, runner mode dispatch
(project_out engages MultiLayerProjector; steer/anchor byte-identical), the
analyze_w13 necessity verdict branches, gemma non-regression, and w13
config/template parity.

SYNTHETIC only (CPU fake model, no torch GPU / no HF). Covers:
  * MultiLayerProjector math on synthetic tensors — project-out removes the
    û-component (post-hook projection onto û ~ 0), norm along û -> 0, other
    dims untouched, removal_frac scales it, all positions/every forward;
  * runner project_out mode engages the projector on gemma AND llama arms, the
    natural −G prompt is evaluated, and a steer arm STILL uses the steerer
    (project_out is a new mode, absent field -> steer path unchanged);
  * analyze_w13: BEHAVIOURAL_NECESSARY_AND_PURE (behav drops below the null band
    with a positive matched-pairs drop, confound+readout do not),
    BEHAVIOURAL_NOT_NECESSARY (project-out leaves betting at baseline),
    NECESSARY_BUT_IMPURE (confound also drops), and the matched-pairs design
    (same-seed => same-state => same-balance) that holds balance fixed per pair;
  * arms_sec4_w13.yaml registry counts/fields + template parity (both models'
    arms listed, escaped need-list quotes, MLC_OUT/SYNC isolation);
  * gemma steering path byte-identical (project_out absent -> steer unchanged).
"""
import json
import re
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from multilayer_causal.src import hooks, runner, sec4_stats
from multilayer_causal.src.registry import load_arms

MLC = Path(__file__).resolve().parents[1]
W13_YAML = MLC / "configs" / "arms_sec4_w13.yaml"
W13_TEMPLATE = MLC / "amlt" / "sec4_w13.yaml.template"
D, NL = runner.D_MODEL, runner.N_LAYERS


# ============================================================ project-out math

def _mock_model():
    class Layer(nn.Module):
        def forward(self, x):
            return (x,)

    class Mock(nn.Module):
        def __init__(self):
            super().__init__()
            inner = nn.Module()
            inner.layers = nn.ModuleList([Layer() for _ in range(4)])
            self.model = inner

        def forward(self, x):
            for l in self.model.layers:
                x = l(x)[0]
            return x

    return Mock()


def test_projector_removes_u_component_all_positions():
    """h - (h·û)û: the post-hook projection onto û is ~0 at every position,
    the û-aligned dim goes to 0, orthogonal dims are untouched, and the removal
    fires on EVERY forward (prefill T>1 and decode-like T=1). û passed NON-unit
    to prove the hook normalises it."""
    m = _mock_model()
    u = torch.zeros(D)
    u[0] = 5.0                                  # non-unit; hook must normalise
    pr = hooks.MultiLayerProjector({1: u, 2: u}, frac=1.0)
    pr.install(m)
    x = torch.ones(1, 3, D)
    out = m(x)
    u_unit = u / u.norm()
    proj = (out[0] * u_unit).sum(-1)
    assert torch.allclose(proj, torch.zeros(3), atol=1e-6)   # û-component gone
    assert torch.allclose(out[0, :, 0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(out[0, :, 1:], torch.ones(3, D - 1))  # ortho untouched
    out_dec = m(torch.ones(1, 1, D))            # decode-like pass fires again
    assert torch.allclose((out_dec[0] * u_unit).sum(-1), torch.zeros(1),
                          atol=1e-6)
    pr.remove()
    assert torch.allclose(m(torch.ones(1, 3, D)), torch.ones(1, 3, D))


def test_projector_removal_frac_scales_partial():
    """removal_frac in (0,1) removes only a FRACTION of the û-component:
    frac=0.5 leaves half, frac=0.0 is a no-op."""
    m = _mock_model()
    u = torch.zeros(D)
    u[0] = 1.0
    for frac, expect in ((0.5, 0.5), (0.0, 1.0)):
        pr = hooks.MultiLayerProjector({1: u}, frac=frac)
        pr.install(m)
        out = m(torch.ones(1, 2, D))
        assert torch.allclose(out[0, :, 0], torch.full((2,), expect), atol=1e-6)
        assert torch.allclose(out[0, :, 1:], torch.ones(2, D - 1))
        pr.remove()


def test_projector_off_diagonal_direction_removes_only_that_axis():
    """A tilted û (mix of dim0+dim1): project-out zeroes the component ALONG û
    while the orthogonal complement survives (general geometric check)."""
    m = _mock_model()
    u = torch.zeros(D)
    u[0] = u[1] = 1.0                           # 45° in the (0,1) plane
    pr = hooks.MultiLayerProjector({1: u}, frac=1.0)
    pr.install(m)
    out = m(torch.ones(1, 1, D))[0, 0]
    u_unit = u / u.norm()
    assert abs(float(out @ u_unit)) < 1e-6      # no component along û
    # h=1s: (h·û)=2/sqrt2=sqrt2; removed sqrt2*û_unit=(1,1,0..) -> dims 0,1 -> 0
    assert torch.allclose(out[:2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(out[2:], torch.ones(D - 2))
    pr.remove()


# ============================================================ runner dispatch

class _Batch(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    default = "I will bet. Final Decision: Bet $20"

    def __call__(self, prompt, return_tensors="pt"):
        ids = [(ord(c) % 97) + 3 for c in prompt[::4]] or [5, 6]
        t = torch.tensor([ids], dtype=torch.long)
        return _Batch(input_ids=t, attention_mask=torch.ones_like(t))

    def decode(self, ids, skip_special_tokens=True):
        return self.default


class _IdLayer(nn.Module):
    def forward(self, x):
        return (x,)


def _fake_model(n_layers, d):
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(_IdLayer() for _ in range(n_layers))
            self.config = types.SimpleNamespace(hidden_size=d)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            x = torch.zeros(1, input_ids.shape[1], d)
            for layer in self.model.layers:
                x = layer(x)[0]
            return x

        def generate(self, input_ids=None, attention_mask=None, **kw):
            self.forward(input_ids=input_ids)
            new = torch.full((1, 3), 11, dtype=torch.long)
            return torch.cat([input_ids, new], dim=1)

    return FakeModel()


def _write_sm_catalog(root: Path):
    for sub in ("gemma_v4_role", "llama_v4_role"):
        sm = root / "slot_machine" / sub
        sm.mkdir(parents=True, exist_ok=True)
        games = []
        for combo in ["", "M", "MH"]:
            decs = [{"balance_before": 100 - 2 * r} for r in range(9)]
            hist = [{"round": r + 1, "bet": 10, "win": r % 2 == 0,
                     "balance": 100 - 2 * (r + 1)} for r in range(8)]
            games.append({"bet_type": "variable", "prompt_combo": combo,
                          "decisions": decs, "history": hist})
        (sm / f"final_{sub}.json").write_text(json.dumps({"results": games}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    _write_sm_catalog(tmp_path / "behavioral")
    monkeypatch.setenv("LLM_ADDICTION_BEHAVIORAL_ROOT",
                       str(tmp_path / "behavioral"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    tok = FakeTok()

    def fake_load(gpu, model_name="gemma"):
        nl = 42 if model_name == "gemma" else 32
        return _fake_model(nl, D), tok, "cpu"

    monkeypatch.setattr(runner, "load_model", fake_load)
    return types.SimpleNamespace(tok=tok, out=tmp_path / "out", mp=monkeypatch,
                                 tmp=tmp_path)


def _axis_npz(tmp_path, name, n_layers=NL):
    dirs = np.zeros((n_layers, D), dtype=np.float64)
    dirs[:, 0] = 1.0
    p = tmp_path / name
    np.savez(p, directions=dirs, scales=np.full(n_layers, 0.5),
             schema_version=1, gate_passed=True)
    return p


def _arm(**kw):
    a = {"model": "gemma", "task": "sm", "n": 1, "phase": "w13dry"}
    a.update(kw)
    return a


def _records(out, arm_id):
    return [json.loads(l) for l in
            (Path(out) / f"{arm_id}.jsonl").read_text().splitlines()]


def test_project_out_engages_projector_gemma(env):
    """A project_out arm engages MultiLayerProjector (NOT the steerer), with the
    arm's per-layer axis rows, evaluating the natural −G prompt."""
    npz = _axis_npz(env.tmp, "behav.npz")
    seen = []
    real = runner.MultiLayerProjector

    class SpyProjector(real):
        def __init__(self, directions, frac=1.0):
            seen.append((directions, frac))
            super().__init__(directions, frac=frac)

    steer_calls = []

    class SpySteerer(runner.MultiLayerSteerer):
        def __init__(self, *a, **k):
            steer_calls.append(a)
            super().__init__(*a, **k)

    env.mp.setattr(runner, "MultiLayerProjector", SpyProjector)
    env.mp.setattr(runner, "MultiLayerSteerer", SpySteerer)
    runner.run_arm(_arm(id="w13_po", mode="project_out",
                        layers=list(range(16, 22)), directions_npz=str(npz)),
                   env.out, smoke=True)
    recs = _records(env.out, "w13_po")
    assert len(recs) == 1 and recs[0]["mode"] == "project_out"
    assert recs[0]["action"] == "bet" and recs[0]["parse_ok"] is True
    assert len(seen) == 1 and not steer_calls        # projector, never steerer
    directions, frac = seen[0]
    assert frac == 1.0
    assert set(directions) == set(range(16, 22))


def test_project_out_removal_frac_passthrough(env):
    npz = _axis_npz(env.tmp, "behav2.npz")
    seen = []

    class SpyProjector(runner.MultiLayerProjector):
        def __init__(self, directions, frac=1.0):
            seen.append(frac)
            super().__init__(directions, frac=frac)

    env.mp.setattr(runner, "MultiLayerProjector", SpyProjector)
    runner.run_arm(_arm(id="w13_half", mode="project_out", removal_frac=0.5,
                        layers=list(range(16, 22)), directions_npz=str(npz)),
                   env.out, smoke=True)
    assert seen == [0.5]


def test_project_out_llama_model_field(env):
    """The llama project_out arm loads via the model field and engages the
    projector at the L14-19 window (32-layer axis)."""
    npz = _axis_npz(env.tmp, "llama_behav.npz", n_layers=32)
    seen = []

    class SpyProjector(runner.MultiLayerProjector):
        def __init__(self, directions, frac=1.0):
            seen.append(directions)
            super().__init__(directions, frac=frac)

    env.mp.setattr(runner, "MultiLayerProjector", SpyProjector)
    runner.run_arm(_arm(id="w13_llama_po", model="llama", mode="project_out",
                        layers=list(range(14, 20)), directions_npz=str(npz)),
                   env.out, smoke=True)
    assert len(seen) == 1 and set(seen[0]) == set(range(14, 20))


def test_steer_arm_still_uses_steerer_not_projector(env):
    """Non-regression: an arm WITHOUT project_out (mode steer) engages the
    steerer and NEVER the projector — project_out is a new sibling mode, the
    steer path is byte-identical."""
    npz = _axis_npz(env.tmp, "behav3.npz")
    proj_calls, steer_calls = [], []

    class SpyProjector(runner.MultiLayerProjector):
        def __init__(self, *a, **k):
            proj_calls.append(a)
            super().__init__(*a, **k)

    class SpySteerer(runner.MultiLayerSteerer):
        def __init__(self, *a, **k):
            steer_calls.append(a)
            super().__init__(*a, **k)

    env.mp.setattr(runner, "MultiLayerProjector", SpyProjector)
    env.mp.setattr(runner, "MultiLayerSteerer", SpySteerer)
    runner.run_arm(_arm(id="w13_steer", mode="steer", alpha=3.0,
                        layers=list(range(16, 22)), directions_npz=str(npz)),
                   env.out, smoke=True)
    assert len(steer_calls) == 1 and not proj_calls
    assert _records(env.out, "w13_steer")[0]["mode"] == "steer"


def test_project_out_random_null_uses_seeded_unit_dirs(env):
    """A random-direction project_out null removes a deterministic per-seed unit
    axis (NOT the npz rows) — the necessity control."""
    npz = _axis_npz(env.tmp, "behav4.npz")
    arm = _arm(id="w13_null", mode="project_out", direction="random",
               dir_seed=2026071301, layers=list(range(16, 22)),
               directions_npz=str(npz))
    d1 = runner._load_projectout_dirs(arm)
    d2 = runner._load_projectout_dirs(arm)
    z = np.load(npz)
    for li in range(16, 22):
        assert torch.equal(d1[li], d2[li])                    # deterministic
        assert torch.linalg.norm(d1[li]).item() == pytest.approx(1.0, abs=1e-5)
        assert not torch.allclose(
            d1[li], torch.tensor(z["directions"][li], dtype=torch.float32))
    runner.run_arm(arm, env.out, smoke=True)
    assert _records(env.out, "w13_null")[0]["mode"] == "project_out"


def test_project_out_failed_gate_excludes_arm(env):
    """A failed reproduction gate in the axis npz auto-excludes the project_out
    arm at load time (same guard as the steerer)."""
    dirs = np.zeros((NL, D)); dirs[:, 0] = 1.0
    bad = env.tmp / "bad_gate.npz"
    np.savez(bad, directions=dirs, scales=np.full(NL, 0.5), gate_passed=False)
    with pytest.raises(AssertionError, match="reproduction gate"):
        runner.run_arm(_arm(id="w13_badgate", mode="project_out",
                            layers=list(range(16, 22)), directions_npz=str(bad)),
                       env.out, smoke=True)


# ============================================================ analyze_w13

BAL_BINS = [(0, "60"), (1, "45"), (2, "30")]   # (trial group, balance) triples


def _write_w13_arm(results_dir, arm_id, mean_bet, rng, n=30, with_proj=False,
                   balance=60):
    p = results_dir / f"{arm_id}.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            seed = 8000042 + i * 997
            bet = float(np.clip(mean_bet + 0.01 * rng.standard_normal(), 0, 1))
            bal = 30 + (i % 3) * 25            # 30 / 55 / 80 across trials
            rec = {"trial_id": i, "seed": seed, "arm": arm_id,
                   "parse_ok": True, "bet_ratio": bet, "action": "bet",
                   "source_state": {"balance": bal}}
            if with_proj:
                rec["vector_log"] = {"layer": 18, "proj": 0.0, "h_norm": 5.0}
            f.write(json.dumps(rec) + "\n")


def _write_w13_model(results_dir, model, rng, base, behav, confound, readout,
                     null=None):
    null = base if null is None else null
    _write_w13_arm(results_dir, f"sec4_w13_{model}_base", base, rng)
    _write_w13_arm(results_dir, f"sec4_w13_{model}_behavioural", behav, rng,
                   with_proj=True)
    _write_w13_arm(results_dir, f"sec4_w13_{model}_confound", confound, rng,
                   with_proj=True)
    _write_w13_arm(results_dir, f"sec4_w13_{model}_readout", readout, rng,
                   with_proj=True)
    for k in (1, 2):
        _write_w13_arm(results_dir, f"sec4_w13_{model}_null_{k}", null, rng)


def test_analyze_w13_behavioural_necessary_and_pure(tmp_path):
    """Behavioural project-out drops betting well below the null band with a
    positive matched-pairs drop; confound + readout stay at baseline =>
    per-model verdict BEHAVIOURAL_NECESSARY_AND_PURE, behav NECESSARY, others
    NOT_NECESSARY."""
    rng = np.random.default_rng(0)
    results = tmp_path / "results"
    results.mkdir()
    for model in ("gemma", "llama"):
        _write_w13_model(results, model, rng, base=0.40, behav=0.18,
                         confound=0.40, readout=0.40)
    res = sec4_stats.analyze_w13(results)
    for model in ("gemma", "llama"):
        m = res["models"][model]
        assert m["verdict"] == "BEHAVIOURAL_NECESSARY_AND_PURE", (model, m["verdict"])
        assert m["axes"]["behavioural"]["verdict"] == "NECESSARY"
        assert m["axes"]["confound"]["verdict"] == "NOT_NECESSARY"
        assert m["axes"]["readout"]["verdict"] == "NOT_NECESSARY"
        # matched-pairs drop present, positive; by_bin is a descriptive read-out
        pair = m["axes"]["behavioural"]["paired"]
        assert pair["n_pairs"] == 30
        assert pair["paired_drop"] > 0.1
        assert len(pair["by_bin"]) >= 2          # multiple shown-balance strata
    assert res["necessity_table"]["gemma"]["behavioural"] == "NECESSARY"


def test_analyze_w13_behavioural_not_necessary(tmp_path):
    """Project-out leaves betting AT baseline for every axis => nothing is
    necessary => BEHAVIOURAL_NOT_NECESSARY."""
    rng = np.random.default_rng(1)
    results = tmp_path / "results"
    results.mkdir()
    _write_w13_model(results, "gemma", rng, base=0.40, behav=0.40,
                     confound=0.40, readout=0.40)
    _write_w13_model(results, "llama", rng, base=0.40, behav=0.40,
                     confound=0.40, readout=0.40)
    res = sec4_stats.analyze_w13(results)
    m = res["models"]["gemma"]
    assert m["verdict"] == "BEHAVIOURAL_NOT_NECESSARY", m["verdict"]
    assert m["axes"]["behavioural"]["verdict"] == "NOT_NECESSARY"
    assert not m["axes"]["behavioural"]["below_null_band"]


def test_analyze_w13_necessary_but_impure(tmp_path):
    """Behavioural AND confound both drop under project-out => the removal is not
    axis-pure => NECESSARY_BUT_IMPURE (the confound could carry it)."""
    rng = np.random.default_rng(2)
    results = tmp_path / "results"
    results.mkdir()
    _write_w13_model(results, "gemma", rng, base=0.40, behav=0.18,
                     confound=0.18, readout=0.40)
    _write_w13_model(results, "llama", rng, base=0.40, behav=0.18,
                     confound=0.40, readout=0.40)
    m = sec4_stats.analyze_w13(results)["models"]["gemma"]
    assert m["verdict"] == "NECESSARY_BUT_IMPURE", m["verdict"]
    assert m["axes"]["behavioural"]["necessary"]
    assert m["axes"]["confound"]["necessary"]


def test_analyze_w13_matched_pairs_hold_balance_fixed_by_construction(tmp_path):
    """Balance is removed as a confound BY CONSTRUCTION, not by post-hoc
    stratification: baseline and project-out arms replay the same seeded slice,
    so within every pair the shown balance is identical across arms. The drop is
    therefore a within-balance drop for every pair, and the descriptive by_bin
    read-out carries a positive drop in each shown-balance stratum."""
    rng = np.random.default_rng(3)
    results = tmp_path / "results"
    results.mkdir()
    _write_w13_model(results, "gemma", rng, base=0.40, behav=0.18,
                     confound=0.40, readout=0.40)
    _write_w13_model(results, "llama", rng, base=0.40, behav=0.40,
                     confound=0.40, readout=0.40)
    pair = sec4_stats.analyze_w13(results)["models"]["gemma"]["axes"][
        "behavioural"]["paired"]
    # positive matched-pairs drop, and every shown-balance stratum shows a drop
    assert pair["paired_drop"] > 0.15
    assert len(pair["by_bin"]) >= 2
    assert all(b["mean_drop"] > 0.15 for b in pair["by_bin"].values())
    # a count-weighted mean over the bins is the paired drop BY CONSTRUCTION
    # (same pairs, partitioned) — the bins are not an independent artefact guard.
    tot = sum(b["n"] for b in pair["by_bin"].values())
    binw = sum(b["n"] * b["mean_drop"] for b in pair["by_bin"].values()) / tot
    assert binw == pytest.approx(pair["paired_drop"], abs=1e-6)


# ============================================================ registry / parity

def test_w13_registry_counts_and_fields():
    arms = load_arms(W13_YAML)
    assert len(arms) == 14
    for a in arms.values():
        assert a["n"] == 200
        assert a["seed_base"] == 8000042
        assert a["phase"] == "sec4_w13"
        assert a.get("task", "sm") == "sm"
        assert a["prompt_set"] == "addiction_role_gm"
    for model, layers, off in (("gemma", list(range(16, 22)), 300),
                               ("llama", list(range(14, 20)), 0)):
        ms = {k: a for k, a in arms.items() if a["model"] == model}
        assert len(ms) == 7, (model, len(ms))
        for a in ms.values():
            assert a["layers"] == layers
            assert a["state_offset"] == off
        # exactly one no-hook baseline
        bases = [a for a in ms.values() if a["mode"] == "anchor_minus"]
        assert len(bases) == 1 and bases[0]["id"] == f"sec4_w13_{model}_base"
        # three full project-out axis arms on the frozen npz
        for axis in ("behavioural", "confound", "readout"):
            a = ms[f"sec4_w13_{model}_{axis}"]
            assert a["mode"] == "project_out" and "removal_frac" not in a
            assert a["directions_npz"].endswith(
                f"{model}_slot_machine_i_ba_{axis}.npz")
        # one partial-removal behavioural arm
        half = ms[f"sec4_w13_{model}_behavioural_half"]
        assert half["mode"] == "project_out" and half["removal_frac"] == 0.5
        # two random-direction project-out nulls, unique seeds
        nulls = [a for a in ms.values() if a.get("direction") == "random"]
        assert len(nulls) == 2
        assert all(a["mode"] == "project_out" for a in nulls)
        assert len({a["dir_seed"] for a in nulls}) == 2


def test_w13_project_out_is_sm_only():
    """project_out is SM-only and removal_frac is project_out-only (registry
    guards)."""
    import yaml
    from multilayer_causal.src import registry
    # a project_out IC arm must be rejected
    bad = {"defaults": {}, "arms": [
        {"id": "x", "mode": "project_out", "task": "ic", "phase": "p",
         "layers": [16, 21], "model": "gemma"}]}
    p = MLC / "configs" / "_tmp_w13_bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    try:
        with pytest.raises(AssertionError, match="project_out is SM-only"):
            registry.load_arms(p)
    finally:
        p.unlink()


def test_w13_template_lists_all_arms():
    arms = load_arms(W13_YAML)
    text = W13_TEMPLATE.read_text()
    runs = re.findall(r"run_arms\.sh \d+ ([^\n]+)", text)
    assert len(runs) == 1, "expected exactly one run_arms.sh fan-out line"
    listed = runs[0].split()
    assert sorted(listed) == sorted(arms), set(listed) ^ set(arms)
    # YAML description guard: never ': ' after the description key (amlt trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # both models' frozen axes referenced; reuse (fetch) not rebuild by default
    for model in ("gemma", "llama"):
        for axis in ("behavioural", "confound", "readout"):
            assert f"{model}_slot_machine_i_ba_{axis}.npz" in text
    # need-list double-quotes MUST be backslash-escaped in the python -c block
    need = re.search(r"need = \[([^\]]+)\]", text)
    assert need, "no need-list in sec4_w13.yaml.template"
    assert re.findall(r'(?<!\\)"', need.group(1)) == []
    assert need.group(1).count('\\"') == 2 * 6      # 6 axes, two quotes each
    # both catalogs bootstrapped before the fan-out
    assert 'ensure_sm_catalog(\\"gemma\\")' in text
    assert 'ensure_sm_catalog(\\"llama\\")' in text
    # isolation env
    assert "MLC_OUT: multilayer_causal/results/sec4_w13" in text
    assert 'MLC_SYNC_EVERY: "50"' in text
    assert "arms_sec4_w13.yaml" in text


# ============================================================ non-regression

def test_gemma_steer_hook_math_unchanged():
    """Byte-identical guard: MultiLayerSteerer still ADDS alpha*scale*dir; the
    project-out mode did not perturb the steering hook."""
    m = _mock_model()
    v = torch.zeros(D)
    v[0] = 1.0
    st = hooks.MultiLayerSteerer({0: v, 3: v}, {0: 2.0, 3: 2.0}, alpha=1.5)
    st.install(m)
    out = m(torch.zeros(1, 2, D))
    assert torch.allclose(out[0, :, 0], torch.full((2,), 2 * 1.5 * 2.0))
    st.remove()


def test_registry_modes_additive():
    from multilayer_causal.src.registry import MODES
    assert "project_out" in MODES
    # prior modes intact
    assert {"anchor_minus", "anchor_plus", "patch", "subspace",
            "steer"} <= MODES
