# Multilayer Causal Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `multilayer_causal/` — an isolated harness that runs E1 (multi-layer paired patching), E2 (rank-r subspace patching) and E3 (multi-layer additive steering) on Gemma-2-9B slot-machine states, with frozen seeds, per-arm JSONL checkpoints, HF latest-state sync/resume, and amlt job packaging.

**Architecture:** One thin runner + three small hook classes + declarative arm registry (`arms.yaml`). Single-decision protocol inherited byte-identically from `sae_v3_analysis/src/run_m3pp_strong_patching.py` (prompt builder, parser, state pool, seeds). No edits to any existing experiment file; `sae_v3_analysis` imported read-only only in the (later, gated) E3b full-game mode.

**Tech Stack:** Python 3.10+, torch/transformers (bf16, GPU only on nodes), numpy/scipy/pyyaml, huggingface_hub. Tests are CPU-only (mock layers, no transformers import).

**Spec:** `docs/superpowers/specs/2026-06-10-multilayer-causal-intervention-design.md`

---

## File map

```
multilayer_causal/
├── README.md                     # Task 10
├── configs/arms.yaml             # Task 5 (E1 arms; E2/E3 appended at gate time)
├── src/
│   ├── __init__.py               # empty
│   ├── prompts.py                # Task 1 — frozen copies of M3'' build_prompt/parse_response
│   ├── states.py                 # Task 2 — −G state pool + HF catalog bootstrap
│   ├── hooks.py                  # Task 3 — cache_layer_outputs, MultiLayerPatcher,
│   │                             #          SubspacePatcher, MultiLayerSteerer
│   ├── checkpoint.py             # Task 4 — ArmCheckpoint (JSONL+HF sync), VectorStore
│   ├── registry.py               # Task 5 — arms.yaml loader/validator
│   ├── runner.py                 # Task 6 — model load, trial loop, mode dispatch
│   ├── analyze.py                # Task 7 — per-arm summaries, anchor tests, G1/S* gates
│   └── pca_basis.py              # Task 9 — phase_a npz → per-layer PCA bases (E2 prep)
├── run_experiment.py             # Task 6 — CLI
├── run_arms.sh                   # Task 8 — shard arms across GPUs inside one job
├── scripts/push_code_to_hf.py    # Task 8 — tarball → HF dataset
├── amlt/smoke.yaml               # Task 8
├── amlt/e1_main.yaml             # Task 8
└── tests/
    ├── test_prompts.py           # Task 1
    ├── test_states.py            # Task 2
    ├── test_hooks.py             # Task 3
    ├── test_checkpoint.py        # Task 4
    └── test_registry.py          # Task 5
```

Conventions: repo root on `sys.path` for tests (`pytest` from repo root); all modules
import as `multilayer_causal.src.<mod>`. Local outputs under `multilayer_causal/out/`
(gitignored). Env: `HF_TOKEN`, `LLM_ADDICTION_BEHAVIORAL_ROOT` (default
`/home/v-seungplee/data/llm-addiction/behavioral`).

---

### Task 1: scaffold + frozen prompt functions

**Files:** Create `multilayer_causal/{src/__init__.py,tests/__init__.py}`, `multilayer_causal/src/prompts.py`, `multilayer_causal/tests/test_prompts.py`, append `multilayer_causal/out/` to `.gitignore`.

- [ ] **Step 1.1: failing test** — `multilayer_causal/tests/test_prompts.py`:

```python
"""Prompt builder/parser must stay byte-identical to the M3'' originals."""
import importlib.util
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_m3pp():
    spec = importlib.util.spec_from_file_location(
        "m3pp", REPO / "sae_v3_analysis" / "src" / "run_m3pp_strong_patching.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_source_parity_with_m3pp():
    from multilayer_causal.src import prompts
    m3pp = _load_m3pp()
    assert inspect.getsource(prompts.build_prompt) == inspect.getsource(m3pp.build_prompt)
    assert inspect.getsource(prompts.parse_response) == inspect.getsource(m3pp.parse_response)


def test_build_prompt_minimal_game():
    from multilayer_causal.src.prompts import build_prompt
    game = {"decisions": [{"balance_before": 80}], "bet_type": "variable",
            "prompt_combo": "", "history": []}
    p = build_prompt(game, 0)
    assert "Current balance: $80" in p
    assert "Bet (choose $5-$80)" in p
    assert "Final Decision:" in p
    # override adds the G component
    pg = build_prompt(game, 0, override_combo="G")
    assert "set a target amount yourself" in pg and "set a target amount" not in p


def test_parse_response_cases():
    from multilayer_causal.src.prompts import parse_response
    assert parse_response("blah\nFinal Decision: Bet $30", 100) == ("bet", 30)
    assert parse_response("Final Decision: Stop", 100) == ("stop", 0)
    assert parse_response("I will stop playing now.", 100) == ("stop", 0)
    assert parse_response("no decision here", 100) == ("bet", 10)      # fallback
    assert parse_response("Final Decision: Bet $500", 60) == ("bet", 60)  # clamp
```

- [ ] **Step 1.2:** `pytest multilayer_causal/tests/test_prompts.py -q` → FAIL (module missing)
- [ ] **Step 1.3:** create `multilayer_causal/src/prompts.py` — header + byte-identical copies of `build_prompt` (run_m3pp_strong_patching.py:49–85) and `parse_response` (:88–105):

```python
"""Frozen copies of the M3'' prompt builder and parser.

PROVENANCE: copied byte-identically from
sae_v3_analysis/src/run_m3pp_strong_patching.py (build_prompt, parse_response).
tests/test_prompts.py::test_source_parity_with_m3pp enforces the freeze.
Do NOT "improve" these — comparability with the M3 family depends on identity.
"""
from __future__ import annotations

# <exact function bodies copied from the original file>
```

- [ ] **Step 1.4:** run tests → PASS. Add `multilayer_causal/out/` to `.gitignore`.
- [ ] **Step 1.5:** `git add -A multilayer_causal .gitignore && git commit -m "feat(mlc): scaffold + frozen M3'' prompt functions"`

### Task 2: state pool + HF catalog bootstrap

**Files:** Create `multilayer_causal/src/states.py`, `multilayer_causal/tests/test_states.py`.

- [ ] **Step 2.1: failing test**

```python
import json
from multilayer_causal.src import states


def _write_catalog(tmp_path, monkeypatch):
    d = tmp_path / "slot_machine" / "gemma_v4_role"
    d.mkdir(parents=True)
    games = []
    for combo in ["", "G", "M"]:
        for bt in ["variable", "fixed"]:
            games.append({"bet_type": bt, "prompt_combo": combo,
                          "decisions": [{"balance_before": 100 - 5 * r} for r in range(8)],
                          "history": []})
    (d / "final_gemma_x.json").write_text(json.dumps({"results": games}))
    monkeypatch.setenv("LLM_ADDICTION_BEHAVIORAL_ROOT", str(tmp_path))


def test_pool_filters_variable_minusG_rounds_2_to_6(tmp_path, monkeypatch):
    _write_catalog(tmp_path, monkeypatch)
    pool = states.load_minusG_states("gemma", n=100)
    # only variable & no-G games: combos "" and "M" → 2 games × rounds 2..6 = 10 states
    assert len(pool) == 10
    assert all(2 <= ri <= 6 for _, ri in pool)
    assert all("G" not in g.get("prompt_combo", "") for g, _ in pool)


def test_pool_order_is_frozen(tmp_path, monkeypatch):
    _write_catalog(tmp_path, monkeypatch)
    a = [(id(g), ri) for g, ri in states.load_minusG_states("gemma", n=100)]
    b = [(id(g), ri) for g, ri in states.load_minusG_states("gemma", n=100)]
    assert [ri for _, ri in a] == [ri for _, ri in b]  # same shuffled order (seed 42)
```

- [ ] **Step 2.2:** run → FAIL
- [ ] **Step 2.3:** implement `states.py`:

```python
"""−G slot-machine state pool (frozen, M3'' semantics) + HF catalog bootstrap."""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

SEED_BASE = 42
HF_REPO = "llm-addiction-research/llm-addiction"


def behavioral_root() -> Path:
    return Path(os.environ.get("LLM_ADDICTION_BEHAVIORAL_ROOT",
                               "/home/v-seungplee/data/llm-addiction/behavioral"))


def ensure_sm_catalog(model: str = "gemma") -> Path:
    """Download behavioral/slot_machine/{model}_v4_role/*.json from HF if absent."""
    dest = behavioral_root() / "slot_machine" / f"{model}_v4_role"
    if dest.exists() and any(dest.glob("*.json")):
        return dest
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    prefix = f"behavioral/slot_machine/{model}_v4_role/"
    files = [f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
             if f.startswith(prefix) and f.endswith(".json")]
    assert files, f"no catalog files under {prefix} on {HF_REPO}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
        (dest / Path(f).name).write_bytes(Path(p).read_bytes())
    return dest


def load_minusG_states(model: str, n: int = 200) -> list:
    """§3 −G variable (game, round_idx) pairs.

    PROVENANCE: logic identical to run_m3pp_strong_patching.load_minusG_states
    (rounds 2–6, variable betting, no 'G' in combo, random.Random(42) shuffle).
    """
    path = behavioral_root() / "slot_machine" / f"{model}_v4_role"
    states = []
    for game_file in sorted(path.glob("*.json")):
        d = json.load(open(game_file))
        games = d.get("results", d.get("games", []))
        if isinstance(games, dict):
            games = list(games.values())
        for game in games:
            if game.get("bet_type") != "variable":
                continue
            if "G" in game.get("prompt_combo", ""):
                continue
            n_decisions = len(game.get("decisions", []))
            for round_idx in range(2, min(7, n_decisions)):
                states.append((game, round_idx))
    rng = random.Random(SEED_BASE)
    rng.shuffle(states)
    return states[:n]
```

- [ ] **Step 2.4:** run → PASS
- [ ] **Step 2.5:** commit `feat(mlc): frozen -G state pool + HF catalog bootstrap`

### Task 3: hooks (patch / subspace / steer) + cache util

**Files:** Create `multilayer_causal/src/hooks.py`, `multilayer_causal/tests/test_hooks.py`.

- [ ] **Step 3.1: failing test** (mock decoder layers; no transformers):

```python
import torch
import torch.nn as nn
from multilayer_causal.src import hooks


class Layer(nn.Module):
    def forward(self, x):
        return (x + 1.0,)          # HF layers return tuples


class Mock(nn.Module):
    """model.model.layers structure, D=8, 4 layers."""
    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([Layer() for _ in range(4)])
        self.model = inner

    def forward(self, x):
        for l in self.model.layers:
            x = l(x)[0]
        return x


def test_patcher_replaces_suffix_once_on_prefill():
    m = Mock()
    cached = {1: torch.full((3, 8), 9.0), 2: torch.full((5, 8), 7.0)}
    p = hooks.MultiLayerPatcher(cached)
    p.install(m)
    out = m(torch.zeros(1, 5, 8))                  # prefill T=5
    # layer1 out would be 2.0; last 3 positions replaced by 9.0 then layer2..4 add 1 each
    # but layer2 replaces ALL 5 positions with 7.0 → final = 7+2 = 9
    assert torch.allclose(out, torch.full((1, 5, 8), 9.0))
    out2 = m(torch.zeros(1, 1, 8))                 # decode step T=1 → no patch
    assert torch.allclose(out2, torch.full((1, 1, 8), 4.0))
    p.remove()
    assert torch.allclose(m(torch.zeros(1, 5, 8)), torch.full((1, 5, 8), 4.0))


def test_subspace_patcher_moves_only_projection():
    m = Mock()
    D, r = 8, 1
    V = torch.zeros(D, r); V[0, 0] = 1.0           # subspace = e0
    plus = torch.full((4, D), 5.0)
    sp = hooks.SubspacePatcher({1: plus}, {1: V})
    sp.install(m)
    out = m(torch.zeros(1, 4, D))
    # live h at layer1 = 2.0 everywhere; delta=3 on all dims, projected → only dim0 moves
    # layer1 dim0: 2+3=5, others stay 2; layers 2..4 add +3 total
    assert torch.allclose(out[0, :, 0], torch.full((4,), 8.0))
    assert torch.allclose(out[0, :, 1:], torch.full((4, D - 1), 5.0))
    sp.remove()


def test_steerer_adds_every_forward_all_positions():
    m = Mock()
    v = torch.zeros(8); v[0] = 1.0
    st = hooks.MultiLayerSteerer({0: v, 3: v}, {0: 2.0, 3: 2.0}, alpha=1.5)
    st.install(m)
    out = m(torch.zeros(1, 2, 8))
    assert torch.allclose(out[0, :, 0], torch.full((2,), 4.0 + 2 * 1.5 * 2.0))
    out2 = m(torch.zeros(1, 1, 8))                 # fires again on decode-like pass
    assert torch.allclose(out2[0, :, 0], torch.full((1,), 4.0 + 2 * 1.5 * 2.0))
    st.remove()
```

- [ ] **Step 3.2:** run → FAIL
- [ ] **Step 3.3:** implement `hooks.py`:

```python
"""Multi-layer forward hooks. Batch size 1 only (M3/replay protocol).

HF decoder layers return tuples; element 0 is hidden states (B, T, D).
MultiLayerPatcher  — E1: replace last-k positions with cached +G twin values,
                     once per generation (first forward with T>1), per layer.
SubspacePatcher    — E2: same positions/timing, but move only the rank-r
                     projection: h += (h_plus − h) @ V @ V^T.
MultiLayerSteerer  — E3: h += alpha * scale_l * v_l at ALL positions on EVERY
                     forward (prefill + each decode step).
"""
from __future__ import annotations
import torch


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _repack(output, out):
    return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out


class _HookSet:
    def __init__(self):
        self._handles = []

    def _layers(self, model):
        return model.model.layers

    def install(self, model):
        layers = self._layers(model)
        for li in self.layer_indices:
            assert 0 <= li < len(layers), f"layer {li} out of range ({len(layers)})"
            self._handles.append(layers[li].register_forward_hook(self._hook_for(li)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class MultiLayerPatcher(_HookSet):
    def __init__(self, cached):
        super().__init__()
        self.cached = {int(k): v for k, v in cached.items()}
        self.layer_indices = sorted(self.cached)
        self._fired = {li: 0 for li in self.layer_indices}

    def _hook_for(self, li):
        patch_h = self.cached[li]

        def hook(module, _input, output):
            out = _hidden(output)
            assert out.shape[0] == 1, "batch size must be 1"
            if out.shape[1] > 1 and self._fired[li] == 0:
                self._fired[li] += 1
                k = min(out.shape[1], patch_h.shape[0])
                out[0, -k:, :] = patch_h[-k:].to(out.dtype).to(out.device)
            return _repack(output, out)

        return hook


class SubspacePatcher(_HookSet):
    def __init__(self, cached, bases):
        super().__init__()
        self.cached = {int(k): v for k, v in cached.items()}
        self.bases = {int(k): v for k, v in bases.items()}
        assert set(self.cached) == set(self.bases)
        self.layer_indices = sorted(self.cached)
        self._fired = {li: 0 for li in self.layer_indices}

    def _hook_for(self, li):
        patch_h, V = self.cached[li], self.bases[li]

        def hook(module, _input, output):
            out = _hidden(output)
            assert out.shape[0] == 1, "batch size must be 1"
            if out.shape[1] > 1 and self._fired[li] == 0:
                self._fired[li] += 1
                k = min(out.shape[1], patch_h.shape[0])
                Vd = V.to(out.dtype).to(out.device)
                delta = patch_h[-k:].to(out.dtype).to(out.device) - out[0, -k:, :]
                out[0, -k:, :] += (delta @ Vd) @ Vd.T
            return _repack(output, out)

        return hook


class MultiLayerSteerer(_HookSet):
    def __init__(self, directions, scales, alpha):
        super().__init__()
        self.directions = {int(k): v for k, v in directions.items()}
        self.scales = {int(k): float(v) for k, v in scales.items()}
        assert set(self.directions) == set(self.scales)
        self.layer_indices = sorted(self.directions)
        self.alpha = float(alpha)

    def _hook_for(self, li):
        add = self.alpha * self.scales[li] * self.directions[li]

        def hook(module, _input, output):
            out = _hidden(output)
            out += add.to(out.dtype).to(out.device)
            return _repack(output, out)

        return hook


def cache_layer_outputs(model, tokenizer, device, prompt_text, layer_indices):
    """One no-grad forward over prompt_text; capture each layer's (T, D) output."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    cached, handles = {}, []
    layers = model.model.layers
    for li in layer_indices:
        def mk(li):
            def hook(module, _i, output):
                cached[li] = _hidden(output)[0].detach().clone()
            return hook
        handles.append(layers[li].register_forward_hook(mk(li)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return cached, inputs["input_ids"][0]
```

- [ ] **Step 3.4:** run → PASS
- [ ] **Step 3.5:** commit `feat(mlc): multi-layer patch/subspace/steer hooks`

### Task 4: checkpoint + vector store (resume & HF latest-state sync)

**Files:** Create `multilayer_causal/src/checkpoint.py`, `multilayer_causal/tests/test_checkpoint.py`.

- [ ] **Step 4.1: failing test**

```python
import json
import numpy as np
from multilayer_causal.src.checkpoint import ArmCheckpoint, VectorStore


def test_record_resume_roundtrip(tmp_path):
    ck = ArmCheckpoint("e1", "full", tmp_path, sync_every=2, hf_enabled=False)
    assert ck.done_seeds() == set()
    ck.record({"seed": 42, "x": 1})
    ck.record({"seed": 1039, "x": 2})
    ck2 = ArmCheckpoint("e1", "full", tmp_path, hf_enabled=False)
    assert ck2.done_seeds() == {42, 1039}
    lines = [json.loads(l) for l in open(tmp_path / "full.jsonl")]
    assert [l["x"] for l in lines] == [1, 2]


def test_sync_called_every_n(tmp_path, monkeypatch):
    calls = []
    ck = ArmCheckpoint("e1", "full", tmp_path, sync_every=2, hf_enabled=True)
    monkeypatch.setattr(ck, "_upload", lambda: calls.append(1))
    for s in range(5):
        ck.record({"seed": s})
    assert len(calls) == 2          # after 2 and 4


def test_vector_store_roundtrip(tmp_path):
    vs = VectorStore(tmp_path / "vec.npz", n_layers=3, d_model=4)
    vs.append(seed=1, minus=np.ones((3, 4)), plus=2 * np.ones((3, 4)))
    vs.save()
    vs2 = VectorStore(tmp_path / "vec.npz", n_layers=3, d_model=4)
    assert vs2.seeds == [1]
    vs2.append(seed=2, minus=np.zeros((3, 4)), plus=np.zeros((3, 4)))
    vs2.save()
    vs3 = VectorStore(tmp_path / "vec.npz", n_layers=3, d_model=4)
    assert vs3.seeds == [1, 2] and vs3.minus.shape == (2, 3, 4)
```

- [ ] **Step 4.2:** run → FAIL
- [ ] **Step 4.3:** implement `checkpoint.py`:

```python
"""Per-arm JSONL checkpointing with HF latest-state sync (preemption-safe).

HF layout (dataset llm-addiction-research/llm-addiction):
  experiments/multilayer_causal/checkpoints/{phase}/{arm}.jsonl   <- overwritten
  experiments/multilayer_causal/checkpoints/{phase}/{arm}_vectors.npz
The same path is overwritten each sync: the dataset always shows the LATEST
resumable state (git history keeps the trail). Jobs download their checkpoint
at start, so a preempted job resumes by plain resubmission.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

HF_REPO = "llm-addiction-research/llm-addiction"
HF_BASE = "experiments/multilayer_causal/checkpoints"


class ArmCheckpoint:
    def __init__(self, phase, arm_id, out_dir, sync_every=10, hf_enabled=None):
        self.phase, self.arm_id = phase, arm_id
        self.path = Path(out_dir) / f"{arm_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hf_path = f"{HF_BASE}/{phase}/{arm_id}.jsonl"
        self.token = os.environ.get("HF_TOKEN")
        self.hf_enabled = bool(self.token) if hf_enabled is None else hf_enabled
        self.sync_every = sync_every
        self._since = 0
        if not self.path.exists() and self.hf_enabled:
            self._download()

    def _download(self):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(HF_REPO, self.hf_path, repo_type="dataset",
                                token=self.token)
            self.path.write_bytes(Path(p).read_bytes())
            print(f"[ckpt] resumed {self.arm_id} from HF "
                  f"({len(self.done_seeds())} trials)", flush=True)
        except Exception:
            pass  # nothing to resume

    def done_seeds(self):
        seeds = set()
        if self.path.exists():
            for line in open(self.path):
                try:
                    seeds.add(json.loads(line)["seed"])
                except Exception:
                    pass
        return seeds

    def record(self, result):
        with open(self.path, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._since += 1
        if self.hf_enabled and self._since % self.sync_every == 0:
            self._upload()

    def final_sync(self):
        if self.hf_enabled:
            self._upload()

    def _upload(self):
        try:
            from huggingface_hub import HfApi
            HfApi(token=self.token).upload_file(
                path_or_fileobj=str(self.path), path_in_repo=self.hf_path,
                repo_id=HF_REPO, repo_type="dataset",
                commit_message=f"ckpt {self.phase}/{self.arm_id}")
        except Exception as e:  # never kill the run over a sync hiccup
            print(f"[ckpt] HF sync failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)


class VectorStore:
    """Per-arm (n, L, D) fp16 stacks of −G / +G last-token hidden vectors."""

    def __init__(self, path, n_layers, d_model):
        self.path = Path(path)
        self.n_layers, self.d_model = n_layers, d_model
        if self.path.exists():
            z = np.load(self.path)
            self.seeds = list(z["seeds"])
            self._minus = [m for m in z["minus"]]
            self._plus = [p for p in z["plus"]]
        else:
            self.seeds, self._minus, self._plus = [], [], []

    @property
    def minus(self):
        return np.stack(self._minus) if self._minus else np.zeros((0, self.n_layers, self.d_model))

    @property
    def plus(self):
        return np.stack(self._plus) if self._plus else np.zeros((0, self.n_layers, self.d_model))

    def append(self, seed, minus, plus):
        assert minus.shape == (self.n_layers, self.d_model)
        assert plus.shape == (self.n_layers, self.d_model)
        self.seeds.append(int(seed))
        self._minus.append(minus.astype(np.float16))
        self._plus.append(plus.astype(np.float16))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, seeds=np.array(self.seeds),
                            minus=self.minus.astype(np.float16),
                            plus=self.plus.astype(np.float16))
```

- [ ] **Step 4.4:** run → PASS
- [ ] **Step 4.5:** commit `feat(mlc): arm checkpoint with HF latest-state sync + vector store`

### Task 5: arm registry (arms.yaml + loader)

**Files:** Create `multilayer_causal/configs/arms.yaml`, `multilayer_causal/src/registry.py`, `multilayer_causal/tests/test_registry.py`.

- [ ] **Step 5.1: failing test**

```python
from multilayer_causal.src.registry import load_arms


def test_e1_registry_complete():
    arms = load_arms()
    e1 = [a for a in arms.values() if a["phase"] == "e1"]
    assert len(e1) == 19                                   # 17 interventions + 2 anchors
    full = arms["e1_full"]
    assert full["mode"] == "patch" and full["layers"] == list(range(42))
    assert full.get("log_vectors") is True
    assert arms["e1_anchor_minus"]["mode"] == "anchor_minus"
    sliding = [a for a in e1 if a["id"].startswith("e1_win")]
    assert len(sliding) == 7
    covered = sorted(l for a in sliding for l in a["layers"])
    assert covered == list(range(42))                      # tiling, no gaps/overlap


def test_ids_unique_and_layers_valid():
    arms = load_arms()
    assert len(arms) == len(set(arms))
    for a in arms.values():
        for l in a.get("layers", []):
            assert 0 <= l <= 41
```

- [ ] **Step 5.2:** run → FAIL
- [ ] **Step 5.3:** write `configs/arms.yaml`:

```yaml
# Arm registry — multilayer causal intervention (spec 2026-06-10).
# layers: [a, b] = INCLUSIVE range; n: default trials; modes:
#   anchor_minus / anchor_plus / patch / subspace / steer
defaults:
  model: gemma
  task: sm
  n: 50

arms:
  # ---- E1: anchors + multi-layer paired patch sweep -----------------------
  - {id: e1_anchor_minus, phase: e1, mode: anchor_minus}
  - {id: e1_anchor_plus,  phase: e1, mode: anchor_plus}
  - {id: e1_full,         phase: e1, mode: patch, layers: [0, 41], log_vectors: true}
  - {id: e1_cum_b8,       phase: e1, mode: patch, layers: [0, 8]}
  - {id: e1_cum_b16,      phase: e1, mode: patch, layers: [0, 16]}
  - {id: e1_cum_b22,      phase: e1, mode: patch, layers: [0, 22]}
  - {id: e1_cum_b30,      phase: e1, mode: patch, layers: [0, 30]}
  - {id: e1_cum_b36,      phase: e1, mode: patch, layers: [0, 36]}
  - {id: e1_cum_t8,       phase: e1, mode: patch, layers: [8, 41]}
  - {id: e1_cum_t16,      phase: e1, mode: patch, layers: [16, 41]}
  - {id: e1_cum_t22,      phase: e1, mode: patch, layers: [22, 41]}
  - {id: e1_cum_t30,      phase: e1, mode: patch, layers: [30, 41]}
  - {id: e1_win_0,        phase: e1, mode: patch, layers: [0, 5]}
  - {id: e1_win_6,        phase: e1, mode: patch, layers: [6, 11]}
  - {id: e1_win_12,       phase: e1, mode: patch, layers: [12, 17]}
  - {id: e1_win_18,       phase: e1, mode: patch, layers: [18, 23]}
  - {id: e1_win_24,       phase: e1, mode: patch, layers: [24, 29]}
  - {id: e1_win_30,       phase: e1, mode: patch, layers: [30, 35]}
  - {id: e1_win_36,       phase: e1, mode: patch, layers: [36, 41]}
  # ---- E2/E3 arms are appended after the E1 gate fixes S* ------------------
```

- [ ] **Step 5.4:** implement `src/registry.py`:

```python
"""Load + validate configs/arms.yaml. layers: [a,b] inclusive → expanded list."""
from __future__ import annotations
from pathlib import Path

import yaml

ARMS_YAML = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"
MODES = {"anchor_minus", "anchor_plus", "patch", "subspace", "steer"}


def load_arms(path=ARMS_YAML):
    cfg = yaml.safe_load(open(path))
    defaults = cfg.get("defaults", {})
    arms = {}
    for raw in cfg["arms"]:
        a = {**defaults, **raw}
        assert a["mode"] in MODES, f"{a['id']}: bad mode {a['mode']}"
        assert a["id"] not in arms, f"duplicate arm id {a['id']}"
        if "layers" in a:
            lo, hi = a["layers"]
            assert 0 <= lo <= hi <= 41, f"{a['id']}: bad layer range"
            a["layers"] = list(range(lo, hi + 1))
        arms[a["id"]] = a
    return arms
```

- [ ] **Step 5.5:** run → PASS; commit `feat(mlc): declarative E1 arm registry`

### Task 6: runner + CLI

**Files:** Create `multilayer_causal/src/runner.py`, `multilayer_causal/run_experiment.py`. No new unit test (GPU path); the smoke job is the integration test. Registry/mode dispatch is covered by Task 5; hooks by Task 3.

- [ ] **Step 6.1:** implement `src/runner.py`:

```python
"""Trial loop: one arm = N single-decision trials (M3'' protocol, multi-layer)."""
from __future__ import annotations
import os
import re
import time
from datetime import datetime

import numpy as np

from .prompts import build_prompt, parse_response
from .states import load_minusG_states, ensure_sm_catalog
from .hooks import (MultiLayerPatcher, SubspacePatcher, MultiLayerSteerer,
                    cache_layer_outputs)
from .checkpoint import ArmCheckpoint, VectorStore

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
    assert len(model.model.layers) == N_LAYERS
    assert model.config.hidden_size == D_MODEL
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
    """directions npz: keys 'directions' (L,D), 'scales' (L,) built by analyze.py."""
    import torch
    z = np.load(arm["directions_npz"])
    dirs = {li: torch.tensor(z["directions"][li], dtype=torch.float32)
            for li in arm["layers"]}
    scales = {li: float(z["scales"][li]) for li in arm["layers"]}
    return dirs, scales


def _load_bases(arm):
    import torch
    z = np.load(arm["basis_npz"])
    if arm.get("basis", "pca") == "random":
        rng = np.random.Generator(np.random.PCG64(int(arm["basis_seed"])))
        bases = {}
        for li in arm["layers"]:
            g = rng.standard_normal((D_MODEL, int(arm["r"])))
            q, _ = np.linalg.qr(g)
            bases[li] = torch.tensor(q, dtype=torch.float32)
        return bases
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
    model, tok, device = load_model(gpu)
    done = ck.done_seeds()
    print(f"[{arm['id']}] n={n} done={len(done)}", flush=True)

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
                cached_pf, _ = cache_layer_outputs(model, tok, device, plus_p,
                                                   list(range(N_LAYERS)))
                vec.append(seed,
                           np.stack([cached_m[l][-1].float().cpu().numpy()
                                     for l in range(N_LAYERS)]),
                           np.stack([cached_pf[l][-1].float().cpu().numpy()
                                     for l in range(N_LAYERS)]))
        elif mode == "subspace":
            cached, _ = cache_layer_outputs(model, tok, device, plus_p, arm["layers"])
            hookset = SubspacePatcher(cached, _load_bases(arm))
        elif mode == "steer":
            dirs, scales = _load_steer_assets(arm)
            hookset = MultiLayerSteerer(dirs, scales, alpha=float(arm["alpha"]))
        elif mode != "anchor_minus":
            raise ValueError(mode)

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
```

- [ ] **Step 6.2:** implement `run_experiment.py`:

```python
#!/usr/bin/env python3
"""CLI: python run_experiment.py --arm e1_full --gpu 0 [--n 50] [--smoke]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from multilayer_causal.src.registry import load_arms          # noqa: E402
from multilayer_causal.src.runner import run_arm              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    arms = load_arms()
    assert args.arm in arms, f"unknown arm {args.arm}; have {sorted(arms)}"
    run_arm(arms[args.arm], args.out, gpu=args.gpu, n=args.n, smoke=args.smoke)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.3:** full test suite `pytest multilayer_causal/tests -q` → PASS (runner not imported by tests; verify `python -c "import ast; ast.parse(open('multilayer_causal/src/runner.py').read())"`).
- [ ] **Step 6.4:** commit `feat(mlc): trial runner + CLI`

### Task 7: analysis + gates

**Files:** Create `multilayer_causal/src/analyze.py`.

- [ ] **Step 7.1:** implement (read all `{out}/{arm}.jsonl`, summarize, evaluate gates):

```python
"""Per-arm summaries, anchor comparisons, G1 gate, S* selection, direction build.

Usage:
  python -m multilayer_causal.src.analyze summary --out multilayer_causal/out
  python -m multilayer_causal.src.analyze directions --vectors out/e1_full_vectors.npz \
      --dest out/directions.npz
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_arm(out_dir, arm_id):
    p = Path(out_dir) / f"{arm_id}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def summarize(trials):
    if not trials:
        return None
    br = np.array([t["bet_ratio"] for t in trials], float)
    stop = np.array([t["action"] == "stop" for t in trials], float)
    return {
        "n": len(trials),
        "bet_ratio_mean": float(br.mean()),
        "bet_ratio_ci95": [float(x) for x in np.percentile(
            [np.random.default_rng(0).choice(br, len(br)).mean()
             for _ in range(1000)], [2.5, 97.5])],
        "stop_rate": float(stop.mean()),
        "extreme_rate": float(np.mean([t["extreme"] for t in trials])),
        "parse_rate": float(np.mean([t["parse_ok"] for t in trials])),
    }


def welch_vs(trials_a, trials_b, key="bet_ratio"):
    a = np.array([t[key] for t in trials_a], float)
    b = np.array([t[key] for t in trials_b], float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def cohen_h(p1, p2):
    import math
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def gate_g1(out_dir):
    """Full-layer patch must be indistinguishable from natural_plusG."""
    full = load_arm(out_dir, "e1_full")
    plus = load_arm(out_dir, "e1_anchor_plus")
    if not full or not plus:
        return {"pass": False, "reason": "missing arms"}
    _, p = welch_vs(full, plus)
    ds = abs(summarize(full)["stop_rate"] - summarize(plus)["stop_rate"])
    return {"pass": bool(p > 0.05 and ds < 0.15), "welch_p_vs_plus": p,
            "stop_rate_gap": ds}


def select_s_star(out_dir, arms):
    """Smallest passing layer set; ties → deeper window. Pre-registered rule."""
    plus = load_arm(out_dir, "e1_anchor_plus")
    passing = []
    for aid, a in arms.items():
        if a["phase"] != "e1" or a["mode"] != "patch":
            continue
        tr = load_arm(out_dir, aid)
        if len(tr) < 30:
            continue
        _, p = welch_vs(tr, plus)
        ds = abs(summarize(tr)["stop_rate"] - summarize(plus)["stop_rate"])
        if p > 0.05 and ds < 0.15:
            passing.append((len(a["layers"]), -min(a["layers"]), aid, a["layers"]))
    if not passing:
        return None
    passing.sort()
    return {"arm": passing[0][2], "layers": passing[0][3],
            "all_passing": [x[2] for x in passing]}


def build_directions(vectors_npz, dest):
    """v̂_l = normalize(mean(plus − minus)); scale_l = 0.03 · median ||minus_l||."""
    z = np.load(vectors_npz)
    minus, plus = z["minus"].astype(np.float32), z["plus"].astype(np.float32)
    assert minus.shape == plus.shape and minus.ndim == 3
    delta = (plus - minus).mean(axis=0)                      # (L, D)
    dirs = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9)
    scales = 0.03 * np.median(np.linalg.norm(minus, axis=2), axis=0)  # (L,)
    np.savez(dest, directions=dirs, scales=scales, n_pairs=minus.shape[0])
    print(f"directions → {dest} (n_pairs={minus.shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    d = sub.add_parser("directions")
    d.add_argument("--vectors", required=True)
    d.add_argument("--dest", required=True)
    args = ap.parse_args()
    if args.cmd == "directions":
        build_directions(args.vectors, args.dest)
        return
    from .registry import load_arms
    arms = load_arms()
    plus = load_arm(args.out, "e1_anchor_plus")
    minus = load_arm(args.out, "e1_anchor_minus")
    rows = {}
    for aid in arms:
        tr = load_arm(args.out, aid)
        if not tr:
            continue
        srow = summarize(tr)
        if plus and aid not in ("e1_anchor_plus",):
            srow["welch_p_vs_plusG"] = welch_vs(tr, plus)[1]
        if minus and aid not in ("e1_anchor_minus",):
            srow["welch_p_vs_minusG"] = welch_vs(tr, minus)[1]
        rows[aid] = srow
    report = {"arms": rows, "gate_g1": gate_g1(args.out),
              "s_star": select_s_star(args.out, arms)}
    dest = Path(args.out) / "summary.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2:** `pytest multilayer_causal/tests -q` still PASS; `python -m multilayer_causal.src.analyze summary --out /tmp/empty` runs (empty report).
- [ ] **Step 7.3:** commit `feat(mlc): analysis, G1 gate, S* selection, direction builder`

### Task 8: job packaging (tarball push, GPU sharding, amlt yamls)

**Files:** Create `multilayer_causal/scripts/push_code_to_hf.py`, `multilayer_causal/run_arms.sh`, `multilayer_causal/amlt/smoke.yaml`, `multilayer_causal/amlt/e1_main.yaml`.

- [ ] **Step 8.1:** `scripts/push_code_to_hf.py`:

```python
#!/usr/bin/env python3
"""Tar the harness (+ read-only deps) and upload LATEST to the HF dataset."""
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "llm-addiction-research/llm-addiction"
DEST = "experiments/multilayer_causal/code/multilayer_causal.tar.gz"
PATHS = ["multilayer_causal"]          # self-contained for E1/E2/E3a


def main():
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "multilayer_causal.tar.gz"
        subprocess.run(["tar", "czf", str(tar), "--exclude", "multilayer_causal/out",
                        "--exclude", "__pycache__", *PATHS],
                       cwd=REPO_ROOT, check=True)
        from huggingface_hub import HfApi
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=str(tar), path_in_repo=DEST,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="code: multilayer_causal latest")
        print(f"pushed {tar.stat().st_size/1e6:.1f}MB → {DEST}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2:** `run_arms.sh`:

```bash
#!/bin/bash
# Shard arms across visible GPUs, sequential per GPU, resume-safe.
# Usage: bash run_arms.sh <NGPUS> <arm1> <arm2> ...
set -uo pipefail
NGPUS=$1; shift
declare -a Q
i=0
for arm in "$@"; do
  g=$((i % NGPUS)); Q[$g]="${Q[$g]:-} $arm"; i=$((i+1))
done
for g in $(seq 0 $((NGPUS-1))); do
  (
    for arm in ${Q[$g]:-}; do
      echo "[gpu$g] start $arm"
      python multilayer_causal/run_experiment.py --arm "$arm" --gpu "$g" \
        || echo "[gpu$g] $arm FAILED (continuing)"
    done
  ) &
done
wait
echo "ALL ARMS DONE"
```

- [ ] **Step 8.3:** `amlt/smoke.yaml` (metacognition-math pattern; HF_TOKEN value inserted at submission time after verifying write access with `HfApi(token=...).whoami()`):

```yaml
description: multilayer-causal SMOKE — env + harness check (full-layer patch, n=3)

target:
  service: sing
  name: msrresrchbasicvc
  workspace_name: msra-sh-aml-ws

environment:
  image: amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04

code:
  local_dir: $CONFIG_DIR/

jobs:
  - name: mlc_smoke
    sku: 80G4-H100
    sla_tier: Standard
    priority: high
    identity: managed
    submit_args:
      max_run_duration_seconds: 7200
      env:
        _AZUREML_SINGULARITY_JOB_UAI: "<same UAI as metacognition-math yamls>"
        HF_TOKEN: "<inserted at submission>"
        HUGGING_FACE_HUB_TOKEN: "<inserted at submission>"
    command:
      - nvidia-smi
      - |
        bash -c '
        set -x
        source /opt/conda/etc/profile.d/conda.sh
        conda activate ptca
        pip install -q "huggingface_hub<1.0" "transformers>=4.44" accelerate scipy pyyaml 2>&1 | tail -1
        mkdir -p /scratch/mlc && cd /scratch/mlc
        python - <<EOF
        from huggingface_hub import hf_hub_download
        import os, tarfile
        p = hf_hub_download("llm-addiction-research/llm-addiction",
                            "experiments/multilayer_causal/code/multilayer_causal.tar.gz",
                            repo_type="dataset", token=os.environ["HF_TOKEN"])
        tarfile.open(p).extractall(".")
        EOF
        export LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/mlc/data/behavioral
        python multilayer_causal/run_experiment.py --arm e1_anchor_plus --gpu 0 --smoke
        python multilayer_causal/run_experiment.py --arm e1_full --gpu 0 --smoke
        echo SMOKE_OK
        '
```

- [ ] **Step 8.4:** `amlt/e1_main.yaml` — same skeleton; command tail:

```yaml
        export LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/mlc/data/behavioral
        bash multilayer_causal/run_arms.sh 4 \
          e1_anchor_minus e1_anchor_plus e1_full \
          e1_cum_b8 e1_cum_b16 e1_cum_b22 e1_cum_b30 e1_cum_b36 \
          e1_cum_t8 e1_cum_t16 e1_cum_t22 e1_cum_t30 \
          e1_win_0 e1_win_6 e1_win_12 e1_win_18 e1_win_24 e1_win_30 e1_win_36
        python -m multilayer_causal.src.analyze summary --out multilayer_causal/out
        python - <<EOF
        import os
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj="multilayer_causal/out/summary.json",
            path_in_repo="experiments/multilayer_causal/results/e1_summary.json",
            repo_id="llm-addiction-research/llm-addiction", repo_type="dataset",
            commit_message="e1 summary")
        EOF
```

with `max_run_duration_seconds: 172800`.

- [ ] **Step 8.5:** commit `feat(mlc): job packaging — HF code push, GPU sharding, amlt yamls`

### Task 9: PCA basis builder (E2 prep, runs on node or big-RAM box)

**Files:** Create `multilayer_causal/src/pca_basis.py`.

- [ ] **Step 9.1:** implement:

```python
"""Build per-layer rank-128 PCA bases from phase_a_hidden_states.npz (Gemma SM).

Run once before E2 (needs ~8GB RAM + the 6GB npz):
  python -m multilayer_causal.src.pca_basis --dest out/pca_bases.npz
Uploads to experiments/multilayer_causal/assets/pca_bases_gemma_sm.npz.
"""
from __future__ import annotations
import argparse
import os

import numpy as np

HF_REPO = "llm-addiction-research/llm-addiction"
SRC = "sae_features_v3/slot_machine/gemma/checkpoint/phase_a_hidden_states.npz"
R_MAX = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--src-local", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    src = args.src_local
    if src is None:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(HF_REPO, SRC, repo_type="dataset", token=token)
    z = np.load(src, mmap_mode="r")
    hs = z["hidden_states"]                     # (n_rounds, L, D) fp16
    n, L, D = hs.shape
    out = {}
    from sklearn.utils.extmath import randomized_svd
    for l in range(L):
        X = np.asarray(hs[:, l, :], dtype=np.float32)
        X -= X.mean(axis=0, keepdims=True)
        _, _, Vt = randomized_svd(X, n_components=R_MAX, random_state=0)
        out[f"L{l}"] = Vt.T.astype(np.float32)  # (D, R_MAX)
        print(f"L{l} done", flush=True)
    np.savez_compressed(args.dest, **out)
    if not args.no_upload and token:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=args.dest,
            path_in_repo="experiments/multilayer_causal/assets/pca_bases_gemma_sm.npz",
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="pca bases gemma sm L0-41 r128")


if __name__ == "__main__":
    main()
```

NOTE: verify `phase_a` array layout (`(n, L, D)` vs dict-of-layers) against
`phase_a_metadata.json` before first run; adjust the indexing line if needed.

- [ ] **Step 9.2:** `pytest -q` still green; commit `feat(mlc): PCA basis builder for E2`

### Task 10: README + final review

**Files:** Create `multilayer_causal/README.md`.

- [ ] **Step 10.1:** README: purpose, spec link, arm table, HF paths
  (`experiments/multilayer_causal/{code,checkpoints,results,assets}`), local quickstart
  (`pytest`, `--smoke`), node flow (push_code → amlt smoke → e1_main), resume semantics,
  DO-NOT-EDIT provenance note for `prompts.py`/`states.py`.
- [ ] **Step 10.2:** run full suite `pytest multilayer_causal/tests -q` → all PASS.
- [ ] **Step 10.3:** commit `docs(mlc): README` and push branch `feat/multilayer-causal`.

---

## Self-review

- **Spec coverage**: E1 arms ✓ (Task 5), E2 subspace + random controls ✓ (hooks Task 3, `_load_bases` Task 6, bases Task 9; E2 arm entries added at gate time by design), E3a steering ✓ (hooks + `_load_steer_assets` + directions builder Task 7), E3b full-game = explicitly gated post-E1 (spec §2.3; will reuse `exact_behavioral_replay` — not in this plan's tasks, planned at gate), checkpoint/resume/HF-latest sync ✓ (Task 4), monitoring = operational (post-plan), amlt ✓ (Task 8), stats/gates ✓ (Task 7).
- **Placeholders**: amlt yamls carry two values resolved at submission time (UAI string copied from metacognition yaml; HF token verified then inserted) — deliberate secret-handling, not a design gap.
- **Type consistency**: arm dict keys (`id/phase/mode/layers/n/r/alpha/basis/basis_seed/directions_npz/basis_npz/log_vectors`) consistent across registry/runner/analyze; `VectorStore.append(seed, minus, plus)` matches runner usage; `ArmCheckpoint(phase, arm_id, out_dir, ...)` consistent.
