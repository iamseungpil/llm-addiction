"""W8 STEP 1 - model-aware infra + LLaMA SM window-scan config/template parity.

SYNTHETIC only (no torch / no HF / no real model): MODEL_DIMS resolution
(gemma default byte-identical, llama additive), model-aware _replicate_to_full
shape, a gemma NON-REGRESSION guard that the frozen wave6 build-path constants
stay 42/3584, the readout-axis llama NotImplementedError guard, and the
arms_sec4_w8scan.yaml <-> amlt/sec4_w8scan.yaml.template consistency.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from multilayer_causal.src import behavior_axis as ba
from multilayer_causal.src import indicator_axes as ia
from multilayer_causal.src import paper_axes as pa
from multilayer_causal.src import runner
from multilayer_causal.src.registry import load_arms

MLC = Path(__file__).resolve().parents[1]
W8_YAML = MLC / "configs" / "arms_sec4_w8scan.yaml"
W8_TEMPLATE = MLC / "amlt" / "sec4_w8scan.yaml.template"
WINDOWS = [(14, 19), (16, 21), (18, 23), (20, 25)]
W8_NPZ = tuple(f"llama_slot_machine_i_ba_behavioural_L{lo}_{hi}.npz"
               for lo, hi in WINDOWS)


# ------------------------------------------------------- MODEL_DIMS registry

def test_model_dims_registry_and_defaults():
    assert pa.MODEL_DIMS["gemma"] == (42, 3584)
    assert pa.MODEL_DIMS["llama"] == (32, 4096)
    # every module-level gemma constant must resolve byte-identically from the
    # single map — existing gemma arms/assets/tests are unchanged.
    assert (pa.N_LAYERS, pa.D_MODEL) == (42, 3584)
    assert (ia.N_LAYERS, ia.D_MODEL) == (42, 3584)
    assert (runner.N_LAYERS, runner.D_MODEL) == (42, 3584)
    # model-aware SM catalog: gemma stays the frozen v4_role file, llama added.
    assert pa.sm_catalog_for("gemma") == pa.SM_CATALOG
    assert "llama_v4_role" in pa.sm_catalog_for("llama")


# ---------------------------------------------------- _replicate_to_full shape

def test_replicate_to_full_model_aware_shape():
    rs = np.random.RandomState(0)
    gemma_layers = [16, 17, 18, 19, 20, 21]
    dirs = rs.randn(len(gemma_layers), 8).astype(np.float32)
    # default (gemma) -> 42 rows, byte-identical to the pre-W8 behaviour
    g = ia._replicate_to_full(dirs, gemma_layers)
    assert g.shape == (42, 8)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=1e-5)
    # llama window -> 32 rows; the requested layers carry the built directions
    llama_layers = [14, 15, 16, 17, 18, 19]
    l = ia._replicate_to_full(dirs, llama_layers, pa.MODEL_DIMS["llama"][0])
    assert l.shape == (32, 8)
    for li, lay in enumerate(llama_layers):
        assert np.allclose(l[lay], dirs[li] / np.linalg.norm(dirs[li]), atol=1e-5)


# -------------------------------------------------- gemma non-regression guard

def test_gemma_wave6_constants_nonregression():
    """The object-frozen gemma build path must still read 42/3584 everywhere."""
    assert (ba.N_LAYERS, ba.D_MODEL) == (42, 3584)
    assert pa.N_LAYERS == 42 and pa.D_MODEL == 3584
    assert ia.N_LAYERS == 42
    # _replicate_to_full / _save_axis default to gemma geometry (42 rows)
    assert ia._replicate_to_full(np.ones((1, 4), np.float32), [0]).shape == (42, 4)


def test_readout_axis_requires_decoder_llama_not_implemented():
    with pytest.raises(NotImplementedError):
        ia.build_readout_axis_from_arrays(
            feats=np.zeros((10, 4)), indicator=np.zeros(10),
            balance=np.zeros(10), rounds=np.zeros(10), groups=np.arange(10),
            decoder=None, layers=[0])


# ------------------------------------------------------ w8scan registry parity

def test_w8_registry_counts_and_fields():
    arms = load_arms(W8_YAML)
    assert len(arms) == 15
    steer = {k: a for k, a in arms.items()
             if a["mode"] == "steer" and a.get("direction") != "random"}
    assert len(steer) == 12  # 4 windows x 3 alphas
    for a in arms.values():
        assert a["model"] == "llama"
        assert a.get("task", "sm") == "sm"
        assert a["n"] == 150
        assert a["seed_base"] == 4000042
        assert a["state_offset"] == 0
        assert a["prompt_set"] == "addiction_role_gm"
    # every (window, alpha) cell present, each pointing at its own window npz
    seen = set()
    for a in steer.values():
        lo, hi = a["layers"][0], a["layers"][-1]
        seen.add((lo, hi, a["alpha"]))
        assert a["directions_npz"].endswith(
            f"llama_slot_machine_i_ba_behavioural_L{lo}_{hi}.npz")
    for lo, hi in WINDOWS:
        for al in (-3, 0, 3):
            assert (lo, hi, al) in seen, (lo, hi, al)
    # one no-steer baseline + exactly 2 random nulls
    assert sum(1 for a in arms.values() if a["mode"] == "anchor_minus") == 1
    assert sum(1 for a in arms.values() if a.get("direction") == "random") == 2
    # llama layer bounds honoured (top window [20,25] <= 31)
    assert all(max(a["layers"]) <= 31 for a in arms.values())


def test_w8_template_lists_exactly_the_registry_arms():
    arms = load_arms(W8_YAML)
    text = W8_TEMPLATE.read_text()
    m = re.search(r"run_arms\.sh \d+ ([^\n]+)", text)
    assert m, "no run_arms.sh line in sec4_w8scan.yaml.template"
    listed = m.group(1).split()
    assert sorted(listed) == sorted(arms), set(listed) ^ set(arms)
    # YAML guard: description must use ' - ', never ': ' (amlt parse trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # build step builds each candidate window via the llama behavioural CLI
    assert "--build-llama-windows" in text and "--model llama" in text
    for lo, hi in WINDOWS:
        assert f"{lo} {hi}" in text  # window passed to --layers
    for npz in W8_NPZ:
        assert npz in text
    # the need-list double-quotes MUST be backslash-escaped (\") inside the
    # bash single-quoted python -c block (an unescaped quote kills submission)
    need = re.search(r"need = \[([^\]]+)\]", text)
    assert need, "no need-list in sec4_w8scan.yaml.template"
    assert re.findall(r'(?<!\\)"', need.group(1)) == []
    assert need.group(1).count('\\"') == 2 * len(W8_NPZ)
    # serial llama catalog bootstrap before the fan-out (escaped-quote form so
    # no literal single-quote breaks the bash -c block)
    assert 'ensure_sm_catalog(\\"llama\\")' in text
    # isolation env
    assert "MLC_OUT: multilayer_causal/results/sec4_w8scan" in text
    assert "MLC_ARMS_YAML: multilayer_causal/configs/arms_sec4_w8scan.yaml" in text
    assert 'MLC_SYNC_EVERY: "50"' in text
