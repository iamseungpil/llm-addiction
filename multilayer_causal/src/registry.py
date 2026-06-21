"""Load + validate configs/arms.yaml.

layers: [a, b] inclusive range → expanded list.
layers_list: explicit (possibly non-contiguous) layer indices → stored as
a['layers']; mutually exclusive with layers (W1 bridge arms).
Optional pass-through fields validated here: probe (bool, SM-only),
task ('sm' | 'ic' | 'mw'), direction ('random', needs dir_seed), seed_base,
state_offset, dir_seed, twin_component ('G' | 'M', patch mode only — W3
+M mediation arms), model ('gemma' 42 layers | 'llama' 32 layers, layer
bounds enforced per model), log_vectors (bool).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Default arms file is configs/arms.yaml; the xtask wave points this at
# configs/arms_xtask.yaml via the MLC_ARMS_YAML env var (additive — with the var
# unset the sm/ic load path is byte-identical).
ARMS_YAML = Path(os.environ.get(
    "MLC_ARMS_YAML",
    str(Path(__file__).resolve().parents[1] / "configs" / "arms.yaml")))
MODES = {"anchor_minus", "anchor_plus", "patch", "subspace", "steer"}
TASKS = {"sm", "ic", "mw"}
MODELS = {"gemma", "llama"}
MAX_LAYER = {"gemma": 41, "llama": 31}  # inclusive top decoder-layer index
TWIN_COMPONENTS = {"G", "M"}


def load_arms(path=ARMS_YAML):
    cfg = yaml.safe_load(open(path))
    defaults = cfg.get("defaults", {})
    arms = {}
    for raw in cfg["arms"]:
        a = {**defaults, **raw}
        assert a["mode"] in MODES, f"{a['id']}: bad mode {a['mode']}"
        assert a["id"] not in arms, f"duplicate arm id {a['id']}"
        assert a.get("task", "sm") in TASKS, f"{a['id']}: bad task {a.get('task')}"
        assert a.get("probe", False) in (True, False), f"{a['id']}: probe must be bool"
        assert not (a.get("probe") and a.get("task") == "ic"), \
            f"{a['id']}: probe is SM-only"
        assert not (a.get("probe") and a.get("task") == "mw"), \
            f"{a['id']}: probe is SM-only"
        assert a.get("model", "gemma") in MODELS, f"{a['id']}: bad model {a.get('model')}"
        assert a.get("log_vectors", False) in (True, False), \
            f"{a['id']}: log_vectors must be bool"
        if "twin_component" in a:
            assert a["twin_component"] in TWIN_COMPONENTS, \
                f"{a['id']}: bad twin_component {a['twin_component']}"
            assert a["mode"] == "patch", \
                f"{a['id']}: twin_component is patch-mode only"
        if "direction" in a:
            assert a["direction"] == "random" and "dir_seed" in a, \
                f"{a['id']}: direction must be 'random' with a dir_seed"
        max_layer = MAX_LAYER[a.get("model", "gemma")]
        if "layers_list" in a:
            assert "layers" not in a, \
                f"{a['id']}: layers_list and layers are mutually exclusive"
            ll = a.pop("layers_list")
            assert ll and all(isinstance(l, int) and 0 <= l <= max_layer for l in ll), \
                f"{a['id']}: bad layers_list {ll}"
            assert len(set(ll)) == len(ll), f"{a['id']}: duplicate layers in layers_list"
            a["layers"] = list(ll)
        elif "layers" in a:
            lo, hi = a["layers"]
            assert 0 <= lo <= hi <= max_layer, f"{a['id']}: bad layer range"
            a["layers"] = list(range(lo, hi + 1))
        arms[a["id"]] = a
    return arms
