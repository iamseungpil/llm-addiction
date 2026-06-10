"""Load + validate configs/arms.yaml.

layers: [a, b] inclusive range → expanded list.
layers_list: explicit (possibly non-contiguous) layer indices → stored as
a['layers']; mutually exclusive with layers (W1 bridge arms).
Optional pass-through fields validated here: probe (bool, SM-only),
task ('sm' | 'ic'), direction ('random', needs dir_seed), seed_base,
state_offset, dir_seed.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ARMS_YAML = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"
MODES = {"anchor_minus", "anchor_plus", "patch", "subspace", "steer"}
TASKS = {"sm", "ic"}


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
        if "direction" in a:
            assert a["direction"] == "random" and "dir_seed" in a, \
                f"{a['id']}: direction must be 'random' with a dir_seed"
        if "layers_list" in a:
            assert "layers" not in a, \
                f"{a['id']}: layers_list and layers are mutually exclusive"
            ll = a.pop("layers_list")
            assert ll and all(isinstance(l, int) and 0 <= l <= 41 for l in ll), \
                f"{a['id']}: bad layers_list {ll}"
            assert len(set(ll)) == len(ll), f"{a['id']}: duplicate layers in layers_list"
            a["layers"] = list(ll)
        elif "layers" in a:
            lo, hi = a["layers"]
            assert 0 <= lo <= hi <= 41, f"{a['id']}: bad layer range"
            a["layers"] = list(range(lo, hi + 1))
        arms[a["id"]] = a
    return arms
