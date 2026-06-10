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
