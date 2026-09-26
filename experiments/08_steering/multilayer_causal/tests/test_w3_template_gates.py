"""W3 submission-list gate discipline (RUN_PLAN_W3.md 구현 원칙).

The builder (src/paper_axes.py) records the reproduction gate INSIDE each
assets/w3 npz; a failed gate auto-excludes that axis's arms from the
submission list (amlt/w3.yaml.template) while the registry entries stay for
the frozen test counts. These tests pin both halves: every submitted steer
arm's direction asset must exist on disk (no FileNotFoundError mid-job), and
no submitted arm may ride a gate_passed=False asset.
"""
import re
from pathlib import Path

import numpy as np

from multilayer_causal.src.registry import load_arms

REPO_ROOT = Path(__file__).resolve().parents[2]
W3_TEMPLATE = REPO_ROOT / "multilayer_causal" / "amlt" / "w3.yaml.template"


def submitted_arm_ids():
    ids = []
    for line in W3_TEMPLATE.read_text().splitlines():
        m = re.search(r"run_arms\.sh \d+ (.+)$", line.strip())
        if m:
            ids.extend(m.group(1).split())
    return ids


def test_w3_template_arms_resolve_and_assets_exist():
    arms = load_arms()
    ids = submitted_arm_ids()
    assert len(ids) == len(set(ids)) and ids, "duplicate or empty submission list"
    for arm_id in ids:
        assert arm_id in arms, f"{arm_id} not in configs/arms.yaml"
        npz_rel = arms[arm_id].get("directions_npz")
        if npz_rel:
            assert (REPO_ROOT / npz_rel).exists(), f"{arm_id}: missing {npz_rel}"


def test_w3_template_excludes_failed_gate_axes():
    arms = load_arms()
    ids = set(submitted_arm_ids())
    for arm_id, arm in arms.items():
        npz_rel = arm.get("directions_npz", "")
        if "/assets/w3/" not in npz_rel:
            continue
        path = REPO_ROOT / npz_rel
        assert path.exists(), f"builder never run for {npz_rel}"
        gate = bool(np.load(path)["gate_passed"])
        if not gate:
            assert arm_id not in ids, \
                f"{arm_id} submitted despite gate_passed=False in {npz_rel}"
    # 2026-06-11 build outcome pinned (post round-1 gate fix): saerd + its
    # eval-excluded refit twin passed; bk_sm and bk_sm_balres pass the
    # pre-registered construct gates (relevance perm p ~ 1e-4, existence
    # pair-cos boot 2.5pct > 0, stability ~ 0.99) with stop-ward
    # orientation calibration; the round-3 discriminative balance controls
    # bk_sm_balorth (residual norm 0.914 > 0.2, exactly orthogonal to the
    # target balance-slope direction) and bk_sm_balslope (|cos(plain,
    # balance-slope)| 0.406 > 0.05) pass their construction-validity gates;
    # bk_ic fails the existence gate (source pair cos 0.02) — so 6 w3a +
    # 6 SM w3bk arms remain submitted, IC excluded.
    w3_asset_arms = {a_id for a_id, a in arms.items()
                     if "/assets/w3/" in a.get("directions_npz", "")}
    assert w3_asset_arms & ids == {
        "w3a_saerd_m20", "w3a_saerd_p20", "w3a_saerd_p40",
        "w3a_saerdx_m20", "w3a_saerdx_p20", "w3a_saerdx_p40",
        "w3bk_sm_m20", "w3bk_sm_p20", "w3bk_sm_p40", "w3bk_smres_p40",
        "w3bk_smorth_p40", "w3bk_smbal_p40"}
