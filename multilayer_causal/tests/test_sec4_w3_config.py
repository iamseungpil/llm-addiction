"""arms_sec4_w3.yaml registry validation + amlt template consistency.

The Wave-3 registry must load through the standard validator (twin /
state_filter fields included), carry exactly the goals-v2 rung-1 arm blocks
(Q2 24, Q3 8, Q1-rung2 6 = 38 arms), and the amlt template's run_arms.sh line
must list exactly the registry ids (a template/config drift would silently
drop arms from the wave).
"""
import re
from pathlib import Path

from multilayer_causal.src.registry import load_arms

MLC = Path(__file__).resolve().parents[1]
W3_YAML = MLC / "configs" / "arms_sec4_w3.yaml"
W3_TEMPLATE = MLC / "amlt" / "sec4_w3.yaml.template"

SHARED3_NPZ = "multilayer_causal/assets/sec4/gemma_shared3_iba_behavioural.npz"
SHARED3_ICSCALE_NPZ = ("multilayer_causal/assets/sec4/"
                       "gemma_shared3_iba_behavioural_icscale.npz")
IC_OWN_NPZ = ("multilayer_causal/assets/sec4/"
              "gemma_investment_choice_i_ba_behavioural.npz")
BEHAV_NPZ = ("multilayer_causal/assets/sec4/"
             "gemma_slot_machine_i_ba_behavioural.npz")


def test_w3_registry_counts_and_fields():
    arms = load_arms(W3_YAML)
    assert len(arms) == 38
    assert all(a["phase"] == "sec4_w3" for a in arms.values())

    # Q2: shared3 ladders on SM and IC + IC positive control + baseline + nulls
    sh3_sm = [a for a in arms.values() if a["id"].startswith("sec4_w3_sh3_sm_")]
    sh3_ic = [a for a in arms.values() if a["id"].startswith("sec4_w3_sh3_ic_")]
    ic_own = [a for a in arms.values() if a["id"].startswith("sec4_w3_ic_own_")]
    nulls = [a for a in arms.values() if a["id"].startswith("sec4_w3_ic_null")]
    assert len(sh3_sm) == 7 and len(sh3_ic) == 7
    assert sorted(a["alpha"] for a in sh3_sm) == [-3, -2, -1, 0, 1, 2, 3]
    assert all(a["directions_npz"] == SHARED3_NPZ for a in sh3_sm)
    # IC ladder rides the icscale twin (same directions, IC sigma-units) so
    # its dose magnitude matches the ic_own control and IC null band
    assert all(a["directions_npz"] == SHARED3_ICSCALE_NPZ for a in sh3_ic)
    # IC arms follow the xtask target-column convention: task ic, offset 0
    for a in sh3_ic + ic_own + nulls + [arms["sec4_w3_ic_baseline"]]:
        assert a["task"] == "ic" and a["state_offset"] == 0, a["id"]
        assert a["n"] == 200
    assert sorted(a["alpha"] for a in ic_own) == [-3, 0, 3]
    assert all(a["directions_npz"] == IC_OWN_NPZ for a in ic_own)
    assert arms["sec4_w3_ic_baseline"]["mode"] == "anchor_minus"
    # thick IC null: 3 directions, each dosed at -3 AND +3 with ONE dir_seed
    assert len(nulls) == 6
    seeds = {}
    for a in nulls:
        assert a["direction"] == "random"
        seeds.setdefault(a["dir_seed"], []).append(a["alpha"])
    assert len(seeds) == 3
    assert all(sorted(v) == [-3.0, 3.0] for v in seeds.values())

    # Q3: +G twin-steer ladder + anchor_plus baseline
    plusg = [a for a in arms.values()
             if re.match(r"sec4_w3_plusG_a[mp]?\d*$", a["id"])]
    assert len(plusg) == 7
    for a in plusg:
        assert a["twin"] == "G" and a["mode"] == "steer"
        assert a["directions_npz"] == BEHAV_NPZ
        assert a["task"] == "sm" and a["state_offset"] == 300
    assert arms["sec4_w3_plusG_baseline"]["mode"] == "anchor_plus"

    # Q1 rung-2: postloss / postwin conditional ladders
    for filt in ("postloss", "postwin"):
        fam = [a for a in arms.values()
               if a["id"].startswith(f"sec4_w3_{filt}_")]
        assert len(fam) == 3
        assert sorted(a["alpha"] for a in fam) == [-3, 0, 3]
        for a in fam:
            assert a["state_filter"] == filt and a["task"] == "sm"
            assert a["directions_npz"] == BEHAV_NPZ

    # frozen replay discipline on every SM arm
    for a in arms.values():
        if a["task"] == "sm":
            assert a["seed_base"] == 2000042 and a["state_offset"] == 300
            assert a["prompt_set"] == "addiction_role_gm"
        assert a["layers"] == list(range(16, 22))


def test_w3_template_lists_exactly_the_registry_arms():
    arms = load_arms(W3_YAML)
    text = W3_TEMPLATE.read_text()
    m = re.search(r"run_arms\.sh \d+ ([^\n]+)", text)
    assert m, "no run_arms.sh line in sec4_w3.yaml.template"
    listed = m.group(1).split()
    assert sorted(listed) == sorted(arms), (
        set(listed) ^ set(arms))  # symmetric diff on failure
    # YAML guard: description must use ' - ', never ': ' (amlt parse trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # the build step builds and verifies the five wave3 axis npz
    assert "--wave3" in text
    for npz in ("gemma_slot_machine_i_ba_behavioural.npz",
                "gemma_investment_choice_i_ba_behavioural.npz",
                "gemma_mystery_wheel_i_ba_behavioural.npz",
                "gemma_shared3_iba_behavioural.npz",
                "gemma_shared3_iba_behavioural_icscale.npz"):
        assert npz in text
    assert "MLC_OUT: multilayer_causal/results/sec4_w3" in text
    assert "MLC_ARMS_YAML: multilayer_causal/configs/arms_sec4_w3.yaml" in text
