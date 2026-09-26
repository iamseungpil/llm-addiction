"""arms_sec4_w6.yaml registry validation + amlt template consistency.

The Wave-6 registry must load through the standard validator, carry exactly
the 7-dose mw_rc ladder (nulls/baseline are REUSED from sec4_w4 at analysis
time, so no null/baseline arms here), and the amlt template's run_arms.sh
line must list exactly the registry ids. The template's need-list quotes must
be backslash-escaped inside the bash-quoted python -c block (an unescaped
quote killed a prior submission).
"""
import re
from pathlib import Path

from multilayer_causal.src.registry import load_arms

MLC = Path(__file__).resolve().parents[1]
W6_YAML = MLC / "configs" / "arms_sec4_w6.yaml"
W6_TEMPLATE = MLC / "amlt" / "sec4_w6.yaml.template"

MW_RC_NPZ = ("multilayer_causal/assets/sec4/"
             "gemma_mystery_wheel_mw_rc_behavioural.npz")
W6_NPZ = ("gemma_mystery_wheel_mw_rc_behavioural.npz",
          "gemma_shared3c_rc_behavioural.npz",
          "gemma_shared3c_rc_behavioural_icscale.npz",
          "gemma_shared3c_rc_behavioural_mwscale.npz")


def test_w6_registry_counts_and_fields():
    arms = load_arms(W6_YAML)
    assert len(arms) == 7
    assert all(a["phase"] == "sec4_w6" for a in arms.values())
    assert all(a["id"].startswith("sec4_w6_mw_rc_") for a in arms.values())
    assert sorted(a["alpha"] for a in arms.values()) == [-3, -2, -1, 0, 1, 2, 3]
    for a in arms.values():
        # frozen MW replay discipline (matches the sec4_w4/w5 MW arms whose
        # nulls/baseline this wave reuses at analysis time)
        assert a["task"] == "mw" and a["state_offset"] == 0, a["id"]
        assert a["mode"] == "steer" and a["n"] == 200
        assert a["seed_base"] == 2000042
        assert a["prompt_set"] == "addiction_role_gm"
        assert a["log_vectors"] is True
        assert a["layers"] == list(range(16, 22))
        assert a["directions_npz"] == MW_RC_NPZ


def test_w6_template_lists_exactly_the_registry_arms():
    arms = load_arms(W6_YAML)
    text = W6_TEMPLATE.read_text()
    m = re.search(r"run_arms\.sh \d+ ([^\n]+)", text)
    assert m, "no run_arms.sh line in sec4_w6.yaml.template"
    listed = m.group(1).split()
    assert sorted(listed) == sorted(arms), (
        set(listed) ^ set(arms))  # symmetric diff on failure
    # YAML guard: description must use ' - ', never ': ' (amlt parse trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # the build step builds and verifies the four wave6 axis npz
    assert "--wave6" in text
    for npz in W6_NPZ:
        assert npz in text
    # the need list lives inside a bash single-quoted python -c block: every
    # double-quote in it MUST be backslash-escaped (\") — an unescaped quote
    # already killed one submission
    need = re.search(r"need = \[([^\]]+)\]", text)
    assert need, "no need-list in sec4_w6.yaml.template"
    assert re.findall(r'(?<!\\)"', need.group(1)) == [], need.group(1)
    assert need.group(1).count('\\"') == 2 * len(W6_NPZ)
    assert "MLC_OUT: multilayer_causal/results/sec4_w6" in text
    assert "MLC_ARMS_YAML: multilayer_causal/configs/arms_sec4_w6.yaml" in text
    assert 'MLC_SYNC_EVERY: "50"' in text
