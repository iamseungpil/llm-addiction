"""arms_sec4_w14.yaml validation + the pool_exclude fix that makes +M real.

W14 re-runs the §4.3 G-specificity contrast (+G vs +M) on a G∩M-free MATCHED
pool. The bug it fixes: prompts.twin_combo is idempotent when the component is
already in the base combo, and the W4/W10b plusM arms' twin-free filter keyed off
twin_component (default "G") not the applied twin ("M"), so +M silently no-op'd on
~52% of eval states. exclude_combo_states + pool_exclude remove every G/M-bearing
base state so all three conditions (−G/+G/+M) are real, matched prompt changes.
"""
import re
from pathlib import Path

from multilayer_causal.src.registry import load_arms
from multilayer_causal.src.states import exclude_combo_states

MLC = Path(__file__).resolve().parents[1]
W14_YAML = MLC / "configs" / "arms_sec4_w14.yaml"
W14_TEMPLATE = MLC / "amlt" / "sec4_w14.yaml.template"
GEMMA_NPZ = "multilayer_causal/assets/sec4/gemma_slot_machine_i_ba_behavioural.npz"
LLAMA_NPZ = "multilayer_causal/assets/sec4/llama_slot_machine_i_ba_behavioural.npz"


def _mk(combo):
    """A (game, round_idx) state with the given prompt_combo (runner tuple shape)."""
    return ({"prompt_combo": combo}, 3)


def test_exclude_combo_states_drops_shared_chars():
    states = [_mk("BASE"), _mk("M"), _mk("MW"), _mk("W"), _mk("MRWP"),
              _mk("P"), _mk("MP"), _mk("RW")]
    kept = exclude_combo_states(states, "GM")
    combos = [s[0]["prompt_combo"] for s in kept]
    # every M-bearing combo is gone; none of the survivors contain G or M
    assert combos == ["BASE", "W", "P", "RW"]
    assert all(not any(c in s[0]["prompt_combo"] for c in "GM") for s in kept)


def test_exclude_combo_states_noop_when_empty():
    states = [_mk("M"), _mk("MW")]
    assert exclude_combo_states(states, "") is states
    assert exclude_combo_states(states, None) is states


def test_pool_exclude_makes_plusM_a_real_change():
    # THE REGRESSION PROOF. Without the filter the plusM eval pool is ~half
    # M-bearing (idempotent twin → no-op); after pool_exclude "GM" it is 0%.
    mixed = [_mk("BASE"), _mk("M"), _mk("MW"), _mk("W"), _mk("MRWP"), _mk("MP")]
    noop_before = sum(1 for s in mixed if "M" in s[0]["prompt_combo"])
    assert noop_before == 4  # 4/6 would silently no-op the +M twin
    after = exclude_combo_states(mixed, "GM")
    noop_after = sum(1 for s in after if "M" in s[0]["prompt_combo"])
    assert noop_after == 0


def test_w14_registry_counts_and_fields():
    arms = load_arms(W14_YAML)
    assert len(arms) == 26
    assert all(a["phase"] == "sec4_w14" for a in arms.values())
    # every arm rides the G∩M-free matched pool
    assert all(a["pool_exclude"] == "GM" for a in arms.values())
    assert all(a["task"] == "sm" and a["n"] == 200 for a in arms.values())
    assert all(a["prompt_set"] == "addiction_role_gm" for a in arms.values())

    for model, npz, layers, seed, off, big in (
            ("gemma", GEMMA_NPZ, list(range(16, 22)), 3100144, 300, True),
            ("llama", LLAMA_NPZ, list(range(14, 20)), 6100144, 0, False)):
        fam = {k: v for k, v in arms.items() if v["model"] == model}
        minusG = [a for a in fam.values() if "_minusG_" in a["id"]]
        plusG = [a for a in fam.values() if "_plusG_" in a["id"]]
        plusM = [a for a in fam.values() if "_plusM_" in a["id"]]
        # −G is the no-twin reference; +G/+M carry the twin field
        assert all("twin" not in a for a in minusG)
        assert all(a["twin"] == "G" and a["mode"] == "steer" for a in plusG)
        assert all(a["twin"] == "M" and a["mode"] == "steer" for a in plusM)
        assert len(minusG) == 3  # −G is a 3-point reference on both models
        if big:  # Gemma: full 7-point +G/+M ladders
            assert len(plusG) == 7 and len(plusM) == 7
            assert sorted(a["alpha"] for a in plusG) == [-3, -2, -1, 0, 1, 2, 3]
            assert sorted(a["alpha"] for a in plusM) == [-3, -2, -1, 0, 1, 2, 3]
        else:    # LLaMA: 3-point +G/+M ladders (matches W9/W10b granularity)
            assert len(plusG) == 3 and len(plusM) == 3
        for a in fam.values():
            assert a["layers"] == layers
            assert a["seed_base"] == seed and a["state_offset"] == off
            assert a["directions_npz"] == npz


def test_w14_template_lists_exactly_the_registry_arms():
    arms = load_arms(W14_YAML)
    text = W14_TEMPLATE.read_text()
    listed = " ".join(re.findall(r"run_arms\.sh \d+ ([^\n]+)", text)).split()
    assert sorted(listed) == sorted(arms), set(listed) ^ set(arms)
    # YAML guard: description must use ' - ', never ': ' (amlt parse trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    assert "MLC_OUT: multilayer_causal/results/sec4_w14" in text
    assert "MLC_ARMS_YAML: multilayer_causal/configs/arms_sec4_w14.yaml" in text
    # reuses existing behavioural axes — must NOT rebuild/overwrite them
    assert "--wave" not in text
    for npz in (GEMMA_NPZ.split("/")[-1], LLAMA_NPZ.split("/")[-1]):
        assert npz in text
