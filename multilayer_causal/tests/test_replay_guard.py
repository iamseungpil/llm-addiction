"""Replay guard + W&B no-op safety (Task 4)."""
import os

import numpy as np
import pytest


def _write_axis_npz(path, prompt_set="addiction_role_gm", build_game_ids=(1, 2, 3),
                    task=None):
    """Minimal stand-in for an indicator_axes-built npz (only the keys the
    replay guard reads); no sae_lens / HF needed."""
    kw = {"task": task} if task is not None else {}
    np.savez(path, directions=np.zeros((42, 8), np.float32),
             scales=np.ones(42, np.float32),
             build_prompt_set=prompt_set,
             build_game_ids=np.asarray(build_game_ids), **kw)
    return str(path)


def test_log_arm_noop_under_wandb_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")
    from multilayer_causal.src import wandb_logger
    # both entry points must be hard no-ops (return None, never raise) even if
    # wandb is not installed / not authenticated.
    assert wandb_logger.init_run("proj", "run") is None
    assert wandb_logger.log_arm("sec4_behavioural_ap3", 3.0,
                                {"mean_bet_ratio": 0.42}) is None


def test_replay_guard_mismatch(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")
    from multilayer_causal.src import runner
    with pytest.raises(ValueError):
        runner.assert_replay_match(arm_hash="a", dir_hash="b")


def test_replay_guard_match_ok(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")
    from multilayer_causal.src import runner
    runner.assert_replay_match(arm_hash="deadbeef", dir_hash="deadbeef")


def test_replay_hash_is_deterministic_and_config_sensitive():
    from multilayer_causal.src import runner
    a = {"id": "x", "prompt_set": "addiction_role_gm",
         "directions_npz": "d.npz", "layers": [16, 21], "alpha": 3.0,
         "mode": "steer"}
    h1 = runner._replay_hash(a)
    assert h1 == runner._replay_hash(dict(a))          # stable
    assert h1 != runner._replay_hash({**a, "alpha": 2.0})  # config-sensitive


def test_run_path_guard_raises_on_prompt_set_mismatch(tmp_path):
    """The WIRED guard (called before every trial loop): an arm whose replay
    prompt_set differs from the axis-build prompt_set must raise, not steer."""
    from multilayer_causal.src import runner
    npz = _write_axis_npz(tmp_path / "ax.npz", prompt_set="addiction_role_gm")
    arm = {"id": "sec4_x", "mode": "steer", "prompt_set": "bare_prompt_v0",
           "directions_npz": npz}
    with pytest.raises(ValueError):
        # twin=None ⇒ prompt-set check only (no −G catalog / HF access)
        runner._assert_replay_ok(arm, "gemma", 5, 0, 100, None)


def test_run_path_guard_ok_when_prompt_set_matches(tmp_path):
    from multilayer_causal.src import runner
    npz = _write_axis_npz(tmp_path / "ax.npz", prompt_set="addiction_role_gm")
    arm = {"id": "sec4_x", "mode": "steer", "prompt_set": "addiction_role_gm",
           "directions_npz": npz}
    runner._assert_replay_ok(arm, "gemma", 5, 0, 100, None)  # no raise


def test_run_path_guard_ic_disjointness(tmp_path):
    """sec4_w3 IC leak guard: an IC-built npz whose build_game_ids overlap the
    IC replay pool must raise; a disjoint build passes; an SM-/shared3-stamped
    npz is NOT compared to IC counters (different game-id space)."""
    from multilayer_causal.src import runner
    arm = {"id": "sec4_w3_ic_own_ap3", "mode": "steer", "task": "ic",
           "prompt_set": "addiction_role_gm"}

    leaky = _write_axis_npz(tmp_path / "ic_leak.npz", build_game_ids=(5, 8, 9),
                            task="investment_choice")
    with pytest.raises(AssertionError, match="replayed IC games leak"):
        runner._assert_replay_ok(dict(arm, directions_npz=leaky), "gemma",
                                 5, 0, 100, None, ic_replay_ids={5, 6, 7})

    clean = _write_axis_npz(tmp_path / "ic_clean.npz", build_game_ids=(8, 9),
                            task="investment_choice")
    runner._assert_replay_ok(dict(arm, directions_npz=clean), "gemma",
                             5, 0, 100, None, ic_replay_ids={5, 6, 7})  # no raise

    # shared3 npz carries SM catalog ids — numeric collision with IC counters
    # must NOT fire the guard (cross-id-space comparison is meaningless).
    sm_space = _write_axis_npz(tmp_path / "sh3.npz", build_game_ids=(5, 6, 7),
                               task="shared3")
    runner._assert_replay_ok(dict(arm, directions_npz=sm_space), "gemma",
                             5, 0, 100, None, ic_replay_ids={5, 6, 7})  # no raise


def test_run_path_guard_noop_for_prior_wave_and_null_arms(tmp_path):
    """Prior-wave assets (no build_prompt_set key) and random-null arms carry no
    sec4 provenance, so the guard is a hard no-op — prior waves stay
    byte-identical even with a deliberately mismatched prompt_set."""
    from multilayer_causal.src import runner
    prior = tmp_path / "prior.npz"
    np.savez(prior, directions=np.zeros((42, 8), np.float32),
             scales=np.ones(42, np.float32))  # no build_prompt_set key
    prior_arm = {"id": "xtaskd_x", "mode": "steer", "prompt_set": "whatever",
                 "directions_npz": str(prior)}
    runner._assert_replay_ok(prior_arm, "gemma", 5, 0, 100, None)  # no raise
    assert runner._sec4_provenance(prior_arm) is None

    sec4_npz = _write_axis_npz(tmp_path / "ax.npz")
    null_arm = {"id": "sec4_null_1", "mode": "steer", "direction": "random",
                "prompt_set": "bare_prompt_v0", "directions_npz": sec4_npz}
    runner._assert_replay_ok(null_arm, "gemma", 5, 0, 100, "G")  # no raise
    assert runner._sec4_provenance(null_arm) is None
