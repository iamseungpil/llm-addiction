"""W3 +M twin parity: build_twin_prompt must compose components exactly like
the frozen build_prompt (and therefore like the original 64-condition script
paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py).

Fixture mirrors tests/test_states.py so the parity claims hold on the same
synthetic-catalog states the pool loader actually serves.
"""
import json
from pathlib import Path

import pytest

from multilayer_causal.src import states
from multilayer_causal.src.prompts import build_prompt, build_twin_prompt, twin_combo

REPO = Path(__file__).resolve().parents[2]
M_TEXT = "Your task is to maximize the reward."


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


def test_m_component_text_matches_original_64cond_script():
    src = (REPO / "paper_experiments" / "slot_machine_6models" / "src" /
           "llama_gemma_experiment.py").read_text()
    assert f'prompt += "{M_TEXT}\\n"' in src


def test_m_twin_of_base_state_equals_combo_M_prompt(tmp_path, monkeypatch):
    _write_catalog(tmp_path, monkeypatch)
    pool = states.load_minusG_states("gemma", n=100)
    base = [(g, ri) for g, ri in pool if g.get("prompt_combo", "") == ""]
    assert base
    for g, ri in base:
        twin = build_twin_prompt(g, ri, twin_component="M")
        assert twin == build_prompt(g, ri, override_combo="M")
        assert twin.count(M_TEXT) == 1


def test_default_G_twin_matches_existing_runner_recipe(tmp_path, monkeypatch):
    _write_catalog(tmp_path, monkeypatch)
    pool = states.load_minusG_states("gemma", n=100)
    assert pool
    for g, ri in pool:
        base_combo = g.get("prompt_combo", "")
        plus_combo = base_combo + "G" if "G" not in base_combo else base_combo
        assert build_twin_prompt(g, ri) == \
            build_prompt(g, ri, override_combo=plus_combo)


def test_m_catalog_game_own_prompt_unchanged_and_twin_idempotent(tmp_path, monkeypatch):
    _write_catalog(tmp_path, monkeypatch)
    pool = states.load_minusG_states("gemma", n=100)
    m_states = [(g, ri) for g, ri in pool if g.get("prompt_combo", "") == "M"]
    assert m_states
    for g, ri in m_states:
        own = build_prompt(g, ri)
        assert own == build_prompt(g, ri, override_combo="M")
        assert build_twin_prompt(g, ri, twin_component="M") == own
        assert own.count(M_TEXT) == 1


def test_twin_component_validation():
    assert twin_combo("", "M") == "M"
    assert twin_combo("M", "M") == "M"
    assert twin_combo("", "G") == "G"
    with pytest.raises(AssertionError):
        twin_combo("", "H")
    with pytest.raises(AssertionError):
        build_twin_prompt({"prompt_combo": "", "bet_type": "variable",
                           "decisions": [{"balance_before": 80}], "history": []},
                          0, twin_component="X")
