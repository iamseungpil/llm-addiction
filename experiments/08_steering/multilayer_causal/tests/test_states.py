import json

from multilayer_causal.src import states


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


def test_pool_filters_variable_minusG_rounds_2_to_6(tmp_path, monkeypatch):
    _write_catalog(tmp_path, monkeypatch)
    pool = states.load_minusG_states("gemma", n=100)
    # only variable & no-G games: combos "" and "M" → 2 games × rounds 2..6 = 10 states
    assert len(pool) == 10
    assert all(2 <= ri <= 6 for _, ri in pool)
    assert all("G" not in g.get("prompt_combo", "") for g, _ in pool)


def test_pool_order_is_frozen(tmp_path, monkeypatch):
    _write_catalog(tmp_path, monkeypatch)
    a = states.load_minusG_states("gemma", n=100)
    b = states.load_minusG_states("gemma", n=100)
    assert [ri for _, ri in a] == [ri for _, ri in b]
    assert [g["prompt_combo"] for g, _ in a] == [g["prompt_combo"] for g, _ in b]
