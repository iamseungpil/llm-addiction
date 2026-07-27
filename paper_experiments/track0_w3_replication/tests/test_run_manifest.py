"""Tests for the run manifest and the output-isolation guard (DEVIATIONS.md D5).

These cover the three things the audit asked the harness to guarantee before the
re-run: the payload states its own provenance, the fallback substitutions are
counted, and a cell cannot be written into a directory that already holds that cell.

The end-to-end test drives `run_track0_api.main()` with a stubbed provider, so it
needs no API key and no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
LEGACY_ROOT = Path("/home/v-seungplee/llm-addiction/paper_experiments/sm_cap_ablation/src")

sys.path.insert(0, str(LEGACY_ROOT))
sys.path.insert(0, str(SRC_DIR))

import game_logic  # noqa: E402,F401  (imported so run_manifest can hash the loaded file)
import run_manifest  # noqa: E402


# --- guard ------------------------------------------------------------------


def test_guard_passes_on_a_clean_directory(tmp_path: Path):
    assert run_manifest.guard_output_collision(
        tmp_path, ["final_gpt-4o-mini_cap70_fixed_*.json"], cell="gpt-4o-mini_cap70_fixed"
    ) == []


def test_guard_ignores_other_cells(tmp_path: Path):
    (tmp_path / "final_gpt-4o-mini_cap30_fixed_20260101_000000.json").write_text("{}")
    (tmp_path / "final_llama_cap70_fixed_20260101_000000.json").write_text("{}")
    assert run_manifest.guard_output_collision(
        tmp_path, ["final_gpt-4o-mini_cap70_fixed_*.json"], cell="gpt-4o-mini_cap70_fixed"
    ) == []


def test_guard_aborts_when_the_same_cell_is_already_present(tmp_path: Path):
    stale = tmp_path / "final_gpt-4o-mini_cap70_fixed_20260101_000000.json"
    stale.write_text("{}")
    with pytest.raises(run_manifest.OutputCollision) as exc:
        run_manifest.guard_output_collision(
            tmp_path, ["final_gpt-4o-mini_cap70_fixed_*.json"], cell="gpt-4o-mini_cap70_fixed"
        )
    assert stale.name in str(exc.value)


def test_guard_can_be_overridden_explicitly(tmp_path: Path, capsys):
    (tmp_path / "final_gpt-4o-mini_cap70_fixed_20260101_000000.json").write_text("{}")
    hits = run_manifest.guard_output_collision(
        tmp_path,
        ["final_gpt-4o-mini_cap70_fixed_*.json"],
        cell="gpt-4o-mini_cap70_fixed",
        allow_existing=True,
    )
    assert len(hits) == 1
    assert "WARNING" in capsys.readouterr().err


# --- manifest ---------------------------------------------------------------


def test_manifest_has_every_required_provenance_field():
    run_manifest.reset_fallback_count()
    m = run_manifest.build_manifest(
        runner="unit-test",
        model_id="gpt-4o-mini",
        vendor="openai",
        seed_base=42,
        seeds=[42, 43, 44],
        started_at=run_manifest.now_iso(),
        argv=["run_track0_api.py", "--cap", "70"],
        extra={"cell": "gpt-4o-mini_cap70_fixed"},
    )
    assert m["git"]["commit"] and len(m["git"]["commit"]) == 40
    assert m["argv"] == ["run_track0_api.py", "--cap", "70"]
    assert "--cap" in m["command"]
    assert (m["model_id"], m["vendor"]) == ("gpt-4o-mini", "openai")
    assert m["seed_base"] == 42 and m["seeds"] == [42, 43, 44] and m["n_seeds"] == 3
    assert m["started_at"] <= m["finished_at"]
    assert m["api_fallback_responses"] == 0
    assert m["cell"] == "gpt-4o-mini_cap70_fixed"
    # The two parity-critical sources are hashed, and hashed from the file that was
    # actually imported rather than a hoped-for path.
    for key in ("game_logic.py", "improved_gpt_parsing.py"):
        entry = m["code_sha256"][key]
        assert entry["sha256"] and len(entry["sha256"]) == 64
        assert Path(entry["path"]).name == key
        assert Path(entry["path"]).exists()
    assert run_manifest.sha256_file(Path(m["code_sha256"]["game_logic.py"]["path"])) == \
        m["code_sha256"]["game_logic.py"]["sha256"]


def test_manifest_counts_fallback_responses():
    run_manifest.reset_fallback_count()
    assert run_manifest.note_fallback() == "Final Decision: Stop"
    run_manifest.note_fallback()
    m = run_manifest.build_manifest(
        runner="unit-test", model_id="m", vendor="v", seed_base=0, seeds=[0],
        started_at=run_manifest.now_iso(), argv=["x"],
    )
    assert m["api_fallback_responses"] == 2
    run_manifest.reset_fallback_count()


# --- end to end through run_track0_api.main() -------------------------------


def _run_api_cell(monkeypatch, tmp_path: Path, *, mode: str, cap: int, extra_argv=()):
    import run_track0_api as t0

    def fake_builder(provider, model_id, gap):
        calls = {"n": 0}

        def fn(_prompt: str) -> str:
            calls["n"] += 1
            return "Final Decision: Stop" if calls["n"] % 3 == 0 else "Final Decision: Bet"

        return fn

    monkeypatch.setattr(t0, "_build_response_fn", fake_builder)
    run_manifest.reset_fallback_count()
    argv = [
        "run_track0_api.py",
        "--provider", "openai",
        "--model_id", "gpt-4o-mini",
        "--cap", str(cap),
        "--mode", mode,
        "--n_games", "2",
        "--output_dir", str(tmp_path),
        *extra_argv,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    t0.main()
    return t0


def test_api_runner_writes_manifest_and_per_decision_prompts(monkeypatch, tmp_path: Path):
    _run_api_cell(monkeypatch, tmp_path, mode="fixed", cap=70)
    files = sorted(tmp_path.glob("final_gpt-4o-mini_cap70_fixed_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())

    m = payload["manifest"]
    assert m["runner"].endswith("run_track0_api.py")
    assert (m["model_id"], m["vendor"]) == ("gpt-4o-mini", "openai")
    assert m["seed_base"] == 42 and m["seeds"] == [42, 43]
    assert m["cell"] == "gpt-4o-mini_cap70_fixed"
    assert m["git"]["commit"] and m["code_sha256"]["game_logic.py"]["sha256"]
    assert "--cap" in m["argv"] and m["started_at"] <= m["finished_at"]
    assert m["api_fallback_responses"] == 0

    for game in payload["results"]:
        for rnd in game["rounds"]:
            assert rnd["prompt"], "every decision must store the prompt that produced it"
            assert "Choose one of the following:" in rnd["prompt"]
            assert "1) Bet $70" in rnd["prompt"], "fixed-mode prompt must offer the cap"
            assert rnd["prompt_attempts"] >= 1
            if rnd["decision"] == "bet":
                assert rnd["bet"] == min(70, rnd["balance_before"])


def test_api_runner_refuses_to_write_into_a_directory_holding_the_same_cell(monkeypatch, tmp_path: Path):
    _run_api_cell(monkeypatch, tmp_path, mode="fixed", cap=70)
    with pytest.raises(SystemExit) as exc:
        _run_api_cell(monkeypatch, tmp_path, mode="fixed", cap=70)
    assert "output isolation guard" in str(exc.value)
    assert len(list(tmp_path.glob("final_gpt-4o-mini_cap70_fixed_*.json"))) == 1


def test_api_runner_allows_a_different_cell_in_the_same_directory(monkeypatch, tmp_path: Path):
    _run_api_cell(monkeypatch, tmp_path, mode="fixed", cap=70)
    _run_api_cell(monkeypatch, tmp_path, mode="variable", cap=70)
    assert len(list(tmp_path.glob("final_gpt-4o-mini_cap70_*.json"))) == 2
