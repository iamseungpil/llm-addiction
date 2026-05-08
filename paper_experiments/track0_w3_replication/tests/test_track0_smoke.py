"""Smoke tests for the Track 0 module.

Live model/API tests are skipped automatically if the relevant resource is missing
(no GPU, no API key). The two synthetic tests run unconditionally and exercise the
analyze/sanity_checks pipelines end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
# Track 0 W3 Plan v5.2 §8: `game_logic` now imports `improved_parse_gpt_response`
# from /home/v-seungplee/llm-addiction/legacy/. Insert that path before the SRC_DIR
# so the legacy parser resolves cleanly when the smoke tests import `game_logic`.
sys.path.insert(0, "/home/v-seungplee/llm-addiction/paper_experiments/sm_cap_ablation/src")
sys.path.insert(0, str(SRC_DIR))

import game_logic  # noqa: E402
import sanity_checks  # noqa: E402


def _cuda_available() -> bool:
    torch = pytest.importorskip("torch", reason="torch required for GPU smoke")
    return bool(torch.cuda.is_available())


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device available; Gemma smoke needs GPU")
def test_smoke_open_weight_gemma_5games(tmp_path: Path):
    cmd = [
        sys.executable, str(SRC_DIR / "run_track0_open_weight.py"),
        "--model", "gemma",
        "--gpu", "0",
        "--cap", "30",
        "--mode", "variable",
        "--n_games", "5",
        "--output_dir", str(tmp_path),
        "--smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr

    files = sorted(tmp_path.glob("final_gemma_cap30_variable_*.json"))
    assert len(files) == 1
    with open(files[0]) as f:
        payload = json.load(f)
    assert payload["model"] == "gemma"
    assert payload["cap"] == 30
    assert payload["mode"] == "variable"
    assert payload["n_games"] == 5
    assert len(payload["results"]) == 5
    bks = sum(1 for r in payload["results"] if r["bankrupt"])
    assert 0 <= bks <= 5


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; GPT-4o-mini smoke needs API",
)
def test_smoke_api_gpt4omini_5games(tmp_path: Path):
    cmd = [
        sys.executable, str(SRC_DIR / "run_track0_api.py"),
        "--provider", "openai",
        "--model_id", "gpt-4o-mini",
        "--cap", "30",
        "--mode", "variable",
        "--n_games", "5",
        "--output_dir", str(tmp_path),
        "--smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr

    files = sorted(tmp_path.glob("final_gpt-4o-mini_cap30_variable_*.json"))
    assert len(files) == 1
    with open(files[0]) as f:
        payload = json.load(f)
    assert payload["provider"] == "openai"
    assert payload["cap"] == 30
    assert payload["mode"] == "variable"
    assert len(payload["results"]) == 5


def _synthetic_payload(model: str, cap: int, mode: str, n_games: int, base_p: float, seed: int) -> dict:
    import random as _r
    _r.seed(seed)
    results = []
    for i in range(n_games):
        bankrupt = int(_r.random() < base_p)
        rounds = _r.randint(5, 20)
        avg_bet = max(5, min(cap - 2, int(cap * (0.5 if mode == "variable" else 1.0))))
        history = [{"round": r + 1, "bet": avg_bet, "result": "L", "balance": 100 - avg_bet * (r + 1), "win": False, "winnings": 0} for r in range(rounds)]
        results.append({
            "game_id": i,
            "model": model,
            "cap": cap,
            "mode": mode,
            "bankrupt": bool(bankrupt),
            "outcome": "bankruptcy" if bankrupt else "voluntary_stop",
            "final_balance": 0 if bankrupt else 50,
            "total_rounds": rounds,
            "total_bet": avg_bet * rounds,
            "total_won": 0,
            "history": history,
            "rounds": [],
        })
    return {
        "track": "0_w3_replication",
        "model": model,
        "cap": cap,
        "mode": mode,
        "n_games": n_games,
        "smoke": True,
        "config_snapshot": {},
        "timestamp": "synthetic",
        "results": results,
    }


def _write_synthetic_dataset(tmp: Path, n_games: int = 50) -> None:
    models = ["gemma", "llama", "gpt-4o-mini", "claude-haiku", "gemini-flash", "gpt-4o"]
    caps = [10, 30, 50, 70]
    modes = ["fixed", "variable"]
    seed = 0
    for m_idx, m in enumerate(models):
        for c in caps:
            for mode in modes:
                seed += 1
                # variable bankrupts more at higher caps (mock the W3 effect)
                base_p = 0.10 + (0.01 * (c // 10)) + (0.05 if mode == "variable" else 0.0) + 0.005 * m_idx
                payload = _synthetic_payload(m, c, mode, n_games, base_p, seed)
                fname = f"final_{m}_cap{c}_{mode}_synth{seed}.json"
                with open(tmp / fname, "w") as f:
                    json.dump(payload, f)


def test_smoke_analyze_pipeline(tmp_path: Path):
    _write_synthetic_dataset(tmp_path, n_games=50)
    out = tmp_path / "summary.json"
    cmd = [
        sys.executable, str(SRC_DIR / "analyze_track0.py"),
        "--input_dir", str(tmp_path),
        "--output_path", str(out),
        "--no_bambi",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    summary = json.loads(out.read_text())
    assert "primary_contrast" in summary
    assert summary["n_games_total"] == 6 * 4 * 2 * 50
    assert "beta_primary_prob" in summary["primary_contrast"]
    assert summary["primary_contrast"]["method"].startswith("bootstrap")


def test_smoke_sanity_checks_synthetic(tmp_path: Path):
    _write_synthetic_dataset(tmp_path, n_games=50)
    out_md = tmp_path / "sanity.md"
    cmd = [
        sys.executable, str(SRC_DIR / "sanity_checks.py"),
        "--input_dir", str(tmp_path),
        "--output_path", str(out_md),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    md = out_md.read_text()
    assert "Track 0 sanity checks" in md
    assert "S1" in md and "S2" in md
    # All 6 mocked models should appear
    for m in ["gemma", "llama", "gpt-4o-mini", "claude-haiku", "gemini-flash", "gpt-4o"]:
        assert m in md


def test_game_logic_cap_none_legacy_semantics():
    g = game_logic.SlotMachineGame()
    assert g.offered_fixed_bet() == 10
    assert g.variable_upper_bound() == 100  # full balance


def test_game_logic_cap_constrains_variable_upper_bound():
    g = game_logic.SlotMachineGame(cap=30, mode="variable")
    assert g.variable_upper_bound() == 30
    g.balance = 20
    assert g.variable_upper_bound() == 20  # min(cap, balance)


def test_game_logic_cap_constrains_fixed_offered_bet():
    g = game_logic.SlotMachineGame(cap=50, mode="fixed")
    assert g.offered_fixed_bet() == 50


def test_game_logic_parse_response_clamps_to_cap():
    g = game_logic.SlotMachineGame(cap=30, mode="variable")
    decision, bet, _ = game_logic.parse_response("I'll bet $90.\nFinal Decision: Bet $90", g)
    assert decision == "continue"
    assert bet == 30  # clamped to cap


def test_game_logic_parse_response_stop():
    g = game_logic.SlotMachineGame(cap=30, mode="variable")
    decision, bet, _ = game_logic.parse_response("It's risky.\nFinal Decision: Stop", g)
    assert decision == "stop"
    assert bet is None


def test_game_logic_run_single_game_with_stub_response():
    calls = {"n": 0}

    def stub(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] >= 3:
            return "Final Decision: Stop"
        return "Final Decision: Bet $10"

    record = game_logic.run_single_game(
        response_fn=stub,
        cap=30,
        mode="variable",
        max_rounds=10,
        seed=123,
    )
    assert record["mode"] == "variable"
    assert record["cap"] == 30
    assert record["total_rounds"] >= 1
    assert record["outcome"] in {"voluntary_stop", "bankruptcy", "max_rounds"}
