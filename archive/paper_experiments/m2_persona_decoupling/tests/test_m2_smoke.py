"""Smoke tests for the Track A1 / M2 module.

Live model/API tests are skipped automatically if the relevant resource is missing
(no GPU, no API key). Synthetic tests run unconditionally and exercise the
analyze + sanity_checks pipelines end-to-end without requiring bambi/pymc.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import prompts as prompts_mod  # noqa: E402
import sanity_checks as sanity_mod  # noqa: E402


def _cuda_available() -> bool:
    torch = pytest.importorskip("torch", reason="torch required for GPU smoke")
    return bool(torch.cuda.is_available())


def test_first_person_prefix_contains_role_instruction():
    prefix = prompts_mod.first_person_prefix("+G")
    # ROLE_INSTRUCTION starts with the canonical first-person language.
    assert "behavioral economics simulation" in prefix
    # No role-play language leaks in.
    low = prefix.lower()
    assert "imagine" not in low
    assert "casino" not in low
    assert "gambler" not in low


def test_role_play_prefix_contains_imagine_keyword():
    prefix = prompts_mod.role_play_gambler_prefix("+G")
    low = prefix.lower()
    assert "imagine" in low
    assert "gambler" in low
    assert "casino" in low


def test_build_prompt_first_person_no_double_role_instruction():
    from game_logic import SlotMachineGame  # type: ignore
    game = SlotMachineGame()
    prompt = prompts_mod.build_prompt(game, condition="+G", framing="first_person")
    # ROLE_INSTRUCTION should appear exactly once (the framing prefix), not duplicated
    # by the canonical create_prompt path.
    assert prompt.count("behavioral economics simulation") == 1
    # +G prompt component must be present in the body.
    assert "First, set a target amount yourself" in prompt


def test_build_prompt_role_play_does_not_carry_first_person():
    from game_logic import SlotMachineGame  # type: ignore
    game = SlotMachineGame()
    prompt = prompts_mod.build_prompt(game, condition="BASE", framing="role_play_gambler")
    assert "behavioral economics simulation" not in prompt
    assert "imagine" in prompt.lower()


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device available; Gemma smoke needs GPU")
def test_smoke_open_weight_5games(tmp_path: Path):
    cmd = [
        sys.executable, str(SRC_DIR / "run_m2_open_weight.py"),
        "--model", "gemma",
        "--gpu", "0",
        "--condition", "+G",
        "--framing", "first_person",
        "--task", "SM",
        "--n_games", "5",
        "--output_dir", str(tmp_path),
        "--smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr
    files = sorted(tmp_path.glob("final_gemma_pG_first_person_SM_*.json"))
    assert len(files) == 1
    with open(files[0]) as f:
        payload = json.load(f)
    assert payload["model"] == "gemma"
    assert payload["condition"] == "+G"
    assert payload["framing"] == "first_person"
    assert payload["task"] == "SM"
    assert payload["n_games"] == 5
    assert len(payload["results"]) == 5


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; GPT-4o-mini smoke needs API",
)
def test_smoke_api_5games(tmp_path: Path):
    cmd = [
        sys.executable, str(SRC_DIR / "run_m2_api.py"),
        "--provider", "openai",
        "--model_id", "gpt-4o-mini",
        "--condition", "+G",
        "--framing", "role_play_gambler",
        "--task", "SM",
        "--n_games", "5",
        "--output_dir", str(tmp_path),
        "--smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr
    files = sorted(tmp_path.glob("final_gpt-4o-mini_pG_role_play_gambler_SM_*.json"))
    assert len(files) == 1
    with open(files[0]) as f:
        payload = json.load(f)
    assert payload["provider"] == "openai"
    assert payload["condition"] == "+G"
    assert payload["framing"] == "role_play_gambler"
    assert len(payload["results"]) == 5


def _synthetic_payload(
    model: str, condition: str, framing: str, task: str, n_games: int, base_p: float,
    role_kw_boost: float, seed: int,
) -> Dict:
    import random as _r
    _r.seed(seed)
    results: List[Dict] = []
    base_kw = 0.5
    if framing == "role_play_gambler" and condition in ("+G", "+GM"):
        kw_per_round = base_kw + role_kw_boost
    elif condition in ("+G", "+GM"):
        kw_per_round = base_kw + role_kw_boost * 0.5  # tiny first-person boost
    else:
        kw_per_round = base_kw
    for i in range(n_games):
        bankrupt = int(_r.random() < base_p)
        rounds = _r.randint(5, 20)
        rs = []
        for r_i in range(rounds):
            kw_count_int = max(0, int(round(kw_per_round + _r.gauss(0, 0.3))))
            kw_words = ("bet risk loss " * kw_count_int).strip() or "stop"
            rs.append({
                "round": r_i + 1,
                "decision": "bet",
                "bet": 10,
                "response": f"I will {kw_words}. Final Decision: Bet $10",
                "balance_before": 100 - r_i * 10,
                "balance_after": 90 - r_i * 10,
            })
        results.append({
            "game_id": i,
            "model": model,
            "condition": condition,
            "framing": framing,
            "task": task,
            "bankrupt": bool(bankrupt),
            "outcome": "bankruptcy" if bankrupt else "voluntary_stop",
            "final_balance": 0 if bankrupt else 50,
            "total_rounds": rounds,
            "total_bet": 10 * rounds,
            "total_won": 0,
            "history": [{"round": r + 1, "bet": 10, "result": "L", "balance": 100 - 10 * (r + 1), "win": False, "winnings": 0} for r in range(rounds)],
            "rounds": rs,
        })
    return {
        "track": "A1_m2_persona_decoupling",
        "model": model,
        "condition": condition,
        "framing": framing,
        "task": task,
        "n_games": n_games,
        "smoke": True,
        "config_snapshot": {},
        "timestamp": "synthetic",
        "results": results,
    }


def _write_synthetic_dataset(tmp: Path, n_games: int = 100, role_kw_boost: float = 1.0) -> None:
    models = ["gemma", "llama", "gpt-4o-mini", "claude-haiku", "gemini-flash", "gpt-4o"]
    conditions = ["BASE", "+G", "+M", "+GM"]
    framings = ["first_person", "role_play_gambler"]
    seed = 0
    for m_idx, m in enumerate(models):
        for c in conditions:
            for fr in framings:
                seed += 1
                # Bake a strong first-person +G boost vs role-play +G so the bootstrap
                # primary contrast is reliably positive at the synthetic sample size.
                base = 0.15 + 0.005 * m_idx
                if c == "+G" and fr == "first_person":
                    base += 0.30
                elif c == "+G" and fr == "role_play_gambler":
                    base += 0.10
                elif c == "+GM" and fr == "first_person":
                    base += 0.35
                elif c == "+GM" and fr == "role_play_gambler":
                    base += 0.15
                payload = _synthetic_payload(m, c, fr, "SM", n_games, base, role_kw_boost, seed)
                cs = c.replace("+", "p")
                fname = f"final_{m}_{cs}_{fr}_SM_synth{seed}.json"
                with open(tmp / fname, "w") as f:
                    json.dump(payload, f)


def test_smoke_analyze_pipeline_synthetic(tmp_path: Path):
    _write_synthetic_dataset(tmp_path, n_games=100)
    out = tmp_path / "summary.json"
    cmd = [
        sys.executable, str(SRC_DIR / "analyze_m2.py"),
        "--input_dir", str(tmp_path),
        "--output_path", str(out),
        "--no_bambi",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    summary = json.loads(out.read_text())
    assert "per_task" in summary
    sm = summary["per_task"]["SM"]
    assert "primary_contrast" in sm
    assert sm["primary_contrast"]["method"].startswith("bootstrap")
    assert sm["n_games_total"] == 6 * 4 * 2 * 100
    # Synthetic data has a +0.20 pp baked first-vs-role gap on +G; bootstrap should
    # recover a positive point estimate (CI gating is left to live data).
    assert sm["primary_contrast"]["delta_gap_prob"] > 0


def test_sanity_check_manipulation_works(tmp_path: Path):
    _write_synthetic_dataset(tmp_path, n_games=30, role_kw_boost=2.0)
    keywords = sanity_mod.DEFAULT_KEYWORDS
    rows = sanity_mod.load_long_records(str(tmp_path), keywords)
    per_model = sanity_mod.compute_per_model(rows)
    assert per_model
    # Synthetic dataset has role-play +G with 2x gambling words vs first-person +G.
    n_pass = sum(1 for r in per_model if r["manipulation_passes"])
    assert n_pass == len(per_model), f"all 6 models should pass; got {n_pass}/{len(per_model)}"
