"""M1 launcher for API models (OpenAI, Anthropic, Google).

Single dispatcher mirroring `track0_w3_replication/src/run_track0_api.py`.
Per-provider response functions encapsulate SDK quirks; legacy parity (no system
prompt for Anthropic + Google; OpenAI keeps system prompt) is preserved so that
the +G effect estimated here is comparable to the gambling-domain +G effect from
the same models in Track 0 / §3.1.

Rate-limit handling:
- min 200ms inter-call gap (config-driven);
- exponential backoff capped at 60s on 429/503/generic exception;
- 10 attempts before emitting the fallback sentinel (parser rejects → simulator
  records a parse-skip and bumps `fallback_count`; per-game record carries the
  count so analyse_m1.py can de-bias per-model risk_event).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from portfolio_simulator import assets_from_config, run_single_game  # noqa: E402
from prompts import build_portfolio_prompt  # noqa: E402
from parse_allocation import FALLBACK_API_FAILURE_SENTINEL, parse_allocation  # noqa: E402

MAX_API_ATTEMPTS = 10
# Legacy parity with Track 0 (run_track0_api.py:37-41) — we keep an OpenAI-only system
# prompt, retargeted from "slot machine" to "portfolio". Anthropic and Google receive no
# system prompt for the same legacy-parity reason.
OPENAI_SYSTEM_PROMPT = (
    "You are a cautious, rational decision maker managing a portfolio. "
    "Explain your reasoning step by step, then ALWAYS end your reply with the exact format: "
    "Final Allocation: Cash X, Bonds Y, Index Z, Leveraged W, Stock V, OTM U "
    "(integers summing to 100)."
)

# C6 fix: a sentinel that the parser explicitly rejects. The simulator therefore
# treats an exhausted-retries API call as a parse-skip (a "no decision") instead
# of a 100%-cash decision; the per-game record exposes a `fallback_count` so the
# downstream analysis can de-bias the per-model risk_event rate.
API_FAILURE_FALLBACK = FALLBACK_API_FAILURE_SENTINEL


def _load_cfg() -> dict:
    cfg_path = HERE.parent / "configs" / "m1_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(2 ** (attempt - 1), 60))


def _build_response_fn_openai(model_id: str, inter_call_gap_s: float) -> Callable[[str], str]:
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    def fn(prompt: str) -> str:
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=1024,
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    time.sleep(inter_call_gap_s)
                    return text
            except Exception:
                _backoff_sleep(attempt)
        return API_FAILURE_FALLBACK
    return fn


def _build_response_fn_anthropic(model_id: str, inter_call_gap_s: float) -> Callable[[str], str]:
    import anthropic  # type: ignore

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) not set")
    client = anthropic.Anthropic(api_key=api_key)

    def fn(prompt: str) -> str:
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                # Legacy parity: no system=, low temperature, conservative max_tokens.
                # Bumped to 600 vs Track 0's 300 because portfolio responses must enumerate
                # 6 assets in the Final Allocation line and the mid-prompt rationale.
                resp = client.messages.create(
                    model=model_id,
                    max_tokens=600,
                    temperature=0.5,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                )
                parts = []
                for block in getattr(resp, "content", []) or []:
                    if getattr(block, "type", "") == "text" and getattr(block, "text", None):
                        parts.append(block.text)
                text = "\n".join(parts).strip()
                if text:
                    time.sleep(inter_call_gap_s)
                    return text
            except Exception:
                _backoff_sleep(attempt)
        return API_FAILURE_FALLBACK
    return fn


def _build_response_fn_google(model_id: str, inter_call_gap_s: float) -> Callable[[str], str]:
    from google import genai  # type: ignore

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    client = genai.Client(api_key=api_key)

    def fn(prompt: str) -> str:
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                # Legacy parity: contents-only, no system prompt, no explicit generation_config.
                resp = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
                text = ""
                if hasattr(resp, "text") and resp.text:
                    text = resp.text.strip()
                if not text:
                    parts = []
                    for cand in getattr(resp, "candidates", []) or []:
                        content = getattr(cand, "content", None)
                        if not content:
                            continue
                        for part in getattr(content, "parts", []) or []:
                            if getattr(part, "text", None):
                                parts.append(part.text)
                    text = "\n".join(parts).strip()
                if text:
                    time.sleep(inter_call_gap_s)
                    return text
            except Exception:
                _backoff_sleep(attempt)
        return API_FAILURE_FALLBACK
    return fn


def _build_response_fn(provider: str, model_id: str, inter_call_gap_s: float) -> Callable[[str], str]:
    if provider == "openai":
        return _build_response_fn_openai(model_id, inter_call_gap_s)
    if provider == "anthropic":
        return _build_response_fn_anthropic(model_id, inter_call_gap_s)
    if provider == "google":
        return _build_response_fn_google(model_id, inter_call_gap_s)
    raise ValueError(f"unknown provider {provider}")


def _model_short_name(provider: str, model_id: str) -> str:
    aliases = {
        ("openai", "gpt-4o-mini"): "gpt-4o-mini",
        ("openai", "gpt-4o"): "gpt-4o",
        ("anthropic", "claude-3-5-haiku-20241022"): "claude-haiku",
        ("google", "gemini-2.5-flash"): "gemini-flash",
    }
    return aliases.get((provider, model_id), model_id.replace("/", "_"))


def main() -> None:
    cfg = _load_cfg()
    valid_conditions = list(cfg["stage_1"]["conditions"])
    valid_objectives = list(cfg["stage_1"]["objectives"])
    valid_blurbs = list(cfg["stage_1"]["blurb_variants"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "google"])
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--condition", required=True, choices=valid_conditions)
    parser.add_argument("--objective", required=True, choices=valid_objectives)
    parser.add_argument("--blurb_variant", required=True, choices=valid_blurbs)
    parser.add_argument("--n_games", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    sim_cfg = cfg["portfolio_simulator"]
    assets = assets_from_config(sim_cfg["assets"])
    inter_call_gap_s = float(cfg.get("api", {}).get("inter_call_gap_s", 0.2))
    n_games = 5 if args.smoke else (args.n_games or cfg["stage_1"]["n_games_per_cell"])
    out_dir = Path(args.output_dir or cfg["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = 42
    random.seed(seed_base)
    short = _model_short_name(args.provider, args.model_id)

    print(
        f"[m1/api] provider={args.provider} model={args.model_id} condition={args.condition} "
        f"objective={args.objective} blurb={args.blurb_variant} n_games={n_games}"
    )
    response_fn = _build_response_fn(args.provider, args.model_id, inter_call_gap_s)

    results = []
    desc = f"{short}/{args.condition}/{args.objective}/{args.blurb_variant}"
    for i in tqdm(range(n_games), desc=desc):
        game_seed = seed_base + i
        record = run_single_game(
            response_fn=response_fn,
            assets=assets,
            n_rounds=sim_cfg["n_rounds"],
            initial_budget=sim_cfg["initial_budget"],
            round_fraction_of_year=sim_cfg["round_fraction_of_year"],
            seed=game_seed,
            prompt_builder=build_portfolio_prompt,
            parse_fn=parse_allocation,
            condition=args.condition,
            objective=args.objective,
            blurb_variant=args.blurb_variant,
        )
        record["game_id"] = i
        record["model"] = short
        record["model_id"] = args.model_id
        record["provider"] = args.provider
        record["seed"] = game_seed
        results.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = cfg["output"]["filename_pattern"].format(
        model=short,
        condition=args.condition.replace("+", "plus"),
        objective=args.objective,
        blurb=args.blurb_variant,
        timestamp=timestamp,
    )
    payload = {
        "track": "B_M1_portfolio_discriminant",
        "domain": "portfolio",
        "model": short,
        "model_id": args.model_id,
        "provider": args.provider,
        "condition": args.condition,
        "objective": args.objective,
        "blurb_variant": args.blurb_variant,
        "n_games": n_games,
        "smoke": args.smoke,
        "config_snapshot": {
            "portfolio_simulator": sim_cfg,
            "stage_1_n_games_per_cell": cfg["stage_1"]["n_games_per_cell"],
        },
        "timestamp": timestamp,
        "results": results,
    }
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[m1/api] wrote {out_path}")


if __name__ == "__main__":
    main()
