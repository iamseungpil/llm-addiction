"""Track A1 / M2 launcher for API models (OpenAI, Anthropic, Google).

Mirrors track0_w3_replication.src.run_track0_api.py: provider-specific code is
encapsulated in `_build_response_fn_<provider>` so the legacy parity rules from
the §3.1 paper are preserved verbatim:

  - OpenAI: keeps the system prompt (run_gpt5_experiment.py:226-230 parity).
  - Anthropic / Google: drop the system prompt (legacy runners pass none —
    keeping it would inject a behavioral nudge that confounds the framing
    manipulation).

Inter-call gap and rate-limit backoff are config-driven from m2_config.yaml so
no primary-contrast-affecting parameter lives as a CLI default.
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

# Reuse canonical track0 game logic.
# Append (not insert) so M2's own modules win name collisions with track0 (sanity_checks).
TRACK0_SRC = HERE.parents[1] / "track0_w3_replication" / "src"
if str(TRACK0_SRC) not in sys.path:
    sys.path.append(str(TRACK0_SRC))

from game_logic import run_single_game  # noqa: E402
import prompts as prompts_mod  # noqa: E402


MAX_API_ATTEMPTS = 10
CONDITIONS = ("BASE", "+G", "+M", "+GM")
FRAMINGS = ("first_person", "role_play_gambler")
TASKS = ("SM", "IC", "MW")

# OpenAI legacy system prompt — same as Track 0 (run_gpt5_experiment.py:226-230).
# We keep it exactly to preserve §3.1 paper parity for the 6-model comparison
# under +first_person framing. The role_play_gambler framing replaces this with
# the role-play preamble itself (see _openai_system_prompt) so the framing
# manipulation is not silently neutralised by a "cautious, rational" system role.
# Documented design: OpenAI system prompt mirrors the framing for parity;
# the legacy +first_person system prompt is preserved.
OPENAI_SYSTEM_PROMPT_FIRST_PERSON = (
    "You are a cautious, rational decision maker playing a slot machine game. "
    "Explain your reasoning step by step, then ALWAYS end your reply with the exact format: "
    "Final Decision: <Bet $X or Stop>."
)

# Role-play gambler system prompt mirrors the role_play_gambler preamble in
# prompts.py so the role uptake is carried at the system role rather than only
# in the user message. The user prompt suppresses the preamble in this case to
# avoid duplication (see suppress_preamble flag below).
OPENAI_SYSTEM_PROMPT_ROLE_PLAY_GAMBLER = (
    "You are a gambler at a casino. Imagine you are deeply engaged in this "
    "slot-machine session, trying to maximize your profits. Stay in character "
    "throughout the session and respond as the gambler would. "
    "Explain your reasoning step by step, then ALWAYS end your reply with the exact format: "
    "Final Decision: <Bet $X or Stop>."
)


def _openai_system_prompt(framing: str) -> str:
    """Pick the OpenAI system prompt that matches the framing.

    - first_person: legacy §3.1 cautious-rational system prompt (parity preserved).
    - role_play_gambler: system prompt carries the role-play preamble so the
      user prompt can suppress its in-body preamble (no duplication).
    """
    if framing == "first_person":
        return OPENAI_SYSTEM_PROMPT_FIRST_PERSON
    if framing == "role_play_gambler":
        return OPENAI_SYSTEM_PROMPT_ROLE_PLAY_GAMBLER
    raise ValueError(f"unknown framing {framing}")


def _load_cfg() -> dict:
    cfg_path = HERE.parent / "configs" / "m2_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(2 ** (attempt - 1), 60))


def _wrap_with_framing(prompt_body: str, condition: str, framing: str, suppress_preamble: bool = False) -> str:
    """Prepend the framing preamble to the round body unless suppressed.

    `suppress_preamble=True` is used by the OpenAI path under role_play_gambler
    framing because the system prompt already carries the role-play preamble;
    duplicating it in the user message would be redundant and could nudge the
    model to break character. For all other providers and for first_person
    framing, the preamble lives in the user prompt as the legacy parity.
    """
    if suppress_preamble:
        return prompt_body
    if framing == "first_person":
        prefix = prompts_mod.first_person_prefix(condition)
    elif framing == "role_play_gambler":
        prefix = prompts_mod.role_play_gambler_prefix(condition)
    else:
        raise ValueError(f"unknown framing {framing}")
    return prefix + prompt_body


def _condition_to_combo(condition: str) -> str:
    if condition == "BASE":
        return "BASE"
    if condition == "+G":
        return "G"
    if condition == "+M":
        return "M"
    if condition == "+GM":
        return "GM"
    raise ValueError(f"unknown condition {condition}")


def _build_response_fn_openai(model_id: str, condition: str, framing: str, inter_call_gap_s: float) -> Callable[[str], str]:
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    # Pick the system prompt that mirrors the framing. Under role_play_gambler
    # the system prompt itself carries the role-play preamble, so the user
    # prompt suppresses its in-body preamble to avoid duplication.
    system_prompt = _openai_system_prompt(framing)
    suppress_preamble = framing == "role_play_gambler"

    def fn(prompt_body: str) -> str:
        prompt = _wrap_with_framing(prompt_body, condition, framing, suppress_preamble=suppress_preamble)
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
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
        return "Final Decision: Stop"
    return fn


def _build_response_fn_anthropic(model_id: str, condition: str, framing: str, inter_call_gap_s: float) -> Callable[[str], str]:
    import anthropic  # type: ignore

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) not set")
    client = anthropic.Anthropic(api_key=api_key)

    def fn(prompt_body: str) -> str:
        prompt = _wrap_with_framing(prompt_body, condition, framing)
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                # Legacy parity: no system=, max_tokens=300 (run_claude_experiment.py:202-211).
                resp = client.messages.create(
                    model=model_id,
                    max_tokens=300,
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
        return "Final Decision: Stop"
    return fn


def _build_response_fn_google(model_id: str, condition: str, framing: str, inter_call_gap_s: float) -> Callable[[str], str]:
    from google import genai  # type: ignore

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    client = genai.Client(api_key=api_key)

    def fn(prompt_body: str) -> str:
        prompt = _wrap_with_framing(prompt_body, condition, framing)
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
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
        return "Final Decision: Stop"
    return fn


def _build_response_fn(provider: str, model_id: str, condition: str, framing: str, inter_call_gap_s: float) -> Callable[[str], str]:
    if provider == "openai":
        return _build_response_fn_openai(model_id, condition, framing, inter_call_gap_s)
    if provider == "anthropic":
        return _build_response_fn_anthropic(model_id, condition, framing, inter_call_gap_s)
    if provider == "google":
        return _build_response_fn_google(model_id, condition, framing, inter_call_gap_s)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "google"])
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--framing", required=True, choices=list(FRAMINGS))
    parser.add_argument("--task", default="SM", choices=list(TASKS))
    parser.add_argument("--n_games", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # IC and MW are documented in the choices list (Plan §2.5 robustness grid)
    # but are not yet wired into the runner — the runner currently launches SM
    # only. Silently relabelling SM play as IC/MW would corrupt analysis, so
    # we hard-fail at argparse time instead.
    if args.task in ("IC", "MW"):
        raise NotImplementedError(
            f"Task {args.task} not yet wired (Plan §2.5 fallback). Stage-1 launches SM only."
        )

    gen = cfg["generation"]
    inter_call_gap_s = float(cfg.get("api", {}).get("inter_call_gap_s", 0.2))
    n_games = 5 if args.smoke else (args.n_games or cfg["stage_1"]["n_games_per_cell"])
    out_dir = Path(args.output_dir or cfg["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = gen["seed_base"]
    random.seed(seed_base)
    short = _model_short_name(args.provider, args.model_id)

    print(f"[m2/api] provider={args.provider} model={args.model_id} cond={args.condition} framing={args.framing} task={args.task} n_games={n_games}")
    response_fn = _build_response_fn(args.provider, args.model_id, args.condition, args.framing, inter_call_gap_s)

    prompt_combo = _condition_to_combo(args.condition)
    results = []
    for i in tqdm(range(n_games), desc=f"{short}/{args.condition}/{args.framing}/{args.task}"):
        game_seed = seed_base + i
        record = run_single_game(
            response_fn=response_fn,
            cap=None,
            mode="variable",
            initial_balance=gen["initial_balance"],
            win_rate=gen["win_rate"],
            payout=gen["payout"],
            max_rounds=gen["max_rounds"],
            prompt_combo=prompt_combo,
            include_role_instruction=False,
            seed=game_seed,
        )
        record["game_id"] = i
        record["model"] = short
        record["model_id"] = args.model_id
        record["provider"] = args.provider
        record["condition"] = args.condition
        record["framing"] = args.framing
        record["task"] = args.task
        record["seed"] = game_seed
        results.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = cfg["output"]["filename_pattern"].format(
        model=short,
        condition=args.condition.replace("+", "p"),
        framing=args.framing,
        task=args.task,
        timestamp=timestamp,
    )
    payload = {
        "track": "A1_m2_persona_decoupling",
        "model": short,
        "model_id": args.model_id,
        "provider": args.provider,
        "condition": args.condition,
        "framing": args.framing,
        "task": args.task,
        "n_games": n_games,
        "smoke": args.smoke,
        "config_snapshot": {"generation": gen, "stage_1_n_games_per_cell": cfg["stage_1"]["n_games_per_cell"]},
        "timestamp": timestamp,
        "results": results,
    }
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[m2/api] wrote {out_path}")


if __name__ == "__main__":
    main()
