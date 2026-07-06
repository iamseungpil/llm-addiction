"""IC (investment choice) state pool + frozen response parser for w1ic arms.

The pool replays decision prompts VERBATIM from the §4 IC corpus
(behavioral/investment_choice/v2_role_gemma — the phase_a IC source); no
prompt reconstruction. The parser is a byte-identical frozen copy of the
parser that generated that corpus (see _FrozenICParser PROVENANCE).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import types
from pathlib import Path
from typing import Dict

from .states import HF_REPO, SEED_BASE, behavioral_root

logger = logging.getLogger(__name__)

# PROVENANCE: exploratory_experiments/alternative_paradigms/src/investment_choice/
# run_experiment.py — prompt Option k (riskiest first) → game choice (4 = riskiest,
# 1 = safe stop). Value-identity with the source is asserted in tests/test_ic.py.
PROMPT_TO_GAME = {1: 4, 2: 3, 3: 2, 4: 1}


def ensure_ic_catalog(model: str = "gemma") -> Path:
    """Download behavioral/investment_choice/v2_role_{model}/*.json from HF if absent."""
    dest = behavioral_root() / "investment_choice" / f"v2_role_{model}"
    if dest.exists() and any(dest.glob("*.json")):
        return dest
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    prefix = f"behavioral/investment_choice/v2_role_{model}/"
    files = [f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
             if f.startswith(prefix) and f.endswith(".json")]
    assert files, f"no IC catalog files under {prefix} on {HF_REPO}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
        # Atomic write: concurrent arm processes glob *.json — a half-written
        # .json must never be visible (same race fix as states.ensure_sm_catalog).
        tmp = dest / (Path(f).name + ".tmp")
        tmp.write_bytes(Path(p).read_bytes())
        os.replace(tmp, dest / Path(f).name)
    print(f"[ic] downloaded {len(files)} catalog files → {dest}", flush=True)
    return dest


def load_ic_states(model: str = "gemma", n: int = 200) -> list:
    """Deterministic IC decision-state pool: (game_counter, prompt, meta) tuples.

    Files in sorted filename order, games in data['results'] order (game_counter
    increments across files — same global-counter convention as
    behavior_axis.load_round_labels). Keeps decisions that were not skipped and
    have a stored prompt; the stored full_prompt is used VERBATIM (actual_prompt
    fallback covers records written before the full/actual split).
    """
    path = behavioral_root() / "investment_choice" / f"v2_role_{model}"
    entries, game_counter = [], 0
    files = sorted(path.glob("*.json"))
    assert files, f"no IC catalog files in {path} — run ensure_ic_catalog first"
    for game_file in files:
        data = json.load(open(game_file))
        assert "results" in data, f"{game_file} has no 'results' key"
        for game in data["results"]:
            game_counter += 1
            for dec in game.get("decisions", []):
                if dec.get("skipped"):
                    continue
                prompt = dec.get("full_prompt") or dec.get("actual_prompt")
                if not prompt:
                    continue
                meta = {
                    "balance_before": dec.get("balance_before"),
                    "choice": dec.get("choice"),
                    "bet_constraint": game.get("bet_constraint"),
                }
                entries.append((game_counter, prompt, meta))
    assert entries, f"IC pool empty after filtering ({game_counter} games scanned)"
    rng = random.Random(SEED_BASE)
    rng.shuffle(entries)
    return entries[:n]


# Pool length the sec4_w3 IC arms load: n(200) + state_offset(0) + the runner's
# +50 pad (_run_arm_ic: load_ic_states(n = n + offset + 50)). The axis builder
# excludes every game in this window so the replayed IC states are held out.
IC_REPLAY_EXCLUDE_N = 250


def replay_game_ids(model: str = "gemma", n: int = IC_REPLAY_EXCLUDE_N) -> set:
    """Game ids (global counters — the behavior_axis id-space load_ic_states
    documents) of the first ``n`` entries of the frozen Random(42) IC pool,
    i.e. every game the sec4_w3 IC arms can replay. indicator_axes.
    load_task_arrays excludes these from the IC axis-build split so the
    replayed states are genuinely held out (mirrors the SM excluded_game_ids
    convention)."""
    return {int(g) for g, _, _ in load_ic_states(model, n=n)}


class _FrozenICParser:
    """Byte-identical frozen copy of the v2_role IC response parser.

    PROVENANCE: exploratory_experiments/alternative_paradigms/src/investment_choice/
    run_experiment.py lines 262-313, InvestmentChoiceExperiment.parse_choice_fixed.
    That class wrote the v2_role_gemma catalog (json header == its run_experiment
    final_output; sae_v3_analysis/src/exact_behavioral_replay.py replays IC through
    the same class). Copied verbatim — single quotes and self-style kept; parity
    enforced by tests/test_ic.py via inspect.getsource. The stub model_loader
    reproduces ModelLoader.MODEL_CONFIGS['gemma'] chat_template=True (instruction-
    tuned path: bare first-digit fallback is UNTRUSTED → parse_ok False).
    """

    def __init__(self):
        self.model_loader = types.SimpleNamespace(config={"chat_template": True})

    def parse_choice_fixed(self, response: str) -> Dict:
        """
        Parse response for fixed betting (choice only).

        Args:
            response: Model response

        Returns:
            Dict with 'choice', 'valid', optional 'reason'
        """
        response_lower = response.strip().lower()

        # Empty response check (allow single digit responses like "2")
        if not response_lower or len(response_lower) < 1:
            return {'choice': None, 'valid': False, 'reason': 'empty_response'}

        # PRIORITY 0: Bare number (prefix-completion "Your choice: Option 2")
        # Model just outputs "2" or "2\n..."
        bare_match = re.match(r'^\s*([1234])\b', response_lower)
        if bare_match:
            prompt_option = int(bare_match.group(1))
            return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'prefix_completion'}

        # PRIORITY 1: Explicit decision patterns (CoT models put decision at the END)
        # Use LAST match to get the final stated decision, not analysis text
        decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([1234])',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([1234])',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*option\s+([1234])',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([1234])',
        ]
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                prompt_option = int(matches[-1].group(1))
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'explicit_decision'}

        # PRIORITY 2: Extract first digit 1-4 (fallback)
        # Base models: trusted (P0 would have caught structured responses)
        # CoT models: UNTRUSTED — first digit is likely from reasoning text, trigger retry
        first_digit = re.search(r'([1234])', response_lower)
        if first_digit:
            prompt_option = int(first_digit.group(1))
            is_base_model = not self.model_loader.config.get('chat_template', True)
            if is_base_model:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'first_digit'}
            else:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': False, 'reason': 'first_digit_cot_retry'}

        # Conservative fallback: Stop (game choice 1 = prompt Option 4)
        logger.warning(f"Could not parse fixed choice, defaulting to Option 4 (Stop)")
        return {'choice': 1, 'valid': False, 'reason': 'parse_failed_default_stop'}


_PARSER = _FrozenICParser()


def parse_ic_response(text):
    """Wrap the frozen parser: (choice 1-4 or None, parse_ok, risky=choice in (3,4)).

    R5 contract: callers must gate behavioural metrics on parse_ok; choice/risky
    are still returned for invalid parses (parser default = 1/stop) for logging.
    """
    parsed = _PARSER.parse_choice_fixed(text or "")
    choice = parsed["choice"]
    parse_ok = bool(parsed["valid"])
    risky = choice in (3, 4)
    return choice, parse_ok, risky
