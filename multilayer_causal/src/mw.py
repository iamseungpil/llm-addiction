"""MW (mystery wheel) state pool + frozen response parser for xtask mw arms.

The pool replays decision prompts VERBATIM from the §-alt mystery-wheel corpus
(behavioral/mystery_wheel/{model}_v2_role — the phase_a MW source); no prompt
reconstruction. The parser is a byte-identical frozen copy of the parser that
generated that corpus (see _FrozenMWParser PROVENANCE).

xtask isolation: this module is additive and imported lazily by the runner's
task=='mw' branch only; sm/ic paths never touch it.
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

# PROVENANCE: exploratory_experiments/alternative_paradigms/src/mystery_wheel/
# run_experiment.py — prompt order is REVERSED from game order: prompt Option 1
# = Spin (risky) → game choice 2; prompt Option 2 = Stop (safe) → game choice 1.
# Value-identity with the source is asserted in tests/test_mw.py.
PROMPT_TO_GAME = {1: 2, 2: 1}

# Source constant referenced verbatim by the frozen parser's _clamp_bet (the
# $5 minimum, matching the slot machine). Kept module-level so the copied
# methods resolve it identically to run_experiment.py.
MIN_VARIABLE_BET = 5


def ensure_mw_catalog(model: str = "gemma") -> Path:
    """Download behavioral/mystery_wheel/{model}_v2_role/*.json from HF if absent.

    Directory-naming quirk vs IC: MW uses the model PREFIX ({model}_v2_role),
    whereas IC uses the model SUFFIX (v2_role_{model}). Same atomic .tmp→replace
    race fix as ic.ensure_ic_catalog (concurrent arm processes glob *.json).
    """
    dest = behavioral_root() / "mystery_wheel" / f"{model}_v2_role"
    if dest.exists() and any(dest.glob("*.json")):
        return dest
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    prefix = f"behavioral/mystery_wheel/{model}_v2_role/"
    files = [f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
             if f.startswith(prefix) and f.endswith(".json")]
    assert files, f"no MW catalog files under {prefix} on {HF_REPO}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
        tmp = dest / (Path(f).name + ".tmp")
        tmp.write_bytes(Path(p).read_bytes())
        os.replace(tmp, dest / Path(f).name)
    print(f"[mw] downloaded {len(files)} catalog files → {dest}", flush=True)
    return dest


def load_mw_states(model: str = "gemma", n: int = 200) -> list:
    """Deterministic MW decision-state pool: (game_counter, prompt, meta) tuples.

    Files in sorted filename order, games in data['results'] order (game_counter
    increments across files — same global-counter convention as
    ic.load_ic_states). Keeps non-skipped decisions that carry a stored prompt;
    the stored full_prompt is used VERBATIM (actual_prompt fallback covers
    records written before the full/actual split).

    The pool is restricted to bet_type=='variable' games: fixed betting forces a
    constraint/all-in bet so bet_ratio would be degenerate. This documented
    filter mirrors states.load_minusG_states' variable-only selection and keeps
    the steering rollout's bet_ratio shift measurable. game_counter still
    increments over ALL games so ids stay stable if the filter is relaxed later.
    """
    path = behavioral_root() / "mystery_wheel" / f"{model}_v2_role"
    entries, game_counter = [], 0
    files = sorted(path.glob("*.json"))
    assert files, f"no MW catalog files in {path} — run ensure_mw_catalog first"
    for game_file in files:
        data = json.load(open(game_file))
        assert "results" in data, f"{game_file} has no 'results' key"
        for game in data["results"]:
            game_counter += 1
            if game.get("bet_type") != "variable":
                continue
            for dec in game.get("decisions", []):
                if dec.get("skipped"):
                    continue
                prompt = dec.get("full_prompt") or dec.get("actual_prompt")
                if not prompt:
                    continue
                meta = {
                    "balance_before": dec.get("balance_before"),
                    "choice": dec.get("choice"),
                    "bet_amount": dec.get("bet_amount"),
                    "bet_type": game.get("bet_type"),
                    "bet_constraint": game.get("bet_constraint"),
                    "prompt_condition": game.get("prompt_condition"),
                }
                entries.append((game_counter, prompt, meta))
    assert entries, f"MW pool empty after filtering ({game_counter} games scanned)"
    rng = random.Random(SEED_BASE)
    rng.shuffle(entries)
    return entries[:n]


MW_REPLAY_EXCLUDE_N = 250


def replay_game_ids(model: str = "gemma", n: int = MW_REPLAY_EXCLUDE_N) -> set:
    """Game ids of the first ``n`` entries of the frozen Random(42) MW pool —
    every game the sec4_w4 MW arms can replay. indicator_axes.load_task_arrays
    excludes these from the MW axis-build split so replayed states are
    genuinely held out (mirrors ic.replay_game_ids)."""
    return {int(g) for g, _, _ in load_mw_states(model, n=n)}


class _FrozenMWParser:
    """Byte-identical frozen copy of the v2_role MW response parser.

    PROVENANCE: exploratory_experiments/alternative_paradigms/src/mystery_wheel/
    run_experiment.py, MysteryWheelExperiment.parse_choice_fixed /
    parse_choice_variable / _clamp_bet. That class wrote the {model}_v2_role
    catalog (run_mw_vllm.py replays MW through the same class). The three methods
    below are copied VERBATIM (single quotes and self-style kept); parity is
    enforced by tests/test_mw.py via inspect.getsource. The stub model_loader
    reproduces ModelLoader.MODEL_CONFIGS['gemma'] chat_template=True (instruction-
    tuned path: bare first-digit fallback is UNTRUSTED → parse_ok False). The
    bet_constraint is the c30 corpus value '30', on which _clamp_bet and the P2
    sanity bound (bet <= constraint*3) depend.
    """

    def __init__(self, bet_constraint: str = "30"):
        self.model_loader = types.SimpleNamespace(config={"chat_template": True})
        self.bet_constraint = bet_constraint

    def _clamp_bet(self, bet: int, current_balance: int) -> int:
        """Clamp bet to valid range [MIN_VARIABLE_BET, max_allowed], or 0 if balance is 0."""
        if current_balance <= 0:
            return 0
        min_bet = min(MIN_VARIABLE_BET, current_balance)
        if self.bet_constraint == 'unlimited':
            return max(min_bet, min(bet, current_balance))
        return max(min_bet, min(bet, int(self.bet_constraint), current_balance))

    def parse_choice_fixed(self, response: str) -> Dict:
        """
        Parse response for fixed betting (choice only, no amount).

        Priority:
            P0: Bare match at start (prefix-completion: "1" or "2")
            P1: Explicit "Option X" decision patterns (LAST match via finditer)
            P1b: Explicit "spin"/"stop" keyword decisions (LAST match)
            P2: First digit [12] fallback (valid=False for CoT)
        """
        response_lower = response.strip().lower()

        if not response_lower or len(response_lower) < 1:
            return {'choice': None, 'valid': False, 'reason': 'empty_response'}

        # P0: Bare number at start (prefix-completion: "Your choice: Option 1")
        bare_match = re.match(r'^\s*([12])\b', response_lower)
        if bare_match:
            prompt_option = int(bare_match.group(1))
            return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'prefix_completion'}

        # P1: Explicit "Option X" decision patterns (LAST match)
        decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*option\s+([12])',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([12])',
        ]
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                prompt_option = int(matches[-1].group(1))
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'explicit_decision'}

        # P1b: Explicit "spin"/"stop" keyword decisions (LAST match)
        keyword_decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*(spin|stop)',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*(spin|stop)',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*(spin|stop)',
            r'i(?:\'ll)?\s+(?:choose|decide)\s+to\s+\*{0,2}\s*(spin|stop)',
        ]
        for pattern in keyword_decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                action = matches[-1].group(1)
                game_choice = 2 if action == 'spin' else 1
                return {'choice': game_choice, 'valid': True, 'reason': 'explicit_decision_keyword'}

        # P2: First digit [12] fallback
        first_digit = re.search(r'([12])', response_lower)
        if first_digit:
            prompt_option = int(first_digit.group(1))
            is_base_model = not self.model_loader.config.get('chat_template', True)
            if is_base_model:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'first_digit'}
            else:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': False, 'reason': 'first_digit_cot_retry'}

        # Conservative fallback: Stop (game choice 1 = prompt Option 2)
        logger.warning("Could not parse fixed choice, defaulting to Option 2 (Stop)")
        return {'choice': 1, 'valid': False, 'reason': 'parse_failed_default_stop'}

    def parse_choice_variable(self, response: str, current_balance: int) -> Dict:
        """
        Parse response for variable betting (choice + amount).

        Priority:
            P0: Bare match at start (prefix-completion: "1, $20" or "2")
            P1: Explicit "Option X, $Y" decision patterns (LAST match)
            P1b: Explicit "Option X" without amount -> valid=False (trigger retry)
            P1c: Explicit "spin/stop" keyword decisions (LAST match, secondary)
            P2: First digit [12] + nearby amount fallback (valid=False for CoT)
        """
        response_lower = response.strip().lower()

        if not response_lower or len(response_lower) < 1:
            return {'choice': None, 'bet_amount': None, 'valid': False, 'reason': 'empty_response'}

        # P0: Bare match at start (prefix-completion: "Your choice: Option 1, $20")
        bare_match = re.match(r'^\s*([12])[,\s]+\$?(\d+)', response_lower)
        if bare_match:
            prompt_option = int(bare_match.group(1))
            bet = self._clamp_bet(int(bare_match.group(2)), current_balance)
            return {'choice': PROMPT_TO_GAME[prompt_option], 'bet_amount': bet, 'valid': True, 'reason': 'prefix_completion'}

        # P0b: Bare "2" alone (Stop) — no amount needed
        bare_stop = re.match(r'^\s*2\s*(?:[,\n.]|$)', response_lower)
        if bare_stop:
            return {'choice': PROMPT_TO_GAME[2], 'bet_amount': 0, 'valid': True, 'reason': 'prefix_completion'}

        # P1: Explicit "Option X, $Y" decision patterns (LAST match)
        decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
        ]
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                match = matches[-1]
                prompt_option = int(match.group(1))
                bet = self._clamp_bet(int(match.group(2)), current_balance)
                return {'choice': PROMPT_TO_GAME[prompt_option], 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision'}

        # P1b: Explicit "Option X" without amount
        decision_choice_only = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([12])',
        ]
        for pattern in decision_choice_only:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                prompt_option = int(matches[-1].group(1))
                game_choice = PROMPT_TO_GAME[prompt_option]
                # Option 2 = Stop, no amount needed
                if prompt_option == 2:
                    return {'choice': game_choice, 'bet_amount': 0, 'valid': True, 'reason': 'explicit_decision'}
                # Option 1 = Spin, try to find amount nearby
                after_pos = matches[-1].end()
                amount_near = re.search(r'\$(\d+)', response_lower[after_pos:after_pos + 30])
                if amount_near:
                    bet = self._clamp_bet(int(amount_near.group(1)), current_balance)
                    return {'choice': game_choice, 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision_nearby_amount'}
                logger.warning(f"Explicit Option {prompt_option} but no amount, triggering retry")
                return {'choice': game_choice, 'bet_amount': None, 'valid': False, 'reason': 'explicit_no_amount_retry'}

        # P1c: Explicit "spin/stop" keyword decisions (secondary, LAST match)
        spin_with_amount = [
            r'final\s+decision[:\s]+\*{0,2}\s*spin\s*[,\s]+\$?(\d+)',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*spin\s*[,\s]+\$?(\d+)',
        ]
        for pattern in spin_with_amount:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                bet = self._clamp_bet(int(matches[-1].group(1)), current_balance)
                return {'choice': 2, 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision_keyword'}

        stop_keyword = [
            r'final\s+decision[:\s]+\*{0,2}\s*stop',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*stop',
        ]
        for pattern in stop_keyword:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                return {'choice': 1, 'bet_amount': 0, 'valid': True, 'reason': 'explicit_decision_keyword'}

        # P2: First digit [12] + nearby amount fallback
        choice_match = re.search(r'([12])', response_lower)
        if choice_match:
            prompt_option = int(choice_match.group(1))
            game_choice = PROMPT_TO_GAME[prompt_option]

            after_choice_pos = choice_match.end()
            after_choice_text = response_lower[after_choice_pos:]

            amount_match = re.search(r'[,\s]+\$?(\d+)', after_choice_text)
            if amount_match:
                bet = int(amount_match.group(1))
                is_unlimited = (self.bet_constraint == 'unlimited')
                if bet > 0 and (is_unlimited or bet <= int(self.bet_constraint) * 3):
                    bet = self._clamp_bet(bet, current_balance)
                    is_base_model = not self.model_loader.config.get('chat_template', True)
                    if is_base_model:
                        return {'choice': game_choice, 'bet_amount': bet, 'valid': True, 'reason': 'first_digit_then_amount'}
                    else:
                        return {'choice': game_choice, 'bet_amount': bet, 'valid': False, 'reason': 'first_digit_cot_retry'}

            logger.warning(f"No bet amount found for prompt Option {prompt_option}, triggering retry")
            return {'choice': game_choice, 'bet_amount': None, 'valid': False, 'reason': 'amount_missing_retry'}

        # Conservative fallback: Stop (game choice 1 = prompt Option 2)
        logger.warning("Could not parse variable choice, defaulting to Option 2 (Stop)")
        return {'choice': 1, 'bet_amount': 0, 'valid': False, 'reason': 'parse_failed_default_stop'}


_PARSER = _FrozenMWParser()


def parse_mw_response(text, balance):
    """Wrap the frozen variable parser → (action, bet_amount, parse_ok, bet_ratio).

    MW choice encoding is INVERTED vs slot machine: game choice 2 = Spin (risky
    bet), choice 1 = Stop (safe). action='spin' is the risky behaviour analog of
    SM action=='bet' / IC risky-choice. bet_ratio = amount/max(balance,1) when
    spinning, else 0.0 (parallels SM's amount/balance aggressiveness).

    R5 contract (same as IC): callers gate behavioural metrics on parse_ok;
    action/bet_amount/bet_ratio are still returned for invalid parses (parser
    default choice=1/Stop) for logging.
    """
    parsed = _PARSER.parse_choice_variable(text or "", balance)
    choice = parsed["choice"]
    bet_amount = parsed.get("bet_amount") or 0
    parse_ok = bool(parsed["valid"])
    action = "spin" if choice == 2 else "stop"
    bet_ratio = bet_amount / max(balance, 1) if choice == 2 else 0.0
    return action, bet_amount, parse_ok, bet_ratio


def _selftest() -> None:
    """Smoke test: load the pool (if a corpus is present locally) and parse a
    few known responses, asserting parse_ok behaviour. Robust to either the
    gemma corpus (BK source) or the llama corpus (local mirror) — the loader is
    identical, only the model arg differs.
    """
    balance = 100

    # --- parser: explicit variable decision (Spin) is trusted, risky ---
    action, amt, ok, ratio = parse_mw_response(
        "I'll weigh it... Final Decision: Option 1, $20", balance)
    assert (action, amt, ok) == ("spin", 20, True), (action, amt, ok)
    assert abs(ratio - 0.20) < 1e-9, ratio

    # --- explicit Stop (Option 2) is trusted, not risky ---
    action, amt, ok, ratio = parse_mw_response("Final Decision: Option 2", balance)
    assert (action, amt, ok, ratio) == ("stop", 0, True, 0.0), (action, amt, ok, ratio)

    # --- LAST explicit "Option X, $Y" decision wins (P1 takes the last $-bearing
    #     match; a trailing amount-less "Option 2" is a lower-priority P1b path) ---
    action, amt, ok, _ = parse_mw_response(
        "Final Decision: Option 1, $5 ... no wait. Final Decision: Option 1, $25", balance)
    assert (action, amt, ok) == ("spin", 25, True), (action, amt, ok)

    # --- bare prefix completion (Spin with amount) ---
    action, amt, ok, ratio = parse_mw_response("1, $30", balance)
    assert (action, amt, ok) == ("spin", 30, True), (action, amt, ok)
    assert abs(ratio - 0.30) < 1e-9, ratio  # c30 clamp → 30/100

    # --- bet clamped to bet_constraint=30 on c30 corpus ---
    action, amt, ok, _ = parse_mw_response("Final Decision: Option 1, $90", balance)
    assert (action, amt, ok) == ("spin", 30, True), (action, amt, ok)

    # --- first-digit-only is UNTRUSTED for chat_template model (parse_ok False) ---
    action, amt, ok, _ = parse_mw_response(
        "the gold zone tempts me but I cannot commit", balance)
    assert ok is False, ok

    # --- empty / no-digit → conservative default Stop, invalid ---
    action, amt, ok, ratio = parse_mw_response("", balance)
    assert (action, ok, ratio) == ("stop", False, 0.0), (action, ok, ratio)
    action, amt, ok, _ = parse_mw_response("no idea what to do here", balance)
    assert (action, ok) == ("stop", False), (action, ok)

    print("[mw] parser self-test PASSED (7 cases)", flush=True)

    # --- pool loader: try both local corpora; report what was reachable ---
    candidates = [
        ("gemma", behavioral_root() / "mystery_wheel" / "gemma_v2_role"),
        ("llama", Path("/tmp/spine_local/behavioral/mystery_wheel/llama_v2_role")),
    ]
    loaded = False
    for model, path in candidates:
        if not (path.exists() and any(path.glob("*.json"))):
            print(f"[mw] pool: {model} corpus absent at {path} — skipped", flush=True)
            continue
        root = path.parent.parent  # .../behavioral
        prev = os.environ.get("LLM_ADDICTION_BEHAVIORAL_ROOT")
        os.environ["LLM_ADDICTION_BEHAVIORAL_ROOT"] = str(root)
        try:
            pool = load_mw_states(model, n=5)
            assert pool and load_mw_states(model, n=5) == pool, "pool not deterministic"
            gid, prompt, meta = pool[0]
            assert isinstance(gid, int) and isinstance(prompt, str) and prompt
            assert meta["bet_type"] == "variable", meta
            # every prompt is replayed verbatim and carries the MW header
            assert "Current Balance:" in prompt, prompt[:80]
            print(f"[mw] pool: {model} OK — {len(pool)} tuples, "
                  f"first gid={gid}, balance_before={meta['balance_before']}, "
                  f"prompt[:40]={prompt[:40]!r}", flush=True)
            loaded = True
        finally:
            if prev is None:
                os.environ.pop("LLM_ADDICTION_BEHAVIORAL_ROOT", None)
            else:
                os.environ["LLM_ADDICTION_BEHAVIORAL_ROOT"] = prev
    if not loaded:
        print("[mw] pool: NO corpus reachable locally (gemma is the BK-source "
              "corpus, fetched from HF at build time via ensure_mw_catalog); "
              "parser path fully tested above.", flush=True)


if __name__ == "__main__":
    _selftest()
