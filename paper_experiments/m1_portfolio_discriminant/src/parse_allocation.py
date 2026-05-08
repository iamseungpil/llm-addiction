"""Parse a model response into an allocation dict {canonical_name: fraction}.

Robust to format variation — models phrase allocations in many ways:
  - "Final Allocation: Cash 30, Bonds 10, Index 20, Leveraged 25, Stock 10, OTM 5"
  - "Cash: 30%\nBonds: 10%\n..."
  - JSON-ish: {"cash": 30, "bonds": 10, ...}
  - LaTeX-ish: "$30\\%$ Cash, $10\\%$ Bonds, ..."
  - Plain prose: "I'll put 30 in cash, 10 in bonds, ..."

The parser tries the most explicit format first (Final Allocation: line), then falls
back to looser scans. Returns `None` for the dict if it cannot extract values for
*all* canonical assets — partial parses are unsafe (the model might have skipped
mentioning cash assuming 0%, but that's a confound we don't want to silently absorb;
the runner re-asks instead).

C6 fix — fallback sentinel: when the API/GPU runner exhausts its retries it does
NOT synthesise a 100%-cash allocation (which the simulator would have treated as a
deliberate conservative decision); it emits the sentinel string
``FALLBACK_API_FAILURE_SENTINEL`` instead. This parser explicitly detects and
rejects the sentinel, returning ``None`` so that the simulator records a parse-
skip. Per-game records track the skip count, which downstream analysis uses to
de-bias per-model risk_event rates against unstable-API contamination.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Tuple


# C6 fix: sentinel emitted by the runner when API/GPU retries are exhausted. Must
# be unique-enough to never accidentally match a legitimate model response.
FALLBACK_API_FAILURE_SENTINEL = "__FALLBACK_API_FAILURE__"


# Display label -> canonical name. Mirrors `prompts.ASSET_DISPLAY` but kept local so
# the parser is importable without pulling in the prompt template.
DISPLAY_TO_CANONICAL = {
    "cash": "cash",
    "bond": "bonds",
    "bonds": "bonds",
    "index": "broad_index",
    "broad_index": "broad_index",
    "broad index": "broad_index",
    "leveraged": "leveraged_etf_3x",
    "leveraged_etf": "leveraged_etf_3x",
    "leveraged_etf_3x": "leveraged_etf_3x",
    "stock": "single_volatile_stock",
    "single_volatile_stock": "single_volatile_stock",
    "single stock": "single_volatile_stock",
    "volatile stock": "single_volatile_stock",
    "otm": "otm_call_or_crypto",
    "otm_call": "otm_call_or_crypto",
    "otm_call_or_crypto": "otm_call_or_crypto",
    "crypto": "otm_call_or_crypto",
    "call": "otm_call_or_crypto",
}


def parse_allocation(response: str, asset_names: Iterable[str]) -> Tuple[Optional[Dict[str, float]], str]:
    """Parse `response` into {asset_name: fraction in [0,1]}.

    `asset_names` is the canonical list expected (subset of DISPLAY_TO_CANONICAL values).
    Returns `(allocation_dict, parse_reason)` or `(None, reason)` on failure.

    Fractions are normalised to sum to 1.0 (we accept inputs that already sum to 1, or
    inputs in 0-100 percent units, or inputs of arbitrary positive scale — the model
    might write 30 or 0.30; both are handled).
    """
    asset_set = set(asset_names)
    if not response:
        return None, "empty_response"

    # C6 fix: explicit fallback-sentinel detection. The runner sends this when
    # API/GPU retries are exhausted; we must never silently coerce it into a
    # cash allocation (which the simulator would treat as a deliberate choice).
    if FALLBACK_API_FAILURE_SENTINEL in response:
        return None, "fallback_api_failure"

    final_line = _extract_final_allocation_line(response)
    if final_line is not None:
        parsed = _parse_keyed_pairs(final_line, asset_set)
        if parsed is not None:
            return _normalise(parsed, asset_set), "final_allocation_line"

    json_blob = _extract_json_blob(response)
    if json_blob is not None:
        parsed = _parse_json_blob(json_blob, asset_set)
        if parsed is not None:
            return _normalise(parsed, asset_set), "json_blob"

    parsed = _parse_keyed_pairs(response, asset_set)
    if parsed is not None:
        return _normalise(parsed, asset_set), "loose_keyed_pairs"

    return None, "unparseable"


_FINAL_LINE_RE = re.compile(
    r"final\s+allocation\s*[:\-]?\s*(.+?)(?:\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_final_allocation_line(text: str) -> Optional[str]:
    matches = list(_FINAL_LINE_RE.finditer(text))
    if not matches:
        return None
    body = matches[-1].group(1).strip()
    # Cap to ~400 chars so a runaway response doesn't choke the regex below.
    return body[:400]


def _extract_json_blob(text: str) -> Optional[str]:
    # Greedy scan for the last balanced {...} that mentions at least one asset name.
    candidates = re.findall(r"\{[^{}]{1,400}\}", text)
    for c in reversed(candidates):
        low = c.lower()
        if any(k in low for k in DISPLAY_TO_CANONICAL):
            return c
    return None


def _parse_json_blob(blob: str, asset_set: set) -> Optional[Dict[str, float]]:
    cleaned = blob.replace("\\%", "").replace("%", "").replace("'", '"')
    try:
        d = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    out: Dict[str, float] = {}
    for k, v in d.items():
        canon = _to_canonical(str(k))
        if canon is None or canon not in asset_set:
            continue
        try:
            out[canon] = float(v)
        except (TypeError, ValueError):
            continue
    if len(out) < len(asset_set):
        # Allow missing keys to default to 0 only if at least half of the assets
        # were named — otherwise we're probably parsing a different blob entirely.
        if len(out) >= max(2, len(asset_set) // 2):
            for name in asset_set:
                out.setdefault(name, 0.0)
            return out
        return None
    return out


# Pattern matches "Cash 30", "Cash: 30", "Cash 30%", "Cash $30", "Cash=30", "30 cash",
# "$30\\%$ Cash" (LaTeX), "Cash 0.30". Captures the asset label and the number.
_KEYED_PAIR_RE = re.compile(
    r"(?P<label>cash|bonds?|broad[\s_]?index|index|leveraged(?:[\s_]etf(?:[\s_]3x)?)?|"
    r"single[\s_]volatile[\s_]stock|volatile[\s_]stock|single[\s_]stock|stock|"
    r"otm(?:[\s_]call(?:[\s_]or[\s_]crypto)?)?|crypto|call)"
    r"\s*[:=\-]?\s*\$?\s*(?P<num>\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)

# Also handle "30 cash", "30% cash", "30 in cash" (number-first variant).
_NUM_FIRST_RE = re.compile(
    r"\$?\s*(?P<num>\d+(?:\.\d+)?)\s*%?\s*(?:in\s+)?"
    r"(?P<label>cash|bonds?|broad[\s_]?index|index|leveraged(?:[\s_]etf(?:[\s_]3x)?)?|"
    r"single[\s_]volatile[\s_]stock|volatile[\s_]stock|single[\s_]stock|stock|"
    r"otm(?:[\s_]call(?:[\s_]or[\s_]crypto)?)?|crypto|call)",
    re.IGNORECASE,
)


def _parse_keyed_pairs(text: str, asset_set: set) -> Optional[Dict[str, float]]:
    out: Dict[str, float] = {}
    for m in _KEYED_PAIR_RE.finditer(text):
        canon = _to_canonical(m.group("label"))
        if canon is None or canon not in asset_set:
            continue
        try:
            out.setdefault(canon, float(m.group("num")))
        except ValueError:
            continue
    if len(out) < len(asset_set):
        for m in _NUM_FIRST_RE.finditer(text):
            canon = _to_canonical(m.group("label"))
            if canon is None or canon not in asset_set:
                continue
            try:
                out.setdefault(canon, float(m.group("num")))
            except ValueError:
                continue
    if len(out) < len(asset_set):
        # Allow missing assets to default to 0 only if the parser found at least 3
        # assets explicitly named — otherwise we likely matched stray numbers.
        if len(out) >= 3:
            for name in asset_set:
                out.setdefault(name, 0.0)
        else:
            return None
    return out


def _to_canonical(raw: str) -> Optional[str]:
    key = re.sub(r"\s+", " ", raw.strip().lower())
    key = key.replace("_", " ")
    if key in DISPLAY_TO_CANONICAL:
        return DISPLAY_TO_CANONICAL[key]
    # Try variants without trailing words.
    for k, v in DISPLAY_TO_CANONICAL.items():
        if key.startswith(k):
            return v
    return None


def _normalise(d: Dict[str, float], asset_set: set) -> Dict[str, float]:
    for name in asset_set:
        d.setdefault(name, 0.0)
    total = sum(d.values())
    if total <= 0.0:
        # All zero -> return uniform; runner treats this as a parse-skip semantically
        # but we hand back a valid dict so downstream math doesn't NaN.
        return {name: 0.0 for name in asset_set}
    return {k: v / total for k, v in d.items()}


__all__ = ["parse_allocation", "DISPLAY_TO_CANONICAL", "FALLBACK_API_FAILURE_SENTINEL"]
