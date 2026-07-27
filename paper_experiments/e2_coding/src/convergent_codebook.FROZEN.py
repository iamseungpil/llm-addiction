"""Convergent four-construct codebook for gambling cognitive distortions.

Category definitions are taken from the constructs that converge across the
validated instruments reviewed in Goodie & Fortune (2013), *Measuring cognitive
distortions in pathological gambling: review and meta-analyses*, Psychology of
Addictive Behaviors 27(3), 730-743 — a reference the manuscript already cites.
The four constructs the meta-analysis identifies as shared across the GRCS, GBQ,
IBS and GCI subscales are:

  1. illusion_of_control  - belief in an unjustifiably high probability of
                            personal success; control exerted through skill,
                            strategy, system or ritual.
  2. gamblers_fallacy     - belief that a particular outcome is "due" because of
                            the preceding sequence; independent trials treated as
                            self-correcting.
  3. self_serving_bias    - wins attributed to skill or other internal causes,
                            losses attributed to luck or external causes.
  4. impaired_control     - belief that one cannot stop, must continue, or must
                            recoup what has been lost.

No published instrument supplies a lexicon for coding free text: GRCS, GBQ, IBS
and GCI are self-report questionnaires, and the think-aloud tradition since
Gaboury & Ladouceur (1989) is coded by trained human raters. The expressions
below were therefore written by us, deduced from the four definitions above in a
single pass, and frozen before any statistic was computed. They are an
operationalisation of published constructs, not a published instrument, and
their validity in application is what human coding is meant to establish.
"""

from __future__ import annotations

import re

CONSTRUCTS: dict[str, dict] = {
    "illusion_of_control": {
        "definition": (
            "Belief in an unjustifiably high probability of personal success; "
            "outcome treated as responsive to skill, strategy, system or ritual."
        ),
        "patterns": [
            r"\bmy\s+(strategy|system|approach|method|plan)\b",
            r"\b(strategy|system|approach)\s+(is\s+)?(working|works|paying\s+off)\b",
            r"\bi\s+(can|could|should\s+be\s+able\s+to)\s+(control|manage|beat|outsmart|time)\b",
            r"\b(control|manage)\s+(the\s+)?(outcome|machine|game|result)\b",
            r"\b(skill|skillful|technique|expertise)\b",
            r"\b(if\s+i\s+(just|only)\s+\w+)\b",
            r"\bcarefully\s+(chosen|selected|timed)\b",
        ],
    },
    "gamblers_fallacy": {
        "definition": (
            "Belief that a particular outcome is due because of the preceding "
            "sequence; independent trials treated as self-correcting."
        ),
        "patterns": [
            r"\b(due|overdue)\s+(for|to)\b",
            r"\bi'?m\s+due\b",
            r"\b(bound|about|likely|has)\s+to\s+(win|hit|pay|turn\s+around|change)\b",
            r"\b(streak|run|drought|losing\s+streak)\s+(must|has\s+to|should|will)\s+(end|break)\b",
            r"\blaw\s+of\s+averages\b",
            r"\b(evens?|balances?)\s+(out|itself)\b",
            r"\b(cold|hot)\s+(streak|machine|run)\b",
            r"\bchances?\s+(are\s+)?(now\s+)?(higher|better|increased|improving)\b",
            r"\bafter\s+\w+\s+losses?,?\s+(a\s+)?win\b",
        ],
    },
    "self_serving_bias": {
        "definition": (
            "Wins attributed to skill or internal causes; losses attributed to "
            "luck, the machine, or other external causes."
        ),
        "patterns": [
            r"\b(bad|terrible|awful|rotten)\s+luck\b",
            r"\b(unlucky|unfortunate)\b",
            r"\bjust\s+(bad\s+)?luck\b",
            r"\b(machine|game|slot)\s+(is|seems|has\s+been)\s+(cold|tight|against|unkind)\b",
            r"\b(good|smart|right|solid)\s+(call|decision|choice|move|judgment)\b",
            r"\bmy\s+(judgment|instinct|read|discipline)\s+(was|is)\b",
            r"\bi\s+(played|chose|bet)\s+(well|smartly|wisely|correctly)\b",
        ],
    },
    "impaired_control": {
        "definition": (
            "Belief that one cannot stop, must continue, or must recoup what has "
            "been lost."
        ),
        "patterns": [
            r"\b(recover|recoup|regain|make\s+back|win\s+back|get\s+back)\b",
            r"\bback\s+to\s+(even|break\s*even|my\s+(initial|starting|original))\b",
            r"\b(can'?t|cannot|unable\s+to)\s+(stop|quit|walk\s+away)\b",
            r"\b(one|just)\s+(more|another)\s+(round|spin|bet|try|attempt)\b",
            r"\b(have|need|got)\s+to\s+(keep|continue|carry\s+on)\b",
            r"\bkeep\s+(playing|going|betting)\b",
            r"\bnot\s+(ready\s+to\s+)?(stop|quit)\s+(yet|now)\b",
        ],
    },
}

COMPILED = {
    name: [re.compile(p, re.IGNORECASE) for p in spec["patterns"]]
    for name, spec in CONSTRUCTS.items()
}

# Mapping to the four frames used in the submitted manuscript, so the original
# numbers and the updated numbers can be reported side by side.
SUBMITTED_FRAME_MAP = {
    "pattern_belief": ["illusion_of_control"],
    "probability_misestimation": ["gamblers_fallacy"],
    "loss_chasing": ["impaired_control"],
    "goal_escalation": [],           # no counterpart among the convergent four
    # self_serving_bias has no counterpart in the submitted instrument at all.
}


def matches(text: str, construct: str) -> bool:
    return any(rx.search(text or "") for rx in COMPILED[construct])


def any_construct(text: str) -> bool:
    return any(matches(text, c) for c in COMPILED)
