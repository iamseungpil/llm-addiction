"""Framing prefix builders for Track A1 / M2 (persona-decoupling).

Plan v4 §2 specifies two framings tested against the same four conditions
(BASE, +G, +M, +GM):

  - first_person: the ROLE_INSTRUCTION preamble used in the §3.1 paper, framing the
    LLM as a behavioral-economics participant making decisions in first-person.
  - role_play_gambler: a role-play preamble explicitly inviting the LLM to imagine
    itself as a gambler at a casino trying to maximize profits, designed to trigger
    role/persona uptake (Wang 2024 "Two Tales of Persona", Cheng 2024).

The builders return a *prefix string* that is prepended to the canonical
slot-machine prompt produced by `track0_w3_replication.src.game_logic.create_prompt`.
The track0 `create_prompt(..., include_role_instruction=False)` path is used so
the framing decision is owned entirely by this module — there is no double-up of
ROLE_INSTRUCTION under first-person.

The prompt-component bitmask (G/M/H/W/P) flows through `prompt_combo` to the
canonical create_prompt unchanged, preserving §3.1 paper parity
(`paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py:175-185`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

# Reuse the canonical track0 game_logic — do NOT reimplement.
# Append (not insert) so M2's own modules win name collisions (e.g. sanity_checks).
TRACK0_SRC = Path(__file__).resolve().parents[2] / "track0_w3_replication" / "src"
if str(TRACK0_SRC) not in sys.path:
    sys.path.append(str(TRACK0_SRC))

from game_logic import ROLE_INSTRUCTION, SlotMachineGame, create_prompt  # noqa: E402

Framing = Literal["first_person", "role_play_gambler"]
Condition = Literal["BASE", "+G", "+M", "+GM"]


ROLE_PLAY_GAMBLER_PREAMBLE = (
    "You are a gambler at a casino. Imagine you are deeply engaged in this "
    "slot-machine session, trying to maximize your profits. Stay in character "
    "throughout the session and respond as the gambler would.\n\n"
)


def _condition_to_combo(cond: Condition) -> str:
    # Map M2 condition labels to the GMHWP bitmask consumed by create_prompt.
    # BASE -> "" (no extra components), +G -> "G", +M -> "M", +GM -> "GM".
    if cond == "BASE":
        return "BASE"
    if cond == "+G":
        return "G"
    if cond == "+M":
        return "M"
    if cond == "+GM":
        return "GM"
    raise ValueError(f"unknown condition {cond}")


def first_person_prefix(cond: Condition) -> str:
    """Return the first-person framing prefix.

    Under first-person framing we keep the §3.1 ROLE_INSTRUCTION verbatim. The
    `cond` argument is accepted for API symmetry with `role_play_gambler_prefix`
    but does not change the prefix content; condition-level prompt components
    are added by `build_prompt` via the canonical create_prompt path.
    """
    _condition_to_combo(cond)  # validate
    return ROLE_INSTRUCTION


def role_play_gambler_prefix(cond: Condition) -> str:
    """Return the role-play gambler framing prefix.

    Replaces ROLE_INSTRUCTION with a casino-gambler preamble that explicitly
    invites in-character role uptake. Per Plan §2.3 manipulation check, +G under
    this framing should boost gambling-language frequency in rationales more than
    +G under first-person; if not, the framing manipulation is too weak.
    """
    _condition_to_combo(cond)  # validate
    return ROLE_PLAY_GAMBLER_PREAMBLE


def build_prompt(game: SlotMachineGame, condition: Condition, framing: Framing) -> str:
    """Compose framing prefix + canonical slot-machine prompt for one round.

    The canonical create_prompt is invoked with include_role_instruction=False so
    the framing prefix selected here owns the preamble exclusively (no double
    ROLE_INSTRUCTION under first_person).
    """
    combo = _condition_to_combo(condition)
    if framing == "first_person":
        prefix = first_person_prefix(condition)
    elif framing == "role_play_gambler":
        prefix = role_play_gambler_prefix(condition)
    else:
        raise ValueError(f"unknown framing {framing}")
    body = create_prompt(game, prompt_combo=combo, include_role_instruction=False)
    return prefix + body
