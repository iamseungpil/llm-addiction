"""Run manifest + output-isolation helpers shared by the Track 0 and E7 runners.

Audit follow-up (2026-07-26, DEVIATIONS.md D5). Three things every confirmatory
cell must carry so a result can be re-derived from the artifact alone:

1. **Provenance** — which commit, which literal bytes of the parity-critical code
   (`game_logic.py` + the legacy `improved_gpt_parsing.py`), which argv, which
   model/vendor, which seeds, when it ran. E7 PREREGISTRATION.md §8.4 requires a
   manifest; the matched-cap re-run adopts the same contract.
2. **API-degradation accounting** — the provider wrappers fall back to the literal
   string "Final Decision: Stop" after `MAX_API_ATTEMPTS` failures. That fallback is
   indistinguishable from a genuine stop in the stored transcript, so it must be
   counted. `note_fallback()` is the single place that emits it.
3. **Output isolation** — a re-run must never interleave its JSON with an earlier
   run's JSON in the same directory. `guard_output_collision` aborts before the
   first API call rather than after.

The module deliberately has no third-party imports so it can be imported from any
of the runners (API, open-weight, E7) without pulling extra dependencies.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent

# Fallback locations, used only if the module object is not importable at manifest
# time (it always is in the runners, which import `game_logic` at module load).
_DEFAULT_GAME_LOGIC = HERE / "game_logic.py"
_DEFAULT_PARSER = (
    HERE.parent.parent / "sm_cap_ablation" / "src" / "improved_gpt_parsing.py"
)

# --------------------------------------------------------------------------- fallback

# The literal text every provider wrapper substitutes when the API never returns a
# usable completion. Keep it in one place so the counter cannot drift from the string.
FALLBACK_RESPONSE = "Final Decision: Stop"

_FALLBACK_COUNT = {"n": 0, "run": 0}

# A fallback is written only after the provider wrapper has exhausted its retries, so a run
# of them back to back is not bad luck: it is the vendor refusing every call, which is what a
# depleted prepayment balance looks like. On 2026-07-28 that state produced five cells whose
# fallbacks sat in one contiguous block of games (34 of 344 decisions in the worst), each of
# them silently recorded as the model choosing to stop. They had to be discarded, and the hours
# spent producing them were wasted because nothing stopped the runner.
#
# Abort the cell instead. A cell that dies loudly can be relaunched; a cell that finishes with
# manufactured stops has to be caught downstream, and biases ruin downward if it is not.
_MAX_CONSECUTIVE_FALLBACKS = int(os.getenv("TRACK0_MAX_CONSECUTIVE_FALLBACKS", "5"))


class VendorUnavailable(RuntimeError):
    """Raised when the provider has failed every retry for several decisions in a row."""


def note_fallback() -> str:
    """Record one API/generation fallback and return the substituted response text."""
    _FALLBACK_COUNT["n"] += 1
    _FALLBACK_COUNT["run"] += 1
    if 0 < _MAX_CONSECUTIVE_FALLBACKS <= _FALLBACK_COUNT["run"]:
        raise VendorUnavailable(
            f"{_FALLBACK_COUNT['run']} consecutive API fallbacks after full retry budget; "
            f"the vendor is not serving this key (check billing/quota). Aborting the cell "
            f"rather than writing substituted stops. Set TRACK0_MAX_CONSECUTIVE_FALLBACKS=0 "
            f"to disable this guard."
        )
    return FALLBACK_RESPONSE


def note_api_success() -> None:
    """Clear the consecutive-failure run. Called by each provider wrapper on a real completion."""
    _FALLBACK_COUNT["run"] = 0


def fallback_count() -> int:
    return _FALLBACK_COUNT["n"]


def reset_fallback_count() -> None:
    _FALLBACK_COUNT["n"] = 0


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _module_path(module_name: str, default: Path) -> Path:
    """Hash the file that was actually imported, not a path we hope is the same one."""
    mod = sys.modules.get(module_name)
    file_attr = getattr(mod, "__file__", None) if mod is not None else None
    return Path(file_attr).resolve() if file_attr else default


def code_hashes() -> Dict[str, Dict[str, Optional[str]]]:
    """sha256 of the two parity-critical sources (Plan v5.2 §8)."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for key, module_name, default in (
        ("game_logic.py", "game_logic", _DEFAULT_GAME_LOGIC),
        ("improved_gpt_parsing.py", "improved_gpt_parsing", _DEFAULT_PARSER),
    ):
        path = _module_path(module_name, default)
        out[key] = {"path": str(path), "sha256": sha256_file(path)}
    return out


# --------------------------------------------------------------------------- git


def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def _find_repo_root(start: Path) -> Optional[Path]:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def git_info(start: Optional[Path] = None) -> Dict[str, Optional[object]]:
    repo = _find_repo_root((start or HERE).resolve())
    if repo is None:
        return {"repo": None, "commit": None, "branch": None, "dirty": None}
    status = _git(repo, "status", "--porcelain")
    return {
        "repo": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


# --------------------------------------------------------------------------- manifest


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def build_manifest(
    *,
    runner: str,
    model_id: str,
    vendor: str,
    seed_base: int,
    seeds: Sequence[int],
    started_at: str,
    finished_at: Optional[str] = None,
    argv: Optional[Sequence[str]] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Assemble the run manifest (E7 PREREGISTRATION.md §8.4 / DEVIATIONS.md D5).

    `seeds` is the *actual* per-game seed list, not a formula, so a replay does not
    have to trust that the seed rule in the code was the seed rule at run time.
    `api_fallback_responses` counts how many decisions were served by the substituted
    "Final Decision: Stop" text instead of a real completion.
    """
    argv_list: List[str] = list(argv if argv is not None else sys.argv)
    seeds_list = [int(s) for s in seeds]
    manifest: Dict = {
        "runner": runner,
        "git": git_info(),
        "code_sha256": code_hashes(),
        "argv": argv_list,
        "command": shlex.join([sys.executable, *argv_list]),
        "cwd": os.getcwd(),
        "model_id": model_id,
        "vendor": vendor,
        "seed_base": seed_base,
        "seeds": seeds_list,
        "n_seeds": len(seeds_list),
        "started_at": started_at,
        "finished_at": finished_at if finished_at is not None else now_iso(),
        "api_fallback_responses": fallback_count(),
        "api_fallback_text": FALLBACK_RESPONSE,
        "env": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


# --------------------------------------------------------------------------- guard


class OutputCollision(RuntimeError):
    """Raised when a cell would write into a directory that already holds that cell."""


def guard_output_collision(
    out_dir: Path,
    patterns: Iterable[str],
    *,
    cell: str,
    allow_existing: bool = False,
) -> List[Path]:
    """Abort if `out_dir` already contains output for this cell.

    The re-run must not interleave with the pre-fix artifacts (DEVIATIONS.md D5):
    the analysis globs by cell name, so one stale file silently doubles a cell and
    mixes $10-executed fixed games into $cap-executed ones. Called before any model
    is loaded or any API call is made, so a collision costs nothing.

    Returns the colliding paths (empty when clean). `allow_existing=True` downgrades
    the abort to a loud warning and must be passed explicitly on the command line.
    """
    out_dir = Path(out_dir)
    existing: List[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(out_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                existing.append(path)
    if not existing:
        return []
    listing = "\n".join(f"    {p.name}" for p in existing[:10])
    more = "" if len(existing) <= 10 else f"\n    ... and {len(existing) - 10} more"
    message = (
        f"output isolation guard: {out_dir} already contains {len(existing)} file(s) "
        f"for cell {cell!r}:\n{listing}{more}\n"
        "Refusing to run so the re-run cannot be mixed with the existing artifacts. "
        "Point --output_dir at a fresh directory (or move the old files aside). "
        "Pass --allow_existing_cell only if mixing is genuinely intended."
    )
    if allow_existing:
        print(f"[guard] WARNING (--allow_existing_cell): {message}", file=sys.stderr)
        return existing
    raise OutputCollision(message)
