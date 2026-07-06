"""Thin per-arm Weights & Biases logger for the §4 causal wave (sec4).

Deliberately minimal and crash-proof: the node job must never die because
W&B is missing, unauthenticated, or rate-limited. Two entry points —
``init_run`` (once per job) and ``log_arm`` (once per arm at DONE). Both are
hard NO-OPs when ``WANDB_DISABLED`` is set in the environment or when the
``wandb`` package is not importable, so the runner can call them
unconditionally on every wave (sm/ic/mw/sec4) without a guard.
"""
from __future__ import annotations

import os

_RUN = None  # module-level handle so log_arm can find the active run


def _disabled() -> bool:
    return bool(os.environ.get("WANDB_DISABLED"))


def init_run(project: str, name: str):
    """Start a W&B run (idempotent per process). Returns the run handle or
    None when logging is disabled / wandb is unavailable — never raises."""
    global _RUN
    if _disabled():
        return None
    try:
        import wandb
    except Exception:
        return None
    try:
        if _RUN is None:
            _RUN = wandb.init(project=project, name=name,
                              reinit=False, resume="allow")
    except Exception:
        _RUN = None
    return _RUN


def log_arm(arm_id: str, dose, metrics: dict) -> None:
    """Log one arm's summary at DONE. ``dose`` is the steering alpha (or None);
    ``metrics`` is a flat {name: number} dict. NO-OP when disabled/unavailable;
    swallows every error so a logging hiccup never kills the trial loop."""
    if _disabled():
        return
    try:
        import wandb
    except Exception:
        return
    try:
        run = _RUN if _RUN is not None else wandb.run
        if run is None:
            return
        payload = {"arm_id": arm_id, "dose": dose}
        payload.update({str(k): v for k, v in (metrics or {}).items()})
        run.log(payload)
    except Exception:
        pass
