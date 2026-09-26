"""Preflight gate audit (scripts/preflight_w3.py) against the REAL registry
and REAL assets/w3 npz files — the submission-path half of the GATE_FAILED
enforcement (the runner half lives in test_dryrun.py gate tests)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_w3.py"
spec = importlib.util.spec_from_file_location("preflight_w3", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
sys.modules["preflight_w3"] = preflight
spec.loader.exec_module(preflight)

from multilayer_causal.src.registry import load_arms  # noqa: E402


def _w3_rows():
    return {aid: preflight.audit_arm(a)
            for aid, a in load_arms().items() if aid.startswith("w3")}


def test_every_w3_arm_gets_a_verdict():
    rows = _w3_rows()
    assert rows, "no w3 arms in registry"
    for aid, (rid, _npz, status, fields) in rows.items():
        assert rid == aid
        assert status in ("SUBMIT", "EXCLUDE", "MISSING")
        assert isinstance(fields, dict)


def test_verdicts_mirror_recorded_gate_fields():
    """audit_arm must agree byte-for-byte with what the builder recorded:
    gate_passed False → EXCLUDE, True → SUBMIT, absent/random → SUBMIT."""
    for aid, arm in load_arms().items():
        if not aid.startswith("w3"):
            continue
        _rid, npz_rel, status, _fields = preflight.audit_arm(arm)
        if arm.get("mode") != "steer" or npz_rel is None:
            assert status == "SUBMIT"
            continue
        path = preflight.REPO_ROOT / npz_rel
        if not path.exists():
            assert status == "MISSING"
            continue
        z = np.load(path)
        if arm.get("direction") == "random" or "gate_passed" not in z.files:
            assert status == "SUBMIT", aid
        else:
            expected = "SUBMIT" if bool(z["gate_passed"]) else "EXCLUDE"
            assert status == expected, aid


def test_w3rnd_random_controls_are_exempt():
    rows = _w3_rows()
    rnd = {aid: r for aid, r in rows.items() if aid.startswith("w3rnd")}
    assert rnd
    for aid, (_rid, _npz, status, fields) in rnd.items():
        assert status == "SUBMIT", aid
        assert "random control" in fields.get("note", "")


def test_main_blocks_when_any_gate_failed_or_missing(capsys):
    """main() exit code = whether the printed submit list is safe to amlt-run.
    Derived from the live verdicts so the test stays true as assets evolve."""
    rows = _w3_rows()
    blocked = any(s in ("EXCLUDE", "MISSING") for _r, _n, s, _f in rows.values())
    rc = preflight.main(["--prefix", "w3"])
    out = capsys.readouterr().out
    assert rc == (1 if blocked else 0)
    assert "submit list" in out
    for aid, (_rid, _npz, status, _fields) in rows.items():
        if status in ("EXCLUDE", "MISSING"):
            assert f"{aid}[{status}]" in out


def test_unknown_prefix_asserts():
    with pytest.raises(AssertionError):
        preflight.main(["--prefix", "no_such_wave"])
