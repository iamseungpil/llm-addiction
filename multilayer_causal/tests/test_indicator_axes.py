"""Array-core axis builders: shape / unit-norm / AUC / behavioural deconfound.

Synthetic arrays only — sae_lens / HF are never touched (the CLI + load_task_arrays
paths that reference HF are exercised at run time, not here).
"""
import numpy as np

from multilayer_causal.src import indicator_axes as ia


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def test_readout_axis_smoke():
    d = ia.build_readout_axis_from_arrays(
        feats=np.random.RandomState(0).randn(200, 64),
        indicator=np.random.RandomState(1).randn(200),
        balance=np.random.RandomState(2).rand(200),
        rounds=np.random.RandomState(3).randint(1, 20, 200),
        groups=np.arange(200) // 4,
        decoder=np.random.RandomState(4).randn(64, 8),
        layers=[0, 1],
    )
    assert d["directions"].shape == (2, 8)
    assert np.allclose(np.linalg.norm(d["directions"], axis=1), 1.0, atol=1e-5)
    assert "auc" in d and "provenance" in d
    assert d["scales"].shape == (2,)


def test_behavioural_and_confound_axes():
    rs = np.random.RandomState(0)
    hidden = rs.randn(300, 2, 8)
    ind = rs.randn(300)
    bal = rs.rand(300)
    rnd = rs.randint(1, 20, 300)
    grp = np.arange(300) // 3
    b = ia.build_behavioural_axis_from_arrays(hidden, ind, bal, rnd, grp,
                                              layers=[0, 1], q=0.25)
    c = ia.build_confound_axis_from_arrays(hidden, bal, rnd, layers=[0, 1])
    assert b["directions"].shape == (2, 8) and c["directions"].shape == (2, 8)
    assert np.allclose(np.linalg.norm(b["directions"], axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(c["directions"], axis=1), 1, atol=1e-5)
    assert b["scales"].shape == (2,) and (b["scales"] > 0).all()


def test_behavioural_deconfound_recovers_signal_not_balance():
    """The behavioural axis must track the BALANCE-RESIDUAL indicator signal,
    not the balance-confounded component: with indicator = 2*balance + signal
    and hidden carrying signal along u and balance along v, the deconfounded
    top/bottom split must align the axis with u, NOT v."""
    rs = np.random.RandomState(7)
    N, d = 400, 16
    u = _unit(rs.randn(d))            # behaviour (residual-signal) direction
    v = _unit(rs.randn(d))            # balance (confound) direction
    balance = rs.rand(N)
    signal = rs.randn(N)              # independent of balance
    rounds = rs.randint(1, 25, N)
    indicator = 2.0 * balance + signal            # confounded target
    groups = np.arange(N) // 4
    hidden = np.zeros((N, 2, d))
    for l in range(2):
        hidden[:, l, :] = (signal[:, None] * u
                           + 3.0 * balance[:, None] * v
                           + 0.05 * rs.randn(N, d))
    b = ia.build_behavioural_axis_from_arrays(
        hidden, indicator, balance, rounds, groups, layers=[0, 1], q=0.25)
    axis = _unit(b["directions"][0])
    cos_u = abs(float(axis @ u))
    cos_v = abs(float(axis @ v))
    assert cos_u > 0.7, cos_u
    assert cos_u > cos_v, (cos_u, cos_v)


def _w3_task_data(rs, d=8, n=120):
    """Synthetic load_task_arrays-shaped dict (the fields _build_wave3_axes
    reads) with the I_BA signal along a task-specific direction."""
    u = _unit(rs.randn(d))
    ind = rs.randn(n)
    hidden = np.zeros((n, 2, d))
    for l in range(2):
        hidden[:, l, :] = ind[:, None] * u + 0.05 * rs.randn(n, d)
    return {"hidden": hidden, "indicators": {"i_ba": ind},
            "balance": rs.rand(n), "rounds": rs.randint(1, 20, n),
            "groups": np.arange(n) // 4}


def test_build_wave3_axes_per_task_plus_shared3():
    """Wave-3 cross-task build: one behavioural I_BA axis per task, the 3-task
    SVD-top1 shared3 axis (SM sigma-unit scales) plus its IC-scale twin
    (identical directions, IC sigma-units — the sh3_ic dose is norm-matched to
    the ic_own control/null band), and the pairwise cross-task cosines."""
    rs = np.random.RandomState(11)
    task_data = {t: _w3_task_data(rs) for t in ia.W3_TASKS}
    built, cos_pairs = ia._build_wave3_axes(task_data, layers=[0, 1])

    assert set(built) == set(ia.W3_TASKS) | {"shared3", "shared3_icscale",
                                             "shared3_mwscale"}
    for name, b in built.items():
        assert b["directions"].shape == (2, 8), name
        assert np.allclose(np.linalg.norm(b["directions"], axis=1), 1,
                           atol=1e-5), name
    # shared3 borrows the SM behavioural scales (matched sigma-units on SM)
    assert np.allclose(built["shared3"]["scales"],
                       built["slot_machine"]["scales"])
    assert np.isnan(built["shared3"]["auc"])
    # the icscale twin: SAME directions, IC behavioural scales
    assert np.array_equal(built["shared3_icscale"]["directions"],
                          built["shared3"]["directions"])
    assert np.allclose(built["shared3_icscale"]["scales"],
                       built["investment_choice"]["scales"])
    # reconnaissance: all 3 unordered task pairs, cosines in [-1, 1]
    assert len(cos_pairs) == 3
    assert all(-1.0 <= c <= 1.0 for c in cos_pairs.values())
    # shared3 is the common component: no farther from any task axis than the
    # least-aligned task pair is from each other
    for t in ia.W3_TASKS:
        cs = [abs(float(_unit(built["shared3"]["directions"][li])
                        @ _unit(built[t]["directions"][li])))
              for li in range(2)]
        assert min(cs) > 0.2, (t, cs)


def test_mw_rc_labels_spin_stop_none():
    """Verified MW catalog semantics: choice 2 WITH bet>0 = spin (1.0),
    choice 1 = stop (0.0), choice 2 without a bet = parse edge (NaN), and
    unjoinable rows (choice NaN) stay NaN."""
    choices = np.array([2.0, 1.0, 2.0, 2.0, np.nan, 1.0])
    bets = np.array([20.0, np.nan, np.nan, 0.0, 15.0, 0.0])
    rc = ia._mw_rc_labels(choices, bets)
    assert rc[0] == 1.0          # spin: choice 2, bet>0
    assert rc[1] == 0.0          # stop: choice 1 (no bet needed)
    assert np.isnan(rc[2])       # parse edge: choice 2, no bet
    assert np.isnan(rc[3])       # parse edge: choice 2, bet 0
    assert np.isnan(rc[4])       # unjoinable: no choice (bet alone never labels)
    assert rc[5] == 0.0          # stop with a stored 0 bet is still a stop


def test_keep_mask_default_frozen_rc_keep_superset():
    """The default keep rule must be BIT-IDENTICAL to the frozen Wave-1..5
    formula (split & isfinite(i_ba)); rc_keep only ADDS the finite-i_rc rows
    (stop rows), never removes or reorders anything."""
    rs = np.random.RandomState(5)
    n = 200
    split = rs.rand(n) > 0.3
    i_ba = rs.rand(n)
    i_ba[rs.rand(n) > 0.6] = np.nan          # stop / unjoinable rows
    i_rc = np.where(np.isfinite(i_ba), 1.0, np.nan)
    i_rc[np.isnan(i_ba) & (rs.rand(n) > 0.5)] = 0.0  # labelled stop rows
    frozen = split & np.isfinite(i_ba)
    default = ia._keep_mask(split, i_ba, i_rc)
    assert np.array_equal(default, frozen)
    rc = ia._keep_mask(split, i_ba, i_rc, rc_keep=True)
    # superset: every default row kept, extras are exactly the labelled
    # finite-i_rc rows inside the split
    assert (rc & default).sum() == default.sum()
    assert np.array_equal(rc, split & (np.isfinite(i_ba) | np.isfinite(i_rc)))
    assert rc.sum() > default.sum()  # the fixture has genuine stop rows


def test_keep_mask_nonregression_wave5_iba_build():
    """Loader-shaped non-regression: a wave5-style i_ba behavioural build on
    the default keep rows is byte-identical before/after the Wave-6 change,
    and the rc_keep rows subset by the iba_finite mask reproduce EXACTLY the
    default rows (so the Wave-6 mw_iba diagnostic sees the Wave-3/5 rows)."""
    rs = np.random.RandomState(9)
    n, d = 240, 8
    split = np.ones(n, dtype=bool)
    i_ba = rs.rand(n)
    stop = rs.rand(n) > 0.7
    i_ba[stop] = np.nan                      # stop rows: no bet
    i_rc = np.where(stop, 0.0, 1.0)          # every row choice-labelled
    hidden = rs.randn(n, 2, d)
    bal, rnd = rs.rand(n), rs.randint(1, 20, n).astype(np.float64)
    grp = np.arange(n) // 4

    frozen = split & np.isfinite(i_ba)       # pre-Wave-6 keep formula
    default = ia._keep_mask(split, i_ba, i_rc)
    build = lambda k: ia.build_behavioural_axis_from_arrays(
        hidden[k], np.nan_to_num(i_ba)[k], bal[k], rnd[k], grp[k], layers=[0, 1])
    old = build(frozen)
    new = build(default)
    assert np.array_equal(old["directions"], new["directions"])
    assert np.array_equal(old["scales"], new["scales"])

    rc = ia._keep_mask(split, i_ba, i_rc, rc_keep=True)
    iba_finite = np.isfinite(i_ba)[rc]       # the loader's iba_finite key
    assert np.array_equal(np.flatnonzero(rc)[iba_finite], np.flatnonzero(frozen))
    sub = ia.build_behavioural_axis_from_arrays(
        hidden[rc][iba_finite], np.nan_to_num(i_ba)[rc][iba_finite],
        bal[rc][iba_finite], rnd[rc][iba_finite], grp[rc][iba_finite],
        layers=[0, 1])
    assert np.array_equal(old["directions"], sub["directions"])


def _w6_task_data(rs, d=8, n=600, task="sm"):
    """Synthetic load_task_arrays-shaped dict for _build_wave6_axes: SM carries
    an i_ba signal; IC and MW carry a binary i_rc signal (MW with the rc_keep
    extras: NaN i_ba on the stop rows and the iba_finite mask)."""
    u = _unit(rs.randn(d))
    base = {"balance": rs.rand(n), "rounds": rs.randint(1, 20, n),
            "groups": np.arange(n) // 4}
    if task == "sm":
        ind = rs.randn(n)
        hidden = ind[:, None, None] * u[None, None, :] + 0.05 * rs.randn(n, 2, d)
        return {**base, "hidden": hidden,
                "indicators": {"i_ba": ind, "i_rc": np.full(n, np.nan)},
                "iba_finite": np.ones(n, dtype=bool)}
    rc = (rs.rand(n) > 0.4).astype(np.float64)
    rc[rs.rand(n) > 0.9] = np.nan            # a few unlabeled rows
    hidden = (np.nan_to_num(rc)[:, None, None] * u[None, None, :]
              + 0.05 * rs.randn(n, 2, d))
    iba = np.where(rc == 1.0, rs.rand(n), np.nan)  # spins carry a bet
    return {**base, "hidden": hidden,
            "indicators": {"i_ba": np.nan_to_num(iba), "i_rc": rc},
            "iba_finite": np.isfinite(iba)}


def test_build_wave6_axes_mw_rc_plus_shared3c():
    """Wave-6 build: mw_rc behavioural axis on the finite-i_rc MW subset plus
    shared3c = SVD-top1(sm_iba, ic_rc, mw_rc) in three sigma-unit variants
    (identical directions), and the six diagnostic/loading cosines."""
    rs = np.random.RandomState(13)
    task_data = {"slot_machine": _w6_task_data(rs, task="sm"),
                 "investment_choice": _w6_task_data(rs, task="ic"),
                 "mystery_wheel": _w6_task_data(rs, task="mw")}
    built, cos_pairs = ia._build_wave6_axes(task_data, layers=[0, 1])

    assert set(built) == {"mw_rc", "shared3c", "shared3c_icscale",
                          "shared3c_mwscale"}
    for name, b in built.items():
        assert b["directions"].shape == (2, 8), name
        assert np.allclose(np.linalg.norm(b["directions"], axis=1), 1,
                           atol=1e-5), name
    # the three shared3c variants share ONE direction stack; only scales vary
    assert np.array_equal(built["shared3c_icscale"]["directions"],
                          built["shared3c"]["directions"])
    assert np.array_equal(built["shared3c_mwscale"]["directions"],
                          built["shared3c"]["directions"])
    assert np.allclose(built["shared3c_mwscale"]["scales"],
                       built["mw_rc"]["scales"])
    assert np.isnan(built["shared3c"]["auc"])
    # W7 pre-registration numbers: 3 mw_rc diagnostics + 3 shared3c loadings
    assert set(cos_pairs) == {"mw_rc~mw_iba", "mw_rc~sm_iba", "mw_rc~ic_rc",
                              "shared3c~sm_iba", "shared3c~ic_rc",
                              "shared3c~mw_rc"}
    assert all(-1.0 <= c <= 1.0 for c in cos_pairs.values())
    # shared3c is the common component of its three inputs
    assert abs(cos_pairs["shared3c~mw_rc"]) > 0.2


def test_confound_axis_tracks_balance():
    """The confound axis must point along the balance direction it regresses on."""
    rs = np.random.RandomState(3)
    N, d = 300, 12
    v = _unit(rs.randn(d))
    balance = rs.rand(N)
    rounds = rs.randint(1, 20, N)
    hidden = np.zeros((N, 1, d))
    hidden[:, 0, :] = 4.0 * balance[:, None] * v + 0.05 * rs.randn(N, d)
    c = ia.build_confound_axis_from_arrays(hidden, balance, rounds, layers=[0])
    assert abs(float(_unit(c["directions"][0]) @ v)) > 0.9
