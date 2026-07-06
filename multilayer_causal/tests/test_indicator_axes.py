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
