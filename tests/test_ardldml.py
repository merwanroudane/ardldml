"""Test suite for ardldml."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ardldml as ad
from ardldml import (
    DMLBounds,
    build_balanced_design,
    classical_bounds_test,
    hblock_folds,
    n_restrictions,
    pss_reference,
    sample_use,
    simulate_pss_bounds,
    statsmodels_offset,
)

SEED = 20260625


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_system(T=180, rho=0.5, d_ctl=12, delta=0.0, seed=1, burn=60):
    """Y = D + u with u AR(1); mixed I(0)/I(1) controls and a latent trend."""
    rng = np.random.default_rng(seed)
    n = T + burn
    trend = np.cumsum(rng.standard_normal(n))
    cols, integ = {}, []
    for j in range(d_ctl // 2):
        name = f"w1_{j}"
        cols[name] = np.cumsum(rng.standard_normal(n)) + 0.5 * trend
        integ.append(name)
    for j in range(d_ctl // 2):
        s = np.zeros(n)
        e = rng.standard_normal(n)
        for t in range(1, n):
            s[t] = 0.5 * s[t - 1] + e[t]
        cols[f"w0_{j}"] = s
    e = rng.standard_normal(n)
    xi = rng.standard_normal(n)
    v = delta * e + np.sqrt(max(1 - delta**2, 0.0)) * xi
    D = np.cumsum(v) + 0.3 * trend
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    idx = pd.RangeIndex(n)
    return (
        pd.Series(D + u, index=idx, name="Y").iloc[burn:],
        pd.Series(D, index=idx, name="D").iloc[burn:],
        pd.DataFrame(cols, index=idx).iloc[burn:],
        integ,
    )


@pytest.fixture(scope="module")
def system():
    return make_system()


# ---------------------------------------------------------------------------
# critical values: the simulator must reproduce print
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,case", [(1, 3), (4, 4), (2, 1), (4, 3)])
def test_simulator_reproduces_published_bounds(k, case):
    """Simulated bounds must match the published table within Monte Carlo error."""
    pub = pss_reference(k, case)
    sim = simulate_pss_bounds(k=k, case=case, T=1000, nsim=6000, seed=SEED)
    # The 10% and 5% quantiles settle fastest; the 1% tail needs far more draws.
    for level, tol in ((0.10, 0.25), (0.05, 0.35)):
        assert abs(sim.loc[level, "I(0)"] - pub.loc[level, "I(0)"]) < tol
        assert abs(sim.loc[level, "I(1)"] - pub.loc[level, "I(1)"]) < tol


def test_bounds_ordered():
    cv = simulate_pss_bounds(k=2, case=3, T=400, nsim=1500, seed=SEED)
    assert (cv["I(0)"] <= cv["I(1)"]).all()
    assert cv.loc[0.10, "I(1)"] <= cv.loc[0.01, "I(1)"]


def test_finite_sample_bounds_exceed_asymptotic():
    """Small-T bounds should be larger; this is Narayan's point."""
    small = simulate_pss_bounds(k=1, case=3, T=40, nsim=3000, seed=SEED)
    big = simulate_pss_bounds(k=1, case=3, T=1000, nsim=3000, seed=SEED)
    assert small.loc[0.05, "I(1)"] > big.loc[0.05, "I(1)"]


def test_n_restrictions():
    assert n_restrictions(1, 3) == 2
    assert n_restrictions(1, 4) == 3
    assert n_restrictions(4, 2) == 6
    with pytest.raises(ValueError):
        n_restrictions(1, 9)


def test_statsmodels_offset_is_real():
    """Document the off-by-one: the shifted row must differ from the correct one."""
    off = statsmodels_offset(k=1, case=3)
    correct, shifted = off.iloc[0], off.iloc[1]
    assert correct["I(1)"] > shifted["I(1)"]
    assert abs(correct["I(1)"] - 5.73) < 0.02


def test_simulate_pss_rejects_bad_input():
    with pytest.raises(ValueError):
        simulate_pss_bounds(k=1, case=7)
    with pytest.raises(ValueError):
        simulate_pss_bounds(k=1, case=3, T=3)


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
def test_eval_blocks_partition_the_sample():
    bs = hblock_folds(100, n_blocks=5, buffer=4)
    allidx = np.sort(np.concatenate(bs.eval_blocks))
    assert np.array_equal(allidx, np.arange(100))


def test_buffer_excludes_neighbours():
    bs = hblock_folds(100, n_blocks=4, buffer=5)
    for train, ev in bs:
        assert not set(train) & set(ev)
        assert min(abs(t - e) for t in train for e in (ev[0], ev[-1])) > 5 - 1


def test_sample_use_decreasing_in_buffer():
    assert sample_use(200, 5, 0) > sample_use(200, 5, 5) > sample_use(200, 5, 15)


def test_buffer_too_wide_raises():
    with pytest.raises(ValueError):
        hblock_folds(30, n_blocks=2, buffer=40)


# ---------------------------------------------------------------------------
# balanced design
# ---------------------------------------------------------------------------
def test_balance_differences_integrated_controls_only(system):
    y, d, W, integ = system
    des = build_balanced_design(y, d, W, lags=2, integrated=integ)
    for c in integ:
        assert f"D.{c}" in des.X.columns, "integrated controls must enter differenced"
        assert c not in des.X.columns, "integrated controls must not enter in levels"
    for c in W.columns:
        if c not in integ:
            assert c in des.X.columns, "stationary controls enter in levels"
    # The Z projection uses levels for everything.
    assert list(des.Wlev.columns) == list(W.columns)
    assert list(des.Z.columns) == ["Y.L1", "D.L1"]


def test_design_has_no_missing(system):
    y, d, W, integ = system
    des = build_balanced_design(y, d, W, lags=4, integrated=integ)
    assert not des.X.isna().any().any()
    assert not des.Z.isna().any().any()
    assert des.n == len(des.index)


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------
def test_recovers_true_long_run_coefficient(system):
    """Y = D + u, so theta should be near 1."""
    y, d, W, integ = system
    res = DMLBounds(y, d, W, lags=2, n_blocks=5, buffer=3, integrated=integ).fit()
    assert abs(res.theta - 1.0) < 0.25
    assert 0.0 < res.alpha < 1.0
    assert np.isfinite(res.stat) and res.stat > 0


def test_statistic_has_no_intercept_by_default(system):
    """Equation (10) is a no-intercept projection; the option must change the fit."""
    y, d, W, integ = system
    a = DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ).fit()
    b = DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ,
                  include_constant=True).fit()
    assert a.spec.include_constant is False
    assert a.stat != pytest.approx(b.stat)


def test_null_statistic_smaller_than_alternative():
    y0, d0, W0, i0 = make_system(rho=1.0, seed=3)
    y1, d1, W1, i1 = make_system(rho=0.5, seed=3)
    s0 = DMLBounds(y0, d0, W0, lags=2, n_blocks=5, buffer=3, integrated=i0).fit().stat
    s1 = DMLBounds(y1, d1, W1, lags=2, n_blocks=5, buffer=3, integrated=i1).fit().stat
    assert s1 > s0


def test_high_dimensional_is_estimable():
    """d > T must still return a statistic; the unpenalised ECM cannot."""
    y, d, W, integ = make_system(T=90, d_ctl=120, seed=5)
    res = DMLBounds(y, d, W, lags=1, n_blocks=4, buffer=2, integrated=integ).fit()
    assert np.isfinite(res.stat)


def test_rejects_non_dataframe_controls(system):
    y, d, W, _ = system
    with pytest.raises(TypeError):
        DMLBounds(y, d, W["w1_0"])


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_returns_valid_pvalue(system):
    y, d, W, integ = system
    res = (
        DMLBounds(y, d, W, lags=2, n_blocks=5, buffer=3, integrated=integ)
        .fit()
        .bootstrap(B=49, seed=SEED)
    )
    assert 0.0 <= res.pvalue <= 1.0
    assert np.isfinite(res.critical_value)
    assert res.boot["n_failed"] == 0
    assert len(res.boot["draws"]) == 49


def test_bootstrap_null_not_rejected():
    y, d, W, integ = make_system(rho=1.0, seed=11)
    res = (
        DMLBounds(y, d, W, lags=2, n_blocks=5, buffer=3, integrated=integ)
        .fit()
        .bootstrap(B=99, seed=SEED)
    )
    assert res.pvalue > 0.05, "a true null should not be rejected here"


def test_schemes_agree_under_exogeneity():
    """delta = 0: system and fixed-regressor schemes should broadly coincide."""
    y, d, W, integ = make_system(delta=0.0, seed=13)
    base = DMLBounds(y, d, W, lags=2, n_blocks=5, buffer=3, integrated=integ).fit()
    p_sys = base.bootstrap(B=99, seed=SEED, scheme="system").pvalue
    p_fix = base.bootstrap(B=99, seed=SEED, scheme="fixed").pvalue
    assert abs(p_sys - p_fix) < 0.30


def test_conditional_model_absorbs_endogeneity():
    """
    Under strong endogeneity the *residual* correlation is still small.

    This is the paper's own Section 7.4 explanation for why the fixed-regressor
    scheme does not visibly deteriorate: the restricted conditional model
    includes the contemporaneous dD term, so the resampled conditional error has
    already been orthogonalised against the regressor innovation before any
    resampling happens. The paper calls this a knife-edge property of the
    conditional specification rather than of the resampling scheme -- which is
    why the system form is the default anyway.
    """
    y, d, W, integ = make_system(delta=0.8, seed=17)
    res = (
        DMLBounds(y, d, W, lags=2, n_blocks=5, buffer=3, integrated=integ)
        .fit()
        .bootstrap(B=29, seed=SEED, scheme="system")
    )
    assert abs(res.boot["corr_eps_v"]) < 0.35, (
        "the contemporaneous dD term should have absorbed most of the correlation"
    )
    assert 0.0 <= res.pvalue <= 1.0
    assert np.isfinite(res.critical_value)


def test_system_scheme_valid_under_endogeneity():
    """A true null must not be rejected when the focal regressor is endogenous."""
    y, d, W, integ = make_system(rho=1.0, delta=0.8, seed=19)
    res = (
        DMLBounds(y, d, W, lags=2, n_blocks=5, buffer=3, integrated=integ)
        .fit()
        .bootstrap(B=99, seed=SEED, scheme="system")
    )
    assert res.pvalue > 0.05


def test_bad_scheme_raises(system):
    y, d, W, integ = system
    res = DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ).fit()
    with pytest.raises(ValueError):
        res.bootstrap(B=5, scheme="nonsense")


def test_summary_warns_before_bootstrap(system):
    y, d, W, integ = system
    res = DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ).fit()
    assert res.pvalue is None
    assert "no bootstrap" in res.summary().lower()
    assert res.decision() == "no bootstrap run"


# ---------------------------------------------------------------------------
# classical benchmark
# ---------------------------------------------------------------------------
def test_classical_three_steps(system):
    y, d, W, _ = system
    res = classical_bounds_test(y, d.to_frame(), lags=2, order=2, case=3)
    assert res.k == 1
    assert np.isfinite(res.f_stat) and res.f_stat > 0
    assert res.t_stat < 0
    assert 0.0 <= res.theta_pvalue <= 1.0
    assert abs(res.long_run["D"] - 1.0) < 0.3


def test_classical_bounds_are_generated(system):
    y, d, W, _ = system
    res = classical_bounds_test(y, d.to_frame(), lags=2, order=2, case=3)
    cv = res.bounds(T=500, nsim=1200, seed=SEED)
    assert list(cv.columns) == ["I(0)", "I(1)"]
    assert (cv["I(0)"] <= cv["I(1)"]).all()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "regime,n", [("1973-1985", 156), ("1986-1998", 156), ("1999-2007", 108), ("2008-2020", 156)]
)
def test_regime_sample_sizes_match_paper(regime, n):
    assert len(ad.load_passthrough(regime=regime)) == n


def test_bundled_data_complete():
    df = ad.load_passthrough()
    assert not df.isna().any().any()
    assert list(df.columns) == ["cpi", "neer", "m2", "ffr", "ip", "unrate", "oil", "gs10", "baa"]
    assert str(df.index.min().date()) == "1973-01-01"


def test_log_transform_applied():
    raw = ad.load_passthrough(log=False)
    log = ad.load_passthrough(log=True)
    assert np.allclose(np.log(raw["cpi"]), log["cpi"])
    assert np.allclose(raw["ffr"], log["ffr"]), "rates must stay in levels"


def test_bad_regime_raises():
    with pytest.raises(ValueError):
        ad.load_passthrough(regime="1066-1067")


# ---------------------------------------------------------------------------
# simulation harness
# ---------------------------------------------------------------------------
def test_simulate_design_shapes():
    y, d, W, integ = ad.simulate_design("C", T=120, d=20, seed=SEED)
    assert len(y) == len(d) == len(W) == 120
    assert W.shape[1] == 20
    assert len(integ) == 10


def test_design_a_is_stationary_only():
    _, _, _, integ = ad.simulate_design("A", T=100, seed=SEED)
    assert integ == []


def test_bad_design_raises():
    with pytest.raises(ValueError):
        ad.simulate_design("Z", T=100)


# ---------------------------------------------------------------------------
# smoke: plots and tables
# ---------------------------------------------------------------------------
def test_plots_and_tables_run(system):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    y, d, W, integ = system
    res = (
        DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ)
        .fit()
        .bootstrap(B=29, seed=SEED)
    )
    assert ad.plot_bracket(k=8, k_tilde=4) is not None
    assert ad.plot_bootstrap_null(res) is not None
    assert ad.plot_block_structure(res.first_stage.folds) is not None

    tab = ad.result_table(res, labels=["(1)"])
    assert "F" in tab.index
    tex = ad.to_latex(tab, caption="cap", label="tab:x", notes="a note")
    for token in ("\\begin{table}", "\\caption{cap}", "\\label{tab:x}",
                  "\\toprule", "\\bottomrule", "a note", "\\end{table}"):
        assert token in tex


def test_plot_null_requires_bootstrap(system):
    pytest.importorskip("matplotlib")
    y, d, W, integ = system
    res = DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ).fit()
    with pytest.raises(ValueError):
        ad.plot_bootstrap_null(res)


def test_stars():
    assert ad.stars(0.004) == "***"
    assert ad.stars(0.03) == "**"
    assert ad.stars(0.08) == "*"
    assert ad.stars(0.5) == ""
    assert ad.stars(None) == ""
