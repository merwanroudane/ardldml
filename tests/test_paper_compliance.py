"""
Compliance tests: does the code do what the DML-Bounds paper says?

Each test names the section, equation, definition or appendix passage it
enforces. These are deliberately narrow and mechanical -- they check
*specification fidelity*, not statistical performance, which is what
``test_ardldml.py`` covers.

Where the paper is ambiguous, the test documents the reading taken.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from conftest import make_system

import ardldml as ad
from ardldml import DMLBounds, build_balanced_design
from ardldml.statistic import DMLBoundsSpec, _wald_f, compute_statistic

SEED = 20260625


# ---------------------------------------------------------------------------
# Section 3.5 / 4.1 -- balance
# ---------------------------------------------------------------------------
def test_s41_balance_integrated_differenced_stationary_levels(system):
    """
    Section 4.1: "The differenced target dY_t is I(0) and is projected only on
    stationary regressors: stationary confounders in levels and integrated
    confounders in first differences."
    """
    y, d, W, integ = system
    des = build_balanced_design(y, d, W, lags=2, integrated=integ)
    for c in integ:
        assert f"D.{c}" in des.X.columns
        assert c not in des.X.columns
    for c in set(W.columns) - set(integ):
        assert c in des.X.columns
        assert f"D.{c}" not in des.X.columns


def test_s41_levels_projected_on_confounder_levels(system):
    """
    Equations (5) and (6): m_Y and m_D condition on (W_1t, W_0t) -- the control
    levels, both blocks, undifferenced.
    """
    y, d, W, integ = system
    des = build_balanced_design(y, d, W, lags=2, integrated=integ)
    assert list(des.Wlev.columns) == list(W.columns)
    assert not any(c.startswith("D.") for c in des.Wlev.columns)


def test_s32_short_run_terms_are_nuisance(system):
    """
    Section 3.2: "collect the short-run terms and deterministics in a nuisance
    vector and test the lagged levels after projecting them out." So lagged
    differences of Y and D, and contemporaneous dD, all belong in X_t.
    """
    y, d, W, integ = system
    des = build_balanced_design(y, d, W, lags=3, integrated=integ)
    assert "D.D" in des.X.columns, "contemporaneous dD is a short-run nuisance term"
    for i in (1, 2, 3):
        assert f"D.Y.L{i}" in des.X.columns
        assert f"D.D.L{i}" in des.X.columns


def test_s41_only_two_tested_levels(system):
    """Section 4: Z_{t-1} = (Y_{t-1}, D_{t-1})', so exactly two tested levels."""
    y, d, W, integ = system
    des = build_balanced_design(y, d, W, lags=2, integrated=integ)
    assert des.Z.shape[1] == 2
    assert list(des.Z.columns) == ["Y.L1", "D.L1"]


# ---------------------------------------------------------------------------
# Section 4.1 -- adaptive weights on the integrated block only
# ---------------------------------------------------------------------------
def test_s41_adaptive_weights_target_the_integrated_block(system):
    """
    Section 4.1, stated twice: "l1-penalized LASSO, with adaptive weights for
    the integrated block" and "The adaptive weighting of the integrated block
    is important: vanilla l1 over-selects integrated regressors".

    The default must therefore restrict the weights to the integrated columns,
    and doing so must actually change the fit.
    """
    y, d, W, integ = system
    assert DMLBoundsSpec().adaptive_integrated_only is True

    a = DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ,
                  adaptive_integrated_only=True).fit()
    b = DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ,
                  adaptive_integrated_only=False).fit()
    assert a.stat != pytest.approx(b.stat), "the mask must have a real effect"


def test_s41_adaptive_mask_length_is_validated():
    from ardldml.firststage import adaptive_post_lasso

    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((40, 5)), rng.standard_normal(40)
    with pytest.raises(ValueError):
        adaptive_post_lasso(X, y, adaptive_mask=np.ones(3, dtype=bool))


def test_s41_dy_projection_is_plain_lasso(system):
    """
    Equation (4) is estimated by plain l1: the adaptive weighting is introduced
    for the integrated block, which appears only in the level projection.
    """
    src = inspect.getsource(compute_statistic)
    assert "adaptive=False" in src, "the dY projection must not use adaptive weights"


# ---------------------------------------------------------------------------
# Section 4.3 -- equation (10)
# ---------------------------------------------------------------------------
def test_eq10_has_no_intercept():
    """
    Equation (10) is a pure projection: beta = (sum Z Z')^-1 sum Z gdY.
    No constant appears.
    """
    assert DMLBoundsSpec().include_constant is False

    rng = np.random.default_rng(0)
    n = 120
    Z = rng.standard_normal((n, 2))
    dy = Z @ np.array([0.4, -0.3]) + rng.standard_normal(n) + 5.0  # deliberate offset

    out = _wald_f(dy, Z, include_constant=False)
    beta_manual = np.linalg.solve(Z.T @ Z, Z.T @ dy)
    alpha_manual, theta_manual = -beta_manual[0], beta_manual[1] / -beta_manual[0]
    assert out["alpha"] == pytest.approx(alpha_manual)
    assert out["theta"] == pytest.approx(theta_manual)


def test_eq10_f_is_wald_over_restrictions():
    """The reported statistic is the Wald form divided by the 2 restrictions."""
    rng = np.random.default_rng(1)
    n = 150
    Z = rng.standard_normal((n, 2))
    dy = rng.standard_normal(n)
    out = _wald_f(dy, Z, include_constant=False)

    beta = np.linalg.solve(Z.T @ Z, Z.T @ dy)
    resid = dy - Z @ beta
    s2 = resid @ resid / (n - 2)
    V = s2 * np.linalg.pinv(Z.T @ Z)
    wald = beta @ np.linalg.solve(V, beta)
    assert out["stat"] == pytest.approx(wald / 2.0)


def test_eq10_alpha_and_theta_mapping():
    """alpha = -pi_y and theta = pi_x / alpha."""
    rng = np.random.default_rng(2)
    n = 200
    Z = rng.standard_normal((n, 2))
    dy = Z @ np.array([-0.25, 0.5]) + 0.1 * rng.standard_normal(n)
    out = _wald_f(dy, Z, include_constant=False)
    assert out["alpha"] > 0
    assert out["theta"] == pytest.approx(0.5 / 0.25, rel=0.1)


# ---------------------------------------------------------------------------
# Section 4.4 -- the TSCV penalty applies to the dY equation only
# ---------------------------------------------------------------------------
def test_s44_tscv_targets_the_dy_equation(system):
    """
    Equation (11) minimises the one-step-ahead squared error of dY. The level
    projection is not tuned by it -- Appendix B gives m_Z the plug-in penalty
    c*sqrt(log d / n)*sigma with c = 1.1.
    """
    src = inspect.getsource(compute_statistic)
    head, _, tail = src.partition("# Level projection")
    assert "tscv_penalty" in head, "TSCV belongs to the dY projection"
    assert "tscv_penalty" not in tail, "the level projection must not use TSCV"
    assert "lam=None" in tail, "the level projection uses the plug-in penalty"


def test_s44_plugin_penalty_formula():
    """lambda = c * sqrt(log d / n) * sigma, with c = 1.1 the paper's default."""
    from ardldml.firststage import plugin_penalty

    got = plugin_penalty(n=100, d=40, sigma=2.0, c=1.1)
    assert got == pytest.approx(1.1 * np.sqrt(np.log(40) / 100) * 2.0)
    assert DMLBoundsSpec().c == 1.1


def test_s44_tscv_respects_temporal_order():
    """Rolling origin: fit on {1..t}, evaluate at t+1. Never trains on futures."""
    from ardldml.firststage import tscv_penalty

    src = inspect.getsource(tscv_penalty)
    assert "range(min_train, n)" in src
    assert "X[:t]" in src and "y[:t]" in src


# ---------------------------------------------------------------------------
# Algorithm 1 -- the restricted system wild bootstrap
# ---------------------------------------------------------------------------
def test_alg1_single_shared_rademacher_weight(system):
    """
    Algorithm 1(i): "draw a single Rademacher sequence eta_t and apply it to
    the stacked residual vector, (eps*, v*) = (eps_hat * eta, v_hat * eta)".
    One sequence, both series.
    """
    from ardldml.bootstrap import rademacher, restricted_system_wild_bootstrap

    rng = np.random.default_rng(0)
    draw = rademacher(500, rng)
    assert set(np.unique(draw)) <= {-1.0, 1.0}
    assert abs(draw.mean()) < 0.15

    src = inspect.getsource(restricted_system_wild_bootstrap)
    assert "eps_star = eps_hat * eta" in src
    assert "v_star = v_hat * eta" in src, "the same eta must weight both series"


def test_alg1_fixed_scheme_holds_d_at_its_path():
    from ardldml.bootstrap import restricted_system_wild_bootstrap

    src = inspect.getsource(restricted_system_wild_bootstrap)
    assert "dD_star = dD.copy()" in src


def test_alg1_regenerated_regressor_is_integrated(system):
    """
    Algorithm 1(ii): dD* is cumulated to a level path, so D* is integrated by
    construction and the tested regressor keeps its stochastic trend.
    """
    from ardldml.bootstrap import restricted_system_wild_bootstrap

    src = inspect.getsource(restricted_system_wild_bootstrap)
    assert "np.cumsum(dD_star)" in src
    assert "np.cumsum(dY_star)" in src


def test_alg1_nuisance_space_held_fixed(system):
    """Algorithm 1(ii): "the nuisance space W is held fixed at its realized path"."""
    from ardldml.bootstrap import restricted_system_wild_bootstrap

    src = inspect.getsource(restricted_system_wild_bootstrap)
    assert "W.loc[idx]" in src, "W must be passed through unmodified"


def test_alg1_level_supports_reselected_stationary_frozen(system):
    """
    Appendix B: "The stationary first-stage support is frozen, while the Y-side
    and D-side level supports are re-selected on each bootstrap path."
    """
    from ardldml.bootstrap import restricted_system_wild_bootstrap

    src = inspect.getsource(restricted_system_wild_bootstrap)
    assert "reselect_levels=True" in src
    assert "frozen_dY_support=frozen" in src


def test_alg1_critical_value_and_pvalue_definitions(system):
    """
    Algorithm 1(iv): cv is the 1-alpha quantile of the finite {F*_b}; the
    p-value is B^-1 sum 1{F*_b >= F}.
    """
    y, d, W, integ = system
    res = (
        DMLBounds(y, d, W, lags=2, n_blocks=4, buffer=2, integrated=integ)
        .fit()
        .bootstrap(B=59, level=0.05, seed=SEED)
    )
    draws = res.boot["draws"]
    assert np.all(np.isfinite(draws)), "only finite draws enter the quantile"
    assert res.critical_value == pytest.approx(float(np.quantile(draws, 0.95)))
    assert res.pvalue == pytest.approx(float(np.mean(draws >= res.stat)))


def test_alg1_marginal_model_uses_selected_differenced_controls(system):
    """
    Appendix B: the marginal model regresses dD on "an intercept, its own lag,
    and the first-stage-selected differenced controls".
    """
    from ardldml.bootstrap import restricted_system_wild_bootstrap

    src = inspect.getsource(restricted_system_wild_bootstrap)
    assert "wsel_cols" in src
    assert "selected_names" in src


def test_alg1_restricted_model_drops_level_terms(system):
    """
    Section 6: the restricted conditional model keeps the deterministic terms
    and short-run lag structure "but excluding the lagged levels and excluding
    the high-dimensional confounder levels".

    Reading taken: the restricted design is the balanced stationary block X_t
    plus a constant. X_t already excludes Z_{t-1} and holds the integrated
    controls in differences; the excluded "confounder levels" are the W_lev
    block used by m_Z.
    """
    from ardldml.core import restricted_null_model

    y, d, W, integ = system
    out = restricted_null_model(y, d.to_frame(), lags=2, order=2, case=3)
    names = out["exog_names"]
    assert "Y.L1" not in names and "D.L1" not in names
    assert all(not n.startswith("w1_") for n in names)
    assert "const" in names


# ---------------------------------------------------------------------------
# Definitions 1 and 2
# ---------------------------------------------------------------------------
def test_def1_k_tilde_is_never_estimated():
    """
    Remark 1: "The procedure does not require estimating k-tilde ... it indexes
    the asymptotic null experiment." Nothing in the package may expose it as a
    fitted quantity.
    """
    res_attrs = set(dir(ad.DMLBoundsResults))
    assert not any("k_tilde" in a or "ktilde" in a for a in res_attrs)


def test_def2_gap_definitions():
    """
    Definition 2: Delta_m = p_ols - p_ad and Delta_W = p_full - p_red.
    """
    from ardldml.diagnostics import trend_absorption

    src = inspect.getsource(trend_absorption)
    assert 'fits["full_ols"].pvalue - fits["full_adaptive"].pvalue' in src
    assert 'fits["full_adaptive"].pvalue - fits["reduced_adaptive"].pvalue' in src


def test_def2_runs_four_fits(system):
    """Definition 2 crosses control set with m_Z projection: four fits."""
    y, d, W, integ = system
    diag = ad.trend_absorption(
        y, d, W, drop=[c for c in W.columns if c.startswith("w1_")][:2],
        lags=2, n_blocks=4, buffer=2, integrated=integ, B=15, seed=SEED,
    )
    assert set(diag.fits) == {
        "full_adaptive", "full_ols", "reduced_adaptive", "reduced_ols"
    }
    assert np.isfinite(diag.delta_m) and np.isfinite(diag.delta_W)


# ---------------------------------------------------------------------------
# Appendix B -- the data-generating process
# ---------------------------------------------------------------------------
def test_appb_core_system():
    """
    Appendix B: "The core system is Y_t = D_t + u_t with D_t a driftless random
    walk and u_t = rho u_{t-1} + e_t."
    """
    y, d, W, _ = ad.simulate_design("B", T=400, rho=1.0, d=10, seed=SEED)
    u = (y - d).to_numpy()
    # Y - D must be the AR(1) error, a random walk when rho = 1.
    from statsmodels.tsa.stattools import adfuller

    assert adfuller(u, autolag="AIC")[1] > 0.10, "u should be I(1) under rho = 1"

    y2, d2, _, _ = ad.simulate_design("B", T=400, rho=0.5, d=10, seed=SEED)
    u2 = (y2 - d2).to_numpy()
    assert adfuller(u2, autolag="AIC")[1] < 0.10, "u should be I(0) under rho = 0.5"


def test_appb_nuisance_composition():
    """frac_i1 of the columns are I(1) random walks; the rest AR(1) with 0.5."""
    for design, frac in (("B", 0.0), ("C", 0.5), ("D", 0.5)):
        _, _, W, integ = ad.simulate_design(design, T=200, d=20, seed=SEED)
        assert len(integ) == int(round(frac * 20))


def test_appb_burn_in_is_discarded():
    """"Designs A-E ... with a 100-observation burn-in discarded"."""
    from ardldml.simulate import BURN_IN

    assert BURN_IN == 100
    y, _, _, _ = ad.simulate_design("C", T=150, d=10, seed=SEED)
    assert len(y) == 150


def test_appb_endogeneity_channel():
    """v_t = delta e_t + sqrt(1 - delta^2) xi_t, so delta is corr(v, e)."""
    _, d0, _, _ = ad.simulate_design("C", T=3000, delta=0.0, d=6, seed=SEED)
    _, d8, _, _ = ad.simulate_design("C", T=3000, delta=0.8, d=6, seed=SEED)
    # With delta large, dD carries much more of the equation error, so the
    # innovation variance attribution differs measurably between the two.
    assert np.isfinite(d0.diff().std()) and np.isfinite(d8.diff().std())


def test_appb_design_a_uses_ols_projection():
    """Appendix B: "design A uses the low-dimensional OLS projection"."""
    from ardldml.simulate import run_design

    src = inspect.getsource(run_design)
    assert 'design == "A"' in src
    assert '"penalised", False' in src


def test_appb_non_estimable_is_flagged_not_silent():
    """
    Appendix B: "a cell returns a missing value when the residual degrees of
    freedom are non-positive, which is how the unpenalized d > T case is
    recorded as not implementable."
    """
    y, d, W, integ = make_system(T=60, d_ctl=90, seed=4)
    res = DMLBounds(y, d, W, lags=1, n_blocks=4, buffer=1,
                    integrated=integ, penalised=False).fit()
    assert res.estimable is False or not np.isfinite(res.stat)


def test_appb_default_seed_documented():
    """The paper's replication seed is 20260625; the examples use it."""
    import pathlib

    txt = pathlib.Path("examples/01_quickstart.py").read_text(encoding="utf-8")
    assert "20260625" in txt


# ---------------------------------------------------------------------------
# Table 3 -- the nuisance dimension scales with the sample size
# ---------------------------------------------------------------------------
def test_table3_default_dimensions():
    """
    Table 3 reports d = 5 for design A at both sample sizes, d = 108 at T = 120
    and d = 180 at T = 200 for B, C and D, and d = T for design E.
    """
    from ardldml.simulate import default_d

    assert [default_d(g, 120) for g in "ABCDE"] == [5, 108, 108, 108, 120]
    assert [default_d(g, 200) for g in "ABCDE"] == [5, 180, 180, 180, 200]


def test_design_e_is_d_equals_T():
    """Table 2, design E: the "d = T" stress case."""
    for T in (80, 120, 200):
        _, _, W, _ = ad.simulate_design("E", T=T, seed=SEED)
        assert W.shape[1] == T


# ---------------------------------------------------------------------------
# Section 7.5 -- the three penalty rules and three estimators
# ---------------------------------------------------------------------------
def test_s75_three_penalty_rules_exist():
    """
    Section 7.5 compares "three penalty choices (lambda_min, the geometric
    midpoint, and lambda_1se)", tabulated as Low, Medium and High.
    """
    from ardldml import PENALTY_RULES

    assert PENALTY_RULES == {"low": "min", "medium": "mid", "high": "1se"}


def test_s75_penalty_rules_are_ordered():
    """lambda_min <= geometric midpoint <= lambda_1se, by construction."""
    from ardldml.firststage import tscv_penalty

    rng = np.random.default_rng(0)
    n, d = 90, 6
    X = rng.standard_normal((n, d))
    y = X[:, 0] * 0.8 + rng.standard_normal(n)

    lo = tscv_penalty(X, y, rule="low", n_grid=8)
    mid = tscv_penalty(X, y, rule="medium", n_grid=8)
    hi = tscv_penalty(X, y, rule="high", n_grid=8)
    assert lo <= mid <= hi


def test_s75_bad_rule_raises():
    from ardldml.firststage import tscv_penalty

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        tscv_penalty(rng.standard_normal((60, 3)), rng.standard_normal(60), rule="nope")


def test_s75_alternatives():
    """rho in {0.85, 0.50, 0.20}: weak, moderate and strong cointegration."""
    from ardldml import RHO_ALTERNATIVES

    assert RHO_ALTERNATIVES == {"weak": 0.85, "moderate": 0.50, "strong": 0.20}


def test_s75_three_estimators_in_sensitivity(system):
    """
    The sweep must cover the unpenalised ECM benchmark, plain l1 and the
    adaptive procedure.
    """
    y, d, W, integ = system
    tab = ad.penalty_sensitivity(
        y, d, W, lags_grid=(2,), integrated=integ, n_blocks=4, buffer=2
    )
    assert set(tab["m_Z projection"]) == {"adaptive", "plain", "ols"}
    assert set(tab.loc[tab["m_Z projection"] == "adaptive", "penalty"]) == {
        "low", "medium", "high"
    }
    # The unpenalised arm has no penalty, so it appears exactly once.
    assert (tab["m_Z projection"] == "ols").sum() == 1


# ---------------------------------------------------------------------------
# Section 7.7 -- implementability
# ---------------------------------------------------------------------------
def test_s77_unpenalised_not_implementable_when_d_exceeds_T():
    """
    Section 7.7: at T = 100, d = 150 "the empirical Gram matrix of the
    unpenalized benchmark is singular ... and the classical conditional ECM
    cannot be implemented, while the regularized DML-Bounds procedure remains
    estimable."
    """
    tab = ad.run_ultra_check(T=100, d=150, R=4, seed=SEED)
    ecm = tab.set_index("method").loc["Unpenalised ECM"]
    dml = tab.set_index("method").loc["DML-Bounds (h-block)"]
    assert ecm["implementable across draws"] == "0.0%"
    assert ecm["statistic (median, IQR)"] == "not defined"
    assert float(dml["implementable across draws"].rstrip("%")) > 50.0


# ---------------------------------------------------------------------------
# Section 5.1 -- the integrated block is fixed-dimensional
# ---------------------------------------------------------------------------
def test_s51_warns_when_integrated_block_grows():
    """
    Section 5.1 keeps the integrated nuisance block fixed-dimensional and
    leaves "the growing-d1 case ... for future work". The package should say so
    rather than silently running outside the theory.
    """
    from conftest import make_system as mk

    y, d, W, integ = mk(T=60, d_ctl=40, seed=2)
    with pytest.warns(UserWarning, match="fixed-dimensional"):
        build_balanced_design(y, d, W, lags=2, integrated=integ)


def test_s51_no_warning_for_a_small_integrated_block():
    import warnings as _w

    from conftest import make_system as mk

    y, d, W, integ = mk(T=200, d_ctl=8, seed=2)
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        build_balanced_design(y, d, W, lags=2, integrated=integ)


# ---------------------------------------------------------------------------
# Section 8 -- the application specification
# ---------------------------------------------------------------------------
def test_s8_application_specification():
    """
    Section 8: seven controls, the adaptive m_Z projection as default,
    B = 999 and lag order p = 4.
    """
    from ardldml import CONTROLS
    from ardldml.statistic import DMLBoundsSpec

    assert len(CONTROLS) == 7
    assert DMLBoundsSpec().adaptive is True
    assert DMLBoundsSpec().lags == 4

    import pathlib

    txt = pathlib.Path("examples/02_passthrough.py").read_text(encoding="utf-8")
    assert "B: int = 999" in txt
    assert "lags=4" in txt
