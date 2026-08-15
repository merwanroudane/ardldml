"""
The conditional error correction model and the classical bounds statistic.

This module is the *classical* half of :mod:`ardldml`: the unrestricted
conditional ECM of Pesaran, Shin and Smith (2001) and the ordinary bounds
statistic computed from it. DML-Bounds uses it in three places --

* as the benchmark the orthogonalised test is compared against, which is the
  "borrowed bound" column of every Monte Carlo table in the paper;
* as the low-dimensional corner of the framework, recovered when the nuisance
  space carries no integrated component;
* as the *restricted* model that the bootstrap resamples from, built here by
  :func:`restricted_null_model`.

Critical values are not stored anywhere in this package. See
:mod:`ardldml.critvals` for why, and for the generator that replaces them.

The conditional ECM
-------------------
With :math:`y_t` the outcome, :math:`x_t` the long-run forcing regressors and
:math:`z_t` optional stationary regressors that shift short-run dynamics only,

.. math::

    \\Delta y_t = c_0 + c_1 t + \\pi_y y_{t-1} + \\pi_x' x_{t-1}
                + \\sum_{i=1}^{p-1}\\psi_{yi}\\Delta y_{t-i}
                + \\sum_{i=0}^{q-1}\\psi_{xi}'\\Delta x_{t-i}
                + \\gamma' z_t + u_t

from which the speed of adjustment is :math:`\\alpha = -\\pi_y` and the long-run
coefficients are :math:`\\theta = \\pi_x/\\alpha`.

The three-step procedure
------------------------
Rejecting the joint ``F`` null is not sufficient evidence of a level
relationship, because two degenerate cases survive it. The full procedure is

1. ``F`` test of :math:`\\pi_y = 0 \\cap \\pi_x = 0` (plus the restricted
   deterministic term in cases 2 and 4);
2. ``t`` test of :math:`\\pi_y = 0` against :math:`\\pi_y < 0`, ruling out the
   case where :math:`y_t` is :math:`I(1)` but not cointegrated with anything;
3. a conventional Wald test of :math:`\\theta = 0`, ruling out the case where
   :math:`y_t` is stationary but unrelated to :math:`x_t`. Step 3 uses ordinary
   critical values because :math:`\\hat\\theta` is asymptotically normal.

:func:`classical_bounds_test` returns all three.

References
----------
Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing approaches to
the analysis of level relationships. *Journal of Applied Econometrics*, 16(3),
289-326.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .critvals import CASE_LABELS, n_restrictions, simulate_pss_bounds

__all__ = [
    "trend_for_case",
    "conditional_ecm",
    "restricted_null_model",
    "ClassicalBounds",
    "classical_bounds_test",
]


def trend_for_case(case: int) -> str:
    """
    The :mod:`statsmodels` ``trend`` string implied by a deterministic case.

    Cases 2 and 3 both estimate an intercept; they differ only in whether that
    intercept is inside the tested null. The same holds for the trend in cases
    4 and 5. Estimation is therefore identical within each pair, and the case
    only bites at the testing stage.
    """
    if case not in CASE_LABELS:
        raise ValueError(f"case must be one of {sorted(CASE_LABELS)}; got {case!r}")
    return {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}[int(case)]


def conditional_ecm(
    y: pd.Series,
    x: pd.DataFrame,
    lags: int = 1,
    order: int = 1,
    case: int = 3,
    fixed: Optional[pd.DataFrame] = None,
    causal: bool = False,
):
    """
    Fit the unrestricted conditional error correction model.

    A documented wrapper on :class:`statsmodels.tsa.ardl.UECM`.

    Parameters
    ----------
    y : pandas.Series
        Outcome in levels.
    x : pandas.DataFrame
        Long-run forcing regressors in levels.
    lags, order : int
        ARDL orders ``p`` and ``q``.
    case : {1, 2, 3, 4, 5}
        Deterministic case; see :data:`ardldml.critvals.CASE_LABELS`.
    fixed : pandas.DataFrame, optional
        Regressors entered contemporaneously and never lagged -- the ``z_t``
        of the ARDL literature. They shift short-run dynamics but are excluded
        from the long-run relationship, which is what you want for a variable
        that is a genuine predictor but cannot plausibly cointegrate with the
        outcome.
    causal : bool
        If ``True``, drop lag 0 of ``x``.

    Returns
    -------
    statsmodels.tsa.ardl.UECMResults
    """
    from statsmodels.tsa.ardl import UECM

    model = UECM(
        y,
        lags,
        x,
        order=order,
        trend=trend_for_case(case),
        fixed=fixed,
        causal=causal,
    )
    return model.fit()


def restricted_null_model(
    y: pd.Series,
    x: pd.DataFrame,
    lags: int = 1,
    order: int = 1,
    case: int = 3,
    fixed: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Fit the conditional model with the null of no level relationship imposed.

    This is step (2) of Algorithm 1. Following the paper, the restricted model
    keeps "the same deterministic terms and short-run lag structure as the
    empirical specification, but excludes the lagged levels and excludes the
    high-dimensional confounder levels". Everything is in differences, so the
    regression is balanced and the residuals :math:`\\hat\\epsilon_t` are the
    innovations the bootstrap reweights.

    Returns
    -------
    dict
        With keys ``resid`` (the :math:`\\hat\\epsilon_t`), ``params``,
        ``exog_names``, ``index`` and ``dy``.
    """
    dy = y.diff()
    dx = x.diff()

    cols: Dict[str, pd.Series] = {}
    for i in range(1, int(lags)):
        cols[f"D.{y.name}.L{i}"] = dy.shift(i)
    for c in x.columns:
        for i in range(0, int(order)):
            cols[f"D.{c}.L{i}"] = dx[c].shift(i)
    if fixed is not None:
        for c in fixed.columns:
            cols[str(c)] = fixed[c]

    design = pd.DataFrame(cols, index=y.index)
    if case != 1:
        design.insert(0, "const", 1.0)
    if case == 5:
        design["trend"] = np.arange(len(design), dtype=float)

    frame = pd.concat([dy.rename("_dy"), design], axis=1).dropna()
    yy = frame["_dy"].to_numpy()
    XX = frame.drop(columns="_dy").to_numpy()

    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX @ beta
    return {
        "resid": pd.Series(resid, index=frame.index, name="eps_hat"),
        "params": pd.Series(beta, index=frame.drop(columns="_dy").columns),
        "exog_names": list(frame.drop(columns="_dy").columns),
        "index": frame.index,
        "dy": pd.Series(yy, index=frame.index),
    }


class ClassicalBounds:
    """
    Result of the ordinary Pesaran, Shin and Smith bounds test.

    Attributes
    ----------
    f_stat : float
        Joint ``F`` statistic on the level terms (step 1).
    t_stat : float
        ``t`` statistic on the speed-of-adjustment coefficient (step 2).
    theta_wald, theta_pvalue : float
        Wald statistic and its :math:`\\chi^2` p-value for
        :math:`\\theta = 0` (step 3), which uses conventional critical values.
    k, case, nobs : int
        Number of forcing regressors (PSS convention), deterministic case and
        effective sample size.
    alpha : float
        Speed of adjustment :math:`\\alpha = -\\pi_y`.
    long_run : pandas.Series
        Long-run coefficients :math:`\\theta`.
    """

    def __init__(
        self,
        f_stat: float,
        t_stat: float,
        theta_wald: float,
        theta_pvalue: float,
        k: int,
        case: int,
        nobs: int,
        alpha: float,
        long_run: pd.Series,
    ) -> None:
        self.f_stat = float(f_stat)
        self.t_stat = float(t_stat)
        self.theta_wald = float(theta_wald)
        self.theta_pvalue = float(theta_pvalue)
        self.k = int(k)
        self.case = int(case)
        self.nobs = int(nobs)
        self.alpha = float(alpha)
        self.long_run = long_run

    def bounds(self, T: Optional[int] = None, nsim: int = 20_000, seed: Optional[int] = 0):
        """
        Generate the classical bracket for this model.

        Parameters
        ----------
        T : int, optional
            Sample size for the simulation. Defaults to the model's own
            ``nobs``, giving finite-sample bounds. Pass ``T=1000`` to reproduce
            the published asymptotic table.
        nsim : int
            Replications.
        """
        return simulate_pss_bounds(
            k=self.k, case=self.case, T=int(T or self.nobs), nsim=nsim, seed=seed
        )

    def summary(self, T: Optional[int] = None, nsim: int = 20_000, seed: Optional[int] = 0) -> str:
        """Text summary with a freshly generated bracket."""
        cv = self.bounds(T=T, nsim=nsim, seed=seed)
        lines = [
            "Classical bounds test (Pesaran, Shin and Smith 2001)",
            f"H0: no level relationship          F = {self.f_stat:.3f}",
            f"Case {self.case} ({CASE_LABELS[self.case]})    t = {self.t_stat:.3f}",
            f"k = {self.k} forcing regressor(s), {self.nobs} observations, "
            f"{n_restrictions(self.k, self.case)} restrictions",
            "",
            f"simulated bounds (T={int(T or self.nobs)}, nsim={nsim}):",
            cv.to_string(float_format=lambda v: f"{v:6.3f}"),
            "",
            f"step 3  Wald(theta=0) = {self.theta_wald:.3f}  (chi2 p = {self.theta_pvalue:.4f})",
            f"speed of adjustment alpha = {self.alpha:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<ClassicalBounds F={self.f_stat:.3f} t={self.t_stat:.3f} k={self.k} case={self.case}>"


def _level_term_names(y_name: str, x_cols, case: int, available: List[str]) -> List[str]:
    """Names of the coefficients restricted under the step-1 null."""
    names = [f"{y_name}.L1"] + [f"{c}.L1" for c in x_cols]
    missing = [n for n in names if n not in available]
    if missing:  # pragma: no cover - defensive
        raise KeyError(f"level terms {missing} not in UECM output; available: {available}")
    if case == 2 and "const" in available:
        names = ["const"] + names
    elif case == 4:
        trend_name = next((n for n in available if n in ("trend", "time")), None)
        if trend_name is not None:
            names = names + [trend_name]
    return names


def classical_bounds_test(
    y: pd.Series,
    x: pd.DataFrame,
    lags: int = 1,
    order: int = 1,
    case: int = 3,
    fixed: Optional[pd.DataFrame] = None,
) -> ClassicalBounds:
    """
    Run all three steps of the classical bounds procedure.

    Notes
    -----
    ``UECMResults.bounds_test`` is deliberately not used: it indexes the
    critical value table with ``k + 1`` and therefore reports bounds that are
    too small. See :func:`ardldml.critvals.statsmodels_offset`. The Wald
    statistic itself is computed here and read against a bracket generated by
    :func:`ardldml.critvals.simulate_pss_bounds`.
    """
    from scipy import stats as sps

    res = conditional_ecm(y, x, lags=lags, order=order, case=case, fixed=fixed)
    k = int(x.shape[1])
    params, cov = res.params, res.cov_params()
    available = list(params.index)
    y_name = y.name if y.name is not None else "y"

    names = _level_term_names(y_name, x.columns, case, available)
    loc = [available.index(n) for n in names]
    R = np.zeros((len(loc), len(available)))
    for i, j in enumerate(loc):
        R[i, j] = 1.0
    coef = R @ params.to_numpy()
    vcv = R @ cov.to_numpy() @ R.T
    f_stat = float(coef @ np.linalg.solve(vcv, coef) / R.shape[0])

    pi_y = float(params[f"{y_name}.L1"])
    se_y = float(np.sqrt(cov.loc[f"{y_name}.L1", f"{y_name}.L1"]))
    t_stat = pi_y / se_y
    alpha = -pi_y

    # Step 3: theta = pi_x / alpha, delta method for its covariance.
    pi_x = np.array([float(params[f"{c}.L1"]) for c in x.columns])
    theta = pi_x / alpha
    idx = [available.index(f"{c}.L1") for c in x.columns]
    iy = available.index(f"{y_name}.L1")
    cov_np = cov.to_numpy()
    J = np.zeros((k, len(available)))
    for r, j in enumerate(idx):
        J[r, j] = 1.0 / alpha
        J[r, iy] = pi_x[r] / (alpha**2)
    v_theta = J @ cov_np @ J.T
    theta_wald = float(theta @ np.linalg.solve(v_theta, theta))
    theta_p = float(sps.chi2.sf(theta_wald, df=k))

    return ClassicalBounds(
        f_stat=f_stat,
        t_stat=t_stat,
        theta_wald=theta_wald,
        theta_pvalue=theta_p,
        k=k,
        case=case,
        nobs=int(res.nobs),
        alpha=alpha,
        long_run=pd.Series(dict(zip(x.columns, theta)), name="long_run"),
    )
