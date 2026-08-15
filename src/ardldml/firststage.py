"""
The balanced first stage: what gets projected on what, and how.

This is the step where DML-Bounds departs from ordinary double machine
learning, and getting it wrong silently invalidates everything downstream.

Balance
-------
The two nuisance projections have *different* regressor sets, because their
targets have different integration orders:

* :math:`\\Delta Y_t` is :math:`I(0)`. Regressing a stationary target on
  integrated levels is unbalanced and spurious, so the integrated controls must
  enter this projection **in first differences**. The stationary design is

  .. math::
      X_t = (W_{0t},\\; \\Delta W_{1t},\\; \\Delta Y_{t-1}, \\ldots)

* :math:`Z_{t-1} = (Y_{t-1}, D_{t-1})'` is :math:`I(1)`. It is projected on the
  control **levels** :math:`(W_{0t}, W_{1t})`. This is the only place trend
  absorption can happen, and therefore the only place the effective integrated
  count :math:`\\tilde k` is determined.

Lemma 1 of the paper is why the split matters: a stationary regressor cannot
track the stochastic trend of an integrated one, so partialling out stationary
controls leaves the unit root intact no matter how many there are. Only
integrated controls can absorb a trend.

Regularisation
--------------
Two distinct roles, so two distinct defaults:

* the :math:`\\Delta Y` equation is an ordinary stationary sparse prediction
  problem; a plain LASSO with Post-LASSO refitting is appropriate, and the
  penalty may be chosen by rolling-origin cross-validation
  (:func:`tscv_penalty`);
* the :math:`m_Z` projection is where over-selection does real damage. An
  unpenalised projection of :math:`Z` onto many independent random walks
  spuriously stationarises :math:`Z` through pure spurious regression, driving
  :math:`\\tilde k` toward zero and destroying the relation the test is meant
  to detect. Vanilla :math:`\\ell_1` over-selects integrated regressors for the
  same reason. The default here follows the paper: **adaptive** LASSO with
  marginal (univariate) slope weights and the plug-in penalty
  :math:`\\lambda = c\\sqrt{\\log d / n}\\,\\hat\\sigma`, :math:`c = 1.1`.

The adaptive weighting is a stabilisation device, not a selection-consistency
theorem: the paper's formal result keeps the integrated block
fixed-dimensional. Treat it as a guard against spurious absorption, and read
the :func:`~ardldml.diagnostics.trend_absorption` gaps to see whether it bound.

Post-LASSO
----------
Following standard practice, the selected support is refit by unpenalised OLS
to remove shrinkage bias. Degrees of freedom charge the selected support size,
so a cell with non-positive residual degrees of freedom is reported as not
estimable rather than silently returning a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .folds import BlockStructure, hblock_folds

__all__ = [
    "BalancedDesign",
    "build_balanced_design",
    "classify_controls",
    "plugin_penalty",
    "adaptive_post_lasso",
    "tscv_penalty",
    "cross_fit_projection",
    "FirstStage",
]


# ---------------------------------------------------------------------------
# Classifying the nuisance space
# ---------------------------------------------------------------------------
def classify_controls(
    W: pd.DataFrame,
    integrated: Optional[Sequence[str]] = None,
    alpha: float = 0.10,
) -> Tuple[list, list]:
    """
    Split the nuisance space into stationary and integrated blocks.

    Parameters
    ----------
    W : pandas.DataFrame
        The full control set in levels.
    integrated : sequence of str, optional
        Names to treat as :math:`I(1)`. If given, classification is taken as
        stated and no testing is done -- which is usually what you want, since
        the whole appeal of the bounds framework is avoiding pre-tests.
    alpha : float
        Significance level for the fallback ADF test, used only when
        ``integrated`` is ``None``.

    Returns
    -------
    (stationary_names, integrated_names)

    Notes
    -----
    Pre-testing is a compromise here, not a principle. The bounds framework
    exists precisely so the integration order of the *tested* variables need
    not be known. But the balanced design must know which *controls* to
    difference, and that is a specification choice. Prefer passing
    ``integrated`` explicitly on economic grounds; the ADF fallback is a
    convenience for exploratory work, and its pre-test error is not accounted
    for in any of the inference downstream.
    """
    if integrated is not None:
        integrated = [c for c in integrated if c in W.columns]
        stationary = [c for c in W.columns if c not in integrated]
        return stationary, list(integrated)

    from statsmodels.tsa.stattools import adfuller

    stationary, i1 = [], []
    for c in W.columns:
        series = W[c].dropna()
        try:
            pval = adfuller(series, autolag="AIC")[1]
        except Exception:  # pragma: no cover - degenerate series
            pval = 1.0
        (stationary if pval < alpha else i1).append(c)
    return stationary, i1


# ---------------------------------------------------------------------------
# Balanced design
# ---------------------------------------------------------------------------
@dataclass
class BalancedDesign:
    """
    The two design matrices of the balanced first stage.

    Attributes
    ----------
    dY : pandas.Series
        The stationary target :math:`\\Delta Y_t`.
    X : pandas.DataFrame
        Stationary regressors for the :math:`\\Delta Y` projection: stationary
        controls in levels, integrated controls in differences, and lagged
        differences of ``Y`` and ``D``.
    Z : pandas.DataFrame
        The tested level terms :math:`Z_{t-1} = (Y_{t-1}, D_{t-1})`.
    Wlev : pandas.DataFrame
        Control **levels** used to project ``Z``.
    index : pandas.Index
        Common index after lag construction and listwise deletion.
    stationary, integrated : list of str
        The control classification actually used.
    """

    dY: pd.Series
    X: pd.DataFrame
    Z: pd.DataFrame
    Wlev: pd.DataFrame
    index: pd.Index
    stationary: list
    integrated: list

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def d_stationary(self) -> int:
        return self.X.shape[1]

    @property
    def d_levels(self) -> int:
        return self.Wlev.shape[1]

    def summary(self) -> str:
        return (
            f"BalancedDesign: n={self.n}\n"
            f"  dY  target        : {self.dY.name}\n"
            f"  X   stationary    : {self.d_stationary} regressors "
            f"({len(self.stationary)} I(0) levels, {len(self.integrated)} I(1) differences, "
            f"plus lagged differences)\n"
            f"  Z   tested levels : {list(self.Z.columns)}\n"
            f"  W   control levels: {self.d_levels}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<BalancedDesign n={self.n} d_stat={self.d_stationary} d_lev={self.d_levels}>"


def build_balanced_design(
    y: pd.Series,
    d: pd.Series,
    W: pd.DataFrame,
    lags: int = 4,
    integrated: Optional[Sequence[str]] = None,
    adf_alpha: float = 0.10,
) -> BalancedDesign:
    """
    Construct the balanced first-stage design.

    Parameters
    ----------
    y : pandas.Series
        Outcome in levels.
    d : pandas.Series
        Focal regressor in levels.
    W : pandas.DataFrame
        Nuisance controls in levels.
    lags : int
        Number of lagged differences of ``y`` and ``d`` to include in the
        stationary design ``X``. This is the short-run lag structure ``p``.
    integrated : sequence of str, optional
        Controls to treat as :math:`I(1)`; see :func:`classify_controls`.
    adf_alpha : float
        Level for the ADF fallback when ``integrated`` is not given.

    Returns
    -------
    BalancedDesign
    """
    stationary, i1 = classify_controls(W, integrated=integrated, alpha=adf_alpha)

    dY = y.diff().rename(f"D.{y.name}")
    dD = d.diff().rename(f"D.{d.name}")

    cols: Dict[str, pd.Series] = {}
    for c in stationary:
        cols[c] = W[c]
    for c in i1:
        cols[f"D.{c}"] = W[c].diff()
    for i in range(1, int(lags) + 1):
        cols[f"D.{y.name}.L{i}"] = dY.shift(i)
        cols[f"D.{d.name}.L{i}"] = dD.shift(i)
    cols[f"D.{d.name}"] = dD

    X = pd.DataFrame(cols, index=y.index)
    Z = pd.DataFrame({f"{y.name}.L1": y.shift(1), f"{d.name}.L1": d.shift(1)}, index=y.index)
    Wlev = W.copy()

    frame = pd.concat([dY, X, Z, Wlev], axis=1).dropna()
    idx = frame.index
    return BalancedDesign(
        dY=dY.loc[idx],
        X=X.loc[idx],
        Z=Z.loc[idx],
        Wlev=Wlev.loc[idx],
        index=idx,
        stationary=list(stationary),
        integrated=list(i1),
    )


# ---------------------------------------------------------------------------
# Penalised estimation
# ---------------------------------------------------------------------------
def plugin_penalty(n: int, d: int, sigma: float, c: float = 1.1) -> float:
    """
    The plug-in penalty :math:`\\lambda = c\\sqrt{\\log d / n}\\,\\hat\\sigma`.

    This is the rule used in the paper's Monte Carlo, with ``c = 1.1``. It is
    the standard high-dimensional choice: large enough to dominate the noise,
    small enough to leave the signal.
    """
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")
    return float(c * np.sqrt(np.log(max(d, 2)) / n) * sigma)


def _standardise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def adaptive_post_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lam: Optional[float] = None,
    adaptive: bool = True,
    c: float = 1.1,
    max_iter: int = 5000,
    tol: float = 1e-4,
    min_dof: int = 1,
) -> Dict[str, object]:
    """
    Adaptive LASSO selection followed by an unpenalised refit.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Regressors.
    y : ndarray, shape (n,)
        Target.
    lam : float, optional
        Penalty. If ``None``, uses :func:`plugin_penalty` with an initial
        :math:`\\hat\\sigma` from the target's standard deviation, refined once.
    adaptive : bool
        If ``True``, weight the penalty on coefficient ``j`` by the reciprocal
        of the absolute univariate slope of ``y`` on column ``j``, which is the
        "marginal slope weights" rule of the paper. If ``False``, plain LASSO.
    c : float
        Constant in the plug-in penalty.
    min_dof : int
        Minimum residual degrees of freedom required for the refit. If the
        selected support is too large, the refit is skipped and the LASSO fit
        is returned with ``estimable=False``.

    Returns
    -------
    dict
        ``fitted``, ``coef``, ``support`` (boolean mask), ``intercept``,
        ``lam``, ``estimable``.

    Notes
    -----
    Selection is done on standardised columns so the penalty is scale-free;
    coefficients are mapped back to the original scale before the refit.
    """
    from sklearn.linear_model import Lasso

    n, d = X.shape
    Xs, mu, sd = _standardise(X)
    ybar = float(y.mean())
    yc = y - ybar

    if adaptive:
        denom = (Xs**2).sum(axis=0)
        denom = np.where(denom < 1e-12, 1.0, denom)
        marg = np.abs(Xs.T @ yc) / denom
        w = np.maximum(marg, 1e-8)  # weight = 1/penalty-weight, so multiply columns
        Xw = Xs * w
    else:
        w = np.ones(d)
        Xw = Xs

    if lam is None:
        sigma0 = float(np.std(yc, ddof=1)) or 1.0
        lam = plugin_penalty(n, d, sigma0, c=c)
        fit0 = Lasso(alpha=lam, max_iter=max_iter, tol=tol, fit_intercept=False).fit(Xw, yc)
        r0 = yc - Xw @ fit0.coef_
        sigma1 = float(np.std(r0, ddof=1)) or sigma0
        lam = plugin_penalty(n, d, sigma1, c=c)

    fit = Lasso(alpha=lam, max_iter=max_iter, tol=tol, fit_intercept=False).fit(Xw, yc)
    coef_w = fit.coef_
    support = np.abs(coef_w) > 1e-10

    coef = np.zeros(d)
    estimable = True
    if support.any():
        n_sel = int(support.sum())
        if n - n_sel - 1 < min_dof:
            estimable = False
            coef[support] = coef_w[support] * w[support] / sd[support]
        else:
            Xsel = Xs[:, support]
            beta_sel, *_ = np.linalg.lstsq(Xsel, yc, rcond=None)
            coef[support] = beta_sel / sd[support]
    intercept = ybar - float(mu @ coef)
    fitted = X @ coef + intercept
    return {
        "fitted": fitted,
        "coef": coef,
        "support": support,
        "intercept": intercept,
        "lam": float(lam),
        "estimable": estimable,
    }


def tscv_penalty(
    X: np.ndarray,
    y: np.ndarray,
    grid: Optional[Sequence[float]] = None,
    min_train: Optional[int] = None,
    adaptive: bool = False,
    n_grid: int = 20,
) -> float:
    """
    Rolling-origin time-series cross-validation for the penalty.

    Implements Section 4.4 of the paper: for origins
    :math:`t = T_0, \\ldots, T-1` the first stage is fitted on
    :math:`\\{1, \\ldots, t\\}` and evaluated one step ahead, and
    :math:`\\lambda` minimises the average out-of-sample squared error

    .. math::
        \\lambda_{opt} = \\arg\\min_\\lambda \\frac{1}{T - T_0}
        \\sum_{t=T_0}^{T-1}\\bigl(\\Delta Y_{t+1} - \\hat\\ell^{(\\lambda)}(W_{t+1})\\bigr)^2

    This respects temporal ordering, unlike ordinary k-fold cross-validation,
    which would train on the future.

    Parameters
    ----------
    grid : sequence of float, optional
        Penalties to search. Defaults to a log grid spanning two decades around
        the plug-in value.
    min_train : int, optional
        :math:`T_0`. Defaults to ``max(20, n // 3)``.
    """
    n, d = X.shape
    if min_train is None:
        min_train = max(20, n // 3)
    if min_train >= n - 1:
        return plugin_penalty(n, d, float(np.std(y, ddof=1)) or 1.0)

    if grid is None:
        base = plugin_penalty(n, d, float(np.std(y, ddof=1)) or 1.0)
        grid = np.geomspace(base / 10.0, base * 10.0, n_grid)

    errors = np.zeros(len(grid))
    counts = np.zeros(len(grid))
    for gi, lam in enumerate(grid):
        for t in range(min_train, n):
            fit = adaptive_post_lasso(X[:t], y[:t], lam=float(lam), adaptive=adaptive)
            pred = float(X[t] @ fit["coef"] + fit["intercept"])
            errors[gi] += (y[t] - pred) ** 2
            counts[gi] += 1
    mse = errors / np.maximum(counts, 1)
    return float(np.asarray(grid)[int(np.argmin(mse))])


# ---------------------------------------------------------------------------
# Cross-fitted projections
# ---------------------------------------------------------------------------
def cross_fit_projection(
    X: np.ndarray,
    y: np.ndarray,
    folds: BlockStructure,
    lam: Optional[float] = None,
    adaptive: bool = True,
    c: float = 1.1,
    penalised: bool = True,
) -> Dict[str, object]:
    """
    Out-of-fold predictions of ``y`` on ``X`` under an h-block partition.

    Each evaluation block is predicted by a model estimated on that block's
    buffered training set, so the first-stage error is decoupled from the
    evaluation-fold innovations (Lemma 2).

    Parameters
    ----------
    penalised : bool
        If ``False``, the projection is unpenalised OLS. This is the
        low-dimensional corner used by Design A of the paper's Monte Carlo, and
        the ``ols`` arm of the trend-absorption diagnostic.

    Returns
    -------
    dict
        ``fitted`` (out-of-fold predictions), ``resid``, ``support_union``
        (columns selected in any fold), ``lams``, ``estimable``.
    """
    n = X.shape[0]
    fitted = np.full(n, np.nan)
    support_union = np.zeros(X.shape[1], dtype=bool)
    lams, estimable = [], True

    for train, ev in folds:
        if penalised:
            fit = adaptive_post_lasso(X[train], y[train], lam=lam, adaptive=adaptive, c=c)
            coef, intercept = fit["coef"], fit["intercept"]
            support_union |= fit["support"]
            lams.append(fit["lam"])
            estimable &= bool(fit["estimable"])
        else:
            Xtr = np.hstack([np.ones((train.size, 1)), X[train]])
            beta, *_ = np.linalg.lstsq(Xtr, y[train], rcond=None)
            intercept, coef = float(beta[0]), beta[1:]
            support_union |= True
            estimable &= (train.size - X.shape[1] - 1) > 0
        fitted[ev] = X[ev] @ coef + intercept

    return {
        "fitted": fitted,
        "resid": y - fitted,
        "support_union": support_union,
        "lams": lams,
        "estimable": estimable,
    }


@dataclass
class FirstStage:
    """
    Output of the balanced, cross-fitted first stage.

    Attributes
    ----------
    dY_resid : numpy.ndarray
        :math:`\\widetilde{\\Delta Y}_t = \\Delta Y_t - \\hat\\ell(W_t)`.
    Z_resid : numpy.ndarray, shape (n, 2)
        :math:`\\tilde Z_{t-1} = Z_{t-1} - \\hat m_Z(W_t)`.
    design : BalancedDesign
    folds : BlockStructure
    supports : dict
        Union of selected columns, per projection.
    estimable : bool
        ``False`` if any projection exhausted its degrees of freedom.
    """

    dY_resid: np.ndarray
    Z_resid: np.ndarray
    design: BalancedDesign
    folds: BlockStructure
    supports: Dict[str, np.ndarray]
    estimable: bool

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<FirstStage n={len(self.dY_resid)} "
            f"selected_dY={int(self.supports['dY'].sum())} "
            f"selected_Z={int(self.supports['Z'].sum())} estimable={self.estimable}>"
        )
