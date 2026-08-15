"""
The DML-Bounds statistic.

Given the balanced, cross-fitted residuals of :mod:`ardldml.firststage`, the
long-run parameters are estimated by an *unpenalised* regression on the
orthogonalised variables

.. math::

    \\widetilde{\\Delta Y}_t = \\tilde Z_{t-1}'\\beta + \\hat e_t, \\qquad
    \\hat\\beta = \\Bigl(\\sum_t \\tilde Z_{t-1}\\tilde Z_{t-1}'\\Bigr)^{-1}
                 \\sum_t \\tilde Z_{t-1}\\widetilde{\\Delta Y}_t

and the statistic is the ``F`` form of the Wald test of :math:`\\beta = 0`,
where :math:`Z_{t-1} = (Y_{t-1}, D_{t-1})'`. Writing
:math:`\\beta = (\\pi_y, \\pi_x)'`, the speed of adjustment is
:math:`\\alpha = -\\pi_y` and the long-run coefficient is
:math:`\\theta = \\pi_x/\\alpha`.

Reading the statistic
---------------------
Do **not** compare it with a tabulated bound. The asymptotic reference sits
somewhere inside the Pesaran-Shin-Smith bracket, at a position governed by the
effective integrated count :math:`\\tilde k`, and the finite-sample law is
further perturbed by generated-regressor error that is
:math:`O_p(s\\log d/\\sqrt{T})` -- roughly 0.78 at :math:`s=3, d=40, T=200`.
The paper's Monte Carlo shows rejection rates against the borrowed 5.73 bound
reaching 0.737 under integrated nuisance at :math:`T=1000`.

The critical value comes from
:func:`~ardldml.bootstrap.restricted_system_wild_bootstrap`. Everything in this
module is set up so the bootstrap can recompute the *entire* residualised
statistic on each regenerated path, first-stage selection included, which is
what makes the simulated null reflect the generated-regressor term.

Why :math:`\\tilde k` is never estimated
----------------------------------------
:math:`\\tilde k = k - r` indexes the limit experiment; it is not a tuning
parameter. The bootstrap conditions on its realised value implicitly, by
holding the nuisance space at its observed path and regenerating the focal
regressor from a marginal model that conditions on the differenced controls, so
any common trend between ``D`` and ``W`` is inherited rather than broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .firststage import (
    FirstStage,
    build_balanced_design,
    cross_fit_projection,
    tscv_penalty,
)
from .folds import hblock_folds

__all__ = ["DMLBoundsSpec", "DMLBounds", "DMLBoundsResults", "compute_statistic"]


@dataclass
class DMLBoundsSpec:
    """
    Everything that defines a DML-Bounds fit, in one object.

    Held separately from the data so the bootstrap can rebuild the identical
    statistic on a regenerated path.

    Attributes
    ----------
    lags : int
        Short-run lag order ``p``.
    case : {1, 3}
        Deterministic case. This does **not** enter the statistic -- equation
        (10) is a no-intercept projection on the orthogonalised levels
        regardless. It controls the deterministic terms of the *restricted
        conditional model* the bootstrap resamples from, which Algorithm 1
        requires to carry "the same deterministic terms and short-run lag
        structure as the empirical specification". Case 3 is the default and
        matches the paper's reference specification.
    include_constant : bool
        Add an intercept to the stage-3 regression. Defaults to ``False``,
        which is equation (10) as written. Exposed only for sensitivity
        checks.
    n_blocks, buffer : int
        Cross-fitting configuration ``K`` and ``h``.
    adaptive : bool
        Adaptive weights on the :math:`m_Z` projection. ``True`` is the paper's
        default; ``False`` gives the plain-LASSO arm of the diagnostic.
    dlags : bool
        Include lags of :math:`\\Delta D` in the conditional design. Defaults
        to ``False``, matching equation (3), which carries only the
        contemporaneous term. ``True`` gives the general ARDL(p, q) structure.
    adaptive_integrated_only : bool
        Restrict the adaptive weights to the **integrated block**, which is
        what Section 4.1 specifies: the weighting exists because "vanilla
        :math:`\\ell_1` over-selects integrated regressors and thereby induces
        spurious trend absorption". Stationary controls stay under the plain
        penalty. Set ``False`` to weight every column, the looser reading of
        Appendix B.
    penalised : bool
        If ``False``, both projections are unpenalised OLS -- the
        low-dimensional corner (Design A) and the ``ols`` arm of the diagnostic.
    penalty : str or float
        How the :math:`\\Delta Y` penalty is chosen.

        * ``"plugin"`` -- :math:`c\\sqrt{\\log d/n}\\hat\\sigma`, the Appendix B
          rule and the default;
        * ``"tscv"`` / ``"low"`` / ``"min"`` -- rolling-origin cross-validation
          at the profile minimum (equation 11);
        * ``"medium"`` / ``"mid"`` -- the geometric midpoint;
        * ``"high"`` / ``"1se"`` -- the one-standard-error rule;
        * a float fixes it directly.

        The last three are the three penalty choices Section 7.5 traces
        robustness across. The level projection always uses the plug-in rule
        regardless, since equation (11) tunes the :math:`\\Delta Y` equation
        only.
    c : float
        Constant in the plug-in penalty.
    integrated : sequence of str, optional
        Controls to treat as :math:`I(1)`.
    """

    lags: int = 4
    case: int = 3
    n_blocks: int = 5
    buffer: int = 0
    adaptive: bool = True
    penalised: bool = True
    penalty: object = "plugin"
    c: float = 1.1
    integrated: Optional[Sequence[str]] = None
    include_constant: bool = False
    adaptive_integrated_only: bool = True
    dlags: bool = False


def _wald_f(
    dy_res: np.ndarray, Z_res: np.ndarray, include_constant: bool = False
) -> Dict[str, float]:
    """
    ``F`` form of the Wald test that both level coefficients are zero.

    Follows equation (10) of the paper exactly: an *unpenalised regression on
    the orthogonalised variables*, with no intercept. The first-stage
    projections already carry intercepts, so the residuals are mean-zero by
    construction and a constant here would be redundant.

    ``include_constant=True`` adds one anyway, which changes the residual
    degrees of freedom and therefore the statistic very slightly. It is exposed
    only so the sensitivity can be checked; it is not the paper's statistic.
    """
    n = Z_res.shape[0]
    if include_constant:
        X = np.hstack([np.ones((n, 1)), Z_res])
        tested = np.arange(1, X.shape[1])
    else:
        X = Z_res
        tested = np.arange(X.shape[1])

    beta, *_ = np.linalg.lstsq(X, dy_res, rcond=None)
    resid = dy_res - X @ beta
    dof = n - X.shape[1]
    if dof <= 0:
        return {"stat": np.nan, "alpha": np.nan, "theta": np.nan, "theta_se": np.nan}
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    V = s2 * xtx_inv

    b = beta[tested]
    Vb = V[np.ix_(tested, tested)]
    try:
        stat = float(b @ np.linalg.solve(Vb, b) / len(tested))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate
        return {"stat": np.nan, "alpha": np.nan, "theta": np.nan, "theta_se": np.nan}

    pi_y, pi_x = float(b[0]), float(b[1])
    alpha = -pi_y
    if abs(alpha) < 1e-12:
        return {"stat": stat, "alpha": alpha, "theta": np.nan, "theta_se": np.nan}
    theta = pi_x / alpha

    # Delta method for theta = pi_x / alpha = -pi_x / pi_y.
    g = np.array([pi_x / pi_y**2, -1.0 / pi_y])
    theta_se = float(np.sqrt(max(g @ Vb @ g, 0.0)))
    return {"stat": stat, "alpha": alpha, "theta": theta, "theta_se": theta_se}


def compute_statistic(
    y: pd.Series,
    d: pd.Series,
    W: pd.DataFrame,
    spec: DMLBoundsSpec,
    reselect_levels: bool = True,
    frozen_dY_support: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Build the balanced design, cross-fit, residualise and test.

    This is the whole pipeline in one call, so the bootstrap can invoke it on a
    regenerated path and obtain a statistic that carries the same
    generated-regressor error as the observed one.

    Parameters
    ----------
    reselect_levels : bool
        Re-run selection for the :math:`m_Z` projection. Algorithm 1 requires
        this on every bootstrap path: "re-selecting the first-stage level
        supports on the regenerated paths so that selection error is reflected
        in the bootstrap law". Freezing them instead "leaves a size distortion
        that grows with ``T``".
    frozen_dY_support : ndarray, optional
        Boolean mask restricting the stationary :math:`\\Delta Y` projection to
        a fixed support. The paper freezes the stationary support across
        bootstrap draws while re-selecting the level supports.

    Returns
    -------
    dict
        ``stat``, ``alpha``, ``theta``, ``theta_se``, ``first_stage``,
        ``design``, ``estimable``.
    """
    design = build_balanced_design(
        y, d, W, lags=spec.lags, integrated=spec.integrated, dlags=spec.dlags
    )
    n = design.n
    folds = hblock_folds(n, n_blocks=spec.n_blocks, buffer=spec.buffer)

    Xs = design.X.to_numpy(dtype=float)
    Wl = design.Wlev.to_numpy(dtype=float)
    dy = design.dY.to_numpy(dtype=float)
    Zl = design.Z.to_numpy(dtype=float)

    if frozen_dY_support is not None and frozen_dY_support.any():
        Xs_use = Xs[:, frozen_dY_support]
    else:
        Xs_use = Xs

    lam_dy = None
    if isinstance(spec.penalty, (int, float)) and not isinstance(spec.penalty, bool):
        lam_dy = float(spec.penalty)
    elif spec.penalised and spec.penalty in ("tscv", "min", "1se", "mid",
                                             "low", "medium", "high"):
        rule = "min" if spec.penalty == "tscv" else spec.penalty
        lam_dy = tscv_penalty(Xs_use, dy, adaptive=False, rule=rule)

    # Stationary projection of dY: plain LASSO is appropriate here.
    fit_dy = cross_fit_projection(
        Xs_use, dy, folds, lam=lam_dy, adaptive=False, c=spec.c, penalised=spec.penalised
    )

    # Level projection of each component of Z on control levels (eq 5 and 6).
    #
    # Two details follow Section 4.1 exactly. First, the adaptive weights apply
    # to the *integrated block only*: "vanilla l1 over-selects integrated
    # regressors and thereby induces spurious trend absorption, whereas
    # adaptive penalization curbs it". Stationary controls stay under the plain
    # penalty. Second, the penalty here is always the plug-in value -- equation
    # (11) defines the TSCV penalty for the dY equation alone, and Appendix B
    # gives the m_Z projection the plug-in rule c*sqrt(log d / n)*sigma.
    integrated_mask = np.array(
        [c in set(design.integrated) for c in design.Wlev.columns], dtype=bool
    )
    z_res = np.empty_like(Zl)
    z_support = np.zeros(Wl.shape[1], dtype=bool)
    estimable = fit_dy["estimable"]
    for j in range(Zl.shape[1]):
        fit_z = cross_fit_projection(
            Wl,
            Zl[:, j],
            folds,
            lam=None,
            adaptive=spec.adaptive,
            c=spec.c,
            penalised=spec.penalised,
            adaptive_mask=integrated_mask if spec.adaptive_integrated_only else None,
        )
        z_res[:, j] = fit_z["resid"]
        z_support |= fit_z["support_union"]
        estimable &= fit_z["estimable"]

    out = _wald_f(fit_dy["resid"], z_res, include_constant=spec.include_constant)

    # When the stationary support is frozen, the projection ran on a subset of
    # columns, so its support mask is indexed against that subset. Expand it
    # back to the full design width or downstream consumers -- and the frozen
    # mask on the next bootstrap path -- would be misaligned.
    dy_support = fit_dy["support_union"]
    if frozen_dY_support is not None and frozen_dY_support.any():
        full = np.zeros(Xs.shape[1], dtype=bool)
        full[np.flatnonzero(frozen_dY_support)[dy_support]] = True
        dy_support = full

    first = FirstStage(
        dY_resid=fit_dy["resid"],
        Z_resid=z_res,
        design=design,
        folds=folds,
        supports={"dY": dy_support, "Z": z_support},
        estimable=bool(estimable),
    )
    out.update({"first_stage": first, "design": design, "estimable": bool(estimable)})
    return out


@dataclass
class DMLBoundsResults:
    """
    Fitted DML-Bounds model.

    Call :meth:`bootstrap` to attach a critical value and p-value; until then
    the statistic has no reference distribution and should not be interpreted.
    """

    stat: float
    alpha: float
    theta: float
    theta_se: float
    spec: DMLBoundsSpec
    first_stage: FirstStage
    y: pd.Series
    d: pd.Series
    W: pd.DataFrame
    estimable: bool
    boot: Optional[Dict[str, object]] = field(default=None)

    @property
    def nobs(self) -> int:
        return self.first_stage.design.n

    @property
    def pvalue(self) -> Optional[float]:
        return None if self.boot is None else float(self.boot["pvalue"])

    @property
    def critical_value(self) -> Optional[float]:
        return None if self.boot is None else float(self.boot["crit"])

    def bootstrap(self, B: int = 999, level: float = 0.05, seed: Optional[int] = None, **kw):
        """
        Attach a bootstrap critical value via Algorithm 1.

        See :func:`ardldml.bootstrap.restricted_system_wild_bootstrap`.
        """
        from .bootstrap import restricted_system_wild_bootstrap

        self.boot = restricted_system_wild_bootstrap(
            self.y, self.d, self.W, self.spec, observed=self.stat,
            B=B, level=level, seed=seed, **kw
        )
        return self

    def decision(self, level: float = 0.05) -> str:
        """``"reject"`` or ``"fail to reject"`` at ``level``, or unknown."""
        if self.boot is None:
            return "no bootstrap run"
        return "reject" if self.pvalue < level else "fail to reject"

    def to_frame(self) -> pd.DataFrame:
        """One-row summary, for stacking across specifications."""
        return pd.DataFrame(
            [{
                "n": self.nobs,
                "F": self.stat,
                "boot_cv95": self.critical_value,
                "boot_p": self.pvalue,
                "alpha": self.alpha,
                "theta": self.theta,
                "theta_se": self.theta_se,
                "estimable": self.estimable,
            }]
        )

    def summary(self) -> str:
        lines = [
            "DML-Bounds test for a conditional long-run relationship",
            f"H0: no level relationship after residualisation      F = {self.stat:.4f}",
            f"n = {self.nobs}, controls = {self.W.shape[1]} "
            f"({len(self.first_stage.design.integrated)} treated as I(1)), "
            f"K = {self.spec.n_blocks}, h = {self.spec.buffer}",
            f"selected: {int(self.first_stage.supports['dY'].sum())} stationary, "
            f"{int(self.first_stage.supports['Z'].sum())} level controls",
            "",
        ]
        if self.boot is None:
            lines += [
                "No bootstrap critical value attached.",
                "The statistic has no tabulated reference: call .bootstrap() before",
                "interpreting it. Comparing it with the classical 5.73 bound",
                "over-rejects when the controls are integrated.",
            ]
        else:
            b = self.boot
            lines += [
                f"restricted system wild bootstrap: B = {b['B']}, "
                f"{b['scheme']} scheme",
                f"  critical value ({int((1 - b['level']) * 100)}%) = {b['crit']:.4f}",
                f"  bootstrap p-value                = {b['pvalue']:.4f}",
                f"  decision at {b['level']:.0%}                    -> {self.decision(b['level'])}",
            ]
        lines += [
            "",
            f"speed of adjustment  alpha = {self.alpha:.4f}",
            f"long-run coefficient theta = {self.theta:.4f}  (se {self.theta_se:.4f})",
        ]
        if not self.estimable:
            lines += ["", "WARNING: a projection exhausted its degrees of freedom."]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = "n/a" if self.pvalue is None else f"{self.pvalue:.3f}"
        return f"<DMLBoundsResults F={self.stat:.3f} p={p} n={self.nobs}>"


class DMLBounds:
    """
    Test for a long-run relationship conditional on a high-dimensional,
    possibly integrated, control set.

    Parameters
    ----------
    y : pandas.Series
        Outcome in levels.
    d : pandas.Series
        Focal regressor in levels.
    W : pandas.DataFrame
        Nuisance controls in levels. May be larger than the sample.
    **spec_kwargs
        Passed to :class:`DMLBoundsSpec`.

    Examples
    --------
    >>> model = DMLBounds(y, d, W, lags=4, integrated=["m2", "oil"])   # doctest: +SKIP
    >>> res = model.fit().bootstrap(B=999, seed=20260625)              # doctest: +SKIP
    >>> print(res.summary())                                           # doctest: +SKIP

    Notes
    -----
    The estimand is conditional. DML-Bounds does not ask whether ``y`` and
    ``d`` cointegrate unconditionally; it asks whether they cointegrate *given*
    ``W``. If a control is itself part of the equilibrium system, partialling
    it out removes the relation rather than confounding, and a non-rejection
    reflects over-absorption. That failure mode is not detectable from a single
    fit -- use :func:`ardldml.diagnostics.trend_absorption`.
    """

    def __init__(self, y: pd.Series, d: pd.Series, W: pd.DataFrame, **spec_kwargs) -> None:
        if not isinstance(W, pd.DataFrame):
            raise TypeError("W must be a DataFrame of controls in levels")
        self.y = y
        self.d = d
        self.W = W
        self.spec = DMLBoundsSpec(**spec_kwargs)

    def fit(self) -> DMLBoundsResults:
        """Run the balanced first stage and compute the statistic."""
        out = compute_statistic(self.y, self.d, self.W, self.spec)
        return DMLBoundsResults(
            stat=float(out["stat"]),
            alpha=float(out["alpha"]),
            theta=float(out["theta"]),
            theta_se=float(out["theta_se"]),
            spec=self.spec,
            first_stage=out["first_stage"],
            y=self.y,
            d=self.d,
            W=self.W,
            estimable=bool(out["estimable"]),
        )
