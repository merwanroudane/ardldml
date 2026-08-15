"""
Algorithm 1: the restricted system wild bootstrap.

This module *is* the inference in :mod:`ardldml`. Because the tabulated bounds
are not operationally valid once the conditioning set is large and persistent,
the critical value is generated here, from the data, under the imposed null.

The procedure
-------------
1. Compute the observed statistic :math:`F`.
2. Estimate two auxiliary models under :math:`H_0`:

   * the **restricted conditional model** for :math:`\\Delta Y_t`, with the same
     deterministic terms and short-run lag structure as the empirical
     specification but excluding the lagged levels and the confounder levels,
     giving residuals :math:`\\hat\\epsilon_t`;
   * the **marginal model** for the focal regressor, :math:`\\Delta D_t` on an
     intercept, its own lags and the differenced controls, giving residuals
     :math:`\\hat v_t`.

3. For :math:`b = 1,\\ldots,B`: draw a *single* Rademacher sequence
   :math:`\\eta_t \\in \\{-1,+1\\}` and apply it to the **stacked** residual
   pair, :math:`(\\epsilon^*_t, v^*_t) = (\\hat\\epsilon_t\\eta_t, \\hat v_t\\eta_t)`.
   Regenerate :math:`\\Delta D^*` recursively from the marginal dynamics with
   the control path fixed and cumulate to :math:`D^*`; regenerate
   :math:`\\Delta Y^*` from the restricted conditional dynamics driven by
   :math:`\\Delta D^*` and :math:`\\epsilon^*`, and cumulate to :math:`Y^*`.
   Recompute the **entire** residualised statistic on
   :math:`(Y^*, D^*, W)` with :math:`W` held at its realised path, re-selecting
   the level supports.
4. The critical value is the :math:`1-\\alpha` quantile of :math:`\\{F^*_b\\}`
   and the p-value is :math:`B^{-1}\\sum_b \\mathbf{1}\\{F^*_b \\ge F\\}`.

Why the weight is shared
------------------------
The Pesaran-Shin-Smith framework exists *because* the focal regressor need not
be exogenous: the conditional ECM absorbs the contemporaneous correlation
between :math:`v_t` and the equation error through the :math:`\\Delta D_t` term.
A wild bootstrap that holds ``D`` at its realised path and reweights
:math:`\\hat\\epsilon_t` alone makes the simulated innovations independent of
the regressor path by construction, so it simulates a world with
:math:`\\mathrm{corr}(\\epsilon, v) = 0` whatever the data say. Applying one
weight to the stacked pair carries the empirical cross-covariance -- and the
conditional heteroskedasticity of both series, since :math:`\\eta_t^2 = 1` --
into every draw.

The fixed-regressor scheme is retained as ``scheme="fixed"``, because Algorithm
1 nests it as the strong-exogeneity special case and the paper's own main grid
was computed under it. Under exogeneity the two agree; the ``delta=0`` rows of
the paper's Table 5 verify this numerically.

Why ``W`` is held fixed
-----------------------
Holding the nuisance space at its realised path is what conditions the
bootstrap on the realised trend content -- the object the trend-absorption
bracket is built on. And because the marginal model for :math:`\\Delta D`
conditions on the differenced controls, any stochastic trend that ``D`` shares
with ``W`` enters :math:`D^*` through that fixed path, so the realised
cointegration configuration, and hence :math:`\\tilde k`, is inherited rather
than broken by independent regeneration.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .firststage import build_balanced_design
from .statistic import DMLBoundsSpec, compute_statistic

__all__ = ["restricted_system_wild_bootstrap", "rademacher"]


def rademacher(n: int, rng: np.random.Generator) -> np.ndarray:
    """A length-``n`` sequence of independent :math:`\\pm 1` draws."""
    return rng.integers(0, 2, size=n).astype(float) * 2.0 - 1.0


def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _split_columns(design, y_name: str, d_name: str):
    """Partition the stationary design into recursive and fixed groups."""
    cols = list(design.X.columns)
    ylag = [c for c in cols if c.startswith(f"D.{y_name}.L")]
    dterm = [c for c in cols if c == f"D.{d_name}" or c.startswith(f"D.{d_name}.L")]
    wfix = [c for c in cols if c not in ylag and c not in dterm]
    return ylag, dterm, wfix


def _lag_order(name: str, prefix: str) -> int:
    """Extract ``i`` from ``D.<var>.L<i>``; contemporaneous terms give 0."""
    if name == prefix:
        return 0
    return int(name.rsplit(".L", 1)[1])


def restricted_system_wild_bootstrap(
    y: pd.Series,
    d: pd.Series,
    W: pd.DataFrame,
    spec: DMLBoundsSpec,
    observed: float,
    B: int = 999,
    level: float = 0.05,
    seed: Optional[int] = None,
    scheme: str = "system",
    freeze_stationary_support: bool = True,
    n_jobs: int = 1,
    progress: bool = False,
) -> Dict[str, object]:
    """
    Generate a critical value and p-value for the DML-Bounds statistic.

    Parameters
    ----------
    y, d, W : pandas.Series, pandas.Series, pandas.DataFrame
        The observed data, in levels.
    spec : DMLBoundsSpec
        The specification whose statistic is being tested. The same object is
        used to recompute the statistic on every bootstrap path, so the
        simulated null carries the same generated-regressor error.
    observed : float
        The observed statistic.
    B : int
        Bootstrap replications. The paper uses 999.
    level : float
        Significance level for the reported critical value.
    seed : int, optional
        Seed. The paper uses 20260625.
    scheme : {"system", "fixed"}
        ``"system"`` is Algorithm 1: joint regeneration with a shared weight.
        ``"fixed"`` holds ``D`` at its realised path and reweights
        :math:`\\hat\\epsilon` alone -- valid only under strong exogeneity.
    freeze_stationary_support : bool
        Freeze the stationary first-stage support across draws while
        re-selecting the level supports, as in the paper's implementation.
    n_jobs : int
        Parallel workers for the bootstrap loop.
    progress : bool
        Print a progress line every 10% of draws.

    Returns
    -------
    dict
        ``crit``, ``pvalue``, ``draws``, ``B``, ``level``, ``scheme``,
        ``n_failed``.
    """
    if scheme not in ("system", "fixed"):
        raise ValueError("scheme must be 'system' or 'fixed'")

    y_name = y.name if y.name is not None else "y"
    d_name = d.name if d.name is not None else "d"

    design = build_balanced_design(y, d, W, lags=spec.lags, integrated=spec.integrated)
    idx = design.index
    n = design.n

    ylag_cols, dterm_cols, wfix_cols = _split_columns(design, y_name, d_name)
    Xall = design.X

    # The observed first stage. Its stationary support is frozen across draws
    # (Appendix B) and also defines the "first-stage-selected differenced
    # controls" that enter the marginal model for dD.
    base = compute_statistic(y, d, W, spec)
    dY_support = base["first_stage"].supports["dY"]
    frozen = dY_support if (freeze_stationary_support and dY_support.any()) else None

    selected_names = set(np.asarray(Xall.columns)[dY_support]) if dY_support.any() else set()
    wsel_cols = [c for c in wfix_cols if c in selected_names] or wfix_cols

    # ---- step 2a: restricted conditional model for dY under H0 ------------
    parts: List[np.ndarray] = [np.ones((n, 1))] if spec.case != 1 else [np.empty((n, 0))]
    order = ylag_cols + dterm_cols + wfix_cols
    parts.append(Xall[order].to_numpy(dtype=float))
    Xr = np.hstack([p for p in parts if p.shape[1]])
    dY = design.dY.to_numpy(dtype=float)
    beta_c, eps_hat = _ols(Xr, dY)

    has_const = 1 if spec.case != 1 else 0
    off = has_const
    coef_ylag = {c: float(beta_c[off + i]) for i, c in enumerate(ylag_cols)}
    off += len(ylag_cols)
    coef_dterm = {c: float(beta_c[off + i]) for i, c in enumerate(dterm_cols)}
    off += len(dterm_cols)
    coef_wfix = beta_c[off: off + len(wfix_cols)]
    const_c = float(beta_c[0]) if has_const else 0.0
    w_contrib = (
        Xall[wfix_cols].to_numpy(dtype=float) @ coef_wfix if wfix_cols else np.zeros(n)
    )

    # ---- step 2b: marginal model for dD -----------------------------------
    dD = design.X[f"D.{d_name}"].to_numpy(dtype=float)
    dlag_only = [c for c in dterm_cols if c != f"D.{d_name}"]
    Xm_parts = [np.ones((n, 1))]
    if dlag_only:
        Xm_parts.append(Xall[dlag_only].to_numpy(dtype=float))
    if wsel_cols:
        Xm_parts.append(Xall[wsel_cols].to_numpy(dtype=float))
    Xm = np.hstack(Xm_parts)
    beta_m, v_hat = _ols(Xm, dD)

    const_m = float(beta_m[0])
    coef_dlag_m = {c: float(beta_m[1 + i]) for i, c in enumerate(dlag_only)}
    coef_wm = beta_m[1 + len(dlag_only):]
    wm_contrib = (
        Xall[wsel_cols].to_numpy(dtype=float) @ coef_wm if wsel_cols else np.zeros(n)
    )

    # Anchors for cumulating differences back to levels.
    y0 = float(y.loc[idx[0]]) - float(dY[0])
    d0 = float(d.loc[idx[0]]) - float(dD[0])

    max_ylag = max([_lag_order(c, f"D.{y_name}") for c in ylag_cols], default=0)
    max_dlag = max([_lag_order(c, f"D.{d_name}") for c in dlag_only], default=0)

    def _one_draw(b: int) -> float:
        rng = np.random.default_rng(None if seed is None else seed + b)
        eta = rademacher(n, rng)
        eps_star = eps_hat * eta

        if scheme == "system":
            v_star = v_hat * eta
            dD_star = np.empty(n)
            for t in range(n):
                val = const_m + wm_contrib[t] + v_star[t]
                for c, coef in coef_dlag_m.items():
                    lag = _lag_order(c, f"D.{d_name}")
                    val += coef * (dD_star[t - lag] if t - lag >= 0 else dD[t - lag])
                dD_star[t] = val
        else:
            dD_star = dD.copy()

        dY_star = np.empty(n)
        for t in range(n):
            val = const_c + w_contrib[t] + eps_star[t]
            for c, coef in coef_ylag.items():
                lag = _lag_order(c, f"D.{y_name}")
                val += coef * (dY_star[t - lag] if t - lag >= 0 else dY[t - lag])
            for c, coef in coef_dterm.items():
                lag = _lag_order(c, f"D.{d_name}")
                if t - lag >= 0:
                    val += coef * dD_star[t - lag]
                else:
                    val += coef * dD[t - lag]
            dY_star[t] = val

        y_star = pd.Series(y0 + np.cumsum(dY_star), index=idx, name=y_name)
        d_star = pd.Series(d0 + np.cumsum(dD_star), index=idx, name=d_name)

        try:
            out = compute_statistic(
                y_star, d_star, W.loc[idx], spec,
                reselect_levels=True, frozen_dY_support=frozen,
            )
            return float(out["stat"])
        except Exception:  # pragma: no cover - a degenerate resample
            return np.nan

    if n_jobs != 1:
        from joblib import Parallel, delayed

        draws = np.asarray(
            Parallel(n_jobs=n_jobs)(delayed(_one_draw)(b) for b in range(B)), dtype=float
        )
    else:
        draws = np.empty(B)
        step = max(B // 10, 1)
        for b in range(B):
            draws[b] = _one_draw(b)
            if progress and (b + 1) % step == 0:
                print(f"  bootstrap {b + 1}/{B}", flush=True)

    good = draws[np.isfinite(draws)]
    if good.size == 0:  # pragma: no cover - pathological
        raise RuntimeError("every bootstrap draw failed; check the specification")

    crit = float(np.quantile(good, 1 - level))
    pval = float(np.mean(good >= observed))
    return {
        "crit": crit,
        "pvalue": pval,
        "draws": good,
        "B": int(B),
        "level": float(level),
        "scheme": scheme,
        "n_failed": int(draws.size - good.size),
        "eps_hat": eps_hat,
        "v_hat": v_hat,
        "corr_eps_v": float(np.corrcoef(eps_hat, v_hat)[0, 1]),
    }
