"""
Critical values, generated rather than tabulated.

DML-Bounds takes the view that borrowed bounds tables are not operationally
valid once the conditioning set is large and persistent: the generated-regressor
remainder is :math:`O_p(s\\log d/\\sqrt{T})`, which is order one at the sample
sizes applied work actually uses, so "the first-order asymptotics have not taken
hold". Every critical value this package reports is therefore *computed*, by one
of three routes:

1. :func:`~ardldml.bootstrap.restricted_system_wild_bootstrap` -- the operational
   critical value for a real dataset. This is Algorithm 1 of the paper and the
   one you should use for inference.
2. :func:`~ardldml.simulate.empirical_critical_value` -- the method-specific
   critical value obtained by simulating the null model. Used to separate size
   from power in Monte Carlo work; infeasible on real data.
3. :func:`simulate_pss_bounds`, in this module -- the classical Pesaran, Shin and
   Smith (2001) bracket, regenerated from their own data-generating process.

Why regenerate the classical bounds
-----------------------------------
The paper uses the classical bracket in exactly two places: as the conceptual
endpoints of the trend-absorption bracket, and as the *borrowed* bound that the
Monte Carlo shows over-rejects (rejection rates up to 0.737 under integrated
nuisance at ``T = 1000``). Both uses need the numbers, neither needs a table.

Pesaran, Shin and Smith obtained their tables by stochastic simulation with
``T = 1000`` and 40,000 replications, and printed the data-generating process in
the notes to Table CI. :func:`simulate_pss_bounds` implements that process
directly, which has three advantages over transcribing the published table:

* it is verifiable -- :func:`pss_reference` holds a handful of published cells
  and the test suite checks the simulator reproduces them;
* it extends to any sample size, so the small-sample bounds that Narayan (2004)
  tabulated for ``n = 30..80`` are obtained by passing ``T=n`` rather than by
  shipping a second table;
* it extends past ``k = 10``, where the published tables stop.

The data-generating process
---------------------------
Under the null, with :math:`y_0 = x_0 = 0` and
:math:`e_t = (\\varepsilon_{1t}, e_{2t}')'` a vector of :math:`k+1` independent
standard normals,

.. math::

    y_t = y_{t-1} + \\varepsilon_{1t}, \\qquad x_t = P x_{t-1} + e_{2t}

with :math:`P = I_k` when the regressors are purely :math:`I(1)` (giving the
upper bound) and :math:`P = 0` when they are purely :math:`I(0)` (the lower
bound). The statistic is the ``F`` form of the Wald test of
:math:`\\phi = 0` in

.. math::

    \\Delta y_t = \\phi' z_{t-1} + a' w_t + \\varepsilon_t

where :math:`z_{t-1}` and :math:`w_t` are set by the deterministic case.

The ``k`` convention
--------------------
``k`` is the number of long-run forcing regressors, as in Pesaran, Shin and
Smith. It does not count the dependent variable. Two nearby conventions exist
and mixing them shifts every bound by one row, so the boundary is enforced here:

* :mod:`statsmodels` feeds ``len(ardl_order) == k + 1`` into a ``k``-indexed
  table inside ``UECMResults.bounds_test``, so that method reports bounds that
  are too small and over-rejects. :func:`statsmodels_offset` reproduces the
  discrepancy. This package never calls it.
* The DML-Bounds paper writes :math:`Z_{t-1} = (Y_{t-1}, D_{t-1})' \\in
  \\mathbb{R}^k`, so its ``k`` counts level terms and is one larger.

References
----------
Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing approaches to
the analysis of level relationships. *Journal of Applied Econometrics*, 16(3),
289-326.

Narayan, P. K. (2004). Reformulating critical values for the bounds F-statistics
approach to cointegration. Monash University Discussion Paper 02/04.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CASE_LABELS",
    "DEFAULT_LEVELS",
    "simulate_pss_bounds",
    "pss_reference",
    "statsmodels_offset",
    "n_restrictions",
]

#: Human-readable description of each deterministic case.
CASE_LABELS: Dict[int, str] = {
    1: "no intercept, no trend",
    2: "restricted intercept, no trend",
    3: "unrestricted intercept, no trend",
    4: "unrestricted intercept, restricted trend",
    5: "unrestricted intercept, unrestricted trend",
}

#: Significance levels reported by default.
DEFAULT_LEVELS: Tuple[float, ...] = (0.10, 0.05, 0.01)


def n_restrictions(k: int, case: int) -> int:
    """
    Number of restrictions in the bounds ``F`` test.

    The level terms are always :math:`y_{t-1}` and the ``k`` lagged forcing
    regressors. Cases 2 and 4 add one more, because the intercept (case 2) or
    the trend (case 4) is restricted and therefore tested jointly with them.

    Examples
    --------
    >>> n_restrictions(k=1, case=3), n_restrictions(k=1, case=4)
    (2, 3)
    """
    if case not in CASE_LABELS:
        raise ValueError(f"case must be one of {sorted(CASE_LABELS)}; got {case!r}")
    return int(k) + 1 + (1 if case in (2, 4) else 0)


def _build_design(y: np.ndarray, x: np.ndarray, case: int, T: int):
    """Split regressors into tested levels ``z`` and untested deterministics ``w``."""
    ylag = y[:-1].reshape(-1, 1)
    xlag = x[:-1]
    one = np.ones((T, 1))
    trend = np.arange(1, T + 1, dtype=float).reshape(-1, 1)

    z = np.hstack([ylag, xlag]) if xlag.shape[1] else ylag
    w = np.empty((T, 0))

    if case == 1:
        pass
    elif case == 2:
        z = np.hstack([z, one])
    elif case == 3:
        w = one
    elif case == 4:
        z = np.hstack([z, trend])
        w = one
    elif case == 5:
        w = np.hstack([one, trend])
    else:  # pragma: no cover - guarded by n_restrictions
        raise ValueError(f"unknown case {case!r}")
    return z, w


def _f_statistic(dy: np.ndarray, z: np.ndarray, w: np.ndarray) -> float:
    """``F`` form of the Wald test that the coefficients on ``z`` are zero."""
    X = np.hstack([z, w]) if w.shape[1] else z
    nrest = z.shape[1]
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return np.nan
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    V = s2 * xtx_inv[:nrest, :nrest]
    b = beta[:nrest]
    try:
        quad = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate draw
        return np.nan
    return quad / nrest


def _simulate_one_side(
    k: int,
    case: int,
    T: int,
    nsim: int,
    integrated: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate the null distribution with regressors purely I(1) or purely I(0)."""
    stats = np.empty(nsim, dtype=float)
    for i in range(nsim):
        e = rng.standard_normal((T + 1, k + 1))
        y = np.cumsum(e[:, 0])
        if k == 0:
            x = np.empty((T + 1, 0))
        elif integrated:
            x = np.cumsum(e[:, 1:], axis=0)
        else:
            x = e[:, 1:]
        z, w = _build_design(y, x, case, T)
        stats[i] = _f_statistic(np.diff(y), z, w)
    return stats[np.isfinite(stats)]


def simulate_pss_bounds(
    k: int,
    case: int = 3,
    T: int = 1000,
    nsim: int = 40_000,
    levels: Sequence[float] = DEFAULT_LEVELS,
    seed: Optional[int] = None,
    return_draws: bool = False,
):
    """
    Generate the Pesaran, Shin and Smith bounds by simulating their DGP.

    Parameters
    ----------
    k : int
        Number of long-run forcing regressors. Not the number of level terms;
        see the module docstring.
    case : {1, 2, 3, 4, 5}
        Deterministic case. See :data:`CASE_LABELS`.
    T : int
        Sample size. ``T=1000`` reproduces the published asymptotic tables.
        Smaller values give the finite-sample bounds that Narayan (2004)
        tabulated for ``n = 30..80``.
    nsim : int
        Replications. The published tables use 40,000. Tail quantiles are the
        slowest to settle, so reduce this only for exploratory work.
    levels : sequence of float
        Significance levels.
    seed : int, optional
        Seed for reproducibility.
    return_draws : bool
        If ``True``, also return the two raw null distributions, which is what
        you want to plot a bracket figure.

    Returns
    -------
    pandas.DataFrame
        Indexed by level, with columns ``I(0)`` (lower bound) and ``I(1)``
        (upper bound). If ``return_draws`` is ``True``, returns
        ``(frame, {"I(0)": array, "I(1)": array})``.

    Notes
    -----
    Accuracy against the published table, at ``T=1000``, improves with ``nsim``.
    At 8,000 draws the lower bounds are typically within 0.03 and the upper
    bounds within 0.1; at the full 40,000 both tighten considerably. The test
    suite checks a set of published cells via :func:`pss_reference`.

    Examples
    --------
    >>> cv = simulate_pss_bounds(k=1, case=3, nsim=2000, seed=0)  # doctest: +SKIP
    >>> cv.loc[0.05]                                              # doctest: +SKIP
    I(0)    4.9...
    I(1)    5.8...
    """
    if int(k) < 0:
        raise ValueError(f"k must be non-negative; got {k!r}")
    if case not in CASE_LABELS:
        raise ValueError(f"case must be one of {sorted(CASE_LABELS)}; got {case!r}")
    if T < n_restrictions(k, case) + 3:
        raise ValueError(
            f"T={T} is too small for k={k}, case={case}: "
            f"needs at least {n_restrictions(k, case) + 3} observations"
        )

    seq = np.random.SeedSequence(seed)
    child = seq.spawn(2)
    draws = {
        "I(0)": _simulate_one_side(k, case, T, nsim, False, np.random.default_rng(child[0])),
        "I(1)": _simulate_one_side(k, case, T, nsim, True, np.random.default_rng(child[1])),
    }
    frame = pd.DataFrame(
        {tag: [float(np.quantile(d, 1 - lv)) for lv in levels] for tag, d in draws.items()},
        index=pd.Index(list(levels), name="level"),
    )
    frame.attrs.update({"k": int(k), "case": int(case), "T": int(T), "nsim": int(nsim)})
    if return_draws:
        return frame, draws
    return frame


# ---------------------------------------------------------------------------
# A small set of published cells, used only to validate the simulator.
# Source: Pesaran, Shin and Smith (2001), Tables CI(i)-CI(v), 10%/5%/1%.
# These are NOT used at runtime; see the module docstring.
# ---------------------------------------------------------------------------
_PSS_REFERENCE: Dict[Tuple[int, int], Dict[float, Tuple[float, float]]] = {
    (1, 1): {0.10: (2.44, 3.28), 0.05: (3.15, 4.11), 0.01: (4.81, 6.02)},
    (2, 1): {0.10: (2.17, 3.19), 0.05: (2.72, 3.83), 0.01: (3.88, 5.30)},
    (1, 2): {0.10: (3.02, 3.51), 0.05: (3.62, 4.16), 0.01: (4.94, 5.58)},
    (1, 3): {0.10: (4.04, 4.78), 0.05: (4.94, 5.73), 0.01: (6.84, 7.84)},
    (2, 3): {0.10: (3.17, 4.14), 0.05: (3.79, 4.85), 0.01: (5.15, 6.36)},
    (4, 3): {0.10: (2.45, 3.52), 0.05: (2.86, 4.01), 0.01: (3.74, 5.06)},
    (4, 4): {0.10: (2.68, 3.53), 0.05: (3.05, 3.97), 0.01: (3.81, 4.92)},
    (4, 5): {0.10: (3.03, 4.06), 0.05: (3.47, 4.57), 0.01: (4.40, 5.72)},
}


def pss_reference(k: int, case: int = 3) -> pd.DataFrame:
    """
    Published Pesaran, Shin and Smith bounds for a validated cell.

    Provided so the simulator can be checked against print, and so the
    "borrowed bound" comparison of the Monte Carlo tables can quote the exact
    number the paper quotes. Only a subset of cells is stored; use
    :func:`simulate_pss_bounds` for anything else.

    Raises
    ------
    KeyError
        If ``(k, case)`` is not one of the stored cells.
    """
    key = (int(k), int(case))
    if key not in _PSS_REFERENCE:
        raise KeyError(
            f"no published reference stored for k={k}, case={case}. "
            f"Available: {sorted(_PSS_REFERENCE)}. Use simulate_pss_bounds() instead."
        )
    cell = _PSS_REFERENCE[key]
    return pd.DataFrame(
        {"I(0)": [cell[lv][0] for lv in DEFAULT_LEVELS],
         "I(1)": [cell[lv][1] for lv in DEFAULT_LEVELS]},
        index=pd.Index(list(DEFAULT_LEVELS), name="level"),
    )


def statsmodels_offset(k: int = 1, case: int = 3, level: float = 0.05) -> pd.DataFrame:
    """
    Reproduce the ``statsmodels`` critical-value offset, for verification.

    ``statsmodels.tsa.ardl.UECMResults.bounds_test`` sets
    ``k = len(model.ardl_order)``, which is the number of forcing regressors
    **plus one**, and indexes a table that is itself indexed by the Pesaran,
    Shin and Smith ``k``. Every reported bound is therefore one row too far
    down, hence too small, and the test over-rejects.

    Returns
    -------
    pandas.DataFrame
        The correct bounds for ``k`` alongside the bounds ``statsmodels``
        actually reports, so the distortion can be inspected.
    """
    try:
        correct = pss_reference(k, case).loc[level]
    except KeyError:
        correct = simulate_pss_bounds(k, case, nsim=4000, seed=0).loc[level]
    try:
        shifted = pss_reference(k + 1, case).loc[level]
    except KeyError:
        shifted = simulate_pss_bounds(k + 1, case, nsim=4000, seed=0).loc[level]
    return pd.DataFrame(
        [correct.tolist(), shifted.tolist()],
        index=[f"correct (k={k})", f"statsmodels reports (k={k + 1} row)"],
        columns=["I(0)", "I(1)"],
    )
