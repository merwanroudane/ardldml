"""
Monte Carlo: the paper's data-generating processes and design grid.

Everything here follows Appendix B of the paper.

The core system
---------------
.. math::
    Y_t = D_t + u_t, \\qquad \\Delta D_t = v_t, \\qquad
    u_t = \\rho u_{t-1} + e_t

with :math:`\\rho = 1` the no-cointegration null and :math:`\\rho = 0.5` the
moderate alternative. The nuisance matrix :math:`W_t` has a fraction
``frac_i1`` of :math:`I(1)` random-walk columns and the remainder stationary
AR(1) with coefficient 0.5. In the common-trend designs a latent :math:`I(1)`
trend drives the :math:`I(1)` nuisance columns *and* loads on the core
regressor, which produces the severe near-collinearity case. A 100-observation
burn-in is discarded.

Endogeneity is introduced through
:math:`v_t = \\delta e_t + \\sqrt{1-\\delta^2}\\,\\xi_t`, so :math:`\\delta` is
the contemporaneous correlation between the marginal innovation of ``D`` and
the equation error. :math:`\\delta = 0` recovers the exogenous designs.

The design grid (Table 2)
-------------------------
========  ==============================  ==========  ============================
Design    Nuisance space                  k-tilde     Purpose
========  ==============================  ==========  ============================
A         none / low-dim stationary       = k         reproduce classical behaviour
B         high-dimensional I(0)           = k         residualisation helps
C         high-dimensional with I(1)      <= k        classical bounds distorted
D         cointegrated I(1) controls      < k         trend absorption reduces dim
E         weak signal, near unit root     varies      robustness, d = T
========  ==============================  ==========  ============================

Two kinds of critical value
---------------------------
* :func:`empirical_critical_value` simulates the null model directly and takes
  the empirical quantile of the statistic. This is *method-specific* and
  *infeasible on real data* -- it needs the true DGP. It exists to separate
  size from power, by holding the inference principle at its infeasible best so
  that any distortion seen is a property of the estimator rather than of the
  bootstrap approximation.
* The feasible critical value is the bootstrap of
  :func:`~ardldml.bootstrap.restricted_system_wild_bootstrap`, which is what
  :func:`run_design` reports alongside the borrowed classical bound.

A note on cost
--------------
A single ``run_design`` cell with ``R=1000`` and ``B=999`` is a million
statistic evaluations. The defaults here are much smaller so that the examples
run; scale up deliberately, and use ``n_jobs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .statistic import DMLBounds, DMLBoundsSpec, compute_statistic

__all__ = [
    "DESIGNS",
    "RHO_ALTERNATIVES",
    "default_d",
    "run_ultra_check",
    "run_robustness_grid",
    "DesignSpec",
    "simulate_design",
    "empirical_critical_value",
    "run_design",
    "run_endogeneity_grid",
]

BURN_IN = 100


@dataclass(frozen=True)
class DesignSpec:
    """One row of the Monte Carlo design grid."""

    name: str
    d: int
    frac_i1: float
    common_trend: bool
    description: str


#: The five designs of Table 2. The ``d`` recorded here is the value at
#: ``T = 120``; :func:`default_d` scales it with the sample size to reproduce
#: Table 3, and an explicit ``d=`` always wins.
DESIGNS: Dict[str, DesignSpec] = {
    "A": DesignSpec("A", 5, 0.0, False, "none / low-dimensional stationary"),
    "B": DesignSpec("B", 108, 0.0, False, "high-dimensional I(0)"),
    "C": DesignSpec("C", 108, 0.5, False, "high-dimensional with I(1) block"),
    "D": DesignSpec("D", 108, 0.5, True, "cointegrated I(1) controls"),
    "E": DesignSpec("E", 120, 0.5, True, "weak signal, near unit root, d = T"),
}

#: Alternatives of Section 7.5: weak, moderate and strong cointegration.
RHO_ALTERNATIVES: Dict[str, float] = {"weak": 0.85, "moderate": 0.50, "strong": 0.20}


def default_d(design: str, T: int) -> int:
    """
    The nuisance dimension the paper uses for a design at sample size ``T``.

    Table 3 reports ``d = 5`` for design A at both sample sizes, ``d = 108`` at
    ``T = 120`` and ``d = 180`` at ``T = 200`` for designs B, C and D -- that
    is, nine tenths of the sample -- and ``d = T`` for design E, which is the
    ``d = T`` stress case its row of Table 2 describes.

    Examples
    --------
    >>> [default_d(g, 120) for g in "ABCDE"]
    [5, 108, 108, 108, 120]
    >>> [default_d(g, 200) for g in "ABCDE"]
    [5, 180, 180, 180, 200]
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {sorted(DESIGNS)}; got {design!r}")
    if design == "A":
        return DESIGNS["A"].d
    if design == "E":
        return int(T)
    return int(round(0.9 * T))


def simulate_design(
    design: str = "C",
    T: int = 200,
    rho: float = 1.0,
    delta: float = 0.0,
    d: Optional[int] = None,
    seed: Optional[int] = None,
    burn_in: int = BURN_IN,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, List[str]]:
    """
    Draw one dataset from a design.

    Parameters
    ----------
    design : {"A", "B", "C", "D", "E"}
        Which row of the grid. See :data:`DESIGNS`.
    T : int
        Post-burn-in sample size.
    rho : float
        Persistence of the equilibrium error. ``1.0`` is the null, ``0.5`` the
        moderate alternative.
    delta : float
        Contemporaneous correlation between the marginal innovation of ``D``
        and the equation error. ``0`` is strong exogeneity.
    d : int, optional
        Number of nuisance controls. Defaults to the design's own ``d``.
    seed : int, optional
    burn_in : int
        Discarded initial observations.

    Returns
    -------
    (y, d_series, W, integrated_names)
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {sorted(DESIGNS)}; got {design!r}")
    spec = DESIGNS[design]
    d_ctl = int(default_d(design, T) if d is None else d)
    if not (0.0 <= abs(delta) <= 1.0):
        raise ValueError("delta must lie in [-1, 1]")

    rng = np.random.default_rng(seed)
    n = T + burn_in

    n_i1 = int(round(spec.frac_i1 * d_ctl))
    n_i0 = d_ctl - n_i1

    latent = np.cumsum(rng.standard_normal(n)) if spec.common_trend else np.zeros(n)

    cols: Dict[str, np.ndarray] = {}
    integrated: List[str] = []
    for j in range(n_i1):
        name = f"w1_{j}"
        cols[name] = np.cumsum(rng.standard_normal(n)) + (0.5 * latent if spec.common_trend else 0.0)
        integrated.append(name)
    for j in range(n_i0):
        s = np.zeros(n)
        eps = rng.standard_normal(n)
        for t in range(1, n):
            s[t] = 0.5 * s[t - 1] + eps[t]
        cols[f"w0_{j}"] = s

    e = rng.standard_normal(n)
    xi = rng.standard_normal(n)
    v = delta * e + np.sqrt(max(1.0 - delta**2, 0.0)) * xi

    D = np.cumsum(v)
    if spec.common_trend:
        D = D + 0.3 * latent

    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    Y = D + u

    idx = pd.RangeIndex(n)
    sl = slice(burn_in, None)
    return (
        pd.Series(Y, index=idx, name="Y").iloc[sl],
        pd.Series(D, index=idx, name="D").iloc[sl],
        pd.DataFrame(cols, index=idx).iloc[sl],
        integrated,
    )


def empirical_critical_value(
    design: str = "C",
    T: int = 200,
    R: int = 200,
    level: float = 0.05,
    d: Optional[int] = None,
    seed: Optional[int] = 20260625,
    n_jobs: int = 1,
    **spec_kwargs,
) -> Dict[str, object]:
    """
    Method-specific critical value from the simulated null distribution.

    Draws ``R`` datasets under :math:`\\rho = 1`, computes the statistic on
    each, and returns the empirical ``1 - level`` quantile.

    This is **infeasible on real data** -- it requires knowing the DGP. Its
    purpose is diagnostic: comparing size against this value isolates the
    finite-sample behaviour of the *estimator* from that of the bootstrap
    approximation.

    Returns
    -------
    dict
        ``crit``, ``draws``, ``R``, ``level``.
    """
    def one(r: int) -> float:
        y, dser, W, integ = simulate_design(
            design, T=T, rho=1.0, d=d, seed=None if seed is None else seed + r
        )
        spec = DMLBoundsSpec(integrated=integ, **spec_kwargs)
        try:
            return float(compute_statistic(y, dser, W, spec)["stat"])
        except Exception:  # pragma: no cover
            return np.nan

    if n_jobs != 1:
        from joblib import Parallel, delayed

        draws = np.asarray(Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in range(R)), dtype=float)
    else:
        draws = np.asarray([one(r) for r in range(R)], dtype=float)

    good = draws[np.isfinite(draws)]
    return {
        "crit": float(np.quantile(good, 1 - level)),
        "draws": good,
        "R": int(R),
        "level": float(level),
    }


def run_design(
    design: str = "C",
    T: int = 200,
    R: int = 100,
    B: int = 199,
    rho_null: float = 1.0,
    rho_alt: float = 0.5,
    delta: float = 0.0,
    d: Optional[int] = None,
    level: float = 0.05,
    borrowed_bound: float = 5.73,
    seed: Optional[int] = 20260625,
    n_jobs: int = 1,
    progress: bool = False,
    **spec_kwargs,
) -> pd.DataFrame:
    """
    One cell of the main Monte Carlo table.

    Reports, for the null and the alternative, the rejection rate against the
    borrowed classical bound and against the feasible bootstrap critical value.
    This is the comparison of the paper's Table 3: the borrowed bound
    over-rejects under integrated nuisance while the bootstrap holds size.

    Parameters
    ----------
    borrowed_bound : float
        The classical bound to compare against. The paper's reference value is
        5.73, the Pesaran-Shin-Smith case III, ``k=1``, 5% upper bound.

    Returns
    -------
    pandas.DataFrame
        One row per ``rho``, with ``rej_borrowed`` and ``rej_boot``.
    """
    # Appendix B: "design A uses the low-dimensional OLS projection".
    kwargs = dict(spec_kwargs)
    if design == "A":
        kwargs.setdefault("penalised", False)

    rows = []
    for rho, label in ((rho_null, "size"), (rho_alt, "power")):
        rej_b, rej_boot, n_ok = 0, 0, 0
        for r in range(R):
            s = None if seed is None else seed + 1000 * int(rho * 10) + r
            y, dser, W, integ = simulate_design(design, T=T, rho=rho, delta=delta, d=d, seed=s)
            try:
                res = (
                    DMLBounds(y, dser, W, integrated=integ, **kwargs)
                    .fit()
                    .bootstrap(B=B, level=level, seed=s)
                )
            except Exception:  # pragma: no cover
                continue
            if not np.isfinite(res.stat):
                continue
            n_ok += 1
            rej_b += int(res.stat > borrowed_bound)
            rej_boot += int(res.pvalue < level)
            if progress and (r + 1) % max(R // 5, 1) == 0:
                print(f"  {design} T={T} rho={rho}: {r + 1}/{R}", flush=True)
        rows.append(
            {
                "design": design,
                "T": T,
                "d": int(d if d is not None else default_d(design, T)),
                "rho": rho,
                "kind": label,
                "R_ok": n_ok,
                f"rej @ {borrowed_bound}": rej_b / max(n_ok, 1),
                "rej @ boot": rej_boot / max(n_ok, 1),
            }
        )
    return pd.DataFrame(rows)


def run_endogeneity_grid(
    deltas: Sequence[float] = (0.0, 0.4, 0.8),
    T: int = 200,
    R: int = 100,
    B: int = 199,
    rho: float = 1.0,
    design: str = "C",
    d: Optional[int] = 40,
    level: float = 0.05,
    seed: Optional[int] = 20260625,
    **spec_kwargs,
) -> pd.DataFrame:
    """
    Compare the system and fixed-regressor bootstrap schemes under endogeneity.

    Reproduces the structure of the paper's Table 5. Both schemes are evaluated
    on the *same* simulated datasets, so the columns differ only in the null
    law each simulates.

    Under :math:`\\delta = 0` the two coincide, which is the sanity check that
    the system regeneration introduces no distortion of its own.
    """
    rows = []
    for delta in deltas:
        rej = {"system": 0, "fixed": 0}
        n_ok = 0
        for r in range(R):
            s = None if seed is None else seed + r
            y, dser, W, integ = simulate_design(
                design, T=T, rho=rho, delta=delta, d=d, seed=s
            )
            try:
                base = DMLBounds(y, dser, W, integrated=integ, **spec_kwargs).fit()
                ok = True
                for scheme in ("system", "fixed"):
                    res = base.bootstrap(B=B, level=level, seed=s, scheme=scheme)
                    if res.pvalue is None:
                        ok = False
                        break
                    rej[scheme] += int(res.pvalue < level)
                if ok:
                    n_ok += 1
            except Exception:  # pragma: no cover
                continue
        rows.append(
            {
                "delta": delta,
                "T": T,
                "rho": rho,
                "R_ok": n_ok,
                "rej @ system": rej["system"] / max(n_ok, 1),
                "rej @ fixed": rej["fixed"] / max(n_ok, 1),
            }
        )
    return pd.DataFrame(rows)


def run_ultra_check(
    T: int = 100,
    d: int = 150,
    R: int = 40,
    seed: Optional[int] = 20260625,
    **spec_kwargs,
) -> pd.DataFrame:
    """
    Section 7.7: the ultra-high-dimensional implementability check.

    At :math:`T = 100`, :math:`d = 150` the empirical Gram matrix of the
    unpenalised conditional ECM is singular in every replication, so no
    statistic is defined and the classical benchmark is simply not
    implementable. The regularised procedure remains estimable.

    This is an *implementability* check, not a size or power comparison, so the
    statistic is summarised by its median and interquartile range across the
    draws where it exists rather than by a rejection rate.

    Returns
    -------
    pandas.DataFrame
        One row per method: the share of replications in which it is
        implementable, and the median and IQR of the statistic.
    """
    rows: Dict[str, List[float]] = {"Unpenalised ECM": [], "DML-Bounds (h-block)": []}
    for r in range(R):
        s = None if seed is None else seed + r
        # Design C's mixed I(0)/I(1) structure, at an explicit dimension.
        y, dser, W, integ = simulate_design("C", T=T, rho=1.0, d=d, seed=s)
        for label, penalised in (("Unpenalised ECM", False), ("DML-Bounds (h-block)", True)):
            kw = dict(spec_kwargs)
            kw.setdefault("lags", 1)
            kw.setdefault("n_blocks", 4)
            kw.setdefault("buffer", 2)
            kw["penalised"] = penalised
            try:
                res = DMLBounds(y, dser, W, integrated=integ, **kw).fit()
                val = res.stat if (res.estimable and np.isfinite(res.stat)) else np.nan
            except Exception:
                val = np.nan
            rows[label].append(val)

    out = []
    for label, vals in rows.items():
        arr = np.asarray(vals, dtype=float)
        ok = arr[np.isfinite(arr)]
        share = ok.size / max(arr.size, 1)
        if ok.size:
            q1, med, q3 = np.percentile(ok, [25, 50, 75])
            stat = f"{med:.2f} ({q3 - q1:.2f})"
        else:
            stat = "not defined"
        out.append(
            {
                "method": label,
                "implementable across draws": f"{share:.1%}",
                "statistic (median, IQR)": stat,
            }
        )
    frame = pd.DataFrame(out)
    frame.attrs["note"] = f"T={T}, d={d}, R={R}, mixed I(0)/I(1) nuisance"
    return frame


def run_robustness_grid(
    T_grid: Sequence[int] = (100, 250, 500, 1000),
    rho_grid: Sequence[float] = (0.85, 0.50, 0.20),
    penalties: Sequence[str] = ("low", "medium", "high"),
    methods: Sequence[str] = ("unpenalised", "plain", "adaptive"),
    d: int = 40,
    R: int = 50,
    level: float = 0.05,
    mixed: bool = False,
    seed: Optional[int] = 20260625,
    **spec_kwargs,
) -> pd.DataFrame:
    """
    Sections 7.5 and 7.6: fixed-dimension sensitivity under *oracle* critical
    values.

    Complements the main grid from the opposite direction. It fixes a moderate
    nuisance dimension (40 variables) and traces size and power across sample
    sizes, penalties and estimators using **method-specific empirical**
    critical values rather than the bootstrap. Holding the inference principle
    at its infeasible best isolates the finite-sample behaviour of the
    estimator from that of the bootstrap approximation.

    Parameters
    ----------
    rho_grid : sequence of float
        The alternatives. :data:`RHO_ALTERNATIVES` names them: weak (0.85),
        moderate (0.50) and strong (0.20) cointegration. The null,
        ``rho = 1``, is always simulated to obtain the critical value.
    penalties : sequence of str
        The three choices of Section 7.5 -- Low is :math:`\\lambda_{\\min}`,
        Medium the geometric midpoint, High the one-standard-error rule.
    methods : sequence of str
        ``"unpenalised"`` (the conditional ECM benchmark), ``"plain"``
        (:math:`\\ell_1`) and ``"adaptive"`` (DML-Bounds).
    mixed : bool
        ``False`` gives Section 7.5's purely stationary nuisance; ``True``
        gives Section 7.6's mixed :math:`I(0)`/:math:`I(1)` design with a
        latent common trend.

    Returns
    -------
    pandas.DataFrame
        Empirical size and size-adjusted power, plus ``cv95``, the
        method-specific empirical critical value. That column is the
        informative one: under mixed integrated nuisance it rises far above the
        borrowed bound and keeps rising with the sample, which is the
        finite-sample counterpart of the inference problem.

    Notes
    -----
    Size-adjusted power recentres a poorly behaved finite-sample statistic
    using a critical value that is unavailable in practice. The unpenalised
    benchmark looks competitive here only for that reason; its operational
    problem is that the classical tables get applied directly.
    """
    design = "D" if mixed else "B"
    rows = []

    for T in T_grid:
        for method in methods:
            for pen in penalties:
                kw = dict(spec_kwargs)
                kw.setdefault("lags", 2)
                kw.setdefault("n_blocks", 5)
                kw.setdefault("buffer", 3)
                if method == "unpenalised":
                    kw["penalised"] = False
                else:
                    kw["penalised"] = True
                    kw["adaptive"] = method == "adaptive"
                    kw["penalty"] = pen

                null_stats = []
                for r in range(R):
                    s = None if seed is None else seed + r
                    y, dser, W, integ = simulate_design(design, T=T, rho=1.0, d=d, seed=s)
                    try:
                        res = DMLBounds(y, dser, W, integrated=integ, **kw).fit()
                        null_stats.append(
                            res.stat if (res.estimable and np.isfinite(res.stat)) else np.nan
                        )
                    except Exception:
                        null_stats.append(np.nan)
                null = np.asarray(null_stats, dtype=float)
                good = null[np.isfinite(null)]

                if good.size == 0:
                    rows.append(
                        {"T": T, "method": method, "penalty": pen,
                         "cv95": np.nan, "implementable": 0.0, "size": np.nan}
                    )
                    if method == "unpenalised":
                        break
                    continue

                cv95 = float(np.quantile(good, 1 - level))
                row = {
                    "T": T,
                    "method": method,
                    "penalty": pen if method != "unpenalised" else "-",
                    "cv95": round(cv95, 3),
                    "implementable": round(good.size / null.size, 3),
                    "size": round(float(np.mean(good > cv95)), 3),
                }
                for rho in rho_grid:
                    rej, n_ok = 0, 0
                    for r in range(R):
                        s = None if seed is None else seed + 5000 + r
                        y, dser, W, integ = simulate_design(design, T=T, rho=rho, d=d, seed=s)
                        try:
                            res = DMLBounds(y, dser, W, integrated=integ, **kw).fit()
                            if res.estimable and np.isfinite(res.stat):
                                n_ok += 1
                                rej += int(res.stat > cv95)
                        except Exception:
                            continue
                    row[f"power rho={rho}"] = round(rej / max(n_ok, 1), 3)
                rows.append(row)

                if method == "unpenalised":
                    break  # the penalty is irrelevant without penalisation

    return pd.DataFrame(rows)
