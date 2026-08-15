"""
The trend-absorption diagnostic.

This is the most important thing in the package to actually run, and the
easiest to skip.

The problem it addresses
------------------------
Residualisation is valid only when the controls remove *nuisance* stochastic
trends, not the equilibrium relation being tested. If a control is itself part
of the equilibrium system, partialling it out absorbs the very trend the test
is meant to detect, driving the effective integrated count toward zero and
destroying the relation rather than cleaning it. The estimand changes: a
non-rejection then reflects over-absorption, not the absence of a long-run
relationship.

That requirement -- the paper's Assumption 5, that the nuisance space "does not
span the cointegrating relation of interest" -- cannot be verified directly.
The diagnostic is the practical substitute.

Definition 2
------------
Let :math:`p_{ad}` and :math:`p_{ols}` be the bootstrap p-values under the
adaptive and the unpenalised :math:`m_Z` projection, and :math:`p_{full}` and
:math:`p_{red}` the bootstrap p-values under the full nuisance set and under a
reduced set omitting the controls most likely to be cointegrated with the
tested relation. The diagnostic is the pair of gaps

.. math::
    \\Delta_m = p_{ols} - p_{ad}, \\qquad \\Delta_W = p_{full} - p_{red}

read together with the stability of the long-run coefficient across the four
fits.

How to read it (Remark 9)
-------------------------
* **A large positive** :math:`\\Delta_W` -- a verdict that rejects under the
  reduced set but not under the full set -- together with a long-run
  coefficient that is more stable and more sharply estimated under the reduced
  or adaptive fit, is evidence that the nuisance space is absorbing part of the
  tested relation. Assumption 5 is locally violated and **the reduced-set
  verdict is the more credible one**.
* **Concordant verdicts** across all four fits indicate the conclusion is not
  an artefact of over-absorption.
* The adaptive projection is the default, because vanilla :math:`\\ell_1`
  over-selects integrated regressors and thereby induces spurious absorption.
  The unpenalised projection is retained only to form :math:`\\Delta_m`.

This is a hypothesis-generating device, not a formal test. It has no size and
no power; it tells you where to look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .statistic import DMLBounds

__all__ = ["TrendAbsorption", "trend_absorption", "penalty_sensitivity"]


@dataclass
class TrendAbsorption:
    """
    Result of the four-fit trend-absorption contrast.

    Attributes
    ----------
    fits : dict
        The four fitted :class:`~ardldml.statistic.DMLBoundsResults`, keyed
        ``"full_adaptive"``, ``"full_ols"``, ``"reduced_adaptive"``,
        ``"reduced_ols"``.
    dropped : list of str
        Controls omitted from the reduced set.
    delta_m, delta_W : float
        The two gaps of Definition 2.
    """

    fits: Dict[str, object]
    dropped: List[str]
    delta_m: float
    delta_W: float

    def to_frame(self) -> pd.DataFrame:
        """One row per fit: statistic, critical value, p-value, theta."""
        rows = []
        for key, r in self.fits.items():
            controls, proj = key.split("_")
            rows.append(
                {
                    "controls": controls,
                    "m_Z projection": proj,
                    "n_controls": r.W.shape[1],
                    "F": r.stat,
                    "boot_cv95": r.critical_value,
                    "boot_p": r.pvalue,
                    "verdict": r.decision(),
                    "alpha": r.alpha,
                    "theta": r.theta,
                    "theta_se": r.theta_se,
                }
            )
        return pd.DataFrame(rows)

    @property
    def theta_spread(self) -> float:
        """Range of the long-run coefficient across the four fits."""
        vals = [r.theta for r in self.fits.values() if np.isfinite(r.theta)]
        return float(max(vals) - min(vals)) if vals else float("nan")

    def verdict(self, level: float = 0.05) -> str:
        """
        Plain-language reading, following Remark 9.

        Deliberately conservative: it flags a *possible* violation rather than
        asserting one, because the contrast cannot separate over-absorption
        from ordinary specification sensitivity.
        """
        full = self.fits["full_adaptive"]
        red = self.fits["reduced_adaptive"]
        if full.pvalue is None or red.pvalue is None:
            return "no bootstrap run"

        rejects_reduced = red.pvalue < level
        rejects_full = full.pvalue < level

        if rejects_reduced and not rejects_full and self.delta_W > 0:
            sharper = red.theta_se < full.theta_se
            msg = (
                f"Possible over-absorption. The reduced set rejects "
                f"(p={red.pvalue:.3f}) where the full set does not "
                f"(p={full.pvalue:.3f}); Delta_W = {self.delta_W:+.3f}."
            )
            if sharper:
                msg += (
                    f" The long-run coefficient is also more sharply estimated under the "
                    f"reduced set (se {red.theta_se:.3f} vs {full.theta_se:.3f}), which "
                    f"strengthens the reading. The reduced-set verdict is the more credible."
                )
            else:
                msg += (
                    " The long-run coefficient is not more sharply estimated under the "
                    "reduced set, so ordinary specification sensitivity cannot be ruled out."
                )
            return msg
        if rejects_reduced == rejects_full:
            return (
                f"Concordant verdicts across control sets "
                f"(full p={full.pvalue:.3f}, reduced p={red.pvalue:.3f}). "
                f"The conclusion does not appear to be an artefact of over-absorption."
            )
        return (
            f"Discordant, but not in the over-absorption direction "
            f"(full p={full.pvalue:.3f}, reduced p={red.pvalue:.3f}). "
            f"Treat both verdicts as fragile."
        )

    def summary(self, level: float = 0.05) -> str:
        lines = [
            "Trend-absorption diagnostic (Definition 2)",
            f"dropped from reduced set: {', '.join(self.dropped) or '(none)'}",
            "",
            self.to_frame().to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            ),
            "",
            f"Delta_m = p_ols  - p_ad  = {self.delta_m:+.4f}",
            f"Delta_W = p_full - p_red = {self.delta_W:+.4f}",
            f"theta spread across fits = {self.theta_spread:.4f}",
            "",
            self.verdict(level),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<TrendAbsorption delta_W={self.delta_W:+.3f} delta_m={self.delta_m:+.3f}>"


def trend_absorption(
    y: pd.Series,
    d: pd.Series,
    W: pd.DataFrame,
    drop: Sequence[str],
    B: int = 999,
    seed: Optional[int] = None,
    progress: bool = False,
    **spec_kwargs,
) -> TrendAbsorption:
    """
    Run the four-fit contrast of Definition 2.

    Parameters
    ----------
    y, d, W : pandas.Series, pandas.Series, pandas.DataFrame
        Outcome, focal regressor and controls, all in levels.
    drop : sequence of str
        Controls to omit from the reduced set. Choose these on *economic*
        grounds: the ones most likely to be cointegrated with the tested
        relation itself. In the paper's pass-through application these are
        money and the oil price; in the consumption-income appendix they are
        the nominal aggregates M2 and CPI.
    B : int
        Bootstrap replications per fit. Four bootstraps are run, so the total
        cost is roughly ``4 * B`` statistic evaluations.
    seed : int, optional
        Base seed. Each fit is offset so the four bootstraps are independent.
    progress : bool
        Print which fit is running.
    **spec_kwargs
        Passed to :class:`~ardldml.statistic.DMLBoundsSpec`. ``adaptive`` is
        overridden per fit.

    Returns
    -------
    TrendAbsorption
    """
    drop = [c for c in drop if c in W.columns]
    W_red = W.drop(columns=drop)

    integrated = spec_kwargs.pop("integrated", None)
    integrated_red = None
    if integrated is not None:
        integrated = list(integrated)
        integrated_red = [c for c in integrated if c not in drop]

    plan = [
        ("full_adaptive", W, True, integrated, 0),
        ("full_ols", W, False, integrated, 1),
        ("reduced_adaptive", W_red, True, integrated_red, 2),
        ("reduced_ols", W_red, False, integrated_red, 3),
    ]

    fits: Dict[str, object] = {}
    for key, Wk, adaptive, integ, offset in plan:
        if progress:
            print(f"  fitting {key} ({Wk.shape[1]} controls)...", flush=True)
        res = (
            DMLBounds(y, d, Wk, adaptive=adaptive, integrated=integ, **spec_kwargs)
            .fit()
            .bootstrap(B=B, seed=None if seed is None else seed + 10_000 * offset)
        )
        fits[key] = res

    delta_m = float(fits["full_ols"].pvalue - fits["full_adaptive"].pvalue)
    delta_W = float(fits["full_adaptive"].pvalue - fits["reduced_adaptive"].pvalue)
    return TrendAbsorption(fits=fits, dropped=list(drop), delta_m=delta_m, delta_W=delta_W)


def penalty_sensitivity(
    y: pd.Series,
    d: pd.Series,
    W: pd.DataFrame,
    c_grid: Sequence[float] = (1.1, 0.75, 0.5, 0.25),
    lags_grid: Sequence[int] = (4,),
    projections: Sequence[str] = ("adaptive", "ols"),
    B: Optional[int] = None,
    seed: Optional[int] = None,
    **spec_kwargs,
) -> pd.DataFrame:
    """
    Sweep the penalty, the lag order and the ``m_Z`` projection.

    The paper's own robustness tables vary all three, and its Table 14 shows a
    verdict rejecting at one penalty and not at another. Reporting a single
    cell from this grid is specification search; reporting the grid is the
    method.

    The column to watch is ``n_selected_Z``, the number of control **levels**
    the ``m_Z`` projection retained. It is the empirical counterpart of the
    effective integrated count:

    * ``0`` means nothing was absorbed, so the test sits at the classical
      ``k-tilde = k`` corner and orthogonalisation did nothing;
    * a large value means heavy absorption, and the long-run coefficient
      should be checked for instability.

    A long-run coefficient that changes sign across this grid is a warning that
    the conditioning set, not the data, is driving the answer.

    Parameters
    ----------
    c_grid : sequence of float
        Constants in the plug-in penalty. ``1.1`` is the paper's default; lower
        values select more aggressively.
    lags_grid : sequence of int
        Short-run lag orders to try.
    projections : sequence of str
        Any of ``"adaptive"``, ``"plain"``, ``"ols"``.
    B : int, optional
        Bootstrap draws per cell. If ``None``, no bootstrap is run and only the
        statistic and coefficients are reported, which is much faster.

    Returns
    -------
    pandas.DataFrame
        One row per cell.
    """
    rows = []
    for lags in lags_grid:
        for proj in projections:
            for c in c_grid:
                kw = dict(spec_kwargs)
                kw["lags"] = lags
                if proj == "ols":
                    kw["penalised"] = False
                else:
                    kw["penalised"] = True
                    kw["adaptive"] = proj == "adaptive"
                    kw["c"] = c
                try:
                    res = DMLBounds(y, d, W, **kw).fit()
                    if B:
                        res = res.bootstrap(B=B, seed=seed)
                    rows.append(
                        {
                            "lags": lags,
                            "m_Z projection": proj,
                            "c": None if proj == "ols" else c,
                            "n_selected_Z": int(res.first_stage.supports["Z"].sum()),
                            "F": res.stat,
                            "boot_cv95": res.critical_value,
                            "boot_p": res.pvalue,
                            "alpha": res.alpha,
                            "theta": res.theta,
                            "theta_se": res.theta_se,
                        }
                    )
                except Exception as exc:  # pragma: no cover - degenerate cell
                    rows.append(
                        {"lags": lags, "m_Z projection": proj, "c": c, "error": str(exc)[:60]}
                    )
                if proj == "ols":
                    break  # the penalty is irrelevant without penalisation
    out = pd.DataFrame(rows)
    if "theta" in out:
        finite = out["theta"].dropna()
        out.attrs["theta_sign_flips"] = bool(
            len(finite) and (finite.min() < 0 < finite.max())
        )
    return out
