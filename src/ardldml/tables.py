"""
Publication-quality tables.

Every builder returns a :class:`pandas.DataFrame` you can inspect or reshape,
and :func:`to_latex` turns any of them into a ``booktabs`` table with a
caption, a label and a notes block -- the format journals expect.

Conventions
-----------
* Significance stars follow the usual econometrics convention: ``***`` p < 0.01,
  ``**`` p < 0.05, ``*`` p < 0.10.
* Standard errors go in parentheses beneath the coefficient when
  ``se_below=True``.
* Numbers are formatted once, as strings, so LaTeX and the console agree.
* Bounds tests get a **two-sided** star convention: because the decision
  depends on both bounds, :func:`bounds_stars` marks significance against the
  lower and upper bound separately, and a conclusive rejection requires both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "stars",
    "bounds_stars",
    "result_table",
    "regime_table",
    "montecarlo_table",
    "diagnostic_table",
    "critical_value_table",
    "to_latex",
    "to_markdown",
]


def stars(p: Optional[float], thresholds: Sequence[float] = (0.01, 0.05, 0.10)) -> str:
    """
    Significance stars for a p-value.

    Examples
    --------
    >>> stars(0.004), stars(0.03), stars(0.08), stars(0.5)
    ('***', '**', '*', '')
    """
    if p is None or not np.isfinite(p):
        return ""
    a, b, c = thresholds
    if p < a:
        return "***"
    if p < b:
        return "**"
    if p < c:
        return "*"
    return ""


def bounds_stars(stat: float, lower: float, upper: float) -> str:
    """
    Mark a statistic against a pair of bounds.

    Returns ``"reject"`` when the statistic exceeds the upper bound,
    ``"inconclusive"`` when it lies between, and ``""`` when it falls below the
    lower bound. Using a single star would hide the inconclusive region, which
    is the whole point of a bounds test.
    """
    if not np.isfinite(stat):
        return ""
    if stat > upper:
        return "reject"
    if stat > lower:
        return "inconclusive"
    return ""


def _fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{nd}f}"


def result_table(
    results: Union[object, Sequence[object]],
    labels: Optional[Sequence[str]] = None,
    nd: int = 3,
    se_below: bool = True,
) -> pd.DataFrame:
    """
    Stack one or more fitted results into an estimation table.

    Parameters
    ----------
    results : DMLBoundsResults or sequence of them
    labels : sequence of str, optional
        Column headers. Defaults to ``(1), (2), ...``.
    se_below : bool
        Put the standard error of the long-run coefficient in parentheses on
        its own row.
    """
    if not isinstance(results, (list, tuple)):
        results = [results]
    labels = list(labels or [f"({i + 1})" for i in range(len(results))])

    rows: Dict[str, list] = {}

    def put(key, vals):
        rows[key] = vals

    put("n", [_fmt(r.nobs) for r in results])
    put("controls", [_fmt(r.W.shape[1]) for r in results])
    put("F", [_fmt(r.stat, nd) for r in results])
    put("bootstrap cv (95%)", [_fmt(r.critical_value, nd) for r in results])
    put("bootstrap p", [
        _fmt(r.pvalue, nd) + stars(r.pvalue) for r in results
    ])
    put("verdict", [r.decision() for r in results])
    put("alpha (speed of adj.)", [_fmt(r.alpha, nd) for r in results])
    put("theta (long run)", [_fmt(r.theta, nd) for r in results])
    if se_below:
        put(" ", [f"({_fmt(r.theta_se, nd)})" for r in results])

    return pd.DataFrame(rows, index=labels).T


def regime_table(
    per_regime: Dict[str, Dict[str, object]],
    nd: int = 3,
) -> pd.DataFrame:
    """
    The paper's Table 11 layout: regimes x conditioning sets.

    Parameters
    ----------
    per_regime : dict
        ``{regime_label: {"full": result, "reduced": result}}``.

    Returns
    -------
    pandas.DataFrame
        Long format, two rows per regime.
    """
    rows = []
    for regime, fits in per_regime.items():
        for setname, r in fits.items():
            rows.append(
                {
                    "regime": regime,
                    "conditioning set": setname,
                    "n": r.nobs,
                    "controls": r.W.shape[1],
                    "F": round(float(r.stat), nd),
                    "bootstrap cv .95": None if r.critical_value is None else round(float(r.critical_value), nd),
                    "p": None if r.pvalue is None else round(float(r.pvalue), nd),
                    "verdict": r.decision(),
                    "theta": round(float(r.theta), nd),
                    "se": round(float(r.theta_se), nd),
                }
            )
    return pd.DataFrame(rows)


def montecarlo_table(frames: Sequence[pd.DataFrame], nd: int = 3) -> pd.DataFrame:
    """
    Stack :func:`~ardldml.simulate.run_design` cells into the Table 3 layout.

    One row per design and sample size, with size and power side by side.
    """
    long = pd.concat(list(frames), ignore_index=True)
    size = long[long["kind"] == "size"].set_index(["design", "T", "d"])
    power = long[long["kind"] == "power"].set_index(["design", "T", "d"])
    borrowed = [c for c in long.columns if c.startswith("rej @ ") and c != "rej @ boot"][0]

    out = pd.DataFrame(
        {
            f"size {borrowed}": size[borrowed],
            "size @ boot": size["rej @ boot"],
            "power @ boot": power["rej @ boot"],
        }
    ).reset_index()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(nd)
    return out.sort_values(["T", "design"]).reset_index(drop=True)


def diagnostic_table(diag, nd: int = 3) -> pd.DataFrame:
    """Format a :class:`~ardldml.diagnostics.TrendAbsorption` for printing."""
    fr = diag.to_frame()
    for c in fr.columns:
        if fr[c].dtype.kind == "f":
            fr[c] = fr[c].round(nd)
    return fr


def critical_value_table(
    k_values: Sequence[int] = (1, 2, 3, 4),
    case: int = 3,
    T: int = 1000,
    nsim: int = 20_000,
    level: float = 0.05,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Generate a bounds table by simulation, for reference or validation.

    This is the honest way to print a bounds table: it says what ``T`` and how
    many replications produced it, and it can be regenerated.
    """
    from .critvals import simulate_pss_bounds

    rows = []
    for k in k_values:
        cv = simulate_pss_bounds(k=k, case=case, T=T, nsim=nsim, seed=seed)
        rows.append({"k": k, "I(0)": cv.loc[level, "I(0)"], "I(1)": cv.loc[level, "I(1)"]})
    out = pd.DataFrame(rows).set_index("k").round(3)
    out.attrs["note"] = f"case {case}, T={T}, nsim={nsim}, level={level:.0%}"
    return out


def to_markdown(frame: pd.DataFrame, **kw) -> str:
    """Markdown rendering, falling back to a fixed-width string."""
    try:
        return frame.to_markdown(**kw)
    except ImportError:  # pragma: no cover - tabulate absent
        return frame.to_string()


def to_latex(
    frame: pd.DataFrame,
    caption: str = "",
    label: str = "",
    notes: str = "",
    path: Optional[Union[str, Path]] = None,
    index: bool = True,
    column_format: Optional[str] = None,
) -> str:
    """
    Render as a ``booktabs`` table with caption, label and notes.

    Requires ``\\usepackage{booktabs}`` and, for the notes block,
    ``\\usepackage{threeparttable}`` in the preamble.
    """
    ncol = frame.shape[1] + (1 if index else 0)
    column_format = column_format or ("l" + "r" * (ncol - 1))

    body = frame.to_latex(
        index=index,
        escape=True,
        column_format=column_format,
        bold_rows=False,
    )
    body = (
        body.replace("\\toprule", "\\toprule")
        .replace("\\midrule", "\\midrule")
        .replace("\\bottomrule", "\\bottomrule")
    )

    parts = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        parts.append(f"\\caption{{{caption}}}")
    if label:
        parts.append(f"\\label{{{label}}}")
    if notes:
        parts += ["\\begin{threeparttable}", body,
                  "\\begin{tablenotes}[flushleft]\\small",
                  f"\\item {notes}", "\\end{tablenotes}", "\\end{threeparttable}"]
    else:
        parts.append(body)
    parts.append("\\end{table}")
    out = "\n".join(parts)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(out, encoding="utf-8")
    return out
