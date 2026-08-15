"""
Publication-quality figures.

Every function returns a :class:`matplotlib.figure.Figure`, applies the active
style (:func:`ardldml.style.apply_active_style`, journal unless you switched to
the sunny one) unless ``style=False``, and accepts an ``ax`` so panels can be
composed.

The set mirrors the figures the paper actually uses:

* :func:`plot_bracket` -- the trend-absorption bracket, its Figure 1: where the
  null sits as a function of the effective integrated count.
* :func:`plot_size_comparison` -- its Figure 2: the borrowed bound over-rejects
  under integrated nuisance while the bootstrap restores size.
* :func:`plot_diagnostic` -- its Figure 3: the full-versus-reduced contrast.
* :func:`plot_bootstrap_null` -- the simulated null distribution with the
  observed statistic and critical value marked. This is the figure to show when
  someone asks why you did not use a table.
* :func:`plot_block_structure` -- what h-block cross-fitting did to the sample.
* :func:`plot_series` -- the data.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .style import COLORS, apply_active_style, lighten

__all__ = [
    "plot_bracket",
    "plot_bootstrap_null",
    "plot_size_comparison",
    "plot_diagnostic",
    "plot_block_structure",
    "plot_series",
    "plot_regimes",
]


def _fig(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def plot_bracket(
    k: int = 10,
    lower: float = 4.94,
    upper: float = 5.73,
    k_tilde: Optional[float] = None,
    ax=None,
    style: bool = True,
    figsize=(6.4, 4.0),
):
    """
    The trend-absorption bracket.

    As the effective integrated count :math:`\\tilde k` falls from ``k`` (no
    absorption) to 0 (full absorption), the limiting null moves from the
    :math:`I(1)` endpoint to the :math:`I(0)` endpoint. Classical bounds testing
    is the right-hand endpoint; high-dimensional residualisation can move the
    relevant null anywhere inside.

    Parameters
    ----------
    lower, upper : float
        The bracket endpoints. Defaults are the Pesaran-Shin-Smith case III,
        ``k=1``, 5% values the paper uses as its reference.
    k_tilde : float, optional
        Mark a particular effective integrated count.

    Notes
    -----
    The interpolation between endpoints is illustrative, not a computed limit.
    The paper's own figure is drawn the same way: the point is *where the null
    sits*, not the precise shape of the path.
    """
    if style:
        apply_active_style()
    fig, ax = _fig(ax, figsize)

    xs = np.arange(0, k + 1)
    ys = lower + (upper - lower) * xs / max(k, 1)
    ax.plot(xs, ys, "-o", color=COLORS["bootstrap"], label="limiting null, 95th pct")
    ax.axhline(lower, ls="--", lw=1.0, color=COLORS["i0"])
    ax.axhline(upper, ls="--", lw=1.0, color=COLORS["i1"])
    ax.annotate(f"PSS lower  I(0) = {lower:.2f}", (0.02, lower), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=8, color=COLORS["i0"])
    ax.annotate(f"PSS upper  I(1) = {upper:.2f}", (0.02, upper), xycoords=("axes fraction", "data"),
                va="top", fontsize=8, color=COLORS["i1"])
    ax.fill_between(xs, lower, ys, color=lighten(COLORS["bootstrap"], 0.85), zorder=0)

    if k_tilde is not None:
        yv = lower + (upper - lower) * float(k_tilde) / max(k, 1)
        ax.plot([k_tilde], [yv], "o", ms=8, mfc="white",
                mec=COLORS["observed"], mew=1.4, zorder=5)
        ax.annotate(f"  k-tilde = {k_tilde:g}", (k_tilde, yv), fontsize=8, va="center")

    ax.annotate("full absorption\nk-tilde = 0", (0, lower), textcoords="offset points",
                xytext=(6, 14), fontsize=8, color=COLORS["null"])
    ax.annotate("no absorption\nk-tilde = k", (k, upper), textcoords="offset points",
                xytext=(-10, -22), fontsize=8, ha="right", color=COLORS["null"])
    ax.set_xlabel("effective integrated count  k-tilde   (0 = trends absorbed, k = none absorbed)")
    ax.set_ylabel("limiting null, 95th percentile")
    ax.set_title("The trend-absorption bracket: where the null sits depends on k-tilde")
    fig.tight_layout()
    return fig


def plot_bootstrap_null(
    result,
    ax=None,
    style: bool = True,
    bins: int = 40,
    borrowed: Optional[float] = 5.73,
    figsize=(6.4, 4.0),
):
    """
    The simulated null distribution, with the observed statistic marked.

    Parameters
    ----------
    result : DMLBoundsResults
        A fitted result with ``.bootstrap()`` already called.
    borrowed : float, optional
        Draw the borrowed classical bound for comparison. Seeing it sit far
        from the simulated critical value is the whole argument for not using
        a table.
    """
    if result.boot is None:
        raise ValueError("call .bootstrap() before plotting the null distribution")
    if style:
        apply_active_style()
    fig, ax = _fig(ax, figsize)

    draws = result.boot["draws"]
    ax.hist(draws, bins=bins, color=lighten(COLORS["bootstrap"], 0.55),
            edgecolor=COLORS["bootstrap"], linewidth=0.6, label="bootstrap null")
    ax.axvline(result.critical_value, color=COLORS["bootstrap"], lw=1.6,
               label=f"bootstrap cv 95% = {result.critical_value:.2f}")
    ax.axvline(result.stat, color=COLORS["observed"], lw=1.8, ls="-",
               label=f"observed F = {result.stat:.2f}")
    if borrowed is not None:
        ax.axvline(borrowed, color=COLORS["borrowed"], lw=1.4, ls="--",
                   label=f"borrowed bound = {borrowed:.2f}")

    ax.set_xlabel("DML-Bounds statistic")
    ax.set_ylabel("frequency")
    ax.set_title(
        f"Restricted system wild bootstrap  (B = {result.boot['B']}, "
        f"p = {result.pvalue:.3f})"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_size_comparison(
    frame: pd.DataFrame,
    design_col: str = "design",
    borrowed_col: Optional[str] = None,
    boot_col: str = "rej @ boot",
    nominal: float = 0.05,
    ax=None,
    style: bool = True,
    figsize=(6.8, 4.0),
    labels: Optional[Dict[str, str]] = None,
):
    """
    Empirical size: borrowed bound against bootstrap, by design.

    Reproduces the paper's Figure 2. Feed it the ``kind == "size"`` rows of
    :func:`~ardldml.simulate.run_design` stacked across designs.
    """
    if style:
        apply_active_style()
    fig, ax = _fig(ax, figsize)

    if borrowed_col is None:
        cands = [c for c in frame.columns if c.startswith("rej @ ") and c != boot_col]
        if not cands:
            raise ValueError("could not infer the borrowed-bound column; pass borrowed_col")
        borrowed_col = cands[0]

    designs = frame[design_col].astype(str).tolist()
    x = np.arange(len(designs))
    w = 0.38
    ax.bar(x - w / 2, frame[borrowed_col], w, label=f"rejection @ {borrowed_col.split('@')[-1].strip()}",
           color=COLORS["borrowed"])
    ax.bar(x + w / 2, frame[boot_col], w, label="rejection @ bootstrap cv",
           color=COLORS["bootstrap"])
    ax.axhline(nominal, ls="--", lw=1.2, color=COLORS["null"])
    ax.annotate(f"nominal {nominal:.0%}", (0.995, nominal), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color=COLORS["null"])

    ax.set_xticks(x)
    ax.set_xticklabels([(labels or {}).get(d, d) for d in designs])
    ax.set_ylabel("empirical size (null)")
    ax.set_title("Borrowed bound over-rejects under integrated nuisance;\nbootstrap restores size")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_diagnostic(
    frames: Dict[str, pd.DataFrame],
    x: str = "p",
    level: float = 0.05,
    ax=None,
    style: bool = True,
    figsize=(6.4, 4.0),
    xlabel: str = "specification",
):
    """
    Full-versus-reduced bootstrap p-values across specifications.

    Reproduces the paper's Figure 3. ``frames`` maps a label ("full controls",
    "drop M2, oil") to a frame with an index to plot against and a column of
    bootstrap p-values.
    """
    if style:
        apply_active_style()
    fig, ax = _fig(ax, figsize)

    palette = [COLORS["full"], COLORS["reduced"], COLORS["borrowed"], COLORS["null"]]
    for i, (label, fr) in enumerate(frames.items()):
        ax.plot(fr.index, fr[x], "-o", color=palette[i % len(palette)], label=label)
    ax.axhline(level, ls="--", lw=1.2, color=COLORS["borrowed"])
    ax.annotate(f"{level:.0%}", (0.995, level), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color=COLORS["borrowed"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("bootstrap p-value")
    ax.set_ylim(0, 1)
    ax.set_title("Trend-absorption diagnostic:\ndoes dropping trend-sharing controls flip the verdict?")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_block_structure(
    folds,
    ax=None,
    style: bool = True,
    figsize=(6.8, 3.2),
):
    """
    Visualise the h-block partition: evaluation windows, training sets, buffers.

    Makes the cost of the buffer visible -- the white gaps either side of each
    evaluation block are observations no model ever trains on for that fold.
    """
    if style:
        apply_active_style()
    fig, ax = _fig(ax, figsize)

    n, K = folds.n, folds.n_blocks
    for b, (train, ev) in enumerate(folds):
        yb = K - b
        mask = np.zeros(n, dtype=int)
        mask[train] = 1
        mask[ev] = 2
        starts = np.flatnonzero(np.diff(np.concatenate([[-1], mask])) != 0)
        ends = np.append(starts[1:], n)
        for s, e in zip(starts, ends):
            kind = mask[s]
            color = {0: "white", 1: lighten(COLORS["bootstrap"], 0.72), 2: COLORS["borrowed"]}[kind]
            ax.barh(yb, e - s, left=s, height=0.62, color=color,
                    edgecolor="#BBBBBB", linewidth=0.4)
        ax.text(-0.012 * n, yb, f"fold {b + 1}", ha="right", va="center", fontsize=8)

    ax.set_yticks([])
    ax.set_xlim(0, n)
    ax.set_ylim(0.3, K + 0.7)
    ax.set_xlabel("observation")
    ax.grid(False)
    share = np.mean([t.size for t in folds.train_blocks]) / n
    ax.set_title(
        f"h-block cross-fitting: K = {K}, h = {folds.buffer}  "
        f"(mean training share {share:.0%})"
    )

    import matplotlib.patches as mpatches

    ax.legend(
        handles=[
            mpatches.Patch(color=lighten(COLORS["bootstrap"], 0.72), label="training"),
            mpatches.Patch(color=COLORS["borrowed"], label="evaluation"),
            mpatches.Patch(facecolor="white", edgecolor="#BBBBBB", label="buffer (discarded)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
    )
    fig.tight_layout()
    return fig


def plot_series(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    ncols: int = 3,
    style: bool = True,
    figsize=None,
    title: Optional[str] = None,
):
    """Small-multiple time-series panel, one facet per column."""
    import matplotlib.pyplot as plt

    if style:
        apply_active_style()
    cols = list(cols or df.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    figsize = figsize or (2.5 * ncols, 1.9 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for a, c in zip(axes, cols):
        a.plot(df.index, df[c], color=COLORS["bootstrap"], lw=1.2)
        a.set_title(c, fontsize=9)
        a.tick_params(labelsize=7)
    for a in axes[len(cols):]:
        a.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_regimes(
    df: pd.DataFrame,
    regimes: Dict[str, tuple],
    col: str,
    ax=None,
    style: bool = True,
    figsize=(6.8, 3.4),
):
    """A single series with regime windows shaded, for the pass-through data."""
    if style:
        apply_active_style()
    fig, ax = _fig(ax, figsize)
    ax.plot(df.index, df[col], color=COLORS["observed"], lw=1.3)
    for i, (label, (a, b)) in enumerate(regimes.items()):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b),
                   color=lighten(COLORS["bootstrap"], 0.80 if i % 2 == 0 else 0.90), zorder=0)
        mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
        ax.annotate(label, (mid, ax.get_ylim()[1]), ha="center", va="top",
                    fontsize=7.5, color=COLORS["null"])
    ax.set_ylabel(col)
    ax.set_title(f"{col} across monetary regimes")
    fig.tight_layout()
    return fig
