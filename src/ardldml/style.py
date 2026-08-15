"""
Figure styling for journal submission.

The defaults target the look of *Journal of Applied Econometrics* and
*Econometrics Journal* figures: serif type, no top or right spine, hairline
reference lines, a light dotted grid, and the colourblind-safe palette of Wong
(2011).

Use it globally::

    from ardldml.style import use_journal_style
    use_journal_style()

or locally::

    with use_journal_style(context=True):
        fig = plot_bootstrap_null(res)

Every function in :mod:`ardldml.plots` applies the style itself unless you pass
``style=False``, so you rarely need to call this by hand.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["PALETTE", "COLORS", "use_journal_style", "journal_rc", "lighten"]

#: Colourblind-safe palette (Wong 2011).
PALETTE = [
    "#0173B2",  # blue       -- the bootstrap / preferred method
    "#D55E00",  # vermillion -- the borrowed bound / comparison
    "#029E73",  # green      -- reduced control set
    "#CC78BC",  # purple
    "#DE8F05",  # orange
    "#555555",  # charcoal   -- neutral
    "#9E4500",  # brown
    "#117733",  # dark teal
]

#: Semantic aliases used across :mod:`ardldml.plots`.
COLORS: Dict[str, str] = {
    "bootstrap": "#0173B2",
    "borrowed": "#D55E00",
    "reduced": "#029E73",
    "full": "#0173B2",
    "adaptive": "#0173B2",
    "ols": "#D55E00",
    "observed": "#000000",
    "null": "#555555",
    "nominal": "#D55E00",
    "i0": "#029E73",
    "i1": "#D55E00",
}


def lighten(hex_color: str, factor: float = 0.45) -> str:
    """Blend a hex colour toward white; ``factor=0`` returns it unchanged."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))
    out = tuple(int(round(c + (255 - c) * factor)) for c in rgb)
    return "#%02X%02X%02X" % out


def journal_rc(base_size: float = 10.0) -> Dict[str, object]:
    """The rcParams dictionary applied by :func:`use_journal_style`."""
    return {
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia", "serif"],
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    }


@contextmanager
def _style_context(base_size: float):
    with plt.rc_context(journal_rc(base_size)):
        yield


def use_journal_style(base_size: float = 10.0, context: bool = False):
    """
    Apply the journal style.

    Parameters
    ----------
    base_size : float
        Base font size in points.
    context : bool
        If ``True``, return a context manager that restores the previous
        settings on exit. If ``False``, mutate the global rcParams.
    """
    if context:
        return _style_context(base_size)
    mpl.rcParams.update(journal_rc(base_size))
    return None
