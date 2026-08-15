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

Every function in :mod:`ardldml.plots` applies the *active* style itself unless
you pass ``style=False``, so you rarely need to call this by hand. The active
style is whichever of :func:`use_journal_style` or :func:`use_sunny_style` you
called last; journal is the default.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "PALETTE",
    "COLORS",
    "SUNNY_PALETTE",
    "SUNNY_COLORS",
    "SUNNY_BG",
    "use_journal_style",
    "use_sunny_style",
    "apply_active_style",
    "active_style",
    "journal_rc",
    "lighten",
]

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

#: The journal semantics, kept so :func:`use_journal_style` can undo
#: :func:`use_sunny_style`.
_JOURNAL_COLORS: Dict[str, str] = dict(COLORS)


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
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "#B0B0B0",
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

    Calling this also restores :data:`COLORS` to the journal semantics and
    makes journal the active style, so it undoes a previous
    :func:`use_sunny_style`.
    """
    if context:
        return _style_context(base_size)
    mpl.rcParams.update(journal_rc(base_size))
    COLORS.update(_JOURNAL_COLORS)
    _ACTIVE.update(name="journal", base_size=base_size, transparent=False)
    return None


# ---------------------------------------------------------------------------
# Active style
# ---------------------------------------------------------------------------
#: Which style was last applied globally, and with what arguments.
_ACTIVE: Dict[str, object] = {"name": "journal", "base_size": 10.0, "transparent": False}


def active_style() -> Dict[str, object]:
    """A copy of the active style record: ``name``, ``base_size``, ``transparent``."""
    return dict(_ACTIVE)


def apply_active_style() -> None:
    """
    Re-apply whichever style is active.

    This is what :mod:`ardldml.plots` calls when ``style=True``. Going through
    it rather than straight to :func:`use_journal_style` is what lets
    :func:`use_sunny_style` stick: a plotting call re-asserts the style you
    chose instead of silently reverting to the journal one.
    """
    if _ACTIVE["name"] == "sunny":
        use_sunny_style(
            base_size=float(_ACTIVE["base_size"]),
            transparent=bool(_ACTIVE["transparent"]),
        )
    else:
        use_journal_style(base_size=float(_ACTIVE["base_size"]))


# ---------------------------------------------------------------------------
# Sunny palette -- for the project website and slides
# ---------------------------------------------------------------------------
#: Warm, high-contrast palette used by the project site. Ordered so the first
#: two entries carry the recurring contrast of the package: the bootstrap
#: against the borrowed bound.
SUNNY_PALETTE = [
    "#E8871A",  # amber      -- the bootstrap / preferred method
    "#D1495B",  # coral red  -- the borrowed bound / comparison
    "#2A9D8F",  # teal       -- reduced control set
    "#F4A259",  # apricot
    "#8E5572",  # plum
    "#5B8E7D",  # sage
    "#F5B841",  # gold
    "#6A4C93",  # violet
]

#: Semantic aliases for the sunny palette, mirroring :data:`COLORS`.
SUNNY_COLORS: Dict[str, str] = {
    "bootstrap": "#E8871A",
    "borrowed": "#D1495B",
    "reduced": "#2A9D8F",
    "full": "#E8871A",
    "adaptive": "#E8871A",
    "ols": "#D1495B",
    "observed": "#3D2C29",
    "null": "#8A7E72",
    "nominal": "#D1495B",
    "i0": "#2A9D8F",
    "i1": "#D1495B",
}

#: Warm off-white page background used by the site figures.
SUNNY_BG = "#FFFDF7"


def use_sunny_style(base_size: float = 10.5, transparent: bool = False):
    """
    Apply the warm website palette.

    Same layout rules as :func:`use_journal_style` -- serif type, no top or
    right spine, light dotted grid -- but with the amber/coral/teal palette and
    a warm off-white ground, so figures sit naturally on the project page.

    Use :func:`use_journal_style` for anything going into a paper; this is for
    the web.

    Parameters
    ----------
    transparent : bool
        Render on a transparent ground instead of the warm off-white, which is
        what you want if the page background is itself a gradient.
    """
    rc = journal_rc(base_size)
    bg = "none" if transparent else SUNNY_BG
    rc.update(
        {
            "axes.prop_cycle": mpl.cycler(color=SUNNY_PALETTE),
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "savefig.facecolor": bg,
            "savefig.transparent": transparent,
            "text.color": "#3D2C29",
            "axes.labelcolor": "#3D2C29",
            "axes.edgecolor": "#B9A88F",
            "xtick.color": "#6B5D52",
            "ytick.color": "#6B5D52",
            "grid.color": "#C9B79C",
        }
    )
    mpl.rcParams.update(rc)
    COLORS.update(SUNNY_COLORS)
    _ACTIVE.update(name="sunny", base_size=base_size, transparent=transparent)
    return None
