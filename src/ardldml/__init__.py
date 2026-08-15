"""
ardldml -- ARDL bounds testing with many persistent controls.

DML-Bounds tests for a long-run (cointegrating) relationship when the
conditioning set is high-dimensional and may itself contain stochastic trends.
It combines a balanced, cross-fitted orthogonalisation of the lagged levels
against the controls with a restricted system wild bootstrap that regenerates
the outcome and the focal regressor jointly under the null.

The idea in one paragraph
-------------------------
Classical ARDL bounds testing brackets one unknown: whether the regressors are
:math:`I(0)` or :math:`I(1)`. With a large, persistent control set a second
unknown appears -- partialling out controls that carry stochastic trends can
*absorb* part of the long-run variation that identifies the error-correction
relation. What governs the null is then not the integration order of the
original regressors but the **effective integrated count**
:math:`\\tilde k = k - r`, the number of stochastic trends that survive
residualisation. :math:`\\tilde k = k` puts the null at the classical
:math:`I(1)` endpoint, :math:`\\tilde k = 0` at the :math:`I(0)` endpoint, and
anything between lands inside the bracket.

Quick start
-----------
>>> from ardldml import DMLBounds, load_passthrough           # doctest: +SKIP
>>> data = load_passthrough(regime="1999-2007", log=True)     # doctest: +SKIP
>>> res = (
...     DMLBounds(data["cpi"], data["neer"], data.drop(columns=["cpi", "neer"]),
...               lags=4, integrated=["m2", "oil", "gs10", "baa"])
...     .fit()
...     .bootstrap(B=999, seed=20260625)
... )                                                          # doctest: +SKIP
>>> print(res.summary())                                       # doctest: +SKIP

Critical values are generated, never looked up
----------------------------------------------
There is no bounds table in this package. Inference comes from
:func:`~ardldml.bootstrap.restricted_system_wild_bootstrap`; the classical
Pesaran-Shin-Smith bracket, when you want it as a benchmark, is regenerated
from their own data-generating process by
:func:`~ardldml.critvals.simulate_pss_bounds`. See :mod:`ardldml.critvals` for
the reasoning.

A caveat worth reading before you use it
----------------------------------------
The estimand is conditional. If a control is itself part of the equilibrium
system, residualisation removes the relation rather than the confounding, and a
non-rejection reflects over-absorption rather than the absence of a long-run
relationship. This cannot be checked from a single fit. Always report
:func:`~ardldml.diagnostics.trend_absorption`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .bootstrap import rademacher, restricted_system_wild_bootstrap
from .core import (
    ClassicalBounds,
    classical_bounds_test,
    conditional_ecm,
    restricted_null_model,
    trend_for_case,
)
from .critvals import (
    CASE_LABELS,
    n_restrictions,
    pss_reference,
    simulate_pss_bounds,
    statsmodels_offset,
)
from .firststage import (
    PENALTY_RULES,
    BalancedDesign,
    FirstStage,
    adaptive_post_lasso,
    build_balanced_design,
    classify_controls,
    cross_fit_projection,
    plugin_penalty,
    tscv_penalty,
)
from .folds import BlockStructure, hblock_folds, sample_use, sample_use_table
from .statistic import DMLBounds, DMLBoundsResults, DMLBoundsSpec, compute_statistic

__all__ = [
    "__version__",
    # main entry points
    "DMLBounds",
    "DMLBoundsResults",
    "DMLBoundsSpec",
    "compute_statistic",
    # inference
    "restricted_system_wild_bootstrap",
    "rademacher",
    # classical benchmark
    "classical_bounds_test",
    "ClassicalBounds",
    "conditional_ecm",
    "restricted_null_model",
    "trend_for_case",
    # critical values
    "simulate_pss_bounds",
    "pss_reference",
    "statsmodels_offset",
    "n_restrictions",
    "CASE_LABELS",
    # first stage
    "build_balanced_design",
    "BalancedDesign",
    "FirstStage",
    "classify_controls",
    "adaptive_post_lasso",
    "cross_fit_projection",
    "plugin_penalty",
    "tscv_penalty",
    "PENALTY_RULES",
    # folds
    "hblock_folds",
    "BlockStructure",
    "sample_use",
    "sample_use_table",
]


_LAZY = {
    "datasets": (
        "load_passthrough", "passthrough_regimes", "data_path",
        "PASSTHROUGH_REGIMES", "CONTROLS", "REDUCED_DROP", "DEFAULT_INTEGRATED",
    ),
    "diagnostics": ("trend_absorption", "TrendAbsorption", "penalty_sensitivity"),
    "simulate": (
        "simulate_design", "empirical_critical_value", "run_design",
        "run_endogeneity_grid", "run_ultra_check", "run_robustness_grid",
        "DESIGNS", "DesignSpec", "default_d", "RHO_ALTERNATIVES",
    ),
    "plots": (
        "plot_bracket", "plot_bootstrap_null", "plot_size_comparison",
        "plot_diagnostic", "plot_block_structure", "plot_series", "plot_regimes",
    ),
    "tables": (
        "result_table", "regime_table", "montecarlo_table", "diagnostic_table",
        "critical_value_table", "to_latex", "to_markdown", "stars", "bounds_stars",
    ),
    "style": ("use_journal_style", "PALETTE", "COLORS"),
}

_LOOKUP = {name: mod for mod, names in _LAZY.items() for name in names}
__all__ += sorted(_LOOKUP)


def __getattr__(name):
    """Lazily expose the heavier optional modules (matplotlib, datasets)."""
    mod = _LOOKUP.get(name)
    if mod is None:
        raise AttributeError(f"module 'ardldml' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{mod}", __name__), name)


def __dir__():
    return sorted(set(__all__))
