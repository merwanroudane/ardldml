# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-08-15

First release.

### Added

**Testing**
- `DMLBounds`, the main entry point: balanced orthogonalisation of the lagged
  levels against a high-dimensional control set, followed by the `F`-form Wald
  test of equation (10).
- `DMLBoundsSpec` to carry a specification, so the bootstrap can rebuild an
  identical statistic on a regenerated path.
- `compute_statistic` for the full pipeline in one call.

**Inference**
- `restricted_system_wild_bootstrap`, Algorithm 1: a single Rademacher weight
  applied to the stacked conditional and marginal residuals, joint regeneration
  of `Y*` and `D*`, the nuisance space held at its realised path, and the entire
  residualised statistic recomputed per draw with the level supports re-selected.
- `scheme="fixed"` retained as the strong-exogeneity special case.

**Critical values, generated rather than tabulated**
- `simulate_pss_bounds` regenerates the classical bracket from the
  data-generating process Pesaran, Shin and Smith printed in the notes to
  Table CI. Works at any `T`, so finite-sample bounds come free, and past
  `k = 10`, where the published tables stop.
- `pss_reference` holds a small set of published cells used to validate the
  simulator in the test suite.
- `empirical_critical_value` for the method-specific null quantile used in
  Monte Carlo work.

**First stage**
- `build_balanced_design`: stationary target on stationary regressors
  (integrated controls differenced), integrated levels on control levels.
- `hblock_folds` with buffered training sets, plus `sample_use_table` so the
  cost of the buffer is visible before you commit to a configuration.
- `adaptive_post_lasso` with marginal slope weights and the plug-in penalty
  `c·sqrt(log d/n)·sigma`, `c = 1.1`; `tscv_penalty` for rolling-origin
  selection.

**Diagnostics**
- `trend_absorption`, Definition 2: the four-fit contrast over control set and
  penalisation, reporting `Delta_m` and `Delta_W` with a conservative verdict.

**Monte Carlo**
- `simulate_design` implementing the Appendix B data-generating process, the
  five designs of Table 2, and the endogeneity channel.
- `run_design` and `run_endogeneity_grid`.

**Data**
- `load_passthrough`: real monthly FRED-MD series, 1973-01 to 2020-12, for the
  exchange-rate pass-through application. The four monetary regimes give 156,
  156, 108 and 156 complete observations, matching the paper's Table 11.

**Output**
- Journal-styled figures: `plot_bracket`, `plot_bootstrap_null`,
  `plot_size_comparison`, `plot_diagnostic`, `plot_block_structure`,
  `plot_series`, `plot_regimes`.
- Tables: `result_table`, `regime_table`, `montecarlo_table`,
  `diagnostic_table`, `critical_value_table`, with `to_latex` producing
  `booktabs` output with caption, label and notes.

**Documentation**
- README with concepts, a full syntax reference, and a stated-limitations
  section.
- `docs/STEP_BY_STEP_GUIDE.md`, a tutorial from zero.
- Three runnable examples and a 42-test suite.

### Notes

- `statsmodels.tsa.ardl.UECMResults.bounds_test` indexes its critical-value
  table with `k + 1` rather than `k`, so it reports bounds that are too small
  and over-rejects. This package never calls it; `statsmodels_offset`
  reproduces the discrepancy for verification, and `classical_bounds_test`
  computes the Wald statistic itself.
- The bundled pass-through results are **not** a replication of the paper's
  Table 11. The paper does not publish its FRED series codes, data vintage,
  per-control transformations, or cross-fitting settings, so the data mapping
  is inferred. The sample design reproduces exactly; the point estimates do not.
