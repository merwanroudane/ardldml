# ardldml 0.1.0

* First release.
* `dml_bounds()` fits the DML-Bounds test of Villena (2026): a balanced,
  cross-fitted first stage orthogonalises the lagged levels against a
  high-dimensional control set, and the null of no level relationship is tested
  on the residualised variables.
* `dml_bootstrap()` implements the restricted system wild bootstrap, which
  regenerates the outcome and the focal regressor jointly under a shared
  Rademacher weight so the endogeneity channel is preserved.
* `trend_absorption()` and `penalty_sensitivity()` report whether a verdict
  survives a change of conditioning set or of penalty.
* `classical_bounds_test()` runs all three steps of the classical procedure and
  reads the statistic against a bracket simulated at the model's own sample
  size, rather than against a table calibrated at T = 1000.
* `simulate_pss_bounds()` regenerates the classical bracket from the
  data-generating process of Pesaran, Shin and Smith (2001); no critical-value
  table is stored in the package.
* Bundled `passthrough` data: nine monthly United States macroeconomic series,
  1973-2020, so every example runs offline.
