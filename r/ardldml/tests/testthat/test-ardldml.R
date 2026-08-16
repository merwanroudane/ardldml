## Reference values are the ones the Python implementation of the same
## procedure produces on the same bundled data. Both are deterministic given the
## specification, so these are exact-agreement regression tests.

test_that("h-block folds partition the sample and buffer the training sets", {
  f <- hblock_folds(100, n_blocks = 4, buffer = 5)
  expect_equal(vapply(f$train, length, numeric(1)), c(70, 65, 65, 70))
  expect_equal(sort(unlist(f$eval)), 1:100)
  expect_equal(f$n_blocks, 4L)
  ## every training set excludes its own evaluation window and the buffer
  for (b in seq_len(f$n_blocks)) {
    expect_length(intersect(f$train[[b]], f$eval[[b]]), 0L)
  }
})

test_that("folds reject impossible configurations", {
  expect_error(hblock_folds(100, n_blocks = 1), "at least 2")
  expect_error(hblock_folds(100, n_blocks = 4, buffer = -1), "non-negative")
  expect_error(hblock_folds(10, n_blocks = 20), "exceeds")
  expect_error(hblock_folds(20, n_blocks = 2, buffer = 40), "empty training set")
})

test_that("sample use falls as the buffer widens", {
  expect_equal(sample_use(100, 4, 0), 0.75)
  expect_lt(sample_use(108, 5, 10), sample_use(108, 5, 0))
  tab <- sample_use_table(108)
  expect_equal(dim(tab), c(4L, 4L))
})

test_that("the number of restrictions follows the deterministic case", {
  expect_equal(n_restrictions(1, 3), 2L)
  expect_equal(n_restrictions(1, 4), 3L)
  expect_equal(n_restrictions(4, 2), 6L)
  expect_error(n_restrictions(1, 9), "case must be")
})

test_that("the balanced design differences the integrated block only", {
  df <- passthrough_regime("1999-2007")
  des <- build_balanced_design(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
                               lags = 4, integrated = DEFAULT_INTEGRATED)
  expect_equal(length(des$dY), 103L)
  expect_equal(des$stationary, "unrate")
  expect_setequal(des$integrated, DEFAULT_INTEGRATED)
  ## unrate enters in levels, the integrated controls in differences
  expect_true("unrate" %in% colnames(des$X))
  expect_true(all(paste0("D.", DEFAULT_INTEGRATED) %in% colnames(des$X)))
  ## Z is the pair of tested levels, W the untouched control levels
  expect_equal(ncol(des$Z), 2L)
  expect_equal(ncol(des$Wlev), length(CONTROLS))
})

test_that("the design refuses malformed input", {
  df <- passthrough_regime("1999-2007")
  W <- as.matrix(df[, CONTROLS])
  expect_error(build_balanced_design(df$cpi, df$neer, W, lags = 0), "at least 1")
  expect_error(build_balanced_design(df$cpi[-1], df$neer, W), "same number")
  dup <- cbind(W, W[, 1, drop = FALSE])
  expect_error(build_balanced_design(df$cpi, df$neer, dup), "duplicate")
})

test_that("adaptive post-LASSO recovers a sparse signal", {
  set.seed(42)
  X <- matrix(stats::rnorm(120 * 6), 120, 6)
  y <- 3 * X[, 1] - 2 * X[, 2] + stats::rnorm(120, sd = 0.1)
  fit <- adaptive_post_lasso(X, y, adaptive = FALSE)
  expect_true(all(which(fit$support) %in% c(1L, 2L)))
  expect_true(all(c(1L, 2L) %in% which(fit$support)))
  ## the refit is unpenalised, so the coefficients are close to the truth
  expect_equal(fit$coef[1], 3, tolerance = 0.05)
  expect_equal(fit$coef[2], -2, tolerance = 0.05)
  expect_true(fit$estimable)
})

test_that("the plug-in penalty scales as expected", {
  expect_gt(plugin_penalty(100, 40, 1), 0)
  expect_lt(plugin_penalty(400, 40, 1), plugin_penalty(100, 40, 1))
  expect_lt(plugin_penalty(100, 40, 0.5), plugin_penalty(100, 40, 1))
  expect_error(plugin_penalty(0, 40, 1), "positive")
})

test_that("the statistic reproduces the reference implementation", {
  df <- passthrough_regime("1999-2007")
  fit <- dml_bounds(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
                    lags = 4, n_blocks = 5, buffer = 6,
                    integrated = DEFAULT_INTEGRATED)
  expect_equal(fit$stat, 0.4043, tolerance = 1e-3)
  expect_equal(fit$theta, -0.9715, tolerance = 1e-3)
  expect_equal(fit$theta_se, 0.8975, tolerance = 1e-3)
  expect_equal(fit$nobs, 103L)
  expect_equal(sum(fit$supports$dY), 6L)
  expect_equal(sum(fit$supports$Z), 1L)
  expect_true(fit$estimable)
  expect_equal(decision(fit), "no bootstrap run")
})

test_that("the statistic reproduces the reference across all four regimes", {
  ref <- c("1973-1985" = 9.111, "1986-1998" = 18.712,
           "1999-2007" = 0.404, "2008-2020" = 3.703)
  for (rg in names(ref)) {
    df <- passthrough_regime(rg)
    fit <- dml_bounds(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
                      lags = 4, n_blocks = 5, buffer = 6,
                      integrated = DEFAULT_INTEGRATED)
    expect_equal(fit$stat, unname(ref[rg]), tolerance = 1e-3,
                 info = paste("regime", rg))
  }
})

test_that("the unpenalised arm differs from the adaptive one", {
  df <- passthrough_regime("1999-2007")
  W <- as.matrix(df[, CONTROLS])
  ad <- dml_bounds(df$cpi, df$neer, W, lags = 4, n_blocks = 5, buffer = 6,
                   integrated = DEFAULT_INTEGRATED)
  ols <- dml_bounds(df$cpi, df$neer, W, lags = 4, n_blocks = 5, buffer = 6,
                    integrated = DEFAULT_INTEGRATED, penalised = FALSE)
  ## the unpenalised projection keeps every control level, by construction
  expect_equal(sum(ols$supports$Z), ncol(W))
  expect_false(isTRUE(all.equal(ad$stat, ols$stat)))
})

test_that("the classical test reproduces the reference implementation", {
  df <- passthrough_regime("1999-2007")
  cb <- classical_bounds_test(df$cpi, cbind(neer = df$neer), lags = 4,
                              nsim = 50, seed = 1)
  expect_equal(cb$f_stat, 1.629, tolerance = 1e-3)
  expect_equal(cb$t_stat, -1.128, tolerance = 1e-3)
  expect_equal(cb$theta_wald, 3.883, tolerance = 1e-3)
  expect_equal(cb$nobs, 104L)
  expect_equal(cb$k, 1L)

  cbf <- classical_bounds_test(df$cpi, cbind(neer = df$neer), lags = 4,
                               fixed = as.matrix(df[, CONTROLS]),
                               nsim = 50, seed = 1)
  expect_equal(cbf$f_stat, 11.084, tolerance = 1e-3)
  expect_equal(cbf$alpha, 0.3063, tolerance = 1e-3)
})

test_that("the bounds verdict respects the inconclusive region", {
  expect_equal(bounds_verdict(6.2, 4.94, 5.73), "reject")
  expect_equal(bounds_verdict(5.2, 4.94, 5.73), "inconclusive")
  expect_equal(bounds_verdict(2.0, 4.94, 5.73), "fail to reject")
  expect_true(is.na(bounds_verdict(NA_real_, 4.94, 5.73)))
})

test_that("the simulator brackets the published cell", {
  skip_on_cran()
  cv <- simulate_pss_bounds(k = 1, case = 3, TT = 1000, nsim = 2000, seed = 7)
  at5 <- cv[cv$level == 0.05, ]
  ## published: I(0) = 4.94, I(1) = 5.73, to Monte Carlo error at this nsim
  expect_equal(at5$I0, 4.94, tolerance = 0.15)
  expect_equal(at5$I1, 5.73, tolerance = 0.15)
  expect_lt(at5$I0, at5$I1)
})

test_that("the simulator guards its arguments", {
  expect_error(simulate_pss_bounds(k = -1), "non-negative")
  expect_error(simulate_pss_bounds(k = 1, case = 7), "case must be")
  expect_error(simulate_pss_bounds(k = 1, case = 3, TT = 4), "too small")
})

test_that("stored published cells are available and ordered", {
  ref <- pss_reference(1, 3)
  expect_equal(ref$I0[ref$level == 0.05], 4.94)
  expect_equal(ref$I1[ref$level == 0.05], 5.73)
  expect_true(all(ref$I0 < ref$I1))
  expect_error(pss_reference(3, 3), "no published reference")
})

test_that("the k-convention offset understates the bounds", {
  off <- k_convention_offset(k = 1, case = 3)
  expect_equal(nrow(off), 2L)
  ## the shifted row is the k+1 row, which is smaller, hence over-rejects
  expect_lt(off$I0[2], off$I0[1])
  expect_lt(off$I1[2], off$I1[1])
})

test_that("the bootstrap attaches a usable critical value", {
  df <- passthrough_regime("1999-2007")
  fit <- dml_bounds(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
                    lags = 4, n_blocks = 5, buffer = 6,
                    integrated = DEFAULT_INTEGRATED)
  fit <- dml_bootstrap(fit, B = 5, seed = 20260625)
  expect_true(is.finite(fit$boot$crit))
  expect_gte(fit$boot$pvalue, 0)
  expect_lte(fit$boot$pvalue, 1)
  expect_equal(fit$boot$B, 5L)
  expect_equal(fit$boot$scheme, "system")
  expect_true(decision(fit) %in% c("reject", "fail to reject"))
  ## the fixed-regressor scheme is the strong-exogeneity special case
  fixed <- dml_bootstrap(fit, B = 5, seed = 20260625, scheme = "fixed")
  expect_equal(fixed$boot$scheme, "fixed")
})

test_that("the bootstrap is reproducible under a seed", {
  df <- passthrough_regime("1999-2007")
  W <- as.matrix(df[, CONTROLS])
  a <- dml_bounds(df$cpi, df$neer, W, lags = 4, n_blocks = 5, buffer = 6,
                  integrated = DEFAULT_INTEGRATED, B = 4, seed = 99)
  b <- dml_bounds(df$cpi, df$neer, W, lags = 4, n_blocks = 5, buffer = 6,
                  integrated = DEFAULT_INTEGRATED, B = 4, seed = 99)
  expect_equal(a$boot$draws, b$boot$draws)
})

test_that("the bundled data has the documented shape", {
  expect_equal(nrow(passthrough), 576L)
  expect_true(all(c("date", "cpi", "neer", CONTROLS) %in% names(passthrough)))
  expect_equal(vapply(names(PASSTHROUGH_REGIMES),
                      function(r) nrow(passthrough_regime(r)), numeric(1)),
               c("1973-1985" = 156, "1986-1998" = 156,
                 "1999-2007" = 108, "2008-2020" = 156))
  ## logs are taken of the quantity and price series only
  raw <- passthrough_regime("1999-2007", log = FALSE)
  lg <- passthrough_regime("1999-2007", log = TRUE)
  expect_equal(lg$cpi, log(raw$cpi))
  expect_equal(lg$unrate, raw$unrate)
  expect_error(passthrough_regime("1066-1067"), "regime must be one of")
})

test_that("palette helpers return the documented lengths", {
  expect_length(parula_colors(6), 6L)
  expect_length(parula_colors(64), 64L)
  expect_match(parula_colors(3), "^#[0-9A-Fa-f]{6}$")
  expect_true(all(c("bootstrap", "borrowed") %in% names(ardl_colors)))
})
