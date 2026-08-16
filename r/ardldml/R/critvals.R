#' Deterministic Case Labels
#'
#' The five deterministic specifications of Pesaran, Shin and Smith (2001).
#' Cases 2 and 4 restrict the intercept or the trend, which puts it inside the
#' tested null and adds one restriction; see \code{\link{n_restrictions}}.
#'
#' @format A named character vector of length 5.
#' @references
#' Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing approaches
#' to the analysis of level relationships. \emph{Journal of Applied
#' Econometrics}, 16(3), 289--326. \doi{10.1002/jae.616}
#' @export
CASE_LABELS <- c(
  "1" = "no intercept, no trend",
  "2" = "restricted intercept, no trend",
  "3" = "unrestricted intercept, no trend",
  "4" = "unrestricted intercept, restricted trend",
  "5" = "unrestricted intercept, unrestricted trend"
)

#' Significance Levels Reported by Default
#'
#' @format A numeric vector of length 3.
#' @export
DEFAULT_LEVELS <- c(0.10, 0.05, 0.01)

#' Number of Restrictions in the Bounds Test
#'
#' The level terms are \eqn{y_{t-1}} and the \eqn{k} lagged forcing regressors.
#' Cases 2 and 4 add one more because the intercept or the trend is restricted
#' and therefore tested jointly with them.
#'
#' @param k Number of long-run forcing regressors. This does \emph{not} count
#'   the dependent variable; mixing that convention up shifts every bound by one
#'   row.
#' @param case Deterministic case, an integer in 1:5.
#' @return An integer.
#' @examples
#' n_restrictions(k = 1, case = 3)
#' n_restrictions(k = 1, case = 4)
#' @export
n_restrictions <- function(k, case = 3L) {
  .check_case(case)
  as.integer(k) + 1L + if (case %in% c(2L, 4L)) 1L else 0L
}

.check_case <- function(case) {
  if (!(as.character(case) %in% names(CASE_LABELS))) {
    stop("case must be one of 1:5; got ", case)
  }
  invisible(TRUE)
}

## F form of the Wald test that the coefficients on the first nrest columns
## of X are zero. Kept internal: it is the same computation in the classical
## test, the simulator and the orthogonalised statistic.
.wald_f <- function(yv, X, nrest) {
  n <- nrow(X)
  p <- ncol(X)
  dof <- n - p
  if (dof <= 0L) return(NA_real_)
  fit <- stats::lm.fit(X, yv)
  if (anyNA(fit$coefficients)) return(NA_real_)
  resid <- fit$residuals
  s2 <- sum(resid^2) / dof
  xtx <- crossprod(X)
  xtx_inv <- tryCatch(solve(xtx), error = function(e) MASS_ginv(xtx))
  if (is.null(xtx_inv)) return(NA_real_)
  idx <- seq_len(nrest)
  V <- s2 * xtx_inv[idx, idx, drop = FALSE]
  b <- fit$coefficients[idx]
  quad <- tryCatch(as.numeric(crossprod(b, solve(V, b))), error = function(e) NA_real_)
  quad / nrest
}

## Moore-Penrose inverse via the SVD, so a rank-deficient cross-product
## degrades the way numpy's pinv does rather than erroring out.
MASS_ginv <- function(A, tol = sqrt(.Machine$double.eps)) {
  s <- svd(A)
  pos <- s$d > max(tol * s$d[1L], 0)
  if (!any(pos)) return(matrix(0, nrow(A), ncol(A)))
  s$v[, pos, drop = FALSE] %*% ((1 / s$d[pos]) * t(s$u[, pos, drop = FALSE]))
}

.pss_design <- function(y, x, case, TT) {
  ylag <- matrix(y[seq_len(TT)], ncol = 1L)
  xlag <- if (ncol(x)) x[seq_len(TT), , drop = FALSE] else matrix(0, TT, 0)
  one <- matrix(1, TT, 1L)
  trend <- matrix(seq_len(TT), ncol = 1L)

  z <- cbind(ylag, xlag)
  w <- matrix(0, TT, 0)
  if (case == 2L) {
    z <- cbind(z, one)
  } else if (case == 3L) {
    w <- one
  } else if (case == 4L) {
    z <- cbind(z, trend)
    w <- one
  } else if (case == 5L) {
    w <- cbind(one, trend)
  }
  list(z = z, w = w)
}

.simulate_one_side <- function(k, case, TT, nsim, integrated) {
  stats_out <- numeric(nsim)
  for (i in seq_len(nsim)) {
    e <- matrix(stats::rnorm((TT + 1L) * (k + 1L)), nrow = TT + 1L)
    y <- cumsum(e[, 1L])
    if (k == 0L) {
      x <- matrix(0, TT + 1L, 0)
    } else if (integrated) {
      x <- apply(e[, -1L, drop = FALSE], 2L, cumsum)
      if (!is.matrix(x)) x <- matrix(x, ncol = k)
    } else {
      x <- e[, -1L, drop = FALSE]
    }
    d <- .pss_design(y, x, case, TT)
    X <- if (ncol(d$w)) cbind(d$z, d$w) else d$z
    stats_out[i] <- .wald_f(diff(y), X, ncol(d$z))
  }
  stats_out[is.finite(stats_out)]
}

#' Generate the Classical Bounds by Simulation
#'
#' Regenerates the Pesaran, Shin and Smith (2001) bracket from the
#' data-generating process printed in the notes to their Table CI, rather than
#' reading a stored table.
#'
#' Under the null, with \eqn{y_0 = x_0 = 0} and \eqn{k+1} independent standard
#' normal innovations, \eqn{y_t = y_{t-1} + \epsilon_{1t}} and
#' \eqn{x_t = P x_{t-1} + e_{2t}}, with \eqn{P = I_k} for the upper bound
#' (regressors purely integrated of order one) and \eqn{P = 0} for the lower
#' bound (purely stationary). The statistic is the \eqn{F} form of the Wald test
#' on the level terms.
#'
#' Generating rather than tabulating has three advantages: it is checkable
#' against print via \code{\link{pss_reference}}; it extends to any sample size,
#' so small-sample bounds come from passing \code{TT = n} rather than shipping a
#' second table; and it extends past \eqn{k = 10}, where the published tables
#' stop.
#'
#' @param k Number of long-run forcing regressors; see \code{\link{n_restrictions}}.
#' @param case Deterministic case, an integer in 1:5.
#' @param TT Sample size. \code{TT = 1000} reproduces the published asymptotic
#'   tables; smaller values give finite-sample bounds.
#' @param nsim Replications. The published tables use 40000. Tail quantiles
#'   settle slowest, so reduce this only for exploratory work.
#' @param levels Significance levels.
#' @param seed Optional seed. Passed to \code{\link[base]{set.seed}}; the
#'   caller's random number state is restored on exit.
#'
#' @return A data frame with columns \code{level}, \code{I0} (the lower,
#'   stationary bound) and \code{I1} (the upper, integrated bound).
#'
#' @references
#' Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing approaches
#' to the analysis of level relationships. \emph{Journal of Applied
#' Econometrics}, 16(3), 289--326. \doi{10.1002/jae.616}
#'
#' Narayan, P. K. (2004). Reformulating critical values for the bounds
#' F-statistics approach to cointegration. Monash University Discussion Paper
#' 02/04.
#'
#' @examples
#' # Small and fast; raise nsim for anything you intend to quote.
#' simulate_pss_bounds(k = 1, case = 3, TT = 120, nsim = 100, seed = 1)
#'
#' \donttest{
#' # Closer to the published asymptotic table.
#' simulate_pss_bounds(k = 1, case = 3, TT = 1000, nsim = 4000, seed = 1)
#' }
#' @export
simulate_pss_bounds <- function(k, case = 3L, TT = 1000L, nsim = 40000L,
                                levels = DEFAULT_LEVELS, seed = NULL) {
  k <- as.integer(k)
  case <- as.integer(case)
  TT <- as.integer(TT)
  nsim <- as.integer(nsim)
  if (k < 0L) stop("k must be non-negative; got ", k)
  .check_case(case)
  need <- n_restrictions(k, case) + 3L
  if (TT < need) {
    stop("TT = ", TT, " is too small for k = ", k, ", case = ", case,
         ": needs at least ", need, " observations")
  }

  if (!is.null(seed)) {
    old <- .save_seed()
    on.exit(.restore_seed(old), add = TRUE)
    set.seed(seed)
  }

  draws <- list(
    I0 = .simulate_one_side(k, case, TT, nsim, integrated = FALSE),
    I1 = .simulate_one_side(k, case, TT, nsim, integrated = TRUE)
  )
  out <- data.frame(
    level = levels,
    I0 = vapply(levels, function(lv) unname(stats::quantile(draws$I0, 1 - lv)), numeric(1)),
    I1 = vapply(levels, function(lv) unname(stats::quantile(draws$I1, 1 - lv)), numeric(1))
  )
  attr(out, "k") <- k
  attr(out, "case") <- case
  attr(out, "TT") <- TT
  attr(out, "nsim") <- nsim
  out
}

.save_seed <- function() {
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv(), inherits = FALSE)
  } else {
    NULL
  }
}

.restore_seed <- function(old) {
  if (is.null(old)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  } else {
    assign(".Random.seed", old, envir = globalenv())
  }
  invisible(NULL)
}

## A handful of published cells, stored only so the simulator can be checked
## against print. Source: Pesaran, Shin and Smith (2001), Tables CI(i)-CI(v).
.PSS_REFERENCE <- list(
  "1,1" = rbind(c(2.44, 3.28), c(3.15, 4.11), c(4.81, 6.02)),
  "2,1" = rbind(c(2.17, 3.19), c(2.72, 3.83), c(3.88, 5.30)),
  "1,2" = rbind(c(3.02, 3.51), c(3.62, 4.16), c(4.94, 5.58)),
  "1,3" = rbind(c(4.04, 4.78), c(4.94, 5.73), c(6.84, 7.84)),
  "2,3" = rbind(c(3.17, 4.14), c(3.79, 4.85), c(5.15, 6.36)),
  "4,3" = rbind(c(2.45, 3.52), c(2.86, 4.01), c(3.74, 5.06)),
  "4,4" = rbind(c(2.68, 3.53), c(3.05, 3.97), c(3.81, 4.92)),
  "4,5" = rbind(c(3.03, 4.06), c(3.47, 4.57), c(4.40, 5.72))
)

#' Published Bounds for a Validated Cell
#'
#' A small set of cells from the printed Pesaran, Shin and Smith (2001) tables,
#' stored so \code{\link{simulate_pss_bounds}} can be checked against print and
#' so the borrowed-bound comparison can quote the exact published number. Only a
#' subset of cells is stored; the simulator has no such gaps.
#'
#' @inheritParams simulate_pss_bounds
#' @return A data frame with columns \code{level}, \code{I0} and \code{I1}.
#' @examples
#' pss_reference(k = 1, case = 3)
#' @export
pss_reference <- function(k, case = 3L) {
  key <- paste(as.integer(k), as.integer(case), sep = ",")
  cell <- .PSS_REFERENCE[[key]]
  if (is.null(cell)) {
    stop("no published reference stored for k = ", k, ", case = ", case,
         ". Available: ", paste(names(.PSS_REFERENCE), collapse = "; "),
         ". Use simulate_pss_bounds() instead.")
  }
  data.frame(level = DEFAULT_LEVELS, I0 = cell[, 1L], I1 = cell[, 2L])
}

#' The k-Convention Offset
#'
#' Two conventions for \eqn{k} sit next to each other in this literature, and
#' mixing them shifts every bound by one row. Pesaran, Shin and Smith count the
#' long-run forcing regressors only; some implementations pass the number of
#' level terms, which is one larger, into a table indexed the first way. Every
#' bound then comes out one row too far down, hence too small, and the test
#' over-rejects.
#'
#' This function puts the correct row next to the shifted one so the size of the
#' distortion can be seen. It is a demonstration, not something the package uses
#' internally: \code{\link{classical_bounds_test}} indexes on \eqn{k}.
#'
#' @inheritParams simulate_pss_bounds
#' @param level Significance level to report.
#' @return A data frame with one row for the correct bounds and one for the
#'   shifted bounds.
#' @examples
#' k_convention_offset(k = 1, case = 3)
#' @export
k_convention_offset <- function(k = 1L, case = 3L, level = 0.05) {
  grab <- function(kk) {
    ref <- tryCatch(pss_reference(kk, case), error = function(e) NULL)
    if (is.null(ref)) ref <- simulate_pss_bounds(kk, case, TT = 500L, nsim = 400L, seed = 0)
    ref[which.min(abs(ref$level - level)), c("I0", "I1")]
  }
  correct <- grab(k)
  shifted <- grab(k + 1L)
  data.frame(
    row = c(sprintf("correct (k = %d)", k),
            sprintf("shifted (k = %d row)", k + 1L)),
    I0 = c(correct$I0, shifted$I0),
    I1 = c(correct$I1, shifted$I1)
  )
}
