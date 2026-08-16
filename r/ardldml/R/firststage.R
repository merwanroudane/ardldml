#' Split the Control Set into Stationary and Integrated Blocks
#'
#' The balanced design has to know which controls to difference, so the two
#' blocks must be named. Passing \code{integrated} explicitly, on economic
#' grounds, is the recommended route: the appeal of the bounds framework is that
#' it avoids pretesting, and a pretest here would put its own error into
#' everything downstream without any of the inference accounting for it.
#'
#' If \code{integrated} is \code{NULL} an Augmented Dickey-Fuller fallback is
#' used, which requires the \pkg{tseries} package.
#'
#' @param W A matrix or data frame of controls in levels, with column names.
#' @param integrated Character vector of control names to treat as integrated of
#'   order one. If supplied, the classification is taken as stated.
#' @param alpha Significance level for the Augmented Dickey-Fuller fallback.
#'
#' @return A list with character vectors \code{stationary} and \code{integrated}.
#' @examples
#' W <- as.matrix(passthrough[, c("m2", "unrate")])
#' classify_controls(W, integrated = "m2")
#' @export
classify_controls <- function(W, integrated = NULL, alpha = 0.10) {
  nms <- colnames(W)
  if (is.null(nms)) stop("W must have column names")
  if (!is.null(integrated)) {
    integrated <- intersect(integrated, nms)
    return(list(stationary = setdiff(nms, integrated), integrated = integrated))
  }
  if (!requireNamespace("tseries", quietly = TRUE)) {
    stop("classify_controls() needs either an explicit 'integrated' argument ",
         "or the 'tseries' package for the Augmented Dickey-Fuller fallback.")
  }
  stationary <- character(0)
  i1 <- character(0)
  for (cn in nms) {
    s <- stats::na.omit(W[, cn])
    p <- tryCatch(suppressWarnings(tseries::adf.test(s)$p.value),
                  error = function(e) 1)
    if (p < alpha) stationary <- c(stationary, cn) else i1 <- c(i1, cn)
  }
  list(stationary = stationary, integrated = i1)
}

#' Build the Balanced First-Stage Design
#'
#' The two nuisance projections have different regressor sets, because their
#' targets have different integration orders. \eqn{\Delta Y_t} is stationary, so
#' the integrated controls enter its projection \strong{in first differences};
#' regressing a stationary target on integrated levels would be unbalanced and
#' spurious. The tested levels \eqn{Z_{t-1} = (Y_{t-1}, D_{t-1})} are integrated
#' and are projected on the control \strong{levels}. That second projection is
#' the only place trend absorption can happen, and therefore the only place the
#' effective integrated count is determined.
#'
#' A stationary regressor cannot track the stochastic trend of an integrated
#' one, so partialling out stationary controls leaves the unit root intact no
#' matter how many there are. Only integrated controls can absorb a trend.
#'
#' @param y Numeric vector: the outcome in levels.
#' @param d Numeric vector: the focal regressor in levels.
#' @param W Matrix or data frame of controls in levels, with column names.
#' @param lags Short-run lag order \eqn{p}: how many lagged differences of
#'   \code{y} (and of \code{d}, if \code{dlags}) enter the stationary design.
#' @param integrated Character vector of controls to treat as integrated of
#'   order one; see \code{\link{classify_controls}}.
#' @param dlags Include lags of \eqn{\Delta D} in the conditional design.
#'   Defaults to \code{FALSE}, which is the paper's equation (3) as written: it
#'   carries the contemporaneous \eqn{\Delta D_t} term and lagged \eqn{\Delta Y}
#'   only. \code{TRUE} gives the general ARDL(p, q) short-run structure, which
#'   is standard practice but is not what the paper specifies. The bootstrap's
#'   marginal model keeps its own lags of \eqn{\Delta D} either way.
#' @param adf_alpha Level for the Augmented Dickey-Fuller fallback.
#'
#' @return An object of class \code{"ardl_design"}: a list with \code{dY},
#'   \code{X} and its column \code{groups}, \code{Z}, \code{Wlev},
#'   \code{dD_lags}, \code{index}, \code{stationary} and \code{integrated}.
#'
#' @examples
#' df <- passthrough_regime("1999-2007")
#' des <- build_balanced_design(df$cpi, df$neer,
#'                              as.matrix(df[, CONTROLS]),
#'                              lags = 4, integrated = DEFAULT_INTEGRATED)
#' des
#'
#' @export
build_balanced_design <- function(y, d, W, lags = 4L, integrated = NULL,
                                  dlags = FALSE, adf_alpha = 0.10) {
  y <- as.numeric(y)
  d <- as.numeric(d)
  W <- as.matrix(W)
  lags <- as.integer(lags)
  if (length(y) != length(d) || length(y) != nrow(W)) {
    stop("y, d and W must have the same number of observations")
  }
  if (is.null(colnames(W))) stop("W must have column names")
  if (anyDuplicated(colnames(W))) {
    stop("duplicate control names: ",
         paste(unique(colnames(W)[duplicated(colnames(W))]), collapse = ", "))
  }
  if (lags < 1L) stop("lags must be at least 1; got ", lags)

  cls <- classify_controls(W, integrated = integrated, alpha = adf_alpha)
  stationary <- cls$stationary
  i1 <- cls$integrated

  n_all <- length(y)
  dY <- .diff1(y)
  dD <- .diff1(d)

  ## Stationary design, in a fixed column order so the bootstrap can address
  ## groups by position rather than by parsing names.
  wfix <- cbind(
    if (length(stationary)) W[, stationary, drop = FALSE] else NULL,
    if (length(i1)) {
      dW <- apply(W[, i1, drop = FALSE], 2L, .diff1)
      colnames(dW) <- paste0("D.", i1)
      dW
    } else NULL
  )
  if (is.null(wfix)) wfix <- matrix(0, n_all, 0)

  ylag <- .lag_matrix(dY, seq_len(lags), prefix = "D.Y.L")
  dlag <- .lag_matrix(dD, seq_len(lags), prefix = "D.D.L")
  dcon <- matrix(dD, ncol = 1L, dimnames = list(NULL, "D.D"))

  X <- cbind(wfix, ylag, if (dlags) dlag else NULL, dcon)
  n_w <- ncol(wfix)
  n_y <- ncol(ylag)
  n_d <- if (dlags) ncol(dlag) else 0L
  groups <- list(
    wfix = seq_len(n_w),
    ylag = if (n_y) n_w + seq_len(n_y) else integer(0),
    dlag = if (n_d) n_w + n_y + seq_len(n_d) else integer(0),
    dD = n_w + n_y + n_d + 1L
  )

  Z <- cbind(Y.L1 = .lag1(y), D.L1 = .lag1(d))
  full <- cbind(dY, X, Z, W, dlag)
  ok <- stats::complete.cases(full)
  idx <- which(ok)
  if (length(idx) < 10L) {
    stop("only ", length(idx), " complete observations after building lags; ",
         "the sample is too short for lags = ", lags)
  }

  ## Section 5.1 keeps the integrated nuisance block fixed-dimensional. Selecting
  ## among many integrated regressors risks picking up irrelevant random walks
  ## through spurious regression, which falsely absorbs stochastic trends. Warn
  ## rather than refuse: the procedure still runs, it is the theory that stops
  ## covering it.
  if (length(i1) > max(10, 0.1 * length(idx))) {
    warning(length(i1), " controls are treated as I(1) on a sample of ",
            length(idx), ". The validity theory holds the integrated block ",
            "fixed-dimensional; selection over a growing integrated block is ",
            "an open problem and can spuriously absorb trends. Read the ",
            "trend-absorption diagnostic.", call. = FALSE)
  }

  structure(
    list(
      dY = dY[idx],
      X = X[idx, , drop = FALSE],
      groups = groups,
      Z = Z[idx, , drop = FALSE],
      Wlev = W[idx, , drop = FALSE],
      dD_lags = dlag[idx, , drop = FALSE],
      index = idx,
      stationary = stationary,
      integrated = i1,
      lags = lags,
      dlags = dlags
    ),
    class = "ardl_design"
  )
}

#' @rdname build_balanced_design
#' @param x An \code{"ardl_design"} object.
#' @param ... Ignored.
#' @export
print.ardl_design <- function(x, ...) {
  cat("Balanced design: n =", length(x$dY), "\n")
  cat("  X  stationary    :", ncol(x$X), "regressors (",
      length(x$stationary), "I(0) levels,", length(x$integrated),
      "I(1) differences, plus lagged differences )\n")
  cat("  Z  tested levels :", paste(colnames(x$Z), collapse = ", "), "\n")
  cat("  W  control levels:", ncol(x$Wlev), "\n")
  invisible(x)
}

.diff1 <- function(v) c(NA_real_, diff(v))

.lag1 <- function(v) c(NA_real_, v[-length(v)])

.lag_matrix <- function(v, lags, prefix) {
  out <- vapply(lags, function(k) {
    c(rep(NA_real_, k), v[seq_len(length(v) - k)])
  }, numeric(length(v)))
  out <- matrix(out, nrow = length(v))
  colnames(out) <- paste0(prefix, lags)
  out
}

#' The Plug-In Penalty
#'
#' \eqn{\lambda = c \sqrt{\log(d)/n} \hat\sigma}, the standard high-dimensional
#' choice: large enough to dominate the noise, small enough to leave the signal.
#' The paper uses \eqn{c = 1.1}.
#'
#' @param n Sample size.
#' @param d Number of regressors.
#' @param sigma Noise scale.
#' @param c Constant.
#' @return A single number.
#' @examples
#' plugin_penalty(n = 100, d = 40, sigma = 1)
#' @export
plugin_penalty <- function(n, d, sigma, c = 1.1) {
  if (n <= 0 || d <= 0) stop("n and d must be positive")
  c * sqrt(log(max(d, 2)) / n) * sigma
}

.standardise <- function(X) {
  mu <- colMeans(X)
  sd <- sqrt(colMeans((X - rep(mu, each = nrow(X)))^2))
  sd[sd < 1e-12] <- 1
  list(Xs = (X - rep(mu, each = nrow(X))) / rep(sd, each = nrow(X)),
       mu = mu, sd = sd)
}

## Single-lambda lasso on a pre-standardised design with no intercept, matching
## the (1/(2n))||y - Xb||^2 + lambda||b||_1 objective. glmnet needs at least two
## columns, so the univariate case is soft-thresholded directly.
.lasso_coef <- function(X, y, lam) {
  d <- ncol(X)
  n <- nrow(X)
  if (d == 0L) return(numeric(0))
  if (d == 1L) {
    xty <- sum(X[, 1L] * y) / n
    xtx <- sum(X[, 1L]^2) / n
    if (xtx < 1e-12) return(0)
    return(sign(xty) * max(abs(xty) - lam, 0) / xtx)
  }
  path <- sort(unique(c(lam * 8, lam * 4, lam * 2, lam)), decreasing = TRUE)
  fit <- glmnet::glmnet(X, y, family = "gaussian", alpha = 1,
                        lambda = path, standardize = FALSE, intercept = FALSE)
  as.numeric(stats::coef(fit, s = lam))[-1L]
}

#' Adaptive Post-LASSO
#'
#' LASSO selection followed by an unpenalised refit on the selected support, with
#' optional adaptive weights on a designated block of columns.
#'
#' The adaptive weighting exists because vanilla \eqn{\ell_1} over-selects
#' integrated regressors and thereby induces spurious trend absorption. Following
#' the paper, the weights apply to the \strong{integrated block only}; stationary
#' columns stay under the plain penalty. The weight on column \eqn{j} is the
#' absolute univariate slope of \code{y} on that column, applied by rescaling the
#' column, which is equivalent to penalising \eqn{|\beta_j|} by its reciprocal.
#'
#' Selection runs on standardised columns so the penalty is scale-free, and the
#' coefficients are mapped back before the refit. Degrees of freedom charge the
#' selected support size, so a cell with non-positive residual degrees of freedom
#' is reported as not estimable rather than silently returning a number.
#'
#' @param X Numeric matrix of regressors.
#' @param y Numeric target.
#' @param lam Penalty. If \code{NULL}, \code{\link{plugin_penalty}} is used with
#'   a noise scale refined once from an initial fit.
#' @param adaptive Enable adaptive weighting. \code{FALSE} gives plain LASSO.
#' @param c Constant in the plug-in penalty.
#' @param adaptive_mask Logical vector marking the columns that receive adaptive
#'   weights. If \code{NULL} and \code{adaptive} is \code{TRUE}, every column is
#'   weighted.
#' @param min_dof Minimum residual degrees of freedom required for the refit.
#'
#' @return A list with \code{fitted}, \code{coef}, \code{support},
#'   \code{intercept}, \code{lam} and \code{estimable}.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 5), 60, 5)
#' y <- X[, 1] + rnorm(60)
#' fit <- adaptive_post_lasso(X, y, adaptive = FALSE)
#' which(fit$support)
#'
#' @references
#' Zou, H. (2006). The adaptive lasso and its oracle properties. \emph{Journal
#' of the American Statistical Association}, 101(476), 1418--1429.
#' \doi{10.1198/016214506000000735}
#' @export
adaptive_post_lasso <- function(X, y, lam = NULL, adaptive = TRUE, c = 1.1,
                                adaptive_mask = NULL, min_dof = 1L) {
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  st <- .standardise(X)
  Xs <- st$Xs
  ybar <- mean(y)
  yc <- y - ybar

  if (adaptive && d > 0L) {
    denom <- colSums(Xs^2)
    denom[denom < 1e-12] <- 1
    w <- pmax(abs(as.numeric(crossprod(Xs, yc))) / denom, 1e-8)
    if (!is.null(adaptive_mask)) {
      mask <- as.logical(adaptive_mask)
      if (length(mask) != d) {
        stop("adaptive_mask has length ", length(mask), " but X has ", d, " columns")
      }
      w[!mask] <- 1
    }
    Xw <- Xs * rep(w, each = n)
  } else {
    w <- rep(1, d)
    Xw <- Xs
  }

  if (is.null(lam)) {
    sigma0 <- stats::sd(yc)
    if (!is.finite(sigma0) || sigma0 == 0) sigma0 <- 1
    lam <- plugin_penalty(n, d, sigma0, c = c)
    b0 <- .lasso_coef(Xw, yc, lam)
    r0 <- yc - as.numeric(Xw %*% b0)
    sigma1 <- stats::sd(r0)
    if (!is.finite(sigma1) || sigma1 == 0) sigma1 <- sigma0
    lam <- plugin_penalty(n, d, sigma1, c = c)
  }

  coef_w <- .lasso_coef(Xw, yc, lam)
  support <- abs(coef_w) > 1e-10
  coef <- numeric(d)
  estimable <- TRUE

  if (any(support)) {
    n_sel <- sum(support)
    if (n - n_sel - 1L < min_dof) {
      estimable <- FALSE
      coef[support] <- coef_w[support] * w[support] / st$sd[support]
    } else {
      Xsel <- Xs[, support, drop = FALSE]
      beta_sel <- stats::lm.fit(Xsel, yc)$coefficients
      beta_sel[is.na(beta_sel)] <- 0
      coef[support] <- beta_sel / st$sd[support]
    }
  }
  intercept <- ybar - sum(st$mu * coef)
  list(fitted = as.numeric(X %*% coef) + intercept,
       coef = coef, support = support, intercept = intercept,
       lam = lam, estimable = estimable)
}

#' Penalty Rules of the Robustness Grid
#'
#' The three penalty choices the paper's robustness tables label Low, Medium and
#' High, mapped onto the points of the cross-validation profile they take.
#'
#' @format A named character vector of length 3.
#' @export
PENALTY_RULES <- c(low = "min", medium = "mid", high = "1se")

#' Rolling-Origin Cross-Validation for the Penalty
#'
#' For origins \eqn{t = T_0, \ldots, T-1} the first stage is fitted on
#' \eqn{\{1, \ldots, t\}} and evaluated one step ahead, and \eqn{\lambda}
#' minimises the average out-of-sample squared error. This respects temporal
#' ordering, unlike ordinary k-fold cross-validation, which would train on the
#' future.
#'
#' The verdict can depend on which point of the profile is taken, which is why
#' \code{\link{penalty_sensitivity}} sweeps all three rather than reporting one.
#'
#' @param X Numeric matrix of regressors.
#' @param y Numeric target.
#' @param grid Penalties to search. Defaults to a log grid spanning two decades
#'   around the plug-in value.
#' @param min_train \eqn{T_0}. Defaults to \code{max(20, n \%/\% 3)}.
#' @param adaptive Adaptive weights during the search.
#' @param n_grid Grid size when \code{grid} is \code{NULL}.
#' @param rule One of \code{"min"} (Low), \code{"1se"} (High) or \code{"mid"}
#'   (Medium, the geometric midpoint); the labels \code{"low"}, \code{"medium"}
#'   and \code{"high"} are accepted too.
#' @param return_profile Also return the grid and its error profile.
#'
#' @return A single penalty, or a list with the profile if
#'   \code{return_profile = TRUE}.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 4), 60, 4)
#' y <- X[, 1] + rnorm(60)
#' tscv_penalty(X, y, n_grid = 4, min_train = 45)
#'
#' @export
tscv_penalty <- function(X, y, grid = NULL, min_train = NULL, adaptive = FALSE,
                         n_grid = 20L, rule = "min", return_profile = FALSE) {
  if (rule %in% names(PENALTY_RULES)) rule <- unname(PENALTY_RULES[rule])
  if (!rule %in% c("min", "1se", "mid")) {
    stop("rule must be one of min/1se/mid (or low/medium/high); got ", rule)
  }
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  if (is.null(min_train)) min_train <- max(20L, n %/% 3L)

  sy <- stats::sd(y)
  if (!is.finite(sy) || sy == 0) sy <- 1
  if (min_train >= n - 1L) {
    lam <- plugin_penalty(n, d, sy)
    return(if (return_profile) list(lam = lam, grid = lam, mse = NA_real_) else lam)
  }
  if (is.null(grid)) {
    base <- plugin_penalty(n, d, sy)
    grid <- exp(seq(log(base / 10), log(base * 10), length.out = n_grid))
  }
  grid <- sort(as.numeric(grid))

  origins <- seq.int(min_train + 1L, n)
  sq <- matrix(NA_real_, nrow = length(grid), ncol = length(origins))
  for (gi in seq_along(grid)) {
    for (j in seq_along(origins)) {
      t <- origins[j]
      fit <- adaptive_post_lasso(X[seq_len(t - 1L), , drop = FALSE],
                                 y[seq_len(t - 1L)],
                                 lam = grid[gi], adaptive = adaptive)
      pred <- sum(X[t, ] * fit$coef) + fit$intercept
      sq[gi, j] <- (y[t] - pred)^2
    }
  }
  mse <- rowMeans(sq)
  i_min <- which.min(mse)
  lam_min <- grid[i_min]
  se <- if (ncol(sq) > 1L) stats::sd(sq[i_min, ]) / sqrt(ncol(sq)) else 0
  within <- which(mse <= mse[i_min] + se)
  lam_1se <- if (length(within)) grid[max(within)] else lam_min
  lam <- switch(rule,
                min = lam_min,
                "1se" = lam_1se,
                mid = sqrt(lam_min * lam_1se))
  if (return_profile) {
    list(lam = lam, grid = grid, mse = mse, lam_min = lam_min, lam_1se = lam_1se)
  } else {
    lam
  }
}

#' Cross-Fitted Projection
#'
#' Out-of-fold predictions of \code{y} on \code{X} under an h-block partition.
#' Each evaluation block is predicted by a model estimated on that block's
#' buffered training set, so the first-stage error is decoupled from the
#' evaluation-fold innovations.
#'
#' @param X Numeric matrix of regressors.
#' @param y Numeric target.
#' @param folds An \code{"ardl_folds"} object from \code{\link{hblock_folds}}.
#' @param lam Penalty passed to \code{\link{adaptive_post_lasso}}.
#' @param adaptive Adaptive weighting.
#' @param c Constant in the plug-in penalty.
#' @param penalised If \code{FALSE}, the projection is unpenalised least
#'   squares: the low-dimensional corner, and the \code{"ols"} arm of the
#'   trend-absorption diagnostic.
#' @param adaptive_mask Columns receiving adaptive weights.
#'
#' @return A list with \code{fitted}, \code{resid}, \code{support_union},
#'   \code{lams} and \code{estimable}.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(80 * 3), 80, 3)
#' y <- X[, 1] + rnorm(80)
#' fit <- cross_fit_projection(X, y, hblock_folds(80, 4, 2), penalised = FALSE)
#' stats::sd(fit$resid)
#'
#' @export
cross_fit_projection <- function(X, y, folds, lam = NULL, adaptive = TRUE,
                                 c = 1.1, penalised = TRUE,
                                 adaptive_mask = NULL) {
  X <- as.matrix(X)
  n <- nrow(X)
  fitted <- rep(NA_real_, n)
  support_union <- rep(FALSE, ncol(X))
  lams <- numeric(0)
  estimable <- TRUE

  for (b in seq_len(folds$n_blocks)) {
    tr <- folds$train[[b]]
    ev <- folds$eval[[b]]
    if (penalised) {
      fit <- adaptive_post_lasso(X[tr, , drop = FALSE], y[tr], lam = lam,
                                 adaptive = adaptive, c = c,
                                 adaptive_mask = adaptive_mask)
      coefs <- fit$coef
      intercept <- fit$intercept
      support_union <- support_union | fit$support
      lams <- c(lams, fit$lam)
      estimable <- estimable && fit$estimable
    } else {
      Xtr <- cbind(1, X[tr, , drop = FALSE])
      beta <- stats::lm.fit(Xtr, y[tr])$coefficients
      beta[is.na(beta)] <- 0
      intercept <- beta[1L]
      coefs <- beta[-1L]
      support_union <- rep(TRUE, ncol(X))
      estimable <- estimable && (length(tr) - ncol(X) - 1L) > 0L
    }
    fitted[ev] <- as.numeric(X[ev, , drop = FALSE] %*% coefs) + intercept
  }

  list(fitted = fitted, resid = y - fitted, support_union = support_union,
       lams = lams, estimable = estimable)
}
