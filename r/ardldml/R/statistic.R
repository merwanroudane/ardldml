#' Specification of a DML-Bounds Fit
#'
#' Everything that defines a fit, held separately from the data so the bootstrap
#' can rebuild the identical statistic on a regenerated path.
#'
#' @param lags Short-run lag order \eqn{p}.
#' @param case Deterministic case, 1 or 3. This does \strong{not} enter the
#'   statistic, which is a no-intercept projection on the orthogonalised levels
#'   regardless. It sets the deterministic terms of the restricted conditional
#'   model the bootstrap resamples from.
#' @param n_blocks Cross-fitting blocks \eqn{K}.
#' @param buffer Cross-fitting buffer \eqn{h}.
#' @param adaptive Adaptive weights on the level projection. \code{TRUE} is the
#'   paper's default; \code{FALSE} gives the plain-LASSO arm of the diagnostic.
#' @param penalised If \code{FALSE}, both projections are unpenalised least
#'   squares -- the low-dimensional corner and the \code{"ols"} arm.
#' @param penalty How the \eqn{\Delta Y} penalty is chosen: \code{"plugin"} (the
#'   default), one of \code{"low"}, \code{"medium"}, \code{"high"} (equivalently
#'   \code{"min"}, \code{"mid"}, \code{"1se"}) for rolling-origin
#'   cross-validation, or a number fixing it directly. The level projection
#'   always uses the plug-in rule.
#' @param c Constant in the plug-in penalty.
#' @param integrated Character vector of controls to treat as integrated of
#'   order one.
#' @param include_constant Add an intercept to the final regression. Defaults to
#'   \code{FALSE}, which is the paper's equation (10) as written; the first-stage
#'   projections already carry intercepts, so the residuals are mean-zero by
#'   construction. Exposed only for sensitivity checks.
#' @param adaptive_integrated_only Restrict the adaptive weights to the
#'   integrated block, which is what the paper specifies.
#' @param dlags Include lags of \eqn{\Delta D} in the conditional design.
#'
#' @return An object of class \code{"dml_spec"}.
#' @examples
#' dml_spec(lags = 4, n_blocks = 5, buffer = 6)
#' @export
dml_spec <- function(lags = 4L, case = 3L, n_blocks = 5L, buffer = 0L,
                     adaptive = TRUE, penalised = TRUE, penalty = "plugin",
                     c = 1.1, integrated = NULL, include_constant = FALSE,
                     adaptive_integrated_only = TRUE, dlags = FALSE) {
  structure(list(lags = as.integer(lags), case = as.integer(case),
                 n_blocks = as.integer(n_blocks), buffer = as.integer(buffer),
                 adaptive = adaptive, penalised = penalised, penalty = penalty,
                 c = c, integrated = integrated,
                 include_constant = include_constant,
                 adaptive_integrated_only = adaptive_integrated_only,
                 dlags = dlags),
            class = "dml_spec")
}

#' @rdname dml_spec
#' @param x A \code{"dml_spec"} object.
#' @param ... Ignored.
#' @export
print.dml_spec <- function(x, ...) {
  cat(sprintf("DML-Bounds specification: lags = %d, K = %d, h = %d\n",
              x$lags, x$n_blocks, x$buffer))
  cat(sprintf("  projection = %s, penalty = %s, c = %.2f, case = %d\n",
              if (!x$penalised) "ols" else if (x$adaptive) "adaptive" else "plain",
              as.character(x$penalty), x$c, x$case))
  invisible(x)
}

## F form of the Wald test that both level coefficients are zero, following
## equation (10): an unpenalised regression on the orthogonalised variables,
## with no intercept.
.wald_levels <- function(dy_res, Z_res, include_constant = FALSE) {
  n <- nrow(Z_res)
  X <- if (include_constant) cbind(1, Z_res) else Z_res
  tested <- if (include_constant) seq.int(2L, ncol(X)) else seq_len(ncol(X))
  bad <- list(stat = NA_real_, alpha = NA_real_, theta = NA_real_,
              theta_se = NA_real_)

  fit <- stats::lm.fit(X, dy_res)
  beta <- fit$coefficients
  if (anyNA(beta)) return(bad)
  resid <- fit$residuals
  dof <- n - ncol(X)
  if (dof <= 0L) return(bad)
  s2 <- sum(resid^2) / dof
  xtx_inv <- tryCatch(solve(crossprod(X)), error = function(e) MASS_ginv(crossprod(X)))
  V <- s2 * xtx_inv

  b <- beta[tested]
  Vb <- V[tested, tested, drop = FALSE]
  stat <- tryCatch(as.numeric(crossprod(b, solve(Vb, b))) / length(tested),
                   error = function(e) NA_real_)
  if (!is.finite(stat)) return(bad)

  pi_y <- b[1L]
  pi_x <- b[2L]
  alpha <- -pi_y
  if (abs(alpha) < 1e-12) {
    return(list(stat = stat, alpha = alpha, theta = NA_real_, theta_se = NA_real_))
  }
  theta <- pi_x / alpha
  ## Delta method for theta = -pi_x / pi_y.
  g <- c(pi_x / pi_y^2, -1 / pi_y)
  theta_se <- sqrt(max(as.numeric(crossprod(g, Vb %*% g)), 0))
  list(stat = unname(stat), alpha = unname(alpha),
       theta = unname(theta), theta_se = unname(theta_se))
}

#' Build the Design, Cross-Fit, Residualise and Test
#'
#' The whole pipeline in one call, so the bootstrap can invoke it on a
#' regenerated path and obtain a statistic carrying the same
#' generated-regressor error as the observed one.
#'
#' @param y Numeric vector: outcome in levels.
#' @param d Numeric vector: focal regressor in levels.
#' @param W Matrix or data frame of controls in levels.
#' @param spec A \code{\link{dml_spec}}.
#' @param frozen_dY_support Optional logical vector fixing the support of the
#'   stationary projection. The paper freezes the stationary support across
#'   bootstrap draws while re-selecting the level supports, so that selection
#'   error is reflected in the bootstrap law.
#'
#' @return A list with \code{stat}, \code{alpha}, \code{theta}, \code{theta_se},
#'   \code{design}, \code{folds}, \code{supports}, \code{resid} and
#'   \code{estimable}.
#'
#' @examples
#' df <- passthrough_regime("1999-2007")
#' out <- compute_statistic(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
#'                          dml_spec(lags = 4, n_blocks = 5, buffer = 6,
#'                                   integrated = DEFAULT_INTEGRATED))
#' round(out$stat, 4)
#'
#' @export
compute_statistic <- function(y, d, W, spec, frozen_dY_support = NULL) {
  design <- build_balanced_design(y, d, W, lags = spec$lags,
                                  integrated = spec$integrated,
                                  dlags = spec$dlags)
  n <- length(design$dY)
  folds <- hblock_folds(n, n_blocks = spec$n_blocks, buffer = spec$buffer)

  Xs <- design$X
  Wl <- design$Wlev
  dy <- design$dY
  Zl <- design$Z

  use_frozen <- !is.null(frozen_dY_support) && any(frozen_dY_support)
  Xs_use <- if (use_frozen) Xs[, frozen_dY_support, drop = FALSE] else Xs

  lam_dy <- NULL
  if (is.numeric(spec$penalty)) {
    lam_dy <- as.numeric(spec$penalty)
  } else if (isTRUE(spec$penalised) &&
             spec$penalty %in% c("tscv", "min", "1se", "mid", "low", "medium", "high")) {
    rule <- if (identical(spec$penalty, "tscv")) "min" else spec$penalty
    lam_dy <- tscv_penalty(Xs_use, dy, adaptive = FALSE, rule = rule)
  }

  ## Stationary projection of dY: plain LASSO is appropriate here.
  fit_dy <- cross_fit_projection(Xs_use, dy, folds, lam = lam_dy,
                                 adaptive = FALSE, c = spec$c,
                                 penalised = spec$penalised)

  ## Level projection of each component of Z on the control levels. The adaptive
  ## weights apply to the integrated block only, and the penalty here is always
  ## the plug-in value.
  integrated_mask <- colnames(Wl) %in% design$integrated
  z_res <- matrix(NA_real_, nrow = n, ncol = ncol(Zl),
                  dimnames = list(NULL, colnames(Zl)))
  z_support <- rep(FALSE, ncol(Wl))
  estimable <- fit_dy$estimable

  for (j in seq_len(ncol(Zl))) {
    fit_z <- cross_fit_projection(
      Wl, Zl[, j], folds, lam = NULL, adaptive = spec$adaptive, c = spec$c,
      penalised = spec$penalised,
      adaptive_mask = if (isTRUE(spec$adaptive_integrated_only)) integrated_mask else NULL
    )
    z_res[, j] <- fit_z$resid
    z_support <- z_support | fit_z$support_union
    estimable <- estimable && fit_z$estimable
  }

  out <- .wald_levels(fit_dy$resid, z_res, include_constant = spec$include_constant)

  ## When the stationary support is frozen the projection ran on a subset of
  ## columns, so expand its mask back to the full design width or the frozen
  ## mask on the next bootstrap path would be misaligned.
  dy_support <- fit_dy$support_union
  if (use_frozen) {
    full <- rep(FALSE, ncol(Xs))
    full[which(frozen_dY_support)[dy_support]] <- TRUE
    dy_support <- full
  }

  c(out, list(design = design, folds = folds,
              supports = list(dY = dy_support, Z = z_support),
              resid = list(dY = fit_dy$resid, Z = z_res),
              estimable = isTRUE(estimable)))
}

#' Test for a Long-Run Relationship with Many Persistent Controls
#'
#' Fits the DML-Bounds test of Villena (2026): the lagged levels
#' \eqn{Z_{t-1} = (Y_{t-1}, D_{t-1})} are orthogonalised against a
#' high-dimensional control set with a balanced, cross-fitted first stage, and
#' the null of no level relationship is tested on the residualised variables.
#'
#' Do \strong{not} compare the statistic with a tabulated bound. The asymptotic
#' reference sits somewhere inside the Pesaran-Shin-Smith bracket, at a position
#' governed by the number of stochastic trends that survive residualisation, and
#' the finite-sample law is further perturbed by generated-regressor error. Call
#' \code{\link{dml_bootstrap}}, or pass \code{B}, to attach a critical value.
#'
#' The estimand is conditional. This does not ask whether \code{y} and \code{d}
#' cointegrate unconditionally; it asks whether they cointegrate \emph{given}
#' \code{W}. If a control is itself part of the equilibrium system, partialling
#' it out removes the relation rather than the confounding, and a non-rejection
#' then reflects over-absorption. That failure mode is not visible in a single
#' fit -- use \code{\link{trend_absorption}}.
#'
#' @param y Numeric vector: outcome in levels.
#' @param d Numeric vector: focal regressor in levels.
#' @param W Matrix or data frame of controls in levels, with column names.
#' @param B Bootstrap replications. \code{0} fits only, which is fast and is
#'   what the short examples use.
#' @param level Significance level for the reported critical value.
#' @param seed Optional seed for the bootstrap.
#' @param ... Passed to \code{\link{dml_spec}}.
#'
#' @return An object of class \code{"dml_bounds"}.
#'
#' @references
#' Villena, M. J. (2026). Testing cointegration with many persistent controls.
#' SSRN working paper. \doi{10.2139/ssrn.6472826}
#'
#' @examples
#' df <- passthrough_regime("1999-2007")
#' fit <- dml_bounds(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
#'                   lags = 4, n_blocks = 5, buffer = 6,
#'                   integrated = DEFAULT_INTEGRATED)
#' fit
#'
#' \donttest{
#' # With inference. The paper uses B = 999; this is smaller so it stays quick.
#' fit <- dml_bootstrap(fit, B = 99, seed = 20260625)
#' summary(fit)
#' }
#'
#' @seealso \code{\link{dml_bootstrap}}, \code{\link{trend_absorption}},
#'   \code{\link{classical_bounds_test}}
#' @export
dml_bounds <- function(y, d, W, B = 0L, level = 0.05, seed = NULL, ...) {
  spec <- dml_spec(...)
  out <- compute_statistic(y, d, W, spec)
  fit <- structure(
    list(stat = out$stat, alpha = out$alpha, theta = out$theta,
         theta_se = out$theta_se, spec = spec, design = out$design,
         folds = out$folds, supports = out$supports, resid = out$resid,
         estimable = out$estimable, nobs = length(out$design$dY),
         n_controls = ncol(out$design$Wlev),
         y = as.numeric(y), d = as.numeric(d), W = as.matrix(W),
         boot = NULL),
    class = "dml_bounds"
  )
  if (B > 0L) fit <- dml_bootstrap(fit, B = B, level = level, seed = seed)
  fit
}

#' @rdname dml_bounds
#' @param x,object A \code{"dml_bounds"} object.
#' @export
print.dml_bounds <- function(x, ...) {
  cat("DML-Bounds test for a conditional long-run relationship\n")
  cat(sprintf("F = %.4f on n = %d, %d controls (%d treated as I(1)), K = %d, h = %d\n",
              x$stat, x$nobs, x$n_controls, length(x$design$integrated),
              x$spec$n_blocks, x$spec$buffer))
  if (is.null(x$boot)) {
    cat("No bootstrap critical value attached; call dml_bootstrap().\n")
  } else {
    cat(sprintf("bootstrap p = %.4f, critical value (%.0f%%) = %.4f -> %s\n",
                x$boot$pvalue, 100 * (1 - x$boot$level), x$boot$crit,
                decision(x)))
  }
  invisible(x)
}

#' @rdname dml_bounds
#' @export
summary.dml_bounds <- function(object, ...) {
  x <- object
  cat("DML-Bounds test for a conditional long-run relationship\n")
  cat(sprintf("H0: no level relationship after residualisation      F = %.4f\n", x$stat))
  cat(sprintf("n = %d, controls = %d (%d treated as I(1)), K = %d, h = %d\n",
              x$nobs, x$n_controls, length(x$design$integrated),
              x$spec$n_blocks, x$spec$buffer))
  cat(sprintf("selected: %d stationary, %d level controls\n\n",
              sum(x$supports$dY), sum(x$supports$Z)))
  if (is.null(x$boot)) {
    cat("No bootstrap critical value attached.\n")
    cat("The statistic has no tabulated reference: call dml_bootstrap() before\n")
    cat("interpreting it. Comparing it with the classical bound over-rejects\n")
    cat("when the controls are integrated.\n")
  } else {
    b <- x$boot
    cat(sprintf("restricted system wild bootstrap: B = %d, %s scheme\n", b$B, b$scheme))
    cat(sprintf("  critical value (%.0f%%) = %.4f\n", 100 * (1 - b$level), b$crit))
    cat(sprintf("  bootstrap p-value       = %.4f\n", b$pvalue))
    cat(sprintf("  decision at %.0f%%          -> %s\n", 100 * b$level, decision(x)))
  }
  cat(sprintf("\nspeed of adjustment  alpha = %.4f\n", x$alpha))
  cat(sprintf("long-run coefficient theta = %.4f  (se %.4f)\n", x$theta, x$theta_se))
  if (!x$estimable) cat("\nWARNING: a projection exhausted its degrees of freedom.\n")
  invisible(x)
}

#' Verdict of a Fitted Test
#'
#' @param object A \code{"dml_bounds"} object.
#' @param level Significance level.
#' @return \code{"reject"}, \code{"fail to reject"}, or \code{"no bootstrap run"}.
#' @examples
#' df <- passthrough_regime("1999-2007")
#' fit <- dml_bounds(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
#'                   lags = 4, n_blocks = 5, buffer = 6,
#'                   integrated = DEFAULT_INTEGRATED)
#' decision(fit)
#' @export
decision <- function(object, level = 0.05) {
  if (is.null(object$boot)) return("no bootstrap run")
  if (object$boot$pvalue < level) "reject" else "fail to reject"
}
