#' The Trend-Absorption Diagnostic
#'
#' The most important thing in the package to actually run, and the easiest to
#' skip.
#'
#' Residualisation is valid only when the controls remove \emph{nuisance}
#' stochastic trends, not the equilibrium relation being tested. If a control is
#' itself part of the equilibrium system, partialling it out absorbs the very
#' trend the test is meant to detect, and a non-rejection then reflects
#' over-absorption rather than the absence of a long-run relationship. That
#' requirement cannot be verified directly; this contrast is the practical
#' substitute.
#'
#' Four fits are run: the full and reduced control sets crossed with the
#' adaptive and the unpenalised level projection. Writing \eqn{p_{ad}} and
#' \eqn{p_{ols}} for the bootstrap p-values under the two projections and
#' \eqn{p_{full}} and \eqn{p_{red}} for those under the two control sets, the
#' diagnostic is the pair of gaps \eqn{\Delta_m = p_{ols} - p_{ad}} and
#' \eqn{\Delta_W = p_{full} - p_{red}}, read together with the stability of the
#' long-run coefficient.
#'
#' A large positive \eqn{\Delta_W} -- a verdict that rejects under the reduced
#' set but not under the full set -- together with a long-run coefficient that is
#' more sharply estimated under the reduced fit, is evidence that the nuisance
#' space is absorbing part of the tested relation, and the reduced-set verdict is
#' then the more credible one. Concordant verdicts across the four fits indicate
#' the conclusion is not an artefact of over-absorption.
#'
#' This is a hypothesis-generating device, not a formal test. It has no size and
#' no power; it tells you where to look.
#'
#' @param y Numeric vector: outcome in levels.
#' @param d Numeric vector: focal regressor in levels.
#' @param W Matrix or data frame of controls in levels.
#' @param drop Character vector of controls to omit from the reduced set. Choose
#'   these on \emph{economic} grounds: the ones most likely to be cointegrated
#'   with the tested relation itself.
#' @param B Bootstrap replications per fit. Four bootstraps are run, so the cost
#'   is roughly \code{4 * B} statistic evaluations.
#' @param seed Base seed; each fit is offset so the four bootstraps are
#'   independent.
#' @param progress Report which fit is running.
#' @param ... Passed to \code{\link{dml_spec}}; \code{adaptive} is set per fit.
#'
#' @return An object of class \code{"trend_absorption"}.
#'
#' @examples
#' \donttest{
#' df <- passthrough_regime("1999-2007")
#' ta <- trend_absorption(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
#'                        drop = REDUCED_DROP, B = 19, seed = 1,
#'                        lags = 4, n_blocks = 5, buffer = 6,
#'                        integrated = DEFAULT_INTEGRATED)
#' ta
#' }
#'
#' @references
#' Villena, M. J. (2026). Testing cointegration with many persistent controls.
#' SSRN working paper. \doi{10.2139/ssrn.6472826}
#' @export
trend_absorption <- function(y, d, W, drop, B = 999L, seed = NULL,
                             progress = FALSE, ...) {
  W <- as.matrix(W)
  drop <- intersect(drop, colnames(W))
  W_red <- W[, setdiff(colnames(W), drop), drop = FALSE]

  args <- list(...)
  integrated <- args$integrated
  args$integrated <- NULL
  integrated_red <- if (is.null(integrated)) NULL else setdiff(integrated, drop)

  plan <- list(
    list(key = "full_adaptive", W = W, adaptive = TRUE, integ = integrated, off = 0L),
    list(key = "full_ols", W = W, adaptive = FALSE, integ = integrated, off = 1L),
    list(key = "reduced_adaptive", W = W_red, adaptive = TRUE, integ = integrated_red, off = 2L),
    list(key = "reduced_ols", W = W_red, adaptive = FALSE, integ = integrated_red, off = 3L)
  )

  fits <- list()
  for (p in plan) {
    if (progress) message("  fitting ", p$key, " (", ncol(p$W), " controls)...")
    call_args <- c(list(y = y, d = d, W = p$W, B = B, seed = if (is.null(seed)) NULL else seed + 10000L * p$off),
                   args, list(adaptive = p$adaptive, integrated = p$integ))
    fits[[p$key]] <- do.call(dml_bounds, call_args)
  }

  structure(list(
    fits = fits,
    dropped = drop,
    delta_m = fits$full_ols$boot$pvalue - fits$full_adaptive$boot$pvalue,
    delta_W = fits$full_adaptive$boot$pvalue - fits$reduced_adaptive$boot$pvalue
  ), class = "trend_absorption")
}

#' @rdname trend_absorption
#' @param x A \code{"trend_absorption"} object.
#' @export
as.data.frame.trend_absorption <- function(x, ...) {
  do.call(rbind, lapply(names(x$fits), function(k) {
    r <- x$fits[[k]]
    parts <- strsplit(k, "_", fixed = TRUE)[[1L]]
    data.frame(controls = parts[1L], projection = parts[2L],
               n_controls = r$n_controls, F = r$stat,
               boot_cv95 = if (is.null(r$boot)) NA_real_ else r$boot$crit,
               boot_p = if (is.null(r$boot)) NA_real_ else r$boot$pvalue,
               verdict = decision(r), alpha = r$alpha,
               theta = r$theta, theta_se = r$theta_se,
               stringsAsFactors = FALSE)
  }))
}

#' @rdname trend_absorption
#' @param level Significance level.
#' @export
absorption_verdict <- function(x, level = 0.05) {
  full <- x$fits$full_adaptive
  red <- x$fits$reduced_adaptive
  if (is.null(full$boot) || is.null(red$boot)) return("no bootstrap run")
  rej_full <- full$boot$pvalue < level
  rej_red <- red$boot$pvalue < level

  if (rej_red && !rej_full && x$delta_W > 0) {
    msg <- sprintf(paste0("Possible over-absorption. The reduced set rejects (p = %.3f) ",
                          "where the full set does not (p = %.3f); Delta_W = %+.3f."),
                   red$boot$pvalue, full$boot$pvalue, x$delta_W)
    if (red$theta_se < full$theta_se) {
      msg <- paste0(msg, sprintf(paste0(" The long-run coefficient is also more sharply ",
                                        "estimated under the reduced set (se %.3f versus %.3f), ",
                                        "which strengthens the reading. The reduced-set verdict ",
                                        "is the more credible."), red$theta_se, full$theta_se))
    } else {
      msg <- paste0(msg, " The long-run coefficient is not more sharply estimated under ",
                    "the reduced set, so ordinary specification sensitivity cannot be ruled out.")
    }
    return(msg)
  }
  if (rej_red == rej_full) {
    return(sprintf(paste0("Concordant verdicts across control sets (full p = %.3f, ",
                          "reduced p = %.3f). The conclusion does not appear to be an ",
                          "artefact of over-absorption."),
                   full$boot$pvalue, red$boot$pvalue))
  }
  sprintf(paste0("Discordant, but not in the over-absorption direction (full p = %.3f, ",
                 "reduced p = %.3f). Treat both verdicts as fragile."),
          full$boot$pvalue, red$boot$pvalue)
}

#' @rdname trend_absorption
#' @export
print.trend_absorption <- function(x, ...) {
  cat("Trend-absorption diagnostic\n")
  cat("dropped from reduced set:",
      if (length(x$dropped)) paste(x$dropped, collapse = ", ") else "(none)", "\n\n")
  print(as.data.frame(x), row.names = FALSE, digits = 4)
  th <- vapply(x$fits, function(f) f$theta, numeric(1))
  th <- th[is.finite(th)]
  cat(sprintf("\nDelta_m = p_ols  - p_ad  = %+.4f\n", x$delta_m))
  cat(sprintf("Delta_W = p_full - p_red = %+.4f\n", x$delta_W))
  if (length(th)) cat(sprintf("theta spread across fits = %.4f\n", max(th) - min(th)))
  cat("\n", absorption_verdict(x), "\n", sep = "")
  invisible(x)
}

#' Sweep the Penalty and the Projection
#'
#' The paper's robustness tables vary the penalty, the lag order and the level
#' projection, and a verdict can turn on that choice. Reporting a single cell
#' from this grid is specification search; reporting the grid is the method.
#'
#' The column to watch is \code{n_selected_Z}, the number of control
#' \strong{levels} the level projection retained. It is the empirical
#' counterpart of the effective integrated count: zero means nothing was
#' absorbed, so the test sits at the classical corner and orthogonalisation did
#' nothing, while a large value means heavy absorption and the long-run
#' coefficient should be checked for instability. A long-run coefficient that
#' changes sign across the grid is a warning that the conditioning set, not the
#' data, is driving the answer.
#'
#' @inheritParams trend_absorption
#' @param rules Penalty rules to sweep: \code{"low"}, \code{"medium"},
#'   \code{"high"}.
#' @param lags_grid Short-run lag orders to try.
#' @param projections Estimators to compare: \code{"adaptive"} (the paper's),
#'   \code{"plain"} (vanilla \eqn{\ell_1}) and \code{"ols"} (the unpenalised
#'   benchmark).
#' @param B Bootstrap draws per cell, or \code{0} for statistics only, which is
#'   much faster.
#'
#' @return A data frame with one row per cell, carrying a
#'   \code{"theta_sign_flips"} attribute.
#'
#' @examples
#' \donttest{
#' df <- passthrough_regime("1999-2007")
#' penalty_sensitivity(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
#'                     lags_grid = 4, n_blocks = 5, buffer = 6,
#'                     integrated = DEFAULT_INTEGRATED)
#' }
#' @export
penalty_sensitivity <- function(y, d, W, rules = c("low", "medium", "high"),
                                lags_grid = 4L,
                                projections = c("adaptive", "plain", "ols"),
                                B = 0L, seed = NULL, ...) {
  rows <- list()
  for (lg in lags_grid) {
    for (proj in projections) {
      for (setting in rules) {
        args <- list(...)
        args$lags <- lg
        if (proj == "ols") {
          args$penalised <- FALSE
        } else {
          args$penalised <- TRUE
          args$adaptive <- proj == "adaptive"
          args$penalty <- setting
        }
        res <- tryCatch(
          do.call(dml_bounds, c(list(y = y, d = d, W = W, B = B, seed = seed), args)),
          error = function(e) e
        )
        if (inherits(res, "error")) {
          rows[[length(rows) + 1L]] <- data.frame(
            lags = lg, projection = proj, penalty = setting,
            n_selected_Z = NA_integer_, F = NA_real_, boot_p = NA_real_,
            alpha = NA_real_, theta = NA_real_, theta_se = NA_real_,
            stringsAsFactors = FALSE)
        } else {
          rows[[length(rows) + 1L]] <- data.frame(
            lags = lg, projection = proj,
            penalty = if (proj == "ols") "-" else setting,
            n_selected_Z = sum(res$supports$Z), F = res$stat,
            boot_p = if (is.null(res$boot)) NA_real_ else res$boot$pvalue,
            alpha = res$alpha, theta = res$theta, theta_se = res$theta_se,
            stringsAsFactors = FALSE)
        }
        if (proj == "ols") break   # the penalty is irrelevant without penalisation
      }
    }
  }
  out <- do.call(rbind, rows)
  fin <- out$theta[is.finite(out$theta)]
  attr(out, "theta_sign_flips") <- length(fin) > 0L && min(fin) < 0 && max(fin) > 0
  out
}
