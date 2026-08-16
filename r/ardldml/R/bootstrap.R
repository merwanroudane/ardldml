#' Rademacher Weights
#'
#' A vector of independent \eqn{\pm 1} draws.
#'
#' @param n Length.
#' @return A numeric vector of \code{-1} and \code{1}.
#' @examples
#' set.seed(1)
#' rademacher(8)
#' @export
rademacher <- function(n) {
  sample(c(-1, 1), size = n, replace = TRUE)
}

.ols <- function(X, y) {
  fit <- stats::lm.fit(X, y)
  b <- fit$coefficients
  b[is.na(b)] <- 0
  list(coef = b, resid = as.numeric(y - X %*% b))
}

#' The Restricted System Wild Bootstrap
#'
#' Attaches a critical value and a p-value to a fitted \code{\link{dml_bounds}}
#' object. This is the inference: because the tabulated bounds are not
#' operationally valid once the conditioning set is large and persistent, the
#' critical value is generated from the data under the imposed null.
#'
#' Two auxiliary models are estimated under \eqn{H_0}: a restricted conditional
#' model for \eqn{\Delta Y_t} carrying the same deterministic terms and
#' short-run lag structure as the empirical specification but excluding the
#' lagged levels and the confounder levels; and a marginal model for
#' \eqn{\Delta D_t} on an intercept, its own lags and the first-stage-selected
#' differenced controls. A \emph{single} Rademacher sequence is then applied to
#' the \strong{stacked} residual pair, both paths are regenerated recursively,
#' and the entire residualised statistic is recomputed with the level supports
#' re-selected.
#'
#' Sharing one weight is what keeps the endogeneity channel alive. The
#' Pesaran-Shin-Smith framework exists \emph{because} the focal regressor need
#' not be exogenous, and the conditional model absorbs the contemporaneous
#' correlation through the \eqn{\Delta D_t} term. A scheme that holds \code{d} at
#' its realised path and reweights the equation error alone simulates a world
#' with zero correlation between the two innovations whatever the data say; it is
#' available as \code{scheme = "fixed"} and is valid only under strong
#' exogeneity.
#'
#' The controls are held at their realised path, which conditions the bootstrap
#' on the realised trend content. Because the marginal model conditions on the
#' differenced controls, any stochastic trend that \code{d} shares with \code{W}
#' is inherited rather than broken.
#'
#' @param object A fitted \code{"dml_bounds"} object.
#' @param B Bootstrap replications. The paper uses 999.
#' @param level Significance level for the reported critical value.
#' @param seed Optional seed.
#' @param scheme \code{"system"} for Algorithm 1, or \code{"fixed"} for the
#'   strong-exogeneity special case.
#' @param freeze_stationary_support Freeze the stationary first-stage support
#'   across draws while re-selecting the level supports, as the paper does.
#' @param progress Print a line every 10\% of draws.
#'
#' @return The object with a \code{boot} component attached, holding
#'   \code{crit}, \code{pvalue}, \code{draws}, \code{B}, \code{level},
#'   \code{scheme}, \code{n_failed} and \code{corr_eps_v}.
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
#' # Tiny B so the example is quick; use B = 999 for anything you report.
#' fit <- dml_bootstrap(fit, B = 5, seed = 1)
#' fit$boot$pvalue
#'
#' @export
dml_bootstrap <- function(object, B = 999L, level = 0.05, seed = NULL,
                          scheme = c("system", "fixed"),
                          freeze_stationary_support = TRUE,
                          progress = FALSE) {
  scheme <- match.arg(scheme)
  if (!inherits(object, "dml_bounds")) stop("object must be a 'dml_bounds' fit")
  B <- as.integer(B)
  spec <- object$spec
  design <- object$design
  idx <- design$index
  n <- length(design$dY)
  g <- design$groups

  X <- design$X
  dY <- design$dY
  dD <- X[, g$dD]
  dlag_obs <- design$dD_lags
  p <- ncol(dlag_obs)

  ## The observed stationary support: frozen across draws, and defining the
  ## "first-stage-selected differenced controls" of the marginal model.
  dY_support <- object$supports$dY
  frozen <- if (freeze_stationary_support && any(dY_support)) dY_support else NULL
  sel <- which(dY_support)
  wsel <- intersect(g$wfix, sel)
  if (!length(wsel)) wsel <- g$wfix

  ## Step 2a: restricted conditional model for dY under H0.
  rec_cols <- c(g$ylag, g$dlag, g$dD)
  Xr <- cbind(
    if (spec$case != 1L) matrix(1, n, 1L) else NULL,
    X[, rec_cols, drop = FALSE],
    X[, g$wfix, drop = FALSE]
  )
  fitc <- .ols(Xr, dY)
  eps_hat <- fitc$resid
  off <- if (spec$case != 1L) 1L else 0L
  const_c <- if (off) fitc$coef[1L] else 0
  n_y <- length(g$ylag)
  n_dl <- length(g$dlag)
  coef_ylag <- fitc$coef[off + seq_len(n_y)]
  coef_dlag_c <- if (n_dl) fitc$coef[off + n_y + seq_len(n_dl)] else numeric(0)
  coef_dcon <- fitc$coef[off + n_y + n_dl + 1L]
  coef_wfix <- fitc$coef[(off + n_y + n_dl + 1L) + seq_len(length(g$wfix))]
  w_contrib <- if (length(g$wfix)) {
    as.numeric(X[, g$wfix, drop = FALSE] %*% coef_wfix)
  } else rep(0, n)

  ## Step 2b: marginal model for dD.
  Xm <- cbind(matrix(1, n, 1L), dlag_obs, X[, wsel, drop = FALSE])
  fitm <- .ols(Xm, dD)
  v_hat <- fitm$resid
  const_m <- fitm$coef[1L]
  coef_dlag_m <- fitm$coef[1L + seq_len(p)]
  coef_wm <- fitm$coef[(1L + p) + seq_len(length(wsel))]
  wm_contrib <- if (length(wsel)) {
    as.numeric(X[, wsel, drop = FALSE] %*% coef_wm)
  } else rep(0, n)

  ## Anchors for cumulating differences back to levels.
  y0 <- object$y[idx[1L]] - dY[1L]
  d0 <- object$d[idx[1L]] - dD[1L]

  ## Observed pre-sample values. For t <= lag the regenerated path has no
  ## history yet, so the recursion is seeded with the observed difference, which
  ## the design already holds correctly aligned.
  ylag_obs <- X[, g$ylag, drop = FALSE]
  dlag_c_obs <- if (n_dl) X[, g$dlag, drop = FALSE] else NULL

  one_draw <- function(b) {
    if (!is.null(seed)) set.seed(seed + b)
    eta <- rademacher(n)
    eps_star <- eps_hat * eta

    if (scheme == "system") {
      v_star <- v_hat * eta
      dD_star <- numeric(n)
      for (t in seq_len(n)) {
        val <- const_m + wm_contrib[t] + v_star[t]
        for (i in seq_len(p)) {
          val <- val + coef_dlag_m[i] *
            (if (t - i >= 1L) dD_star[t - i] else dlag_obs[t, i])
        }
        dD_star[t] <- val
      }
    } else {
      dD_star <- dD
    }

    dY_star <- numeric(n)
    for (t in seq_len(n)) {
      val <- const_c + w_contrib[t] + eps_star[t] + coef_dcon * dD_star[t]
      for (i in seq_len(n_y)) {
        val <- val + coef_ylag[i] *
          (if (t - i >= 1L) dY_star[t - i] else ylag_obs[t, i])
      }
      if (n_dl) {
        for (i in seq_len(n_dl)) {
          val <- val + coef_dlag_c[i] *
            (if (t - i >= 1L) dD_star[t - i] else dlag_c_obs[t, i])
        }
      }
      dY_star[t] <- val
    }

    y_star <- y0 + cumsum(dY_star)
    d_star <- d0 + cumsum(dD_star)
    out <- tryCatch(
      compute_statistic(y_star, d_star, object$W[idx, , drop = FALSE], spec,
                        frozen_dY_support = frozen),
      error = function(e) NULL
    )
    if (is.null(out)) NA_real_ else out$stat
  }

  draws <- numeric(B)
  step <- max(B %/% 10L, 1L)
  for (b in seq_len(B)) {
    draws[b] <- one_draw(b - 1L)
    if (progress && b %% step == 0L) {
      message(sprintf("  bootstrap %d/%d", b, B))
    }
  }

  good <- draws[is.finite(draws)]
  if (!length(good)) stop("every bootstrap draw failed; check the specification")

  object$boot <- list(
    crit = unname(stats::quantile(good, 1 - level)),
    pvalue = mean(good >= object$stat),
    draws = good,
    B = B,
    level = level,
    scheme = scheme,
    n_failed = length(draws) - length(good),
    eps_hat = eps_hat,
    v_hat = v_hat,
    corr_eps_v = stats::cor(eps_hat, v_hat)
  )
  object
}
