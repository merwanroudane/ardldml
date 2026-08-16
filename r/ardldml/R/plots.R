#' Plot the Simulated Null of a Fitted Test
#'
#' The bootstrap null with the observed statistic and, optionally, a borrowed
#' classical bound marked. The gap between the bootstrap critical value and the
#' borrowed bound is the whole argument for not reading the statistic against a
#' table.
#'
#' @param x A \code{"dml_bounds"} object that has been bootstrapped.
#' @param borrowed Optional classical bound to mark, for contrast.
#' @param breaks Passed to \code{\link[graphics]{hist}}.
#' @param main,xlab Plot labels.
#' @param ... Passed to \code{\link[graphics]{hist}}.
#'
#' @return Invisibly, the object.
#'
#' @examples
#' df <- passthrough_regime("1999-2007")
#' fit <- dml_bounds(df$cpi, df$neer, as.matrix(df[, CONTROLS]),
#'                   lags = 4, n_blocks = 5, buffer = 6,
#'                   integrated = DEFAULT_INTEGRATED, B = 5, seed = 1)
#' plot(fit)
#'
#' @export
#' @importFrom graphics hist abline legend axis box par lines points polygon text
plot.dml_bounds <- function(x, borrowed = 5.73, breaks = 30,
                            main = "Bootstrap null", xlab = "F", ...) {
  if (is.null(x$boot)) {
    stop("no bootstrap attached: call dml_bootstrap() before plotting the null")
  }
  d <- x$boot$draws
  rng <- range(c(d, x$stat, if (!is.null(borrowed)) borrowed), na.rm = TRUE)
  graphics::hist(d, breaks = breaks, col = "#BFDCEC", border = "#0173B2",
                 main = main, xlab = xlab, xlim = rng, ...)
  graphics::abline(v = x$boot$crit, col = ardl_colors[["bootstrap"]], lwd = 2)
  graphics::abline(v = x$stat, col = ardl_colors[["observed"]], lwd = 2)
  lab <- c(sprintf("bootstrap cv = %.2f", x$boot$crit),
           sprintf("observed F = %.2f", x$stat))
  cols <- c(ardl_colors[["bootstrap"]], ardl_colors[["observed"]])
  if (!is.null(borrowed)) {
    graphics::abline(v = borrowed, col = ardl_colors[["borrowed"]], lwd = 2, lty = 2)
    lab <- c(lab, sprintf("borrowed bound = %.2f", borrowed))
    cols <- c(cols, ardl_colors[["borrowed"]])
  }
  graphics::legend("topright", legend = lab, col = cols, lwd = 2,
                   lty = c(1, 1, 2)[seq_along(lab)], bty = "n", cex = 0.85)
  invisible(x)
}

#' Plot the Trend-Absorption Bracket
#'
#' Where the limiting null sits as a function of the number of stochastic trends
#' that survive residualisation. As that count falls from \eqn{k} to zero, the
#' null slides from the integrated endpoint to the stationary one; classical
#' bounds testing is the right-hand end of this picture.
#'
#' @param k Number of level terms.
#' @param lower,upper The classical bracket endpoints.
#' @param k_tilde Optional effective count to mark.
#' @param main,xlab,ylab Plot labels.
#' @return Invisibly, a data frame of the plotted curve.
#' @examples
#' plot_bracket(k = 10, k_tilde = 6)
#' @export
plot_bracket <- function(k = 10L, lower = 4.94, upper = 5.73, k_tilde = NULL,
                         main = "Trend-absorption bracket",
                         xlab = "effective integrated count",
                         ylab = "limiting null, 95th percentile") {
  ks <- seq.int(0L, k)
  ys <- lower + (upper - lower) * ks / k
  graphics::plot(ks, ys, type = "o", pch = 16, lwd = 2,
                 col = ardl_colors[["bootstrap"]],
                 ylim = range(c(ys, lower, upper)),
                 main = main, xlab = xlab, ylab = ylab)
  graphics::abline(h = upper, col = ardl_colors[["i1"]], lty = 2)
  graphics::abline(h = lower, col = ardl_colors[["i0"]], lty = 2)
  graphics::text(0, upper, sprintf("upper I(1) = %.2f", upper),
                 adj = c(0, 1.4), col = ardl_colors[["i1"]], cex = 0.8)
  graphics::text(0, lower, sprintf("lower I(0) = %.2f", lower),
                 adj = c(0, -0.6), col = ardl_colors[["i0"]], cex = 0.8)
  if (!is.null(k_tilde)) {
    yk <- lower + (upper - lower) * k_tilde / k
    graphics::points(k_tilde, yk, pch = 21, bg = "white", cex = 1.6, lwd = 2)
    graphics::text(k_tilde, yk, sprintf(" k-tilde = %g", k_tilde), adj = c(0, 0.5), cex = 0.8)
  }
  invisible(data.frame(k_tilde = ks, null_95 = ys))
}
