#' h-Block Cross-Fitting Folds
#'
#' Cuts the sample into \code{n_blocks} contiguous chronological blocks and
#' builds, for each, a training set that excludes the block itself and the
#' \code{buffer} observations either side of it.
#'
#' In cross-sectional double machine learning the folds are random. In a time
#' series they cannot be: randomly held-out observations are adjacent in time to
#' their own training data, so the first-stage error stays correlated with the
#' evaluation innovations. The buffer is what buys the decoupling, and it costs
#' sample -- at most \eqn{h(K-1)} observations across the training sets. Use
#' \code{\link{sample_use_table}} to see the cost before choosing a
#' configuration.
#'
#' @param n Number of observations.
#' @param n_blocks Number of chronological blocks \eqn{K}; at least 2.
#' @param buffer Buffer width \eqn{h} in observations.
#'
#' @return An object of class \code{"ardl_folds"}: a list with \code{n},
#'   \code{n_blocks}, \code{buffer}, and lists \code{eval} and \code{train} of
#'   integer index vectors.
#'
#' @examples
#' f <- hblock_folds(100, n_blocks = 4, buffer = 5)
#' sapply(f$train, length)
#'
#' @seealso \code{\link{sample_use}}, \code{\link{sample_use_table}}
#' @export
hblock_folds <- function(n, n_blocks = 5L, buffer = 0L) {
  n <- as.integer(n)
  n_blocks <- as.integer(n_blocks)
  buffer <- as.integer(buffer)
  if (n_blocks < 2L) stop("n_blocks must be at least 2; got ", n_blocks)
  if (buffer < 0L) stop("buffer must be non-negative; got ", buffer)
  if (n_blocks > n) stop("n_blocks = ", n_blocks, " exceeds n = ", n)

  edges <- as.integer(seq(0, n, length.out = n_blocks + 1L))
  eval_blocks <- vector("list", n_blocks)
  train_blocks <- vector("list", n_blocks)

  for (b in seq_len(n_blocks)) {
    lo <- edges[b]
    hi <- edges[b + 1L]
    ev <- seq.int(lo + 1L, hi)
    keep <- rep(TRUE, n)
    a <- max(0L, lo - buffer)
    z <- min(n, hi + buffer)
    if (z > a) keep[seq.int(a + 1L, z)] <- FALSE
    tr <- which(keep)
    if (length(tr) == 0L) {
      stop("block ", b, " has an empty training set: buffer = ", buffer,
           " is too wide for n = ", n, " with n_blocks = ", n_blocks)
    }
    eval_blocks[[b]] <- ev
    train_blocks[[b]] <- tr
  }

  structure(
    list(n = n, n_blocks = n_blocks, buffer = buffer,
         eval = eval_blocks, train = train_blocks),
    class = "ardl_folds"
  )
}

#' @rdname hblock_folds
#' @param x An \code{"ardl_folds"} object.
#' @param ... Ignored.
#' @export
as.data.frame.ardl_folds <- function(x, ...) {
  data.frame(
    block = seq_len(x$n_blocks),
    eval_start = vapply(x$eval, function(e) e[1L], numeric(1)),
    eval_end = vapply(x$eval, function(e) e[length(e)], numeric(1)),
    n_eval = vapply(x$eval, length, numeric(1)),
    n_train = vapply(x$train, length, numeric(1)),
    share_train = vapply(x$train, length, numeric(1)) / x$n
  )
}

#' @rdname hblock_folds
#' @export
print.ardl_folds <- function(x, ...) {
  cat(sprintf("h-block folds: n = %d, K = %d, h = %d, mean training share = %.2f\n",
              x$n, x$n_blocks, x$buffer,
              mean(vapply(x$train, length, numeric(1))) / x$n))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}

#' Share of the Sample Available for Training
#'
#' Average, across blocks, of the training-set size as a fraction of \code{n}.
#' With no buffer this is \eqn{1 - 1/K}; each unit of \eqn{h} removes roughly
#' \eqn{2h/n} more. This is the quantity that makes the cost of buffering
#' concrete.
#'
#' @inheritParams hblock_folds
#' @return A single number between 0 and 1.
#' @examples
#' sample_use(108, n_blocks = 5, buffer = 6)
#' @export
sample_use <- function(n, n_blocks = 5L, buffer = 0L) {
  f <- hblock_folds(n, n_blocks = n_blocks, buffer = buffer)
  mean(vapply(f$train, length, numeric(1))) / f$n
}

#' Training Share Across a Grid of K and h
#'
#' Tabulates \code{\link{sample_use}} over a grid, so the point at which the
#' buffer costs more than the cross-fitting buys is visible before a
#' configuration is committed to. Infeasible cells are \code{NA}.
#'
#' @param n Number of observations.
#' @param n_blocks Integer vector of block counts \eqn{K}.
#' @param buffers Integer vector of buffer widths \eqn{h}.
#' @return A numeric matrix, rows named by \eqn{K} and columns by \eqn{h}.
#' @examples
#' sample_use_table(108)
#' @export
sample_use_table <- function(n, n_blocks = c(4L, 5L, 6L, 8L),
                             buffers = c(0L, 2L, 5L, 10L)) {
  out <- matrix(NA_real_, nrow = length(n_blocks), ncol = length(buffers),
                dimnames = list(paste0("K=", n_blocks), paste0("h=", buffers)))
  for (i in seq_along(n_blocks)) {
    for (j in seq_along(buffers)) {
      out[i, j] <- tryCatch(
        sample_use(n, n_blocks = n_blocks[i], buffer = buffers[j]),
        error = function(e) NA_real_
      )
    }
  }
  out
}
