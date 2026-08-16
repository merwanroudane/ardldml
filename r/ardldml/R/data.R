#' Exchange-Rate Pass-Through Data
#'
#' Nine monthly United States macroeconomic series, 1973-01 to 2020-12, in raw
#' levels. These are the series behind the paper's main application, bundled so
#' every example runs offline.
#'
#' The sample starts in 1973-01 because that is the first observation of the
#' trade-weighted dollar index, which is also why the first monetary regime
#' starts there. The four regimes give complete samples of 156, 156, 108 and 156
#' observations.
#'
#' @format A data frame with 576 rows and 10 columns:
#' \describe{
#'   \item{date}{First day of the month, class \code{Date}.}
#'   \item{cpi}{Consumer price index, all urban consumers (\code{CPIAUCSL}).}
#'   \item{neer}{Trade-weighted United States dollar index (\code{TWEXAFEGSMTHx}).}
#'   \item{m2}{M2 money stock (\code{M2SL}).}
#'   \item{ffr}{Effective federal funds rate (\code{FEDFUNDS}).}
#'   \item{ip}{Industrial production index (\code{INDPRO}).}
#'   \item{unrate}{Civilian unemployment rate (\code{UNRATE}).}
#'   \item{oil}{Crude oil spot price, West Texas Intermediate (\code{OILPRICEx}).}
#'   \item{gs10}{Ten-year Treasury constant maturity yield (\code{GS10}).}
#'   \item{baa}{Moody's Baa corporate bond yield (\code{BAA}).}
#' }
#'
#' @source FRED-MD, the monthly macroeconomic database of McCracken and Ng
#'   (2016), vintage 2025-06. \doi{10.1080/07350015.2015.1086655}
#'
#' @references
#' McCracken, M. W. and Ng, S. (2016). FRED-MD: A monthly database for
#' macroeconomic research. \emph{Journal of Business & Economic Statistics},
#' 34(4), 574--589. \doi{10.1080/07350015.2015.1086655}
#'
#' @examples
#' str(passthrough)
#' range(passthrough$date)
"passthrough"

#' The Four Monetary Regimes
#'
#' Testing within regimes rather than across them avoids conflating a structural
#' break with a long-run relationship.
#'
#' @format A named list of start and end months, as \code{"YYYY-MM"} strings.
#' @export
PASSTHROUGH_REGIMES <- list(
  "1973-1985" = c("1973-01", "1985-12"),
  "1986-1998" = c("1986-01", "1998-12"),
  "1999-2007" = c("1999-01", "2007-12"),
  "2008-2020" = c("2008-01", "2020-12")
)

#' The Seven Controls of the Full Conditioning Set
#' @format A character vector of length 7.
#' @export
CONTROLS <- c("m2", "ffr", "ip", "unrate", "oil", "gs10", "baa")

#' Controls Dropped to Form the Reduced Set
#'
#' The two most likely to share a stochastic trend with the pass-through
#' relation itself, which is what makes them the interesting ones to drop in the
#' trend-absorption diagnostic.
#'
#' @format A character vector of length 2.
#' @export
REDUCED_DROP <- c("m2", "oil")

#' Series Transformed to Logs
#'
#' The quantity and price series. The four interest and unemployment rates stay
#' in levels: they are already in percentage-point units and can approach zero.
#'
#' @format A character vector of length 5.
#' @export
LOG_SERIES <- c("cpi", "neer", "m2", "ip", "oil")

#' Controls Treated as Integrated of Order One
#'
#' On economic grounds, not by pretest.
#'
#' @format A character vector of length 6.
#' @export
DEFAULT_INTEGRATED <- c("m2", "ip", "oil", "gs10", "baa", "ffr")

#' Load a Regime of the Pass-Through Data
#'
#' @param regime One of the names of \code{\link{PASSTHROUGH_REGIMES}}, or
#'   \code{NULL} for the full 1973-2020 sample.
#' @param log Take logs of \code{\link{LOG_SERIES}}.
#' @param start,end Optional \code{"YYYY-MM"} bounds, applied after
#'   \code{regime}.
#'
#' @return A data frame with a \code{date} column and the nine series.
#'
#' @examples
#' df <- passthrough_regime("1999-2007")
#' nrow(df)
#' head(df[, c("date", "cpi", "neer")])
#' @export
passthrough_regime <- function(regime = NULL, log = TRUE,
                               start = NULL, end = NULL) {
  df <- passthrough
  if (!is.null(regime)) {
    if (!regime %in% names(PASSTHROUGH_REGIMES)) {
      stop("regime must be one of ", paste(names(PASSTHROUGH_REGIMES), collapse = ", "),
           "; got ", regime)
    }
    ab <- PASSTHROUGH_REGIMES[[regime]]
    start <- if (is.null(start)) ab[1L] else start
    end <- if (is.null(end)) ab[2L] else end
  }
  if (!is.null(start)) df <- df[df$date >= as.Date(paste0(start, "-01")), , drop = FALSE]
  if (!is.null(end)) {
    last <- seq(as.Date(paste0(end, "-01")), by = "month", length.out = 2L)[2L] - 1L
    df <- df[df$date <= last, , drop = FALSE]
  }
  if (log) {
    for (cn in LOG_SERIES) {
      if (cn %in% names(df)) {
        if (any(df[[cn]] <= 0, na.rm = TRUE)) {
          stop("cannot take logs of ", cn, ": non-positive values present")
        }
        df[[cn]] <- log(df[[cn]])
      }
    }
  }
  rownames(df) <- NULL
  df
}
