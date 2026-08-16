#' ardldml: Bounds Testing for Cointegration with Many Persistent Controls
#'
#' An implementation of the DML-Bounds procedure of Villena (2026) for testing a
#' long-run relationship when the conditioning set is high-dimensional and may
#' itself carry stochastic trends.
#'
#' @section The problem:
#' The Autoregressive Distributed Lag (ARDL) bounds test of Pesaran, Shin and
#' Smith (2001) brackets one unknown: because the integration order of the
#' regressors is not known, the lower critical value treats them as stationary
#' and the upper as integrated, and the statistic is read against the interval.
#' With many persistent controls a second unknown appears. Projecting the lagged
#' levels onto controls that themselves carry stochastic trends can absorb part
#' of the long-run variation that identifies the error-correction relation, so
#' what governs the null is the number of stochastic trends that survive
#' residualisation rather than the integration order of the original regressors.
#' The null then sits somewhere inside the classical bracket, at a point that is
#' not known.
#'
#' Stationary controls are harmless: a stationary regressor cannot track the
#' stochastic trend of an integrated one, so it cannot absorb one. Only
#' integrated controls make the bracket live.
#'
#' @section Why no bounds table is shipped:
#' Tabulated critical values are not valid in this setting, and the
#' generated-regressor remainder is not negligible at the sample sizes applied
#' work actually uses. Every critical value here is computed instead:
#' \code{\link{dml_bootstrap}} for inference on real data, and
#' \code{\link{simulate_pss_bounds}} to regenerate the classical bracket at any
#' sample size and any \eqn{k}.
#'
#' @section Where to start:
#' \code{\link{dml_bounds}} fits the test and \code{\link{dml_bootstrap}}
#' attaches a critical value. \code{\link{trend_absorption}} is the diagnostic
#' you should run before believing a non-rejection, and
#' \code{\link{penalty_sensitivity}} reports whether the verdict survives a
#' change of penalty. \code{\link{passthrough}} is the bundled data.
#'
#' @references
#' Villena, M. J. (2026). Testing cointegration with many persistent controls.
#' SSRN working paper. \doi{10.2139/ssrn.6472826}
#'
#' Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing approaches
#' to the analysis of level relationships. \emph{Journal of Applied
#' Econometrics}, 16(3), 289--326. \doi{10.1002/jae.616}
#'
#' Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey,
#' W. and Robins, J. (2018). Double/debiased machine learning for treatment and
#' structural parameters. \emph{The Econometrics Journal}, 21(1), C1--C68.
#' \doi{10.1111/ectj.12097}
#'
#' Zou, H. (2006). The adaptive lasso and its oracle properties. \emph{Journal
#' of the American Statistical Association}, 101(476), 1418--1429.
#' \doi{10.1198/016214506000000735}
#'
#' McCracken, M. W. and Ng, S. (2016). FRED-MD: A monthly database for
#' macroeconomic research. \emph{Journal of Business & Economic Statistics},
#' 34(4), 574--589. \doi{10.1080/07350015.2015.1086655}
#'
#' @author Dr Merwan Roudane \email{merwanroudane920@@gmail.com}
#'
#' @keywords internal
"_PACKAGE"
