#' The decision table and the fitted rejection probability, as functions
#'
#' Two conveniences that read like the paper's decision table (Supplementary
#' Table 5).  \code{select_null_method} returns the branch a set's measured
#' quantities put it on, with the thresholds it applied echoed in the result;
#' the manuscript's table is generated from the same constants, so the
#' function and the table cannot disagree.  \code{predict_fpr} returns the
#' conventional test's rejection probability under a true null, given the
#' magnitude of the implied product.
#'
#' Both are pure functions of their arguments: no matrix, no cache, no
#' scoring.  A quantity that was not supplied is reported as unevaluated, not
#' assumed quiet.
#'
#' @name convenience
#' @keywords internal
NULL

#' Logistic fit of the conventional test's rejection on the implied product
#'
#' Pooled over the null runs of the calibration grids (n = 1,680), model
#' "product alone" in \code{grid_A_B_joint_logistic.csv}.  The predictor was
#' standardised on the fit set, so the mean and SD travelled with the
#' coefficients; the cross-language check recomputes both from the grid files
#' and compares.
#'
#' @format A list with the coefficients, the standardisation constants, the
#'   number of runs, the grids fitted on, and the range the fit saw.
#' @export
FPR_MODEL <- list(
    intercept = -0.07245436534023197,
    coef_per_sd = 4.166465012123036,
    mean_abs_implied = 0.0512505673509389,
    sd_abs_implied = 0.06159690289425014,
    n_runs = 1680L,
    grids = c("A_calibration", "B_regime"),
    test = "conventional cell-level Spearman, alpha=0.05, effect=0",
    source = "results/grid_A_B_joint_logistic.csv, model 'product alone'",
    fitted_range = c(0.0, 0.2969))

#' The co-detection gap rules the grid's boundary analysis uses
#'
#' Below the flag the matched families are nominal on the same draws; above
#' the refusal line, and with a depth-linked axis, no null family is safe.
#' The values are pinned to the analysis constants by a test, so the package
#' and the grid rules cannot drift.
#'
#' @export
CO_DETECTION_GAP_FLAG <- 0.005

#' The co-detection gap at which no null family is safe, on a depth-linked
#' axis
#'
#' @format A scalar threshold on the achieved co-detection gap.
#' @export
CO_DETECTION_GAP_REFUSE <- 0.020

#' The axis-depth correlation above which an axis is depth-linked
#'
#' @format A scalar threshold on the correlation of the tested axis with the
#'   cells' detected-gene count.
#' @export
DEPTH_LINKED_AXIS <- 0.2

#' The checklist's decision, from quantities the caller already measured
#'
#' @param tie_break_share Step 1's exact quantity: the share of the set's
#'   inclusion rate that comes from tie-breaking.
#' @param implied_rho_abs The magnitude of the pre-flight implied product.
#' @param chance_band The chance band to read the implied product against
#'   (\code{\link{preflight}} returns it as \code{chance_band}).  Without it
#'   the pre-flight step is reported as unevaluated.
#' @param codetection_gap The achieved co-detection gap of the best matched
#'   draw.  A negative gap means the draws are more co-detected than the
#'   query and never flags.
#' @param axis_depth_rho The correlation of the tested axis with the cells'
#'   detected-gene count.
#' @param config,thresholds Overrides for the package defaults, as elsewhere
#'   in the framework.
#'
#' @return A list with \code{recommended_action} (one of
#'   \code{conventional_null_ok}, \code{expression_matched_null},
#'   \code{refuse_not_identifiable}, \code{refuse_no_null_family_safe}),
#'   \code{steps} reporting what each supplied quantity said,
#'   \code{thresholds_applied} echoing the constants used, and \code{notes}
#'   carrying anything a reader of the output should know.
#'
#' @examples
#' select_null_method(tie_break_share = 0.8, implied_rho_abs = 0.01,
#'                    chance_band = 0.05)
#' @export
select_null_method <- function(tie_break_share = NULL,
                               implied_rho_abs = NULL,
                               chance_band = NULL,
                               codetection_gap = NULL,
                               axis_depth_rho = NULL,
                               config = NULL,
                               thresholds = NULL) {
    cfg <- utils::modifyList(DEFAULT_CONFIG,
                             if (is.null(config)) list() else config)
    th <- utils::modifyList(DEFAULT_THRESHOLDS,
                            if (is.null(thresholds)) list() else thresholds)
    steps <- list()
    notes <- character(0)
    action <- "conventional_null_ok"
    escalate <- function(current, proposed, severity) {
        if (severity[[proposed]] > severity[[current]]) proposed
        else current
    }
    ## A refusal beats a flag and a flag beats a recommendation; the two
    ## refusals carry the same severity -- they are both terminal -- and the
    ## step order decides which label a set gets: the checklist evaluates
    ## sparsity before null construction.
    severity <- c(conventional_null_ok = 1, expression_matched_null = 2,
                  refuse_not_identifiable = 3, refuse_no_null_family_safe = 3)

    if (is.null(implied_rho_abs) || is.null(chance_band)) {
        steps$preflight <- "not_evaluated"
        if (!is.null(implied_rho_abs) && is.null(chance_band)) {
            notes <- c(notes, paste("implied product supplied without a",
                                    "chance band; preflight() computes the",
                                    "band"))
        }
    } else {
        open_channel <- implied_rho_abs > chance_band
        steps$preflight <- if (open_channel) "channel_open" else "channel_closed"
        if (open_channel) {
            action <- escalate(action, "expression_matched_null", severity)
        }
    }

    if (is.null(tie_break_share)) {
        steps$sparsity <- "not_evaluated"
    } else if (tie_break_share >= th$tie_break_share_high) {
        steps$sparsity <- "tie_break_share_high"
        action <- escalate(action, "refuse_not_identifiable", severity)
    } else if (tie_break_share >= th$tie_break_share_moderate) {
        steps$sparsity <- "tie_break_share_moderate"
        action <- escalate(action, "expression_matched_null", severity)
    } else {
        steps$sparsity <- "below_moderate"
    }

    if (is.null(codetection_gap) || is.null(axis_depth_rho)) {
        steps$codetection <- "not_evaluated"
    } else {
        steps$codetection <- sprintf("gap=%.4f", codetection_gap)
        if (codetection_gap > CO_DETECTION_GAP_REFUSE &&
                axis_depth_rho > DEPTH_LINKED_AXIS) {
            action <- escalate(action, "refuse_no_null_family_safe", severity)
            notes <- c(notes, paste("co-detection gap beyond the refusal",
                                    "line on a depth-linked axis; no null",
                                    "family is safe here"))
        } else if (codetection_gap > CO_DETECTION_GAP_FLAG &&
                           axis_depth_rho > DEPTH_LINKED_AXIS) {
            notes <- c(notes, paste("co-detection gap beyond the flag line",
                                    "on a depth-linked axis; prefer the",
                                    "expression-matched family and report",
                                    "the achieved gap"))
        }
    }

    if (action != "conventional_null_ok") {
        notes <- c(notes, paste("if the analysis dichotomises the score,",
                                "measure the rule's false-positive rate by",
                                "permutation rather than assuming the",
                                "nominal level"))
    }

    list(recommended_action = action,
         steps = steps,
         thresholds_applied = list(
             tie_break_share_moderate = th$tie_break_share_moderate,
             tie_break_share_high = th$tie_break_share_high,
             null_matching_tolerance = cfg$null_matching_tolerance,
             co_detection_gap_flag = CO_DETECTION_GAP_FLAG,
             co_detection_gap_refuse = CO_DETECTION_GAP_REFUSE,
             depth_linked_axis = DEPTH_LINKED_AXIS),
         notes = notes)
}

#' The conventional test's rejection probability under a true null
#'
#' @param implied_rho_abs The magnitude of the implied product: how strongly
#'   the score tracks depth times how strongly the axis does.
#'   \code{\link{preflight}} computes it from the depth vector and the
#'   detection profile before any score exists.
#' @param model Override the shipped fit (mainly for the cross-language
#'   test).
#'
#' @return A list with \code{p_reject}, \code{within_fitted_range} saying
#'   whether the input sits inside the range the fit saw, and the fit's
#'   provenance under \code{model}.  The fit is an interpolation over the
#'   calibration grids' null runs; outside that range the number is an
#'   extrapolation and \code{within_fitted_range} says so rather than hiding
#'   it.
#'
#' @examples
#' predict_fpr(0.05)
#' @export
predict_fpr <- function(implied_rho_abs, model = NULL) {
    m <- utils::modifyList(FPR_MODEL, if (is.null(model)) list() else model)
    z <- (implied_rho_abs - m$mean_abs_implied) / m$sd_abs_implied
    logit <- m$intercept + m$coef_per_sd * z
    p_reject <- 1 / (1 + exp(-logit))
    within <- implied_rho_abs >= m$fitted_range[1] &&
        implied_rho_abs <= m$fitted_range[2]
    list(p_reject = p_reject,
         within_fitted_range = within,
         model = list(source = m$source, n_runs = m$n_runs,
                      grids = m$grids, test = m$test))
}
