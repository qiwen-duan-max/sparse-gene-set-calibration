#' The four-step calibration checklist
#'
#' Everything in this package is available piece by piece, and the pieces are
#' meant to be composable.  This is the assembled version: a single call that
#' runs the whole diagnostic sequence on a matrix, a gene set and a proposed
#' axis, and returns a verdict together with the numbers behind it.
#'
#' @section The four steps, and why they are in this order:
#' \enumerate{
#'   \item **Sparsity.** How much of the set is detectable, and what fraction of
#'     cells score zero.  A set that is undetected in most cells cannot carry a
#'     cell-level claim, and no downstream correction changes that.
#'   \item **Null construction.** Build a replacement set matched on detection,
#'     expression and co-detection, and report how well it matched.  A null that
#'     failed to match is not a null, and its P value is not evidence.
#'   \item **Cutpoint calibration.** If the analysis dichotomises the score,
#'     measure the false-positive rate of the rule rather than assuming the
#'     nominal level.  The gap between an outcome-driven and a pre-specified
#'     cutpoint is routinely a factor of five or more.
#'   \item **External replication.** Sign and significance in an independent
#'     cohort.  Because `max_rank` is a fraction of the gene panel, scores from
#'     two cohorts are not comparable as values; only direction and rank
#'     association are compared.
#' }
#'
#' The verdict is deliberately mechanical.  It says what the numbers support,
#' not what the analyst hopes they support, and it prefers `"NOT_IDENTIFIABLE"`
#' to a qualified approval when the sparsity makes the question unanswerable
#' from the data at hand.
#'
#' @references
#' The two cutpoint rules are the ones compared in the accompanying manuscript:
#' a split chosen from the outcome has a false-positive rate several times the
#' nominal level, while a split fixed at the median of the score does not.
#'
#' @name checklist
#' @keywords internal
NULL

#' Defaults for the whole framework
#'
#' These are the settings used in the accompanying manuscript; change them and
#' every reported rate changes with them, which is why they are collected here
#' rather than scattered as literals.
#'
#' @export
DEFAULT_CONFIG <- list(
    sparsity_threshold = 0.50,
    null_matching_tolerance = 0.05,
    auc_significance_level = 0.05,
    replication_cohorts = 2,
    cutpoint_permutations = 1000,
    rank_frac = 0.05,
    n_null = 200,
    n_draws = 25,
    seed = 42,
    # The "10%" of the brief's insufficient-background case.  It is a factor
    # rather than a percentage because it multiplies the detection floor.
    background_loosen_factor = 0.9)

#' The fewest drawable background genes a matched null still means anything for
#'
#' The pool outside the gene set itself, not the pool size: the query set is
#' excluded from every draw by construction, so a 400-gene set drawn from a
#' 3,000-gene background has a large pool and almost nothing to draw from.
#' Below this the draw is choosing among so few candidates that "matched" stops
#' describing it, and the framework says so rather than returning a null that
#' would be read as a test.
#'
#' @export
MIN_BACKGROUND_GENES <- 10L

#' The smallest spread a gene-set score can have and still be correlated
#'
#' A score whose spread across cells is below this cannot carry a correlation
#' with anything: the statistic is undefined, not merely imprecise.
#'
#' @export
MIN_SCORE_SD <- 0.001

#' The AUC at or above which a score's separation is called suspicious
#'
#' An AUC this high means the score separates the two groups almost perfectly,
#' which is a statement about the data rather than about the score.  It is a
#' threshold for a warning and not for a refusal, because a marker that marks
#' exactly what it is supposed to mark also lands here.
#'
#' @export
SUSPICIOUS_AUC <- 0.99

#: Edge case three of the brief.  A set whose genes are never detected in any
#: cell, or detected in every cell, produces the same number for every cell; the
#: correlation with an axis is then 0/0 and the P value that comes out of it is
#: an artefact of whichever convention the arithmetic chose.  Failing here, at
#: the point the score is made, attaches the failure to the gene set that caused
#: it rather than leaving a NaN whose origin has to be traced back.
check_score <- function(score, label = NULL, context = "") {
    sd <- stats::sd(as.numeric(score))
    if (!is.finite(sd) || sd < MIN_SCORE_SD) {
        event("error", "score_has_no_variance", label = label,
              context = context, sd = sd, threshold = MIN_SCORE_SD)
        stop(sprintf(
            paste0("Gene set score has zero variance. Gene set may be all ",
                   "zeros or all highly expressed; consider different genes. ",
                   "(sd = %.3g < %g%s%s)"),
            sd, MIN_SCORE_SD,
            if (is.null(label)) "" else sprintf(", set '%s'", label),
            if (!nzchar(context)) "" else sprintf(", %s", context)),
            call. = FALSE)
    }
    sd
}

#: Edge case two of the brief.  An AUC this extreme is not evidence that the
#: score is good; it is evidence that the score and the grouping share a source,
#: and the brief names the three sources worth checking.  The warning is
#: signalled rather than raised, because an AUC of 0.99 can also be real -- a
#: marker that marks what it is supposed to mark -- and a framework that refused
#: to report it would be hiding the result rather than qualifying it.
#:
#: Both directions are checked.  The brief states the case as "AUC >= 0.99", but
#: [auroc()] here does not fold its argument about 0.5, and a score that lands
#: every high-axis cell below every low-axis cell separates them just as
#: perfectly as one that does the reverse.  Reading only the upper tail would
#: have left exactly the confound the case exists to catch, undetected, in half
#: the ways it can arise.  The record therefore carries both the AUC and the
#: separation, so a reader can tell the two apart.
check_auc <- function(auc, label = NULL, context = "") {
    if (!is.finite(auc)) {
        return(auc)
    }
    separation <- max(auc, 1 - auc)
    if (separation < SUSPICIOUS_AUC) {
        return(auc)
    }
    event("warning", "suspicious_score_separation", label = label,
          context = context, auc = auc, separation = separation,
          threshold = SUSPICIOUS_AUC,
          check = c("batch_effects",
                    "outcome_leakage_into_cell_type_annotation",
                    "extreme_sparsity"))
    auc
}

#' @rdname checklist
#' @keywords internal
merge_config <- function(config = NULL) {
    out <- DEFAULT_CONFIG
    if (!is.null(config)) {
        unknown <- setdiff(names(config), names(out))
        if (length(unknown) > 0) {
            stop(sprintf("unknown configuration keys: %s",
                         paste(sort(unknown), collapse = ", ")), call. = FALSE)
        }
        out[names(config)] <- config
    }
    out
}

#' @rdname checklist
#' @param cache A `RankCache`.
#' @param gene_set Character vector.
#' @param axis Numeric vector, the variable the score is to be tested against.
#'   Without it, steps 3 and 4 cannot run and are reported as not attempted.
#' @param depth Optional per-cell \eqn{D_c}; taken from the cache when omitted.
#' @param label A name for the set, carried into the result.
#' @param used_matched_null Whether the association was tested against a matched
#'   null.  Defaults to `FALSE`, which is what an unqualified claim amounts to.
#' @param cutpoint_method `NULL`, `"optimum"`, `"median"` or `"pre_specified"`.
#'   If the analysis dichotomises the score, this is the rule it uses.
#' @param cutpoint_pre_specified Required for `"pre_specified"`.
#' @param externally_replicated Result of step 4, if it has been run elsewhere.
#' @param n_perm Permutations for the cutpoint calibration.
#' @param config Overrides for [DEFAULT_CONFIG()].
#'
#' @return A list with `step1_sparsity`, `step2_null_construction`,
#'   `step3_false_positive_rate`, `step4_external_replication`,
#'   `overall_verdict`, `severity`, `confidence` and `reasons`.
#'
#' @export
diagnostic_checklist <- function(cache, gene_set, axis = NULL, depth = NULL,
                                 label = NULL, used_matched_null = FALSE,
                                 cutpoint_method = NULL,
                                 cutpoint_pre_specified = NULL,
                                 externally_replicated = NULL, n_perm = NULL,
                                 config = NULL, detection_matrix = NULL,
                                 detection_matrix_genes = NULL, seed = NULL) {
    cfg <- merge_config(config)
    seed <- if (is.null(seed)) cfg$seed else as.integer(seed)
    n_perm <- if (is.null(n_perm)) cfg$cutpoint_permutations else as.integer(n_perm)

    diag <- sparsity_report(cache, gene_set, rank_frac = cfg$rank_frac)
    pos <- cache$positions(gene_set)

    # ---- step 1: can this score carry a cell-level claim at all? ----
    zero <- diag$observed_zero_rate
    step1 <- if (zero >= cfg$sparsity_threshold) {
        list(severity = "HIGH", recommendation = "USE_MATCHED_NULL")
    } else if (zero >= 0.20) {
        list(severity = "MODERATE", recommendation = "USE_MATCHED_NULL")
    } else {
        list(severity = "LOW", recommendation = "CONVENTIONAL_NULL_ACCEPTABLE")
    }
    step1 <- c(list(
        percent_zero = 100 * zero,
        structural_zero_rate = diag$structural_zero_rate,
        zero_rate_excess = diag$zero_rate_excess,
        n_present = diag$n_present, n_missing = diag$n_missing,
        frac_above_floor = diag$frac_above_floor,
        detection_floor = diag$detection_floor,
        mean_detection = diag$mean_detection,
        criterion = diag$criterion), step1)
    # Where depth is known, the share of the set's inclusion that the tie-break
    # supplies is the exact version of what frac_above_floor approximates, so it
    # is carried into the checklist rather than left in the report.
    if (!is.null(diag$tie_break_share)) {
        step1$tie_break_share <- diag$tie_break_share
        step1$mean_inclusion <- diag$mean_inclusion
    }

    # ---- step 2: is there a null that matches? ----
    built <- construct_expression_matched_null(
        cache, gene_set, n_sets = cfg$n_null, kind = "expression",
        detection_matrix = detection_matrix,
        detection_matrix_genes = detection_matrix_genes, seed = seed)
    step2 <- list(
        null_pool_size = built$pool_size, det_floor = built$det_floor,
        target_detection = built$target_detection,
        null_detection = built$null_detection,
        detection_ratio = built$detection_ratio,
        expression_ratio = built$expression_ratio,
        ks_test_pvalue = built$ks_pvalue, ks_statistic = built$ks_statistic,
        max_overlap = built$max_overlap, n_from_query = built$n_from_query,
        status = built$status)

    # ---- step 3: what does the cutpoint rule actually cost? ----
    step3 <- list(attempted = FALSE)
    if (!is.null(axis) && !is.null(cutpoint_method)) {
        score <- aucell(cache, gene_set,
                        max_rank = as.integer(ceiling(cfg$rank_frac *
                                                      cache$n_genes_total)))
        # Edge case three, at the point the score is made: a set that is never
        # detected, or detected everywhere, has no spread and its correlation
        # with the axis is 0/0.
        check_score(score, label = label, context = "cutpoint calibration")
        outcome <- axis > stats::median(axis)
        # Edge case two.  How well the score separates the two outcome groups
        # is the quantity the brief asks to be checked against 0.99, and it is
        # reported whether or not it crosses, so the record shows the number
        # the judgement was made on.
        score_auc <- check_auc(auroc(score, outcome), label = label,
                               context = "cutpoint calibration")
        fpr <- cutpoint_fpr(score, outcome, n_perm = n_perm,
                            method = cutpoint_method,
                            pre_specified = cutpoint_pre_specified, seed = seed)
        base <- cutpoint_fpr(score, outcome, n_perm = n_perm, method = "median",
                             seed = seed)
        step3 <- list(
            attempted = TRUE, method = cutpoint_method,
            score_auc = score_auc,
            optimum_cutpoint_FPR = fpr$fpr, median_cutpoint_FPR = base$fpr,
            pre_specified_cutpoint_FPR = if (identical(cutpoint_method,
                                                       "pre_specified")) {
                fpr$fpr
            } else {
                NA_real_
            },
            observed_p = fpr$observed_p, n_perm = fpr$n_perm,
            recommendation = if (fpr$fpr > 2 * cfg$auc_significance_level) {
                "USE_PRESPECIFIED"
            } else {
                "OK"
            })
    } else if (!is.null(axis)) {
        step3 <- list(
            attempted = FALSE,
            note = paste("no cutpoint rule was declared; step 3 measures the",
                         "rule, and an undeclared rule is the outcome-driven one"))
    }

    # ---- step 4: does it hold anywhere else? ----
    step4 <- list(
        attempted = !is.null(externally_replicated),
        replication_cohorts = as.integer(!is.null(externally_replicated)),
        success = externally_replicated,
        status = if (is.null(externally_replicated)) {
            "NOT_ATTEMPTED"
        } else if (isTRUE(externally_replicated)) {
            "PASS"
        } else {
            "FAIL"
        })

    # ---- the pre-flight, then the verdict ----
    # Computed *before* the verdict rather than attached afterwards, because it
    # is an input to the verdict.  It used to be attached only: the checklist
    # returned `preflight` and the verdict never read it, so a gene set that was
    # sparse-free in a matrix whose axis tracked sequencing depth came back
    # INTERPRETABLE -- the confound the framework exists to catch, reported as a
    # clean pass by the framework itself.
    if (is.null(depth)) {
        depth <- cache$depth
    }
    pre <- NULL
    if (!is.null(axis) && !is.null(depth)) {
        # The pre-flight check needs the per-cell detected counts and nothing
        # else, so it stays available even when no score has been computed.
        pre <- preflight(depth, axis,
                         as.integer(ceiling(cfg$rank_frac * cache$n_genes_total)),
                         cache$n_genes_total,
                         detection = cache$detection[pos], k = length(pos))
    }

    # ---- the verdict ----
    v <- verdict(diag, used_matched_null = isTRUE(used_matched_null),
                 outcome_driven_cutpoint = identical(cutpoint_method, "optimum"),
                 externally_replicated = externally_replicated,
                 preflight = pre)

    out <- list(
        label = label, n_cells = nrow(cache$clipped),
        n_genes_total = cache$n_genes_total,
        max_rank = as.integer(ceiling(cfg$rank_frac * cache$n_genes_total)),
        step1_sparsity = step1, step2_null_construction = step2,
        step3_false_positive_rate = step3, step4_external_replication = step4,
        overall_verdict = v$verdict, severity = v$severity,
        confidence = if (identical(step1$severity, "HIGH")) "HIGH" else "MODERATE",
        reasons = v$reasons, criterion = v$criterion,
        config = c(cfg, list(seed = seed, n_perm = n_perm)))

    if (!is.null(pre)) {
        out$preflight <- pre
    }
    out
}

#' Assemble the three validation tiers into one result
#'
#' Thin wrapper over [run_tiers()] that also carries the cutpoint specification
#' through, so that the tier record states which rule was used rather than
#' leaving it to the prose around it.
#'
#' @param discovery,replication,tier1 Passed to [run_tiers()].
#' @param cutpoint_specification `"pre_specified"`, `"median"` or `"optimum"`.
#'
#' @return A list of the three tiers plus `cutpoint_specification`.
#'
#' @export
validate_tier_framework <- function(discovery, replication = NULL, tier1 = NULL,
                                    cutpoint_specification = "pre_specified") {
    out <- run_tiers(discovery, replication = replication, tier1 = tier1)
    out$cutpoint_specification <- cutpoint_specification
    out$tiers <- list(TIER_1 = out$tier1, TIER_2 = out$tier2, TIER_3 = out$tier3)
    out
}

#' Write a calibration result to JSON
#'
#' The object is the record of an analysis, so it is written in a form a reader
#' can diff rather than in R's own serialisation.
#'
#' @param x A list, typically the output of [diagnostic_checklist()].
#' @param path Output path.
#'
#' @return `path`, invisibly.
#'
#' @details
#' Collection-valued fields are written as JSON arrays whatever their length.
#' `jsonlite` unboxes a length-one vector by default, which would make `reasons`
#' a string when a verdict has one reason and an array when it has three -- a
#' shape that changes with the data.  It also differs from the Python
#' implementation of this framework, whose lists are always arrays, so a
#' consumer of both would have to test the type before indexing.  The fields
#' listed in `json_array_fields` are therefore protected, at any depth of the
#' object; everything else keeps the default unboxing.
#'
#' @export
to_json <- function(x, path) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
        stop("to_json needs the jsonlite package", call. = FALSE)
    }
    jsonlite::write_json(as_json_arrays(x), path, auto_unbox = TRUE,
                         pretty = TRUE, digits = 8, null = "null", na = "null")
    invisible(path)
}

# Fields that name a collection rather than a value.
json_array_fields <- c("reasons", "genes_below_floor", "null_genes", "genes",
                       "null_sets", "labels", "cohorts", "notes", "warnings")

# Walks the object and marks those fields so that jsonlite keeps them as
# arrays.  A list of sets -- the null draws, for instance -- reaches the atomic
# branch once per element, each time under the name of the field it came from,
# so a one-gene null set stays an array too.
as_json_arrays <- function(x, name = NULL) {
    if (is.data.frame(x)) {
        return(x)
    }
    if (is.list(x)) {
        # Rebuilt rather than edited in place: assigning NULL to an element
        # deletes it and shortens the list, which would walk a four-field step
        # record off the end of itself the moment one of the fields is null.
        nms <- names(x)
        out <- lapply(seq_along(x), function(i) {
            as_json_arrays(x[[i]], if (is.null(nms)) NULL else nms[i])
        })
        if (!is.null(nms) && length(nms) == length(out)) {
            names(out) <- nms
        }
        return(out)
    }
    if (!is.null(name) && name %in% json_array_fields) {
        return(I(x))
    }
    x
}
