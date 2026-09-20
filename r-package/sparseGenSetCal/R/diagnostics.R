#' Pre-flight diagnostics: what a score can and cannot resolve
#'
#' These are cheap, need only the expression matrix and the gene set, and are
#' meant to be run *before* any association test.  Their job is to predict which
#' of the failure modes documented in this package is about to bite.
#'
#' @section The structural zero rate:
#' AUCell and UCell scores are sums over the top `max_rank` ranks.  If a gene
#' set of \eqn{k} genes were ranked independently of everything else, the
#' probability that none of its genes lands in the top fraction \eqn{f} would be
#' \eqn{(1-f)^k}, and the score would be exactly zero.  This is the *structural*
#' zero rate, and it is a property of the scoring rule and the size of the
#' matrix, not of the biology.  Observed zero rates far from the structural value
#' are informative: upward means the genes are lost together, downward means they
#' are found together.  Either way the set does not behave like the independent
#' draw the usual null assumes.
#'
#' @section Rank-fraction comparability:
#' \eqn{max\_rank = \lceil f G \rceil} makes every AUCell-family score a function
#' of how many genes happen to be in the matrix.  Two cohorts processed with
#' different gene panels therefore have scores on different scales even when the
#' biology is identical, and comparing score *values* across them is not
#' interpretable without a correction.  [comparability_report()] measures how
#' large that effect is for a given pair of matrices.
#'
#' @name diagnostics
#' @keywords internal
NULL

#' Thresholds for [verdict()]
#'
#' These are not asserted; they are set from the simulation grid in the
#' accompanying manuscript, which measures how the discrepancy between
#' conventional and calibrated inference grows with the zero rate.  Treat them
#' as calibrated defaults and re-derive them for a new scoring rule.
#'
#' `tie_break_share_moderate` and `tie_break_share_high` govern the criterion
#' used when the cache carries per-cell depth.  `detection_floor` and
#' `min_detected_frac` are the fallback for when it does not, and they are kept
#' deliberately separate: the floor is a fixed rate that binds differently on
#' every dataset -- the criticism this package exists to make -- so it is not
#' allowed to silently become the criterion where the exact one is available.
#'
#' @export
DEFAULT_THRESHOLDS <- list(
    zero_rate_low = 0.20,
    zero_rate_high = 0.50,
    tie_break_share_moderate = 0.50,
    tie_break_share_high = 0.75,
    detection_floor = 0.02,
    min_detected_frac = 0.50,
    zero_rate_excess_tolerance = 0.05)

#' Severities in the order they are raised
#'
#' A reason of moderate weight must never lower a verdict that a heavier one has
#' already set.  Comparing the three as strings does exactly that, since
#' `"MODERATE"` sorts above `"HIGH"`.
#'
#' @export
SEVERITY_ORDER <- c("LOW", "MODERATE", "HIGH")

#' Probability that a k-gene set scores exactly zero under independence
#'
#' @param k Number of genes in the set that are present in the matrix.
#' @param n_genes Number of genes in the matrix.
#' @param rank_frac Ranking fraction of the score.
#'
#' @return A single probability.
#'
#' @export
structural_zero_rate <- function(k, n_genes, rank_frac = 0.05) {
    max_rank <- ceiling(rank_frac * n_genes)
    (1 - max_rank / n_genes)^k
}

#' Everything the framework knows about one gene set before testing it
#'
#' @param cache A `RankCache`.
#' @param gene_set Character vector.
#' @param rank_frac Ranking fraction.
#' @param thresholds Optional overrides for [DEFAULT_THRESHOLDS()].
#' @param detection_floor Rate below which a gene counts as not detected.
#'   Omitting it uses [adaptive_detection_floor()], which takes the floor from
#'   the set's own detection profile.  Passing a number restores the fixed floor,
#'   and the report says which was used, because the two answer a different
#'   question.
#' @param depth Optional per-cell \eqn{D_c}.  Taken from the cache when it
#'   carries it, which makes `tie_break_share` available.
#'
#' @return A list: the observed zero rate from the score itself, the structural
#'   zero rate under independence, their difference, per-gene detection
#'   summaries, the fraction of the set clearing the floor, and -- where depth is
#'   known -- `tie_break_share`, the share of the set's inclusion rate that comes
#'   from rank tie-breaking rather than from expression.
#'
#' @export
sparsity_report <- function(cache, gene_set, rank_frac = 0.05,
                            thresholds = NULL, detection_floor = NULL,
                            depth = NULL) {
    present <- cache$present(gene_set)
    if (length(present) == 0) {
        stop("none of the gene set is in the cache", call. = FALSE)
    }
    pos <- cache$positions(present)
    scores <- aucell(cache, present)
    observed_zero <- mean(scores <= 0)
    structural <- structural_zero_rate(length(pos), cache$n_genes_total,
                                       rank_frac = rank_frac)

    det <- cache$detection[pos]
    if (is.null(detection_floor)) {
        floor <- adaptive_detection_floor(cache, present)
        floor_source <- "adaptive (from this set's detection profile)"
    } else {
        floor <- as.numeric(detection_floor)
        floor_source <- "fixed (supplied by the caller)"
    }
    above <- det >= floor

    out <- list(
        n_requested = length(gene_set), n_present = length(present),
        n_missing = length(gene_set) - length(present),
        observed_zero_rate = observed_zero,
        structural_zero_rate = structural,
        zero_rate_excess = observed_zero - structural,
        mean_detection = mean(det), min_detection = min(det),
        max_detection = max(det), median_detection = stats::median(det),
        detection_floor = floor, detection_floor_source = floor_source,
        frac_above_floor = mean(above), n_above_floor = sum(above),
        genes_below_floor = present[!above],
        median_score = stats::median(scores), mean_score = mean(scores))

    if (is.null(depth)) {
        depth <- cache$depth
    }
    if (!is.null(depth)) {
        max_rank <- as.integer(ceiling(rank_frac * cache$n_genes_total))
        out$max_rank <- max_rank
        out$median_depth <- stats::median(depth)
        out$mean_inclusion <- mean(effective_inclusion(det, depth, max_rank,
                                                       cache$n_genes_total))
        out$tie_break_share <- tie_break_share(det, depth, max_rank,
                                               cache$n_genes_total)
        out$criterion <- "tie_break_share"
    } else {
        out$criterion <- "fixed_detection_floor"
    }
    out
}

#' Sequencing depth of the cells, and the UCell `r_max` it implies
#'
#' UCell's documentation recommends setting `r_max` to roughly the median number
#' of detected genes per cell, with 1500 as a default for 10x data.  For shallow
#' data that default caps far above the observed depth, which changes the score;
#' this reports the value the data actually imply.
#'
#' @param X Cells x genes matrix, or a `RankCache`.
#' @param rank_frac Ranking fraction, for the AUCell ceiling.
#'
#' @return A list of depth summaries and the implied `r_max`.
#'
#' @export
depth_report <- function(X, rank_frac = 0.05) {
    if (inherits(X, "RankCache")) {
        detected <- X$depth
        n_genes <- X$n_genes_total
        n_cells <- nrow(X$clipped)
    } else {
        detected <- row_counts(as_dgCMatrix(X) > 0)
        n_genes <- ncol(X)
        n_cells <- nrow(X)
    }
    if (is.null(detected)) {
        stop("this cache does not carry depth", call. = FALSE)
    }
    median_detected <- stats::median(detected)
    list(median_genes_per_cell = median_detected,
         p10_genes_per_cell = unname(stats::quantile(detected, 0.10)),
         p90_genes_per_cell = unname(stats::quantile(detected, 0.90)),
         n_genes_in_matrix = as.integer(n_genes),
         n_cells = as.integer(n_cells),
         aucell_max_rank = as.integer(ceiling(rank_frac * n_genes)),
         ucell_r_max_implied = as.integer(round(median_detected)),
         ucell_default_is_appropriate = median_detected >= 1200 &&
             median_detected <= 2000)
}

#' How far apart two or more matrices place the same gene set
#'
#' @param matrices A named list of gene counts, or of `c(n_genes, n_cells)`
#'   vectors.  Only the gene count is needed to expose the scale dependence.
#' @param rank_frac Ranking fraction.
#' @param k Representative gene-set size, for the structural zero rate.
#'
#' @return A data frame with one row per matrix, carrying `max_rank`, the
#'   effective fraction, and the structural zero rate for a `k`-gene set.  The
#'   attribute `max_rank_ratio` is the largest `max_rank` over the smallest, and
#'   `scale_comparable` says whether that ratio is below 1.05.
#'
#' @details
#' Compare the `max_rank` column across rows: if the gene panels differ, the same
#' biological signal lands on different scores.
#'
#' @export
comparability_report <- function(matrices, rank_frac = 0.05, k = 8) {
    labels <- names(matrices)
    if (is.null(labels)) {
        labels <- as.character(seq_along(matrices))
    }
    n_genes <- vapply(matrices, function(v) as.numeric(v[1]), numeric(1))
    n_cells <- vapply(matrices, function(v) {
        if (length(v) > 1) as.numeric(v[2]) else NA_real_
    }, numeric(1))
    max_rank <- ceiling(rank_frac * n_genes)
    out <- data.frame(
        dataset = labels, n_genes = as.integer(n_genes),
        n_cells = as.integer(n_cells), max_rank = as.integer(max_rank),
        effective_frac = max_rank / n_genes,
        structural_zero_rate = vapply(n_genes, function(g) {
            structural_zero_rate(k, g, rank_frac)
        }, numeric(1)),
        row.names = NULL)
    if (nrow(out) > 1) {
        ratio <- max(out$max_rank) / max(min(out$max_rank), 1)
        attr(out, "max_rank_ratio") <- ratio
        attr(out, "scale_comparable") <- ratio < 1.05
    }
    out
}

#' Turn a diagnostic report into a recommendation
#'
#' The verdict is deliberately conservative and mechanical: it says what the
#' numbers support, not what the analyst hopes they support.  It prefers
#' `"NOT_IDENTIFIABLE"` to a qualified approval when the sparsity makes the
#' question unanswerable from the data at hand.
#'
#' @param report Output of [sparsity_report()].
#' @param used_matched_null Whether the association was tested against a matched
#'   null rather than a conventional one.
#' @param outcome_driven_cutpoint Whether a continuous score was dichotomised at
#'   a threshold chosen from the outcome.
#' @param externally_replicated `TRUE`, `FALSE`, or `NULL` if replication was
#'   not attempted.
#' @param thresholds Optional overrides for [DEFAULT_THRESHOLDS()].
#' @param preflight Output of [preflight()], if the axis and the per-cell depth
#'   are known.  Its `verdict` is `"CLEAR"`, `"CAUTION"` or `"CONFOUNDED"`, and
#'   the last two are folded into this verdict: an axis correlated with
#'   sequencing depth manufactures the association through the tie-break alone,
#'   which no amount of sparsity on the gene set's side prevents and no matched
#'   null removes. It is a separate input rather than something read out of
#'   `report` because it needs the axis, which the sparsity report never sees.
#'
#' @return A list with `verdict` in `{"INTERPRETABLE",
#'   "INTERPRETABLE_WITH_MATCHED_NULL", "NOT_IDENTIFIABLE"}`, the reasons, the
#'   severities behind them, and `criterion` -- which of `"tie_break_share"` and
#'   `"fixed_detection_floor"` the sparsity assessment was made with.  The
#'   former is the exact criterion and is used whenever the cells' depth is
#'   known; the latter is the weaker fallback.  Either can be the criterion
#'   without having fired, in which case the `reasons` field is empty of
#'   sparsity reasons.
#'
#' @export
verdict <- function(report, used_matched_null = FALSE,
                    outcome_driven_cutpoint = FALSE,
                    externally_replicated = NULL, thresholds = NULL,
                    preflight = NULL) {
    th <- DEFAULT_THRESHOLDS
    if (!is.null(thresholds)) {
        th[names(thresholds)] <- thresholds
    }
    reasons <- character(0)
    severity <- "LOW"
    # A reason of moderate weight must never lower a severity a heavier one has
    # already set, so the three are ordered rather than compared as strings.
    raise <- function(level) {
        severity <<- SEVERITY_ORDER[max(match(severity, SEVERITY_ORDER),
                                        match(level, SEVERITY_ORDER))]
    }

    # Two criteria ask "is the score carried by expression or by the ranking's
    # tie-break?".  The exact one needs the cells' depth; the fallback compares
    # the set against a fixed rate instead and is weaker for it, so the report is
    # made to say which one produced the verdict.
    share <- report$tie_break_share
    if (!is.null(share) && is.finite(share)) {
        criterion <- "tie_break_share"
        if (share >= th$tie_break_share_high) {
            reasons <- c(reasons, sprintf(
                "%.0f%% of the set's chance of entering the score comes from rank tie-breaking rather than from expression; the score is largely a depth proxy",
                100 * share))
            severity <- "HIGH"
        } else if (share >= th$tie_break_share_moderate) {
            reasons <- c(reasons, sprintf(
                "%.0f%% of the set's chance of entering the score comes from rank tie-breaking; more of it is the ranking than the biology",
                100 * share))
            raise("MODERATE")
        }
    } else {
        criterion <- "fixed_detection_floor"
        if (report$frac_above_floor < th$min_detected_frac) {
            reasons <- c(reasons, sprintf(
                "%.0f%% of the set is detected in more than %.2f%% of cells, the floor in use; below it the score is carried by rank tie-breaking rather than by expression",
                100 * report$frac_above_floor, 100 * report$detection_floor))
            severity <- "HIGH"
        }
    }

    if (report$observed_zero_rate >= th$zero_rate_high) {
        reasons <- c(reasons, sprintf(
            "%.0f%% of cells score exactly zero, so most cells carry no information about the set",
            100 * report$observed_zero_rate))
        severity <- "HIGH"
    } else if (report$observed_zero_rate >= th$zero_rate_low) {
        reasons <- c(reasons, sprintf(
            "%.0f%% of cells score exactly zero", 100 * report$observed_zero_rate))
        raise("MODERATE")
    }

    excess <- report$zero_rate_excess
    if (!is.null(excess) && is.finite(excess) &&
        abs(excess) > th$zero_rate_excess_tolerance) {
        reasons <- c(reasons, sprintf(
            "the observed zero rate departs from the independent-draw expectation by %+.0f percentage points, so the set is not behaving like an independent draw",
            100 * excess))
        raise("MODERATE")
    }

    if (!used_matched_null) {
        reasons <- c(reasons, paste(
            "the association was not tested against an expression-matched null"))
        raise("MODERATE")
    }

    if (outcome_driven_cutpoint) {
        reasons <- c(reasons, paste(
            "the cutpoint was chosen from the outcome; under a permuted outcome",
            "such a rule rejects far more often than the nominal level"))
        severity <- "HIGH"
    }
    if (identical(externally_replicated, FALSE)) {
        reasons <- c(reasons, "the association did not replicate in the external cohort")
        severity <- "HIGH"
    } else if (is.null(externally_replicated) && severity != "LOW") {
        reasons <- c(reasons, "no independent cohort was tested")
    }

    # The axis's own confound with depth.  It is a reason and not a refusal: a
    # matched null is calibrated *under* this confound -- the replacement sets
    # carry the same tie-break behaviour, which is the framework's central
    # result -- so the confound does not make the recommended analysis
    # untrustworthy.  What it does is say why the conventional analysis is not a
    # substitute for it, which is the one thing `used_matched_null` alone does
    # not convey: "no matched null was used" reads as a missing step until the
    # axis is named as the reason it is not optional.
    pf <- if (is.list(preflight) && !is.null(preflight$verdict)) {
        as.character(preflight$verdict)[1]
    } else {
        NULL
    }
    if (identical(pf, "CONFOUNDED")) {
        reasons <- c(reasons, paste(
            "the axis is correlated with sequencing depth, so the score's",
            "depth-derived component alone reproduces the association"))
        raise("MODERATE")
    } else if (identical(pf, "CAUTION")) {
        reasons <- c(reasons, paste(
            "the axis is weakly correlated with sequencing depth; the",
            "tie-break alone could account for part of the association"))
        raise("MODERATE")
    }

    # A set with nothing wrong with it is only plain INTERPRETABLE if the
    # association was in fact tested against a matched null.  Without one, the
    # best available reading is that the numbers are not interpretable yet,
    # however unremarkable they look.
    v <- if (severity == "HIGH") {
        "NOT_IDENTIFIABLE"
    } else if (!used_matched_null) {
        "INTERPRETABLE_WITH_MATCHED_NULL"
    } else {
        "INTERPRETABLE"
    }
    if (length(reasons) == 0) {
        reasons <- "no diagnostic threshold was crossed"
    }
    list(verdict = v, severity = severity, reasons = reasons,
         criterion = criterion, used_matched_null = used_matched_null,
         outcome_driven_cutpoint = outcome_driven_cutpoint,
         externally_replicated = externally_replicated)
}
