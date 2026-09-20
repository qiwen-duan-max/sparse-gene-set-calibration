#' Three-tier validation, with the tiers made regime-matched
#'
#' The usual three-tier story -- technical, internal, external -- is told as if
#' the tiers were independent hurdles.  They are not, and the failure is
#' specific: a technical check run at one sparsity tells you nothing about the
#' same method at another.  A scoring rule that separates a spiked population
#' cleanly at 5\% zeros can be indistinguishable from noise at 67\% zeros, so
#' "Tier 1 passed" is only meaningful when Tier 1 was run in the regime Tier 2
#' and Tier 3 inhabit.
#'
#' [tier1_technical()] therefore takes the observed sparsity as an input and
#' reports the sparsity it was evaluated at alongside the result, and
#' [run_tiers()] refuses to report a Tier 1 pass whose regime does not match the
#' data the later tiers were computed on.
#'
#' Tier 2 asks whether the association survives a matched null in the discovery
#' cohort.  Tier 3 asks whether it is present in an independent cohort -- and
#' since scores are not comparable across cohorts with different gene panels, it
#' compares *direction and rank*, never raw score values.
#'
#' @name tiers
#' @keywords internal
NULL

#' How far a Tier 1 check's sparsity may sit from the data's
#'
#' Above this gap the check is reported as not applicable rather than passed.
#'
#' @export
REGIME_TOLERANCE <- 0.10

#' Simulate a matrix with a planted programme
#'
#' Three things are generated jointly: a depth axis that depends on the latent
#' programme, a target gene set whose expression depends on depth, and a
#' background with enough gene-activity spread that an expression-matched null
#' has candidates to draw from.
#'
#' @param n_cells,n_genes,n_target Integers.
#' @param median_detected Median genes detected per cell.
#' @param detection Detection rate of a target gene at typical depth.
#' @param effect Log-scale effect of the programme on the target genes; zero is
#'   the null.
#' @param depth_programme_loading How strongly depth tracks the programme.
#' @param rank_frac Rank ceiling as a fraction of `n_genes`; `max_rank` is
#'   `ceiling(rank_frac * n_genes)`.  It is the regime knob: the tie-break
#'   channel opens when `max_rank` exceeds the number of genes a cell detects.
#' @param seed Integer.
#'
#' @return A list with `matrix`, `genes`, `target`, `programme`, `depth` and
#'   `max_rank`.
#'
#' @export
simulate_planted <- function(n_cells = 600, n_genes = 4000, n_target = 8,
                             median_detected = 430, detection = 0.05,
                             effect = 0.6, depth_programme_loading = 0,
                             rank_frac = 0.05, seed = 0) {
    set.seed(seed)
    n_bg <- n_genes - n_target
    if (n_bg < 200) {
        stop("need at least 200 background genes", call. = FALSE)
    }
    genes <- c(sprintf("BG%05d", seq_len(n_bg) - 1L),
               sprintf("TGT%03d", seq_len(n_target) - 1L))
    target <- sprintf("TGT%03d", seq_len(n_target) - 1L)

    programme <- stats::rnorm(n_cells)
    programme <- (programme - mean(programme)) / stats::sd(programme)
    log_depth <- 0.35 * stats::rnorm(n_cells) +
        depth_programme_loading * 0.6 * programme
    depth_factor <- exp(log_depth - mean(log_depth))

    bg_activity <- exp(stats::rnorm(n_bg, 0, 1.5))
    bg_activity <- bg_activity / stats::median(bg_activity)

    # The scale that lands the median detected count on target.  The detected
    # count is monotone in it, so a bisection converges quickly.
    rate_for <- function(scale) {
        rate <- outer(depth_factor, scale * bg_activity)
        mean(rowSums(1 - exp(-pmin(rate, 4))) / n_bg) * n_genes
    }
    lo <- 1e-8
    hi <- 1e4
    for (i in seq_len(40)) {
        mid <- sqrt(lo * hi)
        if (rate_for(mid) < median_detected) lo <- mid else hi <- mid
    }
    scale <- sqrt(lo * hi)

    med_depth <- stats::median(depth_factor)
    centre <- -log1p(-min(max(detection, 1e-5), 0.95)) / med_depth
    offset <- stats::rnorm(n_target, 0, 1.2)
    offset <- offset - stats::median(offset)
    target_lambda <- centre * exp(offset)

    rate_bg <- outer(depth_factor, scale * bg_activity)
    bg <- matrix(stats::rpois(length(rate_bg), pmin(rate_bg, 4)),
                 n_cells, n_bg)
    rate_t <- outer(depth_factor, target_lambda) *
        exp(effect * programme)
    tg <- matrix(stats::rpois(length(rate_t), pmin(rate_t, 4)),
                 n_cells, n_target)

    X <- cbind(bg, tg)
    list(matrix = Matrix::Matrix(X, sparse = TRUE), genes = genes,
         target = target, programme = programme, depth = rowSums(X > 0),
         max_rank = as.integer(ceiling(rank_frac * n_genes)))
}

#' Can the scoring rule recover a planted signal at this sparsity?
#'
#' The question is deliberately narrow: given a gene set of this size, in data
#' this sparse, does the score separate a cell population carrying a planted
#' programme from one that does not?
#'
#' @param target_zero_rate The zero rate the check must be run at, typically
#'   taken from the real data with [sparsity_report()].
#' @param n_cells,n_genes,k Integers.
#' @param effect Log-scale effect planted on the target genes.
#' @param median_detected Median detected count; defaults to 36\% of `n_genes`.
#' @param seed Integer.
#' @param n_repeats Independent datasets; the reported separation is the mean.
#'
#' @return A list with `auc`, the attained `zero_rate`, whether it is close
#'   enough to `target_zero_rate` to speak to it (`regime_match`), and `passed`.
#'
#' @details
#' A single sparsity cannot be dialled in directly -- it is an output of the
#' detection rate, the set size and the matrix -- so the function searches the
#' detection rate for the one that lands nearest the requested sparsity, then
#' measures the score's ability to separate the planted group.  The pass
#' threshold of 0.65 is a convention, not a law: it says the score has some
#' ability to rank cells, not that it is fit for a particular claim.
#'
#' @export
tier1_technical <- function(target_zero_rate, n_cells = 800, n_genes = 8000,
                            k = 8, effect = 0.6, median_detected = NULL,
                            seed = 0, n_repeats = 3) {
    grid <- c(0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.45)
    achieved <- numeric(length(grid))
    aucs <- numeric(length(grid))
    for (i in seq_along(grid)) {
        zs <- numeric(n_repeats)
        as_ <- numeric(n_repeats)
        for (rep in seq_len(n_repeats)) {
            sim <- simulate_planted(
                n_cells = n_cells, n_genes = n_genes, n_target = k,
                median_detected = median_detected %||% round(0.36 * n_genes),
                detection = grid[i], effect = effect,
                depth_programme_loading = 0, seed = seed + 101 * rep)
            cache <- rank_cache(sim$matrix, sim$genes,
                                ceiling = max(sim$max_rank, 30), chunk = 400)
            score <- aucell(cache, sim$target, max_rank = sim$max_rank)
            zs[rep] <- mean(score <= 0)
            as_[rep] <- auroc(score, sim$programme > stats::median(sim$programme))
        }
        achieved[i] <- mean(zs)
        aucs[i] <- mean(as_)
    }
    i <- which.min(abs(achieved - target_zero_rate))
    list(auc = aucs[i], zero_rate = achieved[i],
         target_zero_rate = target_zero_rate,
         regime_gap = abs(achieved[i] - target_zero_rate),
         regime_match = abs(achieved[i] - target_zero_rate) <= REGIME_TOLERANCE,
         detection_rate = grid[i],
         passed = aucs[i] >= 0.65,
         curve = data.frame(detection = grid, zero_rate = achieved, auc = aucs))
}

#' Does the association survive a matched null in the discovery cohort?
#'
#' Returns one entry per null family.  The families are nested -- co-detection
#' matching refines expression matching, which refines random -- so a claim that
#' holds against the strictest one holds against the others, and the interesting
#' case is when they disagree.
#'
#' `passed` and `passed_strictest` are the verdicts of the expression and
#' co-detection families, and each carries the name of the family it came from
#' in `passed_family` / `passed_strictest_family`.  Both are `NA` when that
#' family is not among `kinds`: a family that was not drawn has no verdict, and
#' reporting one would be a claim about a test that never ran.
#'
#' @param score Numeric vector.
#' @param axis Numeric vector.
#' @param builder A `MatchedNullBuilder`.
#' @param gene_set Character vector.
#' @param max_rank Rank ceiling.
#' @param n_null Sets drawn per family.
#' @param seed Integer.
#' @param tail Passed to [empirical_p()].
#' @param kinds Which families to run.
#'
#' @return A list with the observed correlation and one entry per family.
#'
#' @export
tier2_internal <- function(score, axis, builder, gene_set, max_rank,
                           n_null = 200, seed = 0, tail = "two-sided",
                           kinds = c("random", "expression", "codetection")) {
    observed <- spearman(score, axis)$rho
    out <- list(observed_rho = observed, n_cells = length(score))
    for (kind in kinds) {
        if (identical(kind, "codetection")) {
            # Refused, not downgraded.  This branch used to record
            # `passed = FALSE` with a note, which reads downstream as "the
            # set failed the strictest test" when in fact no test was run --
            # and the Python port raises here, so the two disagreed about the
            # same input.  A family that cannot be drawn has no verdict to
            # report, and the caller who asked for it gets the same message
            # from either language.
            tryCatch(builder$codetection(gene_set),
                     error = function(e) stop(
                         "co-detection matching needs detection_matrix at construction",
                         call. = FALSE))
        }
        sets <- builder$sample(gene_set, n_null, kind = kind, seed = seed)
        null_rho <- vapply(sets, function(s) {
            spearman(aucell(builder$cache, s, max_rank = max_rank), axis)$rho
        }, numeric(1))
        s <- null_summary(null_rho, observed, tail = tail)
        out[[kind]] <- list(p_value = s$p_value,
                            p_lower = empirical_p(null_rho, observed,
                                                  tail = "lower"),
                            p_upper = empirical_p(null_rho, observed,
                                                  tail = "upper"),
                            null_median = s$null_median,
                            null_sd = s$null_sd, percentile = s$percentile,
                            passed = isTRUE(s$p_value < ALPHA))
    }
    # Each headline verdict is named for the family it came from, and a verdict
    # is recorded only when that family was actually drawn.  `kinds` is a free
    # parameter, and a caller who asks for a subset has no strictest family to
    # be judged against: `isTRUE(NULL)` would report "failed" for a test that
    # was never run, which is the failure this framework exists to stop.
    for (family in c("expression", "codetection")) {
        key <- if (family == "expression") "passed" else "passed_strictest"
        drawn <- !is.null(out[[family]])
        out[[key]] <- if (drawn) isTRUE(out[[family]]$passed) else NA
        out[[paste0(key, "_family")]] <- if (drawn) family else NA_character_
    }
    out
}

#' Is the association present, with the same sign, in an independent cohort?
#'
#' Scores from two cohorts are not comparable when the gene panels differ --
#' `max_rank` is a fraction of the panel, so an identical biological signal lands
#' on different score values.  This compares signs and significance, never score
#' magnitudes, and reports the two effect sizes side by side rather than their
#' difference.
#'
#' @param discovery_rho,replication_rho Spearman correlations of score against
#'   the axis in each cohort.
#' @param n_replication Cells in the replication cohort.
#'
#' @return A list with the two correlations, the replication P value, whether
#'   the signs agree, and `passed`.
#'
#' @export
tier3_external <- function(discovery_rho, replication_rho, n_replication) {
    p_rep <- two_sided_from_spearman(replication_rho, n_replication)
    same_sign <- sign(discovery_rho) == sign(replication_rho)
    significant <- is.finite(p_rep) && p_rep < ALPHA
    list(discovery_rho = discovery_rho, replication_rho = replication_rho,
         replication_p = p_rep, n_replication = as.integer(n_replication),
         same_sign = same_sign, replication_significant = significant,
         # A same-sign, significant result in the replication cohort is the bar.
         passed = same_sign && significant,
         note = paste("score values are not comparable across cohorts with",
                      "different gene panels; only the direction and the rank",
                      "association are compared here"))
}

#' @rdname tiers
#' @keywords internal
two_sided_from_spearman <- function(rho, n) {
    if (!is.finite(rho) || n < 4) {
        return(NA_real_)
    }
    t_stat <- rho * sqrt((n - 2) / max(1 - rho^2, .Machine$double.eps))
    2 * stats::pt(-abs(t_stat), df = n - 2)
}

#' Assemble the three tiers and say what the combination supports
#'
#' @param discovery,replication Lists with `observed_rho` and `n_cells`.
#' @param tier1 Output of [tier1_technical()].
#'
#' @return A list of the three tiers and the verdict.  The verdict is
#'   intentionally conservative: a Tier 1 check that was not run in the right
#'   regime cannot rescue a failed Tier 3, and is reported as not applicable
#'   rather than as passed.
#'
#' @export
run_tiers <- function(discovery, replication = NULL, tier1 = NULL) {
    t1 <- tier1
    if (!is.null(t1) && length(t1) > 0) {
        if (!isTRUE(t1$regime_match)) {
            t1$applicable <- FALSE
            t1$reason <- sprintf(
                "Tier 1 was evaluated at a zero rate of %.2f, which is %.2f away from the data's %.2f; it does not speak to this gene set",
                t1$zero_rate, t1$regime_gap, t1$target_zero_rate)
        } else {
            t1$applicable <- TRUE
        }
    } else {
        t1 <- list()
    }

    t3 <- NULL
    if (!is.null(replication)) {
        t3 <- tier3_external(discovery$observed_rho, replication$observed_rho,
                             replication$n_cells)
    }

    passed <- character(0)
    if (isTRUE(t1$applicable) && isTRUE(t1$passed)) {
        passed <- c(passed, "TIER_1")
    }
    if (isTRUE(discovery$passed)) {
        passed <- c(passed, "TIER_2")
    }
    if (!is.null(t3) && isTRUE(t3$passed)) {
        passed <- c(passed, "TIER_3")
    }

    verdict <- if (is.null(t3)) {
        "INCOMPLETE_NO_EXTERNAL"
    } else if (isTRUE(t3$passed) && isTRUE(discovery$passed)) {
        "VALIDATED"
    } else if (isTRUE(discovery$passed)) {
        "INTERNAL_ONLY"
    } else {
        "NOT_SUPPORTED"
    }

    list(tier1 = t1, tier2 = discovery, tier3 = t3, tiers_passed = passed,
         verdict = verdict)
}

#' @rdname tiers
#' @keywords internal
`%||%` <- function(a, b) if (is.null(a)) b else a
