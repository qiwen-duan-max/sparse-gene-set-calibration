#' Association, permutation and cutpoint machinery
#'
#' The tests here are the ones an analyst would reach for without a calibration
#' framework, collected in one place so that all of them can be measured on the
#' same data.  Every P value is judged at `ALPHA`, and under a true null every
#' rejection is a false positive with respect to the claim being made.
#'
#' @name stats
#' @keywords internal
NULL

#' Nominal level every test in this package is judged against
#'
#' @export
ALPHA <- 0.05

#' Spearman rank correlation
#'
#' Ties are handled by the mid-rank convention.  Non-finite pairs are dropped.
#'
#' @param x,y Numeric vectors.
#'
#' @return A list with `rho` and `p`.
#'
#' @export
spearman <- function(x, y) {
    x <- as.numeric(x)
    y <- as.numeric(y)
    keep <- is.finite(x) & is.finite(y)
    if (sum(keep) < 3) {
        return(list(rho = NA_real_, p = NA_real_))
    }
    ct <- suppressWarnings(stats::cor.test(x[keep], y[keep], method = "spearman",
                                           exact = FALSE))
    list(rho = unname(ct$estimate), p = ct$p.value)
}

#' Area under the ROC curve
#'
#' Uses the Mann-Whitney form, which is exact under ties and avoids building the
#' ROC curve itself: \eqn{AUC = (R_{pos} - n_{pos}(n_{pos}+1)/2) / (n_{pos} n_{neg})}
#' where \eqn{R_{pos}} is the sum of the mid-ranks of the positive class.
#'
#' @param score Numeric vector.
#' @param positive Logical vector.
#'
#' @return A single number, or `NA` when one class is empty.
#'
#' @export
auroc <- function(score, positive) {
    score <- as.numeric(score)
    positive <- as.logical(positive)
    keep <- is.finite(score)
    score <- score[keep]
    positive <- positive[keep]
    # Double, like the margins in chi2_split() and for the same reason: the
    # product of two counts passes 2^31 long before the cohort gets large, and
    # an integer overflow here would return NA for a perfectly ordinary AUC.
    n_pos <- as.numeric(sum(positive))
    n_neg <- as.numeric(sum(!positive))
    if (n_pos == 0 || n_neg == 0) {
        return(NA_real_)
    }
    ranks <- rank(score, ties.method = "average")
    (sum(ranks[positive]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

#' Monte-Carlo P value with the (k + 1) / (n + 1) correction
#'
#' The correction matters here: without it a statistic that beats every one of
#' 2,000 draws is reported as `P = 0`, which is not a number that exists.
#'
#' @param null_values Numeric vector of null statistics.
#' @param observed The observed statistic.
#' @param tail `"lower"`, `"upper"` or `"two-sided"`. Defaults to
#'   `"two-sided"`, which is the question a user of the framework asks: whether
#'   the set is associated with the axis, not whether it is depleted by it. A
#'   one-sided default answers a narrower question than the caller asked, and it
#'   did here -- the matched nulls were being read in the lower tail while the
#'   conventional analyses they are compared against reject in either direction.
#'   Pass the tail explicitly when a directional hypothesis is what is wanted.
#'
#' @return A single P value.
#'
#' @export
empirical_p <- function(null_values, observed, tail = "two-sided") {
    null_values <- as.numeric(null_values)
    null_values <- null_values[is.finite(null_values)]
    n <- length(null_values)
    if (n == 0) {
        return(NA_real_)
    }
    k <- switch(
        tail,
        lower = sum(null_values <= observed),
        upper = sum(null_values >= observed),
        `two-sided` = {
            centre <- stats::median(null_values)
            sum(abs(null_values - centre) >= abs(observed - centre))
        },
        stop(sprintf("unknown tail '%s'", tail), call. = FALSE))
    (k + 1) / (n + 1)
}

#' Everything worth reporting about one null comparison
#'
#' @param null_values Numeric vector of null statistics.
#' @param observed The observed statistic.
#' @param tail Passed to [empirical_p()].
#'
#' @return A list with the null's mean, SD, quantiles, the observed value's
#'   percentile and z-score within it, and the Monte-Carlo P value.
#'
#' @details
#' The z-score is reported alongside the percentile because a heavy-tailed null
#' can put an observation at an extreme percentile without moving it far in
#' absolute terms.
#'
#' The two-sided default matches [empirical_p()]; see there for why the tail is
#' not a parameter to leave implicit.
#'
#' @export
null_summary <- function(null_values, observed, tail = "two-sided") {
    null_values <- as.numeric(null_values)
    null_values <- null_values[is.finite(null_values)]
    n <- length(null_values)
    qs <- if (n > 0) {
        stats::quantile(null_values, c(0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975),
                        names = FALSE, type = 7)
    } else {
        rep(NA_real_, 7)
    }
    sd_null <- if (n > 1) stats::sd(null_values) else NA_real_
    list(observed = observed,
         null_n = n,
         null_mean = if (n > 0) mean(null_values) else NA_real_,
         null_sd = sd_null,
         null_p2_5 = qs[1], null_p5 = qs[2], null_p25 = qs[3],
         null_median = qs[4], null_p75 = qs[5], null_p95 = qs[6],
         null_p97_5 = qs[7],
         percentile = if (n > 0) 100 * mean(null_values <= observed) else NA_real_,
         p_value = empirical_p(null_values, observed, tail = tail),
         z_score = if (n > 1 && is.finite(sd_null) && sd_null > 0) {
             (observed - mean(null_values)) / sd_null
         } else {
             NA_real_
         })
}

#' Cutpoint of a score that best separates the outcome classes
#'
#' Scans quantiles of the score and returns the one maximising the chi-square
#' statistic of the resulting 2 x 2 table.  This is what "we split patients at
#' the optimal threshold" does, and its cost is measured by [cutpoint_fpr()].
#'
#' @param score Numeric vector.
#' @param outcome Logical vector.
#' @param min_frac Smallest admissible group size, as a fraction of the cells.
#' @param n_grid Number of candidate cutpoints.
#'
#' @return The cutpoint, or `NA` when no split is admissible.
#'
#' @export
optimum_cutpoint <- function(score, outcome, min_frac = 0.10, n_grid = 99) {
    score <- as.numeric(score)
    outcome <- as.logical(outcome)
    keep <- is.finite(score)
    score <- score[keep]
    outcome <- outcome[keep]
    if (length(score) < 20 || all(outcome) || !any(outcome)) {
        return(NA_real_)
    }
    grid <- unique(stats::quantile(score, seq(min_frac, 1 - min_frac,
                                              length.out = n_grid),
                                   names = FALSE, type = 7))
    best <- NA_real_
    best_stat <- -Inf
    for (cut in grid) {
        groups <- score > cut
        if (sum(groups) < 5 || sum(!groups) < 5) {
            next
        }
        stat <- chi2_split(outcome, groups)
        if (stat > best_stat) {
            best_stat <- stat
            best <- cut
        }
    }
    best
}

#' @rdname stats
#' @keywords internal
chi2_split <- function(values, groups) {
    # sum() on a logical vector returns an integer, and the four margins
    # multiplied together pass 2^31 as soon as the table has a few hundred
    # cells per side -- which is every real dataset.  The product would come
    # back NA and the statistic with it, silently turning the best cutpoint
    # into "no cutpoint at all".  The counts are therefore made double before
    # they are multiplied; there is no integer arithmetic below.
    values <- as.logical(values)
    groups <- as.logical(groups)
    a <- as.numeric(sum(values[groups]))
    b <- as.numeric(sum(groups)) - a
    cc <- as.numeric(sum(values[!groups]))
    d <- as.numeric(sum(!groups)) - cc
    n <- a + b + cc + d
    if (n == 0) {
        return(0)
    }
    denom <- (a + b) * (cc + d) * (a + cc) * (b + d)
    if (denom <= 0) {
        return(0)
    }
    n * (a * d - b * cc)^2 / denom
}

#' P value of a 2 x 2 split
#'
#' @param score Numeric vector.
#' @param outcome Logical vector.
#' @param cut The cutpoint.
#'
#' @return A two-sided P value for the association between the split and the
#'   outcome, or `NA` when the table is degenerate.
#'
#' @export
split_pvalue <- function(score, outcome, cut) {
    score <- as.numeric(score)
    outcome <- as.logical(outcome)
    keep <- is.finite(score)
    score <- score[keep]
    outcome <- outcome[keep]
    groups <- score > cut
    if (all(groups) || !any(groups)) {
        return(NA_real_)
    }
    tab <- table(factor(outcome, levels = c(FALSE, TRUE)),
                 factor(groups, levels = c(FALSE, TRUE)))
    if (any(rowSums(tab) == 0) || any(colSums(tab) == 0)) {
        return(NA_real_)
    }
    suppressWarnings(stats::chisq.test(tab)$p.value)
}

#' False-positive rate of a cutpoint rule under a true null
#'
#' The outcome is permuted `n_perm` times, the rule is re-applied each time, and
#' the fraction of permutations declared significant at `alpha` is returned.
#' Under a valid rule this is `alpha`; the whole point is that an
#' outcome-chosen cutpoint is not a valid rule.
#'
#' @param score Numeric vector.
#' @param outcome Logical vector.
#' @param n_perm Number of permutations.
#' @param method `"optimum"` re-selects the best cutpoint in every permutation;
#'   `"median"` uses the median of the score, which does not depend on the
#'   outcome; `"pre_specified"` uses `pre_specified`.
#' @param pre_specified A fixed cutpoint chosen without reference to the outcome.
#' @param alpha Nominal level.
#' @param seed Integer.
#'
#' @return A list with `fpr`, `n_perm`, `method`, and the observed split P value
#'   and cutpoint.
#'
#' @examples
#' set.seed(4)
#' score <- rnorm(300)
#' outcome <- score > 0
#' cutpoint_fpr(score, outcome, n_perm = 100, method = "median")$fpr
#'
#' @export
cutpoint_fpr <- function(score, outcome, n_perm = 1000, method = "optimum",
                         pre_specified = NULL, alpha = ALPHA, seed = 0) {
    score <- as.numeric(score)
    outcome <- as.logical(outcome)
    keep <- is.finite(score)
    score <- score[keep]
    outcome <- outcome[keep]
    if (length(score) < 20 || all(outcome) || !any(outcome)) {
        return(list(fpr = NA_real_, n_perm = 0L, method = method,
                    observed_p = NA_real_, observed_cutpoint = NA_real_))
    }

    cut <- switch(
        method,
        median = stats::median(score),
        pre_specified = {
            if (is.null(pre_specified)) {
                stop("pre_specified cutpoint is required for that method",
                     call. = FALSE)
            }
            as.numeric(pre_specified)
        },
        optimum = optimum_cutpoint(score, outcome),
        stop(sprintf("unknown method '%s'", method), call. = FALSE))
    observed_p <- if (is.finite(cut)) split_pvalue(score, outcome, cut) else NA_real_

    set.seed(seed)
    hits <- 0L
    for (i in seq_len(as.integer(n_perm))) {
        permuted <- sample(outcome)
        c_i <- if (identical(method, "optimum")) {
            optimum_cutpoint(score, permuted)
        } else {
            cut
        }
        if (!is.finite(c_i)) {
            next
        }
        p <- split_pvalue(score, permuted, c_i)
        if (is.finite(p) && p < alpha) {
            hits <- hits + 1L
        }
    }
    list(fpr = hits / as.numeric(n_perm), n_perm = as.integer(n_perm),
         method = method, observed_p = observed_p, observed_cutpoint = cut)
}

#' The four conventional analyses, on one score and one axis
#'
#' Naive correlation, label permutation, median cutpoint and outcome-chosen
#' cutpoint.  These are the alternatives the matched nulls are compared against;
#' a calibrated framework has to show both that they can fail and by how much.
#'
#' @param score Numeric vector.
#' @param axis Numeric vector, the variable the score is tested against.
#' @param n_perm Permutations for the label-permutation test.
#' @param seed Integer.
#'
#' @return A named list of P values: `naive_p`, `perm_p`, `cut_median_p`,
#'   `cut_opt_p`, plus the observed correlations.
#'
#' @export
all_conventional <- function(score, axis, n_perm = 1000, seed = 0) {
    score <- as.numeric(score)
    axis <- as.numeric(axis)
    sp <- spearman(score, axis)

    set.seed(seed)
    hits <- 0L
    for (i in seq_len(as.integer(n_perm))) {
        if (abs(spearman(score, sample(axis))$rho) >= abs(sp$rho)) {
            hits <- hits + 1L
        }
    }
    perm_p <- (hits + 1) / (as.integer(n_perm) + 1)

    outcome <- axis > stats::median(axis)
    med <- cutpoint_fpr(score, outcome, n_perm = n_perm, method = "median",
                        seed = seed)
    opt <- cutpoint_fpr(score, outcome, n_perm = n_perm, method = "optimum",
                        seed = seed)

    list(observed_rho = sp$rho, naive_p = sp$p, permutation_p = perm_p,
         cut_median_fpr = med$fpr, cut_optimal_fpr = opt$fpr,
         cut_median_p = med$observed_p, cut_opt_p = opt$observed_p,
         cut_optimal_at = opt$observed_cutpoint)
}
