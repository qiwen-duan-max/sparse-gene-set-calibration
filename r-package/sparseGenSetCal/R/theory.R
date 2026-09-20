#' The closed form behind the calibration
#'
#' Rank-based gene-set scores are sums over the top ranks of a cell.  In a
#' shallow cell the number of genes actually detected is far smaller than the
#' rank ceiling, and the remaining slots are filled by genes with a count of
#' zero, ordered among themselves by an arbitrary tie-break.  A gene the cell
#' never expressed therefore still enters the score, with a probability that is
#' a deterministic function of the cell's depth.  Depth tracks almost every
#' biological axis one cares about, so this is where spurious association in
#' single-cell gene-set scoring comes from.
#'
#' @section The derivation:
#' Let the matrix have \eqn{G} genes and let cell \eqn{c} have non-zero counts
#' on \eqn{D_c} of them.  Write \eqn{m} for the rank ceiling,
#' \eqn{m = \lceil f G \rceil}.  Ranking places the \eqn{D_c} detected genes at
#' ranks \eqn{1..D_c}; the remaining \eqn{G - D_c} genes all have a count of zero
#' and each is equally likely to land at any rank in \eqn{D_c + 1 .. G}.
#'
#' AUCell gives a gene at rank \eqn{r} the weight \eqn{\max(0, m - r + 1)} and
#' divides the sum over the \eqn{k} genes of the set by \eqn{k m}.  The expected
#' weight of a gene the cell never expressed is therefore
#' \deqn{\tau(D) = \frac{(m-D)(m-D+1)}{2m(G-D)} \qquad (1)}
#' and the score splits into a detected part and a realised tie-break part,
#' \deqn{\mathrm{AUC}_c = \mathrm{Det}_c + \mathrm{TB}_c, \quad
#' \mathbb{E}[\mathrm{TB}_c] = \frac{k - d_c(S)}{k}\tau(D_c) \qquad (2)}
#' where \eqn{d_c(S)} is the number of set genes detected in the cell.  The
#' split is exact, cell by cell -- \code{score_decomposition} returns both
#' halves and their sum reproduces \code{aucell_scores} exactly -- while the
#' closed form is the tie-break's expectation under the uniform-order model, so
#' the residual \eqn{\mathrm{TB}_c - E[\mathrm{TB}_c]} is the realised
#' tie-break centred on its mean.  The tie-break is a deterministic hash, which
#' makes that residual a reproducible property of the matrix rather than
#' sampling noise.
#' \eqn{\mathrm{Det}_c} depends on the gene set; the expected tie-break term is
#' the same for every set with the same detection profile, is a decreasing
#' function of depth, and grows in magnitude as the set gets sparser.  That is
#' the whole problem.
#'
#' Equation (2) also says what a null has to reproduce.  A gene detected in a
#' fraction \eqn{p} of cells enters the top \eqn{m} either because it was
#' counted or, failing that, by tie-break:
#' \deqn{P(\text{in top } m) = p + (1-p)\overline{\frac{m-D}{G-D}} \qquad (3)}
#' Matching on \eqn{p} alone leaves the second term free, and because that term
#' varies with the cell's depth the mismatch does not cancel -- it tracks the
#' confounder.
#'
#' @references
#' Aibar S. et al. SCENIC: single-cell regulatory network inference and
#' clustering. \emph{Nature Methods} 14, 1083-1086 (2017).
#'
#' Andreatta M., Carmona S.J. UCell: robust and scalable single-cell gene
#' signature scoring. \emph{Computational and Structural Biotechnology Journal}
#' 19, 3796-3798 (2021).
#'
#' @keywords internal
"_PACKAGE"

#' Tie-break inclusion probability
#'
#' The probability that a gene the cell never expressed lands inside the top
#' `max_rank` positions, i.e. the fraction in equation (3).  It depends only on
#' how many genes the cell detected, never on the gene set.
#'
#' @param n_detected Integer vector of \eqn{D_c}, one entry per cell.
#' @param max_rank Integer `m`.
#' @param n_genes Integer `G`, the number of genes in the matrix the ranks were
#'   taken over, not the number retained in the cache.
#'
#' @return Numeric vector in \eqn{[0, 1]}, zero where the cell detected at least
#'   `max_rank` genes, since then no slot is left for the tie-break.
#'
#' @examples
#' # A cell detecting 400 of 20,000 genes against a ceiling of 1,000 leaves
#' # 600 slots to be settled at random out of 19,600 undetected genes.
#' tie_break_inclusion(400, 1000, 20000)
#'
#' @export
tie_break_inclusion <- function(n_detected, max_rank, n_genes) {
    D <- as.numeric(n_detected)
    m <- as.numeric(max_rank)
    G <- as.numeric(n_genes)
    pmin(pmax(pmax(m - D, 0) / pmax(G - D, 1), 0), 1)
}

#' The tie-break's mean contribution to the score
#'
#' \eqn{\tau(D)} from equation (1): the expected share of an AUCell score
#' contributed by one gene that the cell did not express.  Multiply it by the
#' fraction of the set that goes undetected and the product is the middle term
#' of equation (2).
#'
#' @inheritParams tie_break_inclusion
#'
#' @return Numeric vector, one entry per cell.
#'
#' @examples
#' # tau falls as cells get deeper, which is why the confound tracks depth.
#' tau <- tie_break_level(c(200, 400, 800), 1000, 20000)
#' stopifnot(all(diff(tau) < 0))
#'
#' @export
tie_break_level <- function(n_detected, max_rank, n_genes) {
    D <- as.numeric(n_detected)
    m <- as.numeric(max_rank)
    G <- as.numeric(n_genes)
    gap <- pmax(m - D, 0)
    gap * (gap + 1) / (2 * m * pmax(G - D, 1))
}

#' Inclusion probability of a gene under the tie-break, equation (3)
#'
#' The probability that a gene enters the top `max_rank` at all -- counted, or
#' placed there by the tie-break failing that -- given its detection rate and
#' the depths of the cells it is measured in.
#'
#' @param detection Numeric vector of per-gene detection rates \eqn{p}.
#' @param depth Integer vector of \eqn{D_c}, one entry per cell.
#' @param max_rank Integer `m`.
#' @param n_genes Integer `G`.
#'
#' @return Numeric vector with one entry per gene, averaged over the cells'
#'   depths.
#'
#' @details
#' This is the quantity an expression-matched null has to reproduce, and the
#' reason matching on \eqn{p} alone is not enough: two genes with the same
#' \eqn{p} have different inclusion when they are expressed in cells of
#' different depth, and depth is where the confound lives.
#'
#' @export
effective_inclusion <- function(detection, depth, max_rank, n_genes) {
    p <- as.numeric(detection)
    share <- tie_break_inclusion(depth, max_rank, n_genes)
    vapply(p, function(pi) mean(pi + (1 - pi) * share), numeric(1))
}

#' The share of the inclusion rate that the tie-break supplies
#'
#' Equation (3) splits a gene's chance of entering the score into the part where
#' it was counted and the part where it was placed there anyway.  This returns
#' the second as a fraction of the total, averaged over the genes of the set,
#' which is the cheapest honest answer to "is this score carried by expression?".
#'
#' @inheritParams effective_inclusion
#'
#' @return A single number in \eqn{[0, 1]}, or `NA` for an empty set.
#'
#' @details
#' Unlike a detection floor, this is a property of the gene set *in this data*
#' rather than of a number someone chose.  A set whose genes are detected in
#' half the cells has a small share however sparse the matrix; the same matrix
#' gives a large share to a set of genes detected in a tenth of them.  It is
#' zero when the cells are deep enough that the tie-break never reaches the
#' ceiling, and approaches one as the set stops being detected at all.
#'
#' @examples
#' # Detected everywhere: no tie-break share, whatever the depth.
#' tie_break_share(rep(1, 30), rep(400, 500), 1000, 20000)
#'
#' @export
tie_break_share <- function(detection, depth, max_rank, n_genes) {
    p <- as.numeric(detection)
    if (length(p) == 0) {
        return(NA_real_)
    }
    mean_p <- mean(p)
    mean_inc <- mean(effective_inclusion(p, depth, max_rank, n_genes))
    if (!is.finite(mean_inc) || mean_inc <= 0) {
        return(NA_real_)
    }
    min(max((mean_inc - mean_p) / mean_inc, 0), 1)
}

#' Split a real score into its detected and tie-broken parts
#'
#' The realised split is exact: `detected + tie_break` reproduces the score
#' cell by cell, to the floating-point gap, and the test suite asserts it.
#' Equation (2)'s closed form is the \emph{expectation} of the tie-break term
#' under the uniform-order model, so `tie_break_expected` is that closed form
#' and the difference between the two columns is the residual of the
#' manuscript's equation (3) -- centred, reproducible (the tie-break hash is
#' deterministic), and measured across the calibration grid.
#'
#' @param cache A `RankCache`.
#' @param gene_set Character vector of gene names.
#' @param detection_matrix A cells x cache-genes binary matrix, sparse or dense.
#' @param max_rank Integer `m`; defaults to `ceil(0.05 * cache$n_genes_total)`.
#' @param depth Optional numeric vector of \eqn{D_c}; counted from
#'   `detection_matrix` when omitted.
#'
#' @return A data frame with one row per cell and columns `cell`, `depth`,
#'   `detected`, `tie_break`, `tie_break_expected` and `total`.  `detected`
#'   plus `tie_break` reproduces [aucell()] exactly.
#'
#' @export
score_decomposition <- function(cache, gene_set, detection_matrix,
                                max_rank = NULL, depth = NULL) {
    present <- cache$present(gene_set)
    if (length(present) == 0) {
        stop("none of the gene set is in the cache", call. = FALSE)
    }
    if (is.null(max_rank)) {
        max_rank <- as.integer(ceiling(0.05 * cache$n_genes_total))
    }
    m <- as.integer(max_rank)
    keep <- cache$positions(present)
    k <- length(keep)

    ranks <- cache$clipped[, keep, drop = FALSE]
    contrib <- ifelse(ranks > 0 & ranks <= m, m - ranks + 1, 0)
    contrib <- matrix(as.numeric(contrib), nrow = nrow(ranks))

    B <- as_binary(detection_matrix)
    if (ncol(B) != length(cache$genes)) {
        stop(sprintf(
            "detection_matrix has %d columns; expected one per cache gene (%d)",
            ncol(B), length(cache$genes)), call. = FALSE)
    }
    # Dense over the set's own columns: a gene set is tens of genes wide, so
    # materialising them costs nothing and keeps the two terms of equation (2)
    # in one arithmetic type.
    det_set <- as.matrix(B[, keep, drop = FALSE] != 0)
    if (is.null(depth)) {
        depth <- row_counts(B)
    }
    depth <- as.numeric(depth)

    # A gene that was detected but ranked beyond the ceiling stores zero and
    # contributes nothing; it still belongs to the detected part rather than to
    # the tie-break, so the split is taken from the detection matrix and not
    # from the ranks.
    detected <- rowSums(contrib * det_set) / (k * m)
    tie <- rowSums(contrib * !det_set) / (k * m)
    n_undetected <- k - rowSums(det_set)

    data.frame(
        cell = seq_along(depth) - 1L,
        depth = depth,
        detected = detected,
        tie_break = tie,
        tie_break_expected = n_undetected / k *
            tie_break_level(depth, m, cache$n_genes_total),
        total = detected + tie,
        row.names = NULL)
}

#' Will the tie-break manufacture an association with this axis?
#'
#' Everything here comes from the depth vector and the axis.  No gene set is
#' scored and no matrix is touched, which is the point: this is the check to run
#' *before* committing to an analysis.
#'
#' @param depth Integer vector of \eqn{D_c}.
#' @param axis Numeric vector, the variable the score will be tested against.
#' @param max_rank Integer `m`.
#' @param n_genes Integer `G`.
#' @param detection Optional per-gene detection rates of the set under study.
#' @param k Optional gene-set size, used with `detection`.
#' @param score Optional already-computed score for the same cells.
#'
#' @return A list.  `tau_axis_rho` is the correlation of the score's
#'   depth-derived component with the axis; it is exact, needs no score, and is
#'   the headline number.
#'
#'   `implied_rho` scales it by the mean undetected share of a set with the
#'   supplied detection profile -- the coefficient on \eqn{\tau} in equation (2)
#'   -- giving the correlation of the score's *conditional mean* with the axis.
#'   The realised correlation is smaller, because the tie-break contributes a
#'   random amount in each cell and that noise sits in the score's variance
#'   without moving its mean: `attenuation` is \eqn{sd(\tau)/sd(score)} and
#'   `realised_implied_rho` is the product, which is where the observed
#'   correlation should land if nothing else is going on.
#'
#'   `chance_band` is what chance alone produces at this sample size, and
#'   `verdict` is `"CLEAR"`, `"CAUTION"` or `"CONFOUNDED"` against it.
#'
#'   The other half of the score, \eqn{\mathrm{Det}_c}, also varies with depth,
#'   in the opposite direction: a deeper cell detects more of the set, and a
#'   detected gene outranks every undetected one.  Which term wins is an
#'   empirical question about the data, answered by [score_decomposition()],
#'   not by this function.
#'
#' @examples
#' set.seed(1)
#' depth <- round(exp(rnorm(500, log(450), 0.4)))
#' axis <- rnorm(500)
#' preflight(depth, axis, 1000, 20000)
#'
#' @export
preflight <- function(depth, axis, max_rank, n_genes, detection = NULL,
                      k = NULL, score = NULL) {
    depth <- as.numeric(depth)
    axis <- as.numeric(axis)
    tau <- tie_break_level(depth, max_rank, n_genes)
    sd_tau <- stats::sd(tau)
    rho_tau_axis <- spearman(tau, axis)$rho

    out <- list(
        n_cells = length(depth),
        max_rank = as.integer(max_rank),
        n_genes = as.integer(n_genes),
        median_detected = stats::median(depth),
        depth_ratio = max_rank / max(stats::median(depth), 1),
        tau_axis_rho = rho_tau_axis,
        tau_sd = sd_tau,
        tau_range = range(tau),
        # A cell deeper than the ceiling contributes no tie-break at all, so if
        # every cell is, `tau` is the constant zero and its correlation with the
        # axis does not exist rather than being small.  The verdict is CLEAR
        # either way -- a component that never varies cannot carry an
        # association -- but "rho is near zero" and "rho is not defined" are
        # different statements about the data, and the record should say which
        # one it is making.
        #
        # Constancy is tested by comparing the extremes and not by asking
        # whether the standard deviation is zero, so that the test does not
        # depend on which algorithm computed the standard deviation.
        tau_constant = identical(min(tau), max(tau)),
        depth_axis_rho = spearman(depth, axis)$rho)

    if (!is.null(detection) && !is.null(k)) {
        p <- as.numeric(detection)
        # Mean undetected share of a k-gene set with this detection profile --
        # the coefficient on tau in equation (2).
        gamma <- 1 - mean(p)
        out$detection_mean <- mean(p)
        out$gamma <- gamma
        out$implied_rho <- gamma * rho_tau_axis
        out$mean_inclusion <- mean(effective_inclusion(p, depth, max_rank,
                                                       n_genes))
        out$tie_break_share <- tie_break_share(p, depth, max_rank, n_genes)
    } else {
        out$implied_rho <- rho_tau_axis
    }

    # What chance alone produces at this sample size.  A fixed cut would call a
    # 400-cell dataset clear at |rho| = 0.06 and a 50,000-cell one confounded at
    # the same value, so the band comes from the null distribution of the
    # correlation rather than from a constant.
    band <- 1.96 / sqrt(max(length(depth) - 3, 1))
    out$chance_band <- band

    if (!is.null(score)) {
        s <- as.numeric(score)
        sd_s <- stats::sd(s)
        out$attenuation <- if (sd_s > 0) sd_tau / sd_s else NA_real_
        out$realised_implied_rho <- out$implied_rho * out$attenuation
        out$observed_rho <- spearman(s, axis)$rho
        out$observed_vs_implied <- out$observed_rho - out$realised_implied_rho
    }

    a <- abs(out$tau_axis_rho)
    out$verdict <- if (!is.finite(a) || a <= band) {
        "CLEAR"
    } else if (a < 0.15) {
        "CAUTION"
    } else {
        "CONFOUNDED"
    }
    out
}

#' The range of correlations a score with no biology would still show
#'
#' A null-calibrated P value answers "is this correlation larger than the null
#' produces"; this answers the cruder question a reader asks first -- how large
#' does a correlation have to be before the null cannot produce it at all.
#'
#' @param null_rho The correlation the null produces, typically from a matched
#'   null family.
#' @param n_cells Number of cells.
#' @param level Confidence level for the band.
#'
#' @return A list with `level`, `low`, `high` and `n_cells`.
#'
#' @export
confound_band <- function(null_rho, n_cells, level = 0.95) {
    z <- stats::qnorm(0.5 + level / 2)
    half <- z / sqrt(max(n_cells - 3, 1))
    atanh_r <- atanh(min(max(null_rho, -0.999), 0.999))
    list(level = level,
         low = tanh(atanh_r - half),
         high = tanh(atanh_r + half),
         n_cells = as.integer(n_cells))
}
