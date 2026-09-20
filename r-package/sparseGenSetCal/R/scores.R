#' Gene-set scores, implemented against their published definitions
#'
#' Two of the methods are rank-based and evaluate from a `RankCache`; the other
#' two are expression-based and need the matrix.  All four are here so that a
#' calibration result can be reported as a property of *the analysis* rather
#' than of one particular scoring function.
#'
#' @references
#' Aibar S. et al. \emph{Nature Methods} 14, 1083-1086 (2017).
#'
#' Andreatta M., Carmona S.J. \emph{Computational and Structural Biotechnology
#' Journal} 19, 3796-3798 (2021).
#'
#' @name scores
#' @keywords internal
NULL

#' AUCell scores
#'
#' Area under the recovery curve over the top `max_rank` ranks, normalised by
#' `length(genes) * max_rank`.
#'
#' @param cache A `RankCache`.
#' @param genes Character vector.  Genes absent from the cache are dropped.
#' @param max_rank Rank ceiling; defaults to `ceil(0.05 * cache$n_genes_total)`.
#'
#' @return Numeric vector of scores in \eqn{[0, 1]}, one per cell.
#'
#' @examples
#' X <- make_synthetic_matrix(n_cells = 100, n_genes = 500, seed = 2)
#' cache <- rank_cache(X$matrix, X$genes, ceiling = 40)
#' head(aucell(cache, X$genes[1:10]))
#'
#' @export
aucell <- function(cache, genes, max_rank = NULL) {
    pos <- cache$positions(genes)
    if (length(pos) == 0) {
        stop("none of the requested genes are in the cache", call. = FALSE)
    }
    if (is.null(max_rank)) {
        max_rank <- as.integer(ceiling(0.05 * cache$n_genes_total))
    }
    max_rank <- as.integer(max_rank)
    if (max_rank > cache$ceiling) {
        # A rank above the ceiling was never stored, so the score silently
        # omits it: the answer is a lower bound rather than an AUCell score.
        # This is easy to do by accident, because the default ceiling is
        # ceil(0.05 * G) and the cache may have been built with less.
        warning(sprintf(
            "max_rank %d is above the cache ceiling %d, whose ranks were not stored; the score omits them and is a lower bound. Rebuild the cache with ceiling >= %d.",
            max_rank, cache$ceiling, max_rank), call. = FALSE)
    }
    r <- cache$clipped[, pos, drop = FALSE]
    contrib <- ifelse(r > 0 & r <= max_rank, max_rank - r + 1, 0)
    rowSums(contrib) / (length(pos) * max_rank)
}

#' UCell scores
#'
#' Mann-Whitney U of the set's capped ranks, normalised to \eqn{[0, 1]}.  The
#' UCell documentation recommends setting `r_max` to the median number of
#' detected genes per cell, which [depth_report()] reports; the package default
#' of 1500 suits 10x data with typical depth and is far too high for shallow
#' data.
#'
#' @inheritParams aucell
#' @param r_max Rank cap.
#' @param normalisation `"v2"` uses \eqn{U_{max} = k r_{max} - k(k+1)/2}, the
#'   corrected maximum introduced in UCell v2; `"v1"` uses the original paper's
#'   \eqn{U_{max} = k r_{max}}.
#'
#' @return Numeric vector of scores.
#'
#' @details
#' The published cap is \eqn{r' = r_{max} + 1} for ranks beyond the cap, so the
#' largest value the U statistic can take is
#' \eqn{k(r_{max}+1) - k(k+1)/2}, while the v2 normaliser is
#' \eqn{k r_{max} - k(k+1)/2}.  The two differ by \eqn{k}, so the score floors
#' at \eqn{-1/(r_{max} - (k+1)/2)} rather than exactly zero: invisible at
#' \eqn{r_{max} = 1500}, but \eqn{-0.018} at \eqn{r_{max} = 60} with \eqn{k = 10}.
#' Scores are returned unclipped so that they reproduce the reference
#' implementation exactly.
#'
#' @export
ucell <- function(cache, genes, r_max = 1500, normalisation = "v2") {
    pos <- cache$positions(genes)
    if (length(pos) == 0) {
        stop("none of the requested genes are in the cache", call. = FALSE)
    }
    r_max <- as.integer(r_max)
    if (r_max > cache$ceiling) {
        stop(sprintf(
            "cache ceiling %d is below r_max %d; rebuild the cache with a ceiling of at least %d",
            cache$ceiling, r_max, r_max), call. = FALSE)
    }
    k <- length(pos)
    r <- cache$clipped[, pos, drop = FALSE]
    capped <- ifelse(r > 0 & r <= r_max, r, r_max + 1)
    u_stat <- rowSums(capped) - k * (k + 1) / 2
    u_max <- if (identical(normalisation, "v1")) {
        k * r_max
    } else {
        k * r_max - k * (k + 1) / 2
    }
    1 - u_stat / u_max
}

#' Mean expression of a gene set, minus expression-matched controls
#'
#' The scanpy default.  For each gene of the set, `ctrl_size` control genes are
#' drawn from the same expression bin, and the score is the mean expression of
#' the set minus the mean of the pooled controls.  It produces no null
#' distribution on its own; it is here so that a calibration result can be
#' checked across scoring rules.
#'
#' @param X Cells x genes matrix.
#' @param genes Gene names, one per column of `X`.
#' @param gene_set Character vector.
#' @param ctrl_size Number of control genes drawn per gene of the set.
#' @param n_bins Number of expression bins.
#' @param seed Integer.
#'
#' @return Numeric vector of scores, one per cell.  Not bounded; higher means
#'   the set is more expressed than its expression-matched controls.
#'
#' @export
score_genes <- function(X, genes, gene_set, ctrl_size = 50, n_bins = 25,
                        seed = 0) {
    genes <- as.character(genes)
    pos <- match(intersect(as.character(gene_set), genes), genes)
    if (length(pos) == 0) {
        stop("none of the genes in gene_set are present", call. = FALSE)
    }
    X <- as_dgCMatrix(X)
    mean_expression <- as.numeric(Matrix::colMeans(X))
    labels <- expression_bins(mean_expression, n_bins)

    set.seed(seed)
    controls <- integer(0)
    for (t in pos) {
        candidates <- which(labels == labels[t] & !(seq_along(genes) %in% pos))
        if (length(candidates) == 0) {
            candidates <- which(!(seq_along(genes) %in% pos))
        }
        controls <- c(controls,
                      sample(candidates, min(ctrl_size, length(candidates))))
    }
    as.numeric(Matrix::rowMeans(X[, pos, drop = FALSE]) -
               Matrix::rowMeans(X[, controls, drop = FALSE]))
}

#' @rdname scores
#' @keywords internal
expression_bins <- function(values, n_bins) {
    cut(rank(values, ties.method = "first"),
        breaks = unique(stats::quantile(
            seq_along(values) / (length(values) + 1),
            probs = seq(0, 1, length.out = n_bins + 1))),
        include.lowest = TRUE, labels = FALSE)
}

#' Score the same set by every method the framework supports
#'
#' @param cache A `RankCache`.
#' @param X The expression matrix the cache was built from, or `NULL` to skip
#'   the expression-based methods.
#' @param gene_set Character vector.
#' @param genes Gene names for the columns of `X`.
#' @param ucell_r_max Rank cap for UCell.
#' @param seed Integer, for the expression-matched controls.
#'
#' @return A data frame with columns `method`, `cell` and `score`.
#'
#' @export
score_all <- function(cache, X, genes, gene_set, ucell_r_max = 1500,
                      seed = 0) {
    frames <- list(
        data.frame(method = "AUCell", cell = seq_len(nrow(cache$clipped)),
                   score = aucell(cache, gene_set)),
        data.frame(method = "UCell", cell = seq_len(nrow(cache$clipped)),
                   score = ucell(cache, gene_set, r_max = ucell_r_max)))
    if (!is.null(X)) {
        frames[[length(frames) + 1L]] <- data.frame(
            method = "score_genes", cell = seq_len(nrow(cache$clipped)),
            score = score_genes(X, genes, gene_set, seed = seed))
    }
    do.call(rbind, frames)
}
