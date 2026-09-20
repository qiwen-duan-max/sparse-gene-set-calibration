#' Internal helpers
#'
#' Small coercions and counts shared across the package.  None of these are
#' exported; they exist so that the same conversion is not written twice and
#' disagreed with once.
#'
#' @name sparseGenSetCal-internals
#' @keywords internal
NULL

#' @rdname sparseGenSetCal-internals
as_dgCMatrix <- function(X) {
    X <- Matrix::Matrix(X, sparse = TRUE)
    if (methods::is(X, "dMatrix")) {
        return(methods::as(X, "dgCMatrix"))
    }
    # A logical or pattern matrix -- which is what a detection matrix is --
    # is coerced through the virtual "dMatrix" class rather than straight to
    # "dgCMatrix"; the direct route is deprecated and warns on every call.
    methods::as(X, "dMatrix")
}

#' @rdname sparseGenSetCal-internals
as_binary <- function(X) {
    if (is.matrix(X) || inherits(X, "Matrix")) {
        return(as_dgCMatrix(X) > 0)
    }
    stop("detection matrix must be a matrix or Matrix", call. = FALSE)
}

#' @rdname sparseGenSetCal-internals
#'
#' @details `row_counts()` and `col_counts()` count *non-zero* entries, not the
#'   sum of the values.  The distinction is the whole subject of this package:
#'   summing a column of the expression matrix gives mean expression, counting
#'   its non-zeros gives a detection rate, and only the second one is what the
#'   tie-break argument is about.
row_counts <- function(X) {
    if (inherits(X, "Matrix")) {
        as.numeric(Matrix::rowSums(X != 0))
    } else {
        as.numeric(rowSums(X != 0))
    }
}

#' @rdname sparseGenSetCal-internals
col_counts <- function(X) {
    if (inherits(X, "Matrix")) {
        as.numeric(Matrix::colSums(X != 0))
    } else {
        as.numeric(colSums(X != 0))
    }
}

#' @rdname sparseGenSetCal-internals
is_sparse <- function(X) inherits(X, "Matrix") && !is.matrix(X)

#' A synthetic expression matrix with a realistic gene-activity gradient
#'
#' Three orders of magnitude separate the most and least active genes in a real
#' panel, and that gradient is what makes an expression-matched null possible at
#' all.  This generates a matrix with the same shape so that examples and tests
#' have something to run on that is not a uniform random matrix.
#'
#' The values are Poisson counts, not continuous draws.  Counts are what a
#' single-cell matrix actually holds, and the ties they produce are the reason
#' this package exists: a continuous matrix has almost no exact ties, so a test
#' built on one exercises the tie-break not at all.
#'
#' @param n_cells Integer.
#' @param n_genes Integer.
#' @param density Median fraction of the panel a cell detects.  This is the
#'   regime: it is what decides whether the rank ceiling can be filled with
#'   measured values, and hence how much of a score is tie-break.
#' @param seed Integer.
#'
#' @return A list with `matrix` (a `dgCMatrix` of counts), `genes` (character),
#'   and `cells` (character).
#'
#' @examples
#' X <- make_synthetic_matrix(n_cells = 100, n_genes = 500, seed = 3)
#' dim(X$matrix)
#'
#' @export
make_synthetic_matrix <- function(n_cells = 400, n_genes = 1200, density = 0.06,
                                  seed = 7) {
    set.seed(seed)
    activity <- exp(stats::rnorm(n_genes, 0, 1.5))
    activity <- activity / stats::median(activity)
    cell_scale <- exp(stats::rnorm(n_cells, 0, 0.45))
    shape <- outer(cell_scale, activity)

    # One scalar fixes the depth of the whole matrix, and the depth is the
    # regime.  Solve for the scalar that puts the median cell at the requested
    # density: bisection on the log scale, because detection is a sum of
    # saturating exponentials and has no closed-form inverse.
    target <- density * n_genes
    detected_at <- function(log_scale) {
        stats::median(rowSums(1 - exp(-pmin(shape * exp(log_scale), 30))))
    }
    lo <- -30
    hi <- 5
    for (i in seq_len(200)) {
        mid <- (lo + hi) / 2
        if (detected_at(mid) < target) lo <- mid else hi <- mid
    }
    rate <- pmin(shape * exp((lo + hi) / 2), 30)
    values <- matrix(stats::rpois(n_cells * n_genes, as.vector(rate)),
                     n_cells, n_genes)
    list(matrix = Matrix::Matrix(values, sparse = TRUE),
         genes = sprintf("G%05d", seq_len(n_genes) - 1L),
         cells = sprintf("C%05d", seq_len(n_cells) - 1L))
}
