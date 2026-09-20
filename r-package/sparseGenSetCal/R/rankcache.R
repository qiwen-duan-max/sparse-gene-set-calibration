#' Clipped-rank cache
#'
#' Every rank-based gene-set score is a function of each gene's rank *within a
#' cell*, and in practice only of the ranks below a ceiling.  Building the full
#' rank matrix is \eqn{O(n_{cells} n_{genes} \log n_{genes})} and costs four
#' bytes per entry, which makes a calibration sweep over hundreds of gene sets
#' and thousands of null draws impossible on ordinary hardware.  This reference
#' class builds the ranks once and keeps only what scoring needs: for each cell
#' and each gene of interest, the exact rank when that rank is at or below the
#' ceiling, and zero otherwise.  Ranks above the ceiling contribute nothing to
#' any score implemented here, so the compression is lossless.
#'
#' @section Tie-breaking:
#' Ties are settled by a deterministic key derived from the cell index, the gene
#' index and the seed -- never by a draw from the random number generator.  Pass
#' the same `seed` and the same matrix and the cache is identical, whatever
#' `chunk` is and whichever language built it.  See [tie_break_keys()] for why
#' the generator had to go.
#'
#' @field clipped Integer matrix, cells x retained genes.  Entry \eqn{(i, j)} is
#'   the one-based rank of gene \eqn{j} in cell \eqn{i} when that rank is at
#'   most `ceiling`, and zero otherwise.
#' @field genes Character vector of gene names for the columns.
#' @field n_genes_total `G`, the number of genes in the matrix the ranks came
#'   from, including genes not retained.  Rank-based scores depend on it, so it
#'   is part of the cache's identity.
#' @field ceiling The rank ceiling the cache was built with.
#' @field detection Per-gene detection rate over the cells built from.
#' @field mean_expression Per-gene mean expression over the same cells.
#' @field depth \eqn{D_c}: how many genes each cell detected, counted over the
#'   *whole* panel rather than over the retained columns.  It is the argument of
#'   the closed form in [tie_break_level()], so the cache carries it rather than
#'   making every caller recompute it from a matrix it may no longer have.
#'
#' @examples
#' X <- make_synthetic_matrix(n_cells = 200, n_genes = 800, seed = 1)
#' cache <- rank_cache(X$matrix, X$genes, ceiling = 60, seed = 2024)
#' cache$n_genes_total
#'
#' @export
RankCache <- R6::R6Class(
    "RankCache",
    public = list(
        clipped = NULL,
        genes = NULL,
        n_genes_total = NULL,
        ceiling = NULL,
        detection = NULL,
        mean_expression = NULL,
        depth = NULL,

        #' @description Build a cache from a sparse expression matrix.
        #' @param X Cells x genes matrix, sparse or dense.
        #' @param genes Gene names, one per column of `X`.
        #' @param ceiling Largest rank that must be stored exactly.
        #' @param genes_of_interest Restrict the cache to these genes.  Defaults
        #'   to every gene, which is rarely what is wanted: the point of the
        #'   cache is to drop the columns no gene set will ever query.
        #' @param seed Seed of the tie-breaking key.  A different seed settles
        #'   the ties differently; it does not change any rank that the data
        #'   already decided.
        #' @param chunk Cells per block.  It changes how long the build takes
        #'   and how much memory it needs, and nothing else: the keys do not
        #'   depend on the blocking, so neither do the ranks.
        build = function(X, genes, ceiling, genes_of_interest = NULL,
                         seed = 2024, chunk = 400) {
            if (is.null(genes)) {
                genes <- sprintf("G%05d", seq_len(ncol(X)) - 1L)
            }
            genes <- as.character(genes)
            n_cells <- nrow(X)
            n_genes <- ncol(X)
            if (length(genes) != n_genes) {
                stop(sprintf(
                    "genes has length %d but X has %d columns",
                    length(genes), n_genes), call. = FALSE)
            }
            ceiling <- as.integer(ceiling)
            if (is.na(ceiling) || ceiling < 1 || ceiling >= n_genes) {
                stop(sprintf("ceiling must lie in [1, %d], got %s",
                             n_genes - 1, ceiling), call. = FALSE)
            }

            if (is.null(genes_of_interest)) {
                keep <- seq_len(n_genes)
            } else {
                keep <- match(as.character(genes_of_interest), genes)
                keep <- keep[!is.na(keep)]
                if (length(keep) == 0) {
                    stop("none of genes_of_interest are present in X",
                         call. = FALSE)
                }
            }

            X <- as_dgCMatrix(X)
            detection <- as.numeric(col_counts(X)) / n_cells
            mean_expression <- as.numeric(Matrix::colMeans(X))
            depth <- as.integer(row_counts(X))

            clipped <- matrix(0L, nrow = n_cells, ncol = length(keep))
            for (lo in seq_len(if (n_cells > 0L) ceiling(n_cells / chunk) else 0L)) {
                lo <- (lo - 1L) * chunk + 1L
                hi <- min(lo + chunk - 1L, n_cells)
                rows <- seq.int(lo, hi)
                block <- as.matrix(X[rows, , drop = FALSE])
                # Keys are computed from the *original* row index -- zero-based,
                # to match the reference implementation -- so a cell's rank
                # inside its block is the rank it would have had if the whole
                # matrix had been ranked at once, in either language.
                keys <- tie_break_keys(rows - 1L, n_genes, seed)
                top <- ranked_columns(block, keys, ceiling)

                # Scatter the ceiling ranks into their columns.  A gene outside
                # genes_of_interest takes its rank nowhere, which is the point
                # of restricting the cache.
                block_size <- hi - lo + 1L
                gene_col <- match(as.vector(top), keep)
                present <- which(!is.na(gene_col))
                # as.vector() runs down each column in turn, so the flattened
                # position p counts rows fastest and ranks slowest.
                row_here <- ((present - 1L) %% block_size) + 1L
                rank_here <- ((present - 1L) %/% block_size) + 1L
                clipped[cbind(rows[row_here], gene_col[present])] <- rank_here
            }

            self$clipped <- clipped
            self$genes <- genes[keep]
            self$n_genes_total <- n_genes
            self$ceiling <- ceiling
            self$detection <- detection[keep]
            self$mean_expression <- mean_expression[keep]
            self$depth <- depth
            invisible(self)
        },

        #' @description Gene names of the set that are in the cache.
        #' @param gene_set Character vector.
        present = function(gene_set) {
            intersect(as.character(gene_set), self$genes)
        },

        #' @description Column positions of the set that are in the cache.
        #' @param gene_set Character vector.
        positions = function(gene_set) {
            match(self$present(gene_set), self$genes)
        },

        #' @description Ranks of the named genes, one row per cell.
        #' @param gene_set Character vector.
        ranks = function(gene_set) {
            pos <- self$positions(gene_set)
            if (length(pos) == 0) {
                stop("none of the gene set is in the cache", call. = FALSE)
            }
            self$clipped[, pos, drop = FALSE]
        },

        #' @description Keep only the rows of a subset of cells.
        #' @param idx Row indices.
        subset_cells = function(idx) {
            keep <- self$clipped[idx, , drop = FALSE]
            depth <- if (is.null(self$depth)) NULL else self$depth[idx]
            cache <- RankCache$new()
            cache$clipped <- keep
            cache$genes <- self$genes
            cache$n_genes_total <- self$n_genes_total
            cache$ceiling <- self$ceiling
            cache$detection <- as.numeric(colMeans(keep > 0))
            cache$mean_expression <- self$mean_expression
            cache$depth <- depth
            cache
        },

        #' @description A short description of the cache.
        #' @param ... Ignored.
        print = function(...) {
            cat(sprintf(
                "<RankCache> %d cells x %d cached genes (of %d), ceiling %d, depth %s\n",
                nrow(self$clipped), length(self$genes), self$n_genes_total,
                self$ceiling,
                if (is.null(self$depth)) "absent" else "recorded"))
            invisible(self)
        }
    )
)

#' Build a clipped-rank cache
#'
#' Convenience wrapper around `RankCache$build()`.
#'
#' @param X Cells x genes matrix.
#' @param genes Gene names.
#' @param ceiling Largest rank to store exactly.
#' @param genes_of_interest Columns to retain.
#' @param seed Tie-break key seed.
#' @param chunk Cells per block.
#'
#' @return A `RankCache`.
#'
#' @export
rank_cache <- function(X, genes, ceiling, genes_of_interest = NULL,
                       seed = 2024, chunk = 400) {
    RankCache$new()$build(X, genes, ceiling, genes_of_interest = genes_of_interest,
                          seed = seed, chunk = chunk)
}
