#' How a tie is settled
#'
#' Most of the entries of a single-cell expression matrix are zero, so most of
#' the comparisons a rank-based score makes are between two values that are
#' equal.  The order the zeros end up in is not a detail: it decides which
#' undetected genes occupy the ranks inside the ceiling, and hence what fraction
#' of a gene set's score was never measured at all.  That is the subject of
#' [tie_break_level()].
#'
#' The reference AUCell implementation breaks the ties by adding a uniform
#' random perturbation to the expression values before ranking.  Drawing that
#' perturbation from the language's own random number generator has two
#' consequences a calibration tool cannot live with.  The ranks depend on how
#' the matrix was blocked, because a generator produces a stream and a block of
#' 500 cells consumes a different part of it than two blocks of 250; and they
#' depend on which language is running, because R's Mersenne-Twister and NumPy's
#' PCG64 do not produce the same numbers, so a port can only be tested for
#' statistical similarity and never for equality.
#'
#' This file replaces the stream with a *key*: a pure function of the cell
#' index, the gene index and the seed, with no generator state and no dependence
#' on how the work is divided up.  Ranking is by `(expression descending, key
#' ascending)`, so a tie is settled by the key and everything else is settled by
#' the data.
#'
#' @section Why the arithmetic is written out by hand:
#' The key is MurmurHash3's 32-bit finaliser over a linear mix of the
#' coordinates.  It has to produce the same integers as the Python reference
#' implementation, so the multiplication is done in 16-bit halves: a product of
#' two 32-bit numbers is up to 64 bits wide, which a double cannot hold exactly,
#' but the low and high halves of each operand fit in 16 bits each and every
#' intermediate stays below \eqn{2^{53}}.  Shifts and exclusive-ors are done the
#' same way, through [bitwXor()] on the halves.
#'
#' @name tie_break_keys
#' @keywords internal
NULL

#: 2**32, the modulus every key operation works in.
MASK32 <- 4294967296

# Odd multipliers with good bit dispersion.  The first two are the golden-ratio
# and MurmurHash3 constants; the third separates the seeds.
.C_CELL <- 0x9E3779B1
.C_GENE <- 0x85EBCA6B
.C_SEED <- 0xC2B2AE35

#' @rdname tie_break_keys
#' @param a,b Numeric vectors with entries in \eqn{[0, 2^{32})}.
#' @return `mul32()` returns `a * b` modulo \eqn{2^{32}}.
mul32 <- function(a, b) {
    a0 <- a %% 65536
    a1 <- (a - a0) / 65536
    b0 <- b %% 65536
    b1 <- (b - b0) / 65536
    lo <- a0 * b0
    mid <- ((a0 * b1 + a1 * b0) %% 65536) * 65536
    (lo + mid) %% MASK32
}

#' @rdname tie_break_keys
#' @return `xor32()` returns the bitwise exclusive-or of `a` and `b`.  Both
#'   operands must be whole numbers below \eqn{2^{32}}; the halves go through
#'   [bitwXor()], which needs values that fit in a 32-bit signed integer.
xor32 <- function(a, b) {
    # bitwXor() takes integers, and as.integer() on a matrix returns a bare
    # vector, so the shape has to be put back by hand.  Everything downstream
    # indexes the key matrix by row and column, and a silently flattened key
    # would be indexed by position instead -- which is the failure mode that
    # looks like a wrong hash rather than a lost dimension.
    shape <- dim(a)
    a0 <- a %% 65536
    a1 <- (a - a0) / 65536
    b0 <- b %% 65536
    b1 <- (b - b0) / 65536
    out <- bitwXor(as.integer(a0), as.integer(b0)) +
        bitwXor(as.integer(a1), as.integer(b1)) * 65536
    if (!is.null(shape)) dim(out) <- shape
    out
}

#' @rdname tie_break_keys
#' @param h A numeric vector of keys.
#' @param bits Number of places to shift right.
#' @return `rshift32()` returns the logical right shift of `h`.
rshift32 <- function(h, bits) floor(h / 2^bits)

#' @rdname tie_break_keys
#' @return `fmix32()` returns the finalised keys.
fmix32 <- function(h) {
    h <- xor32(h, rshift32(h, 16))
    h <- mul32(h, 0x7FEB352D)
    h <- xor32(h, rshift32(h, 15))
    h <- mul32(h, 0x846CA68B)
    xor32(h, rshift32(h, 16))
}

#' Tie-break keys for a block of cells against a whole panel of genes
#'
#' @param cells Integer vector of cell coordinates.  These are **zero-based**,
#'   matching the reference implementation's own row indices, and they have to
#'   be the coordinates in the *original* matrix rather than the position within
#'   the block.  R's own row indices are one-based, so a caller holding them has
#'   to subtract one: the key is a function of the coordinate, and a cell keyed
#'   as 11 here and as 12 in Python would rank differently in the two languages
#'   for no reason a reader could see.  [rank_cache()] does this for you.
#' @param n_genes Integer `G`, the number of genes in the panel.  Gene
#'   coordinates are zero-based as well, and they are positions in the panel,
#'   not in `genes_of_interest`.
#' @param seed Integer.
#'
#' @return A `length(cells)` x `n_genes` numeric matrix of keys in
#'   \eqn{[0, 2^{32})}.  The keys are uniformly distributed over the range and
#'   independent across coordinates, so comparing two of them is a fair coin.
#'
#' @examples
#' # The key of a cell is the same whether the block is large or small, which is
#' # what makes the ranks independent of the chunk size.
#' whole <- sparseGenSetCal:::tie_break_keys(0:9, 50, seed = 7)
#' band <- sparseGenSetCal:::tie_break_keys(2:4, 50, seed = 7)
#' identical(whole[3:5, ], band)
#'
#' @keywords internal
tie_break_keys <- function(cells, n_genes, seed = 2024) {
    cells <- as.numeric(cells)
    genes <- seq_len(n_genes) - 1
    h <- outer(mul32(cells, .C_CELL), mul32(genes, .C_GENE), "+")
    h <- (h + mul32(as.numeric(seed) %% MASK32, .C_SEED)) %% MASK32
    fmix32(h)
}

#' Gene columns taking ranks 1..ceiling in each row of a block
#'
#' The total order is `(expression descending, key ascending)`.  A detected gene
#' therefore outranks an undetected one whatever the keys say -- a count of one
#' beats a zero even when the zero's key is smaller -- and among the undetected
#' genes the keys give the uniform random order that makes equation (1) an
#' expectation.
#'
#' This matters more than it looks.  Count data is full of exact ties: a cell
#' with a thousand detected genes has many of them at a count of one.  So
#' "which of these equally expressed genes takes the last rank inside the
#' ceiling" is decided in essentially every cell, and [order()] settles it from
#' the key rather than from whatever the selection algorithm happens to do with
#' equal inputs.
#'
#' @param block Dense cells x genes matrix of expression values.
#' @param keys Numeric matrix of the same shape, from [tie_break_keys()].
#' @param ceiling Integer `m`.
#'
#' @return An integer matrix with `ceiling` columns holding, for each cell, the
#'   gene column that takes each rank in turn.
#'
#' @keywords internal
ranked_columns <- function(block, keys, ceiling) {
    if (!identical(dim(block), dim(keys))) {
        stop("keys must have the same shape as the block", call. = FALSE)
    }
    ceiling <- as.integer(ceiling)
    n_genes <- ncol(block)
    if (is.na(ceiling) || ceiling < 1L || ceiling > n_genes) {
        stop(sprintf("ceiling must lie in [1, %d], got %s", n_genes, ceiling),
             call. = FALSE)
    }
    out <- matrix(0L, nrow = nrow(block), ncol = ceiling)
    for (i in seq_len(nrow(block))) {
        # Negating the expression is what puts the detected genes first; order()
        # is a stable lexicographic sort, so the key decides exactly the ties.
        out[i, ] <- order(-block[i, ], keys[i, ])[seq_len(ceiling)]
    }
    out
}
