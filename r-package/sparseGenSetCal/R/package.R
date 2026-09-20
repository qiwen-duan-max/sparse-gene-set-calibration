#' sparseGenSetCal: calibration for sparse gene-set scores
#'
#' Gene-set scores computed on sparse single-cell data are routinely interpreted
#' against a null the data cannot support.  This package supplies what is needed
#' to replace that null with one the data can support, and to find out before
#' running anything whether it will matter.
#'
#' The mechanism it is built around is a closed form.  Every rank-based score
#' asks where a gene sits among all genes in a cell, and a cell that expresses
#' fewer genes than the score's rank ceiling has no count to place in the
#' remaining positions -- the tie-break fills them with genes the cell never
#' expressed.  The expected share of the score contributed by one such gene is
#' \deqn{\tau(D) = \frac{(m-D)(m-D+1)}{2m(G-D)},}
#' with \eqn{m} the rank ceiling, \eqn{G} the panel size and \eqn{D} the number
#' of genes the cell detected.  That is a function of the cell's depth and not
#' of the gene set, and depth tracks almost every biological axis one cares
#' about.
#'
#' Start with [preflight()], which needs no score and no gene set beyond its
#' detection rates, or with [Calibrator] and [diagnostic_checklist()] for the
#' assembled four-step surface.  [score_decomposition()] splits a real score
#' into the part the data supports and the part the tie-break supplied, and
#' [construct_expression_matched_null()] draws the null that the split implies.
#'
#' @importFrom R6 R6Class
#' @importFrom utils combn
#'
#' @references
#' Aibar S. et al. \emph{Nature Methods} 14, 1083-1086 (2017).
#'
#' Andreatta M., Carmona S.J. \emph{Computational and Structural Biotechnology
#' Journal} 19, 3796-3798 (2021).
#'
#' @keywords internal
"_PACKAGE"
