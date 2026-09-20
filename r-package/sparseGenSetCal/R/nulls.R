#' A detection floor low enough that it does not bind for this gene set
#'
#' [MatchedNullBuilder] excludes genes detected in fewer than `det_floor` of
#' cells, so that the pool is not filled with genes that are almost never
#' observed and matching degenerates.  A *fixed* floor does something else as
#' well, and that something else is a bias: when the gene set's own genes are
#' sparser than the floor, they are excluded from the pool of candidates that
#' could replace them, and every replacement is denser than the gene it stands
#' in for.  The null is then systematically richer than the observed set and the
#' test is conservative in exactly the regime the calibration is meant to
#' address.
#'
#' In a simulated panel at a target detection of 0.01, a fixed floor of 0.01
#' leaves 22.6\% of the background ineligible and makes the drawn null sets 19%
#' denser than the set they replace.  This rule brings that to 1.6\%.
#'
#' @param cache A `RankCache`.
#' @param gene_set Character vector.
#' @param cap The largest floor this will return; 0.01 is appropriate for a
#'   well-detected set.
#' @param frac The floor is this fraction of the set's 10th-percentile detection.
#' @param floor Absolute lower bound.  A set containing genes that are never
#'   detected cannot be matched at all -- the pool has no member with zero
#'   detection to stand in for them -- and this is where the function stops
#'   trying.
#'
#' @return A single number.
#'
#' @export
adaptive_detection_floor <- function(cache, gene_set, cap = 0.01, frac = 0.25,
                                     floor = 1e-4) {
    pos <- cache$positions(gene_set)
    if (length(pos) == 0) {
        return(floor)
    }
    q10 <- as.numeric(stats::quantile(cache$detection[pos], 0.10, names = FALSE))
    max(floor, min(cap, frac * q10))
}

#' Build expression- and co-detection-matched null gene sets
#'
#' A null gene set has to look like the observed one in everything except the
#' biology under test: the same number of genes, detected in the same fraction
#' of cells, expressed at the same level, and -- in the strongest family -- seen
#' in the same cells.  Matching on size alone is what a "random gene set" null
#' does, and it is not enough, because detection rate enters the score's
#' variance and co-detection enters its mean.
#'
#' @section The families:
#' \describe{
#'   \item{`"random"`}{Uniform draw from the background universe.  Controls size
#'     and nothing else; included because it is the null most analyses use.}
#'   \item{`"expression_bin"`}{Draws from the same detection-rate stratum.
#'     Controls the mean detection rate, not its spread.}
#'   \item{`"expression"`}{Nearest neighbours in a relative distance combining
#'     detection rate and mean expression, one replacement per gene.  The
#'     default, and the weakest family that is defensible.}
#'   \item{`"codetection"`}{As `"expression"`, then refined against the observed
#'     co-detection pattern, so that genes that appear together in the observed
#'     set are replaced by genes that appear together.  Needs `detection_matrix`.}
#' }
#'
#' @section Sharing genes with the observation:
#' No replacement is drawn from the set being replaced.  Excluding only the gene
#' it stands in for is not enough: the other members of the set are still
#' eligible, and a null that shares genes with the observation is not a null --
#' its score is correlated with the observed score by construction, which makes
#' every P value drawn from it conservative.  Where the pool cannot supply
#' distinct, non-member genes, the draw relaxes in the order *distinctness first,
#' then exclusion*, and reports how often it had to.
#'
#' @field cache The `RankCache` the builder draws from; its gene list defines
#'   the background universe.
#' @field pool Column positions of the genes clearing `det_floor` -- the
#'   candidates a replacement may be drawn from.
#' @field universe Gene names of `pool`.
#' @field stratum Detection-rate stratum of every gene in the cache.
#' @field edges Quantile edges of the strata, taken over the pool so that each
#'   stratum holds a comparable number of genes.
#' @field n_bins Number of detection-rate strata.
#' @field n_strata Strata actually realised.  Not always `n_bins`: the stratum
#'   edges are quantiles of a heavily tied detection distribution, and repeated
#'   quantiles are dropped rather than kept as empty strata.
#' @field near_frac Fraction of a stratum kept as the sampling pool for each
#'   gene.
#' @field min_candidates Floor on that pool size, so genes in sparse strata
#'   still have room.
#' @field det_floor Detection floor applied to the background universe.
#'
#' @examples
#' X <- make_synthetic_matrix(n_cells = 300, n_genes = 900, seed = 11)
#' cache <- rank_cache(X$matrix, X$genes, ceiling = 60)
#' builder <- MatchedNullBuilder$new(cache, det_floor = 0.005, seed = 3)
#' str(builder$sample(cache$genes[1:8], n_sets = 3), max.level = 1)
#'
#' @export
MatchedNullBuilder <- R6::R6Class(
    "MatchedNullBuilder",
    public = list(
        cache = NULL,
        pool = NULL,
        universe = NULL,
        stratum = NULL,
        edges = NULL,
        n_bins = NULL,
        n_strata = NULL,
        near_frac = NULL,
        min_candidates = NULL,
        det_floor = NULL,

        #' @description Build a null builder over a cache.
        #' @param cache A `RankCache`.
        #' @param detection_matrix Optional cells x cache-genes binary matrix,
        #'   needed for co-detection matching.
        #' @param detection_matrix_genes Gene names for its columns.
        #' @param n_bins Number of detection-rate strata.
        #' @param near_frac Within a stratum, the fraction of closest candidates
        #'   kept as the sampling pool for each gene.
        #' @param min_candidates Floor on the pool size, so that genes in sparse
        #'   strata still have room.
        #' @param det_floor Detection floor for the background universe.
        #' @param seed Integer.
        initialize = function(cache, detection_matrix = NULL,
                              detection_matrix_genes = NULL, n_bins = 10,
                              near_frac = 0.10, min_candidates = 20,
                              det_floor = 0.02, seed = 2024) {
            self$cache <- cache
            self$n_bins <- as.integer(n_bins)
            self$near_frac <- as.numeric(near_frac)
            self$min_candidates <- as.integer(min_candidates)
            self$det_floor <- as.numeric(det_floor)
            private$seed <- as.integer(seed)

            det <- cache$detection
            self$pool <- which(det >= self$det_floor)
            if (length(self$pool) < 50) {
                stop(sprintf(
                    "only %d genes clear the detection floor %g; lower det_floor or supply a denser matrix",
                    length(self$pool), self$det_floor), call. = FALSE)
            }
            # Stratum edges are quantiles over the background, so every stratum
            # holds a comparable number of genes.  Detection rates are heavily
            # tied -- that is what a sparse matrix is -- so the quantiles can
            # repeat, and `cut()` refuses repeated breaks.  Dropping the repeats
            # leaves fewer than `n_bins` strata, which is recorded in
            # `n_strata` rather than passed over: a caller reading a null's
            # composition needs to know how finely it was matched.
            breaks <- unique(as.numeric(stats::quantile(
                det[self$pool],
                seq(0, 1, length.out = self$n_bins + 1)[-c(1, self$n_bins + 1)],
                names = FALSE)))
            self$edges <- breaks
            self$n_strata <- length(breaks) + 1L
            self$stratum <- as.integer(cut(det, c(-Inf, breaks, Inf),
                                           labels = FALSE))
            self$universe <- cache$genes[self$pool]

            private$B <- NULL
            if (!is.null(detection_matrix)) {
                if (is.null(detection_matrix_genes)) {
                    stop("detection_matrix_genes is required with detection_matrix",
                         call. = FALSE)
                }
                col <- match(self$universe, as.character(detection_matrix_genes))
                if (anyNA(col)) {
                    stop(sprintf(
                        "%d background genes are absent from the detection matrix",
                        sum(is.na(col))), call. = FALSE)
                }
                private$B <- as_binary(detection_matrix)[, col, drop = FALSE]
            }
            private$pool_cache <- new.env(parent = emptyenv())
        },

        #' @description Draw `n_sets` replacement sets of the same size as
        #'   `gene_set`.
        #' @param gene_set Character vector.
        #' @param n_sets Integer.
        #' @param kind One of the four families described above.
        #' @param seed Integer; overrides the builder's seed.
        #' @param n_draws Refinement rounds for the co-detection family.
        #' @return A list of character vectors, one per drawn set.
        sample = function(gene_set, n_sets = 1, kind = "expression", seed = NULL,
                          n_draws = 25) {
            s <- if (is.null(seed)) private$seed else as.integer(seed)
            out <- vector("list", as.integer(n_sets))
            for (i in seq_len(as.integer(n_sets))) {
                # Each set gets its own stream, derived from the seed and the
                # set's index, so that drawing 5 sets and drawing 3 give the
                # same first 3 -- a property that makes a result reproducible
                # without having to record how many sets were asked for.
                set.seed(s + i)
                out[[i]] <- switch(
                    kind,
                    random = self$random_set(length(gene_set), exclude = gene_set),
                    expression_bin = self$expression_bin_set(gene_set),
                    expression = self$expression_matched_set(gene_set),
                    codetection = self$codetection_matched_set(gene_set,
                                                               n_draws = n_draws),
                    stop(sprintf("unknown null kind '%s'", kind), call. = FALSE))
            }
            out
        },

        #' @description A uniform draw from the background universe.
        #' @param k Set size.
        #' @param exclude Genes that must not appear -- for a null, the members
        #'   of the set being replaced.  A uniform draw that can return them is
        #'   not a null for that set.
        #' @return A character vector.
        random_set = function(k, exclude = NULL) {
            pool <- self$pool
            if (!is.null(exclude)) {
                drop <- match(intersect(as.character(exclude),
                                        self$cache$genes), self$pool)
                drop <- drop[!is.na(drop)]
                if (length(drop) > 0) {
                    pool <- pool[-drop]
                }
            }
            if (length(pool) < k) {
                pool <- self$pool
            }
            self$cache$genes[sample(pool, k)]
        },

        #' @description Match the *mean* detection rate only, ignoring per-gene
        #'   structure.
        #' @param gene_set Character vector.
        #' @return A character vector.
        expression_bin_set = function(gene_set) {
            pos <- self$cache$positions(gene_set)
            if (length(pos) == 0) {
                stop("no gene of the set is in the background universe",
                     call. = FALSE)
            }
            target <- mean(self$cache$detection[pos])
            # The stratum whose detection range brackets the observed mean.
            stratum <- as.integer(cut(target, c(-Inf, self$edges, Inf),
                                      labels = FALSE))
            candidates <- self$pool[self$stratum[self$pool] == stratum]
            candidates <- setdiff(candidates, pos)
            if (length(candidates) < length(pos)) {
                candidates <- setdiff(self$pool, pos)
            }
            if (length(candidates) < length(pos)) {
                candidates <- self$pool
            }
            self$cache$genes[sample(candidates, length(pos))]
        },

        #' @description One expression-matched replacement set.
        #' @param gene_set Character vector.
        #' @return A character vector of the same length as `gene_set`.
        expression_matched_set = function(gene_set) {
            pos <- self$cache$positions(gene_set)
            if (length(pos) == 0) {
                stop("no gene of the set is in the background universe",
                     call. = FALSE)
            }
            excluded <- rep(FALSE, length(self$cache$genes))
            excluded[pos] <- TRUE
            used <- rep(FALSE, length(self$cache$genes))

            picks <- integer(0)
            n_from_set <- 0L
            n_duplicate <- 0L
            for (j in pos) {
                near <- self$expression_pool(j)
                allowed <- near[!excluded[near]]
                allowed <- allowed[!used[allowed]]
                if (length(allowed) == 0) {
                    # Nothing distinct and outside the set: give up distinctness
                    # first, since a repeated gene inflates the null's variance
                    # less than a gene of the set biases its mean.
                    allowed <- near[!excluded[near]]
                    if (length(allowed) > 0) {
                        n_duplicate <- n_duplicate + 1L
                    }
                }
                if (length(allowed) == 0) {
                    allowed <- near
                    n_from_set <- n_from_set + 1L
                    n_duplicate <- n_duplicate + 1L
                }
                pick <- if (length(allowed) == 1) {
                    allowed
                } else {
                    sample(allowed, 1)
                }
                picks <- c(picks, pick)
                used[pick] <- TRUE
            }
            private$last_stats <- list(n_from_query = n_from_set,
                                       n_duplicate = n_duplicate,
                                       set_size = length(pos))
            self$cache$genes[picks]
        },

        #' @description Nearest neighbours in detection and expression.
        #' @param gene_position Column position in the cache.
        #' @return Integer positions of the candidates that may replace it.
        expression_pool = function(gene_position) {
            key <- as.character(gene_position)
            if (!is.null(private$pool_cache[[key]])) {
                return(private$pool_cache[[key]])
            }
            det <- self$cache$detection
            mean_expr <- self$cache$mean_expression
            j <- gene_position
            candidates <- self$pool[self$stratum[self$pool] == self$stratum[j]]
            if (length(candidates) < self$min_candidates) {
                candidates <- self$pool
            }
            # Relative distance: a gene at 2% detection is matched on a 2x
            # scale, a gene at 60% on a 1.6x scale.  Absolute distance would let
            # the pool drift upward for sparse genes, which is exactly the
            # regime of interest.
            d <- abs(det[candidates] - det[j]) / max(det[j], 1e-6) +
                abs(mean_expr[candidates] - mean_expr[j]) / max(mean_expr[j], 1e-6)
            d[candidates == j] <- Inf
            take <- max(self$min_candidates,
                        as.integer(self$near_frac * length(candidates)))
            near <- candidates[order(d)][seq_len(take)]
            private$pool_cache[[key]] <- near
            near
        },

        #' @description Co-detection of a set: the mean Jaccard overlap of its
        #'   gene pairs across cells.  The Python implementation of the same
        #'   builder uses the mean pairwise cosine between the binary
        #'   detection columns instead -- a correlated but different
        #'   statistic -- so co-detection values are comparable within one
        #'   language, not across the two.
        #' @param gene_set Character vector.
        #' @return A single number, or `NA` for a set of fewer than two genes or
        #'   a builder built without a detection matrix.
        codetection = function(gene_set) {
            if (is.null(private$B)) {
                stop("co-detection matching needs detection_matrix at construction",
                     call. = FALSE)
            }
            col <- match(gene_set, self$universe)
            col <- col[!is.na(col)]
            if (length(col) < 2) {
                return(NA_real_)
            }
            B <- private$B[, col, drop = FALSE]
            n <- ncol(B)
            pairs <- utils::combn(n, 2)
            both <- as.matrix(B[, pairs[1, ], drop = FALSE] &
                              B[, pairs[2, ], drop = FALSE])
            either <- as.matrix(B[, pairs[1, ], drop = FALSE] |
                                B[, pairs[2, ], drop = FALSE])
            num <- Matrix::colSums(both)
            den <- Matrix::colSums(either)
            mean(ifelse(den > 0, num / den, 0))
        },

        #' @description A co-detection-matched replacement set: expression
        #'   matching first, then refinement against the observed co-detection.
        #' @param gene_set Character vector.
        #' @param n_draws Refinement rounds.
        #' @return A character vector.
        codetection_matched_set = function(gene_set, n_draws = 25) {
            target <- self$codetection(gene_set)
            if (!is.finite(target)) {
                return(self$expression_matched_set(gene_set))
            }
            best <- self$expression_matched_set(gene_set)
            # `codetection` returns NA for a replacement that lost all but one
            # of its genes; comparing NA would stop() mid-refinement, so a
            # non-finite gap simply never wins (as in the Python builder).
            best_gap <- self$codetection(best) - target
            if (!is.finite(best_gap)) best_gap <- Inf
            for (i in seq_len(as.integer(n_draws))) {
                candidate <- self$expression_matched_set(gene_set)
                gap <- self$codetection(candidate) - target
                if (is.finite(gap) && abs(gap) < abs(best_gap)) {
                    best <- candidate
                    best_gap <- gap
                }
            }
            best
        },

        #' @description How much the last draw had to relax.
        #' @return A list with `n_from_query`, `n_duplicate` and `set_size`.
        last_draw_stats = function() private$last_stats
    ),
    private = list(
        seed = 2024L,
        B = NULL,
        pool_cache = NULL,
        last_stats = list(n_from_query = 0L, n_duplicate = 0L, set_size = 0L)
    )
)

#' Draw replacement gene sets and report how well they matched
#'
#' The measurement is the point.  A null that fails to match is not a null, and
#' the failure is invisible in the P value it produces -- so the achieved
#' detection, expression and co-detection are reported alongside it, together
#' with how many genes the drawn sets share with the observation.
#'
#' @param cache A `RankCache`.
#' @param gene_set Character vector.
#' @param n_sets Number of replacement sets.  A P value needs a distribution, so
#'   one set is a demonstration and not a test.
#' @param kind One of `"random"`, `"expression_bin"`, `"expression"`,
#'   `"codetection"`.  The nesting is the point: each family controls something
#'   the previous one did not.
#' @param detection_matrix Optional cells x cache-genes binary matrix; required
#'   for the co-detection family.
#' @param detection_matrix_genes Gene names for its columns.
#' @param n_draws Refinement rounds for the co-detection family.
#' @param tolerance Accepted relative gap between the drawn sets' mean detection
#'   and the observed set's.  Reported and used for `status`; it does not
#'   constrain the draw, which is a nearest-neighbour match by construction.
#'   Defaults to `DEFAULT_CONFIG$null_matching_tolerance`.
#' @param seed Integer; defaults to `DEFAULT_CONFIG$seed`.
#' @param config Overrides for [DEFAULT_CONFIG()].  `background_loosen_factor`
#'   is the one this function reads, in the insufficient-background case.
#'
#' @return A list with `null_genes`, `target_detection`, `null_detection`, the
#'   ratios, `ks_statistic` and `ks_pvalue`, `max_overlap`, `n_from_query`,
#'   `draw_relaxations`, `duplicate_picks` and `status`.  `status` is `"PASS"`
#'   when the family's mean detection gap is within `tolerance` and no drawn set
#'   reuses a gene.  The gap is averaged over draws rather than maximised: a
#'   maximum can only grow with the number of draws.  `mean_detection_gap`,
#'   `worst_detection_gap` and `frac_within_tolerance` are all reported.
#'
#'   `pool_floor` is the detection floor the null was actually drawn at,
#'   `det_floor_loosened` whether it had to be moved to get there, and
#'   `n_available_background` how many genes could stand in for the set -- the
#'   pool minus the set itself, which is the number that decides whether a
#'   matched null is possible at all.
#'
#' @section Insufficient background genes:
#' When fewer than `MIN_BACKGROUND_GENES` genes can replace the set, the
#' detection floor is lowered by `background_loosen_factor` (10% by default) and
#' the widened draw is announced as a `background_insufficient` record rather
#' than performed quietly.  If the loosened floor still leaves too few
#' candidates the call stops with `"Cannot construct matched null. Consider
#' different gene set."` and the counts behind it -- the alternative is a null
#' drawn from a handful of genes, which is not a null and would be read as a
#' test.
#'
#' @export
construct_expression_matched_null <- function(cache, gene_set, n_sets = 1,
                                              kind = "expression",
                                              detection_matrix = NULL,
                                              detection_matrix_genes = NULL,
                                              n_draws = 25, tolerance = NULL,
                                              seed = NULL, config = NULL) {
    cfg <- merge_config(config)
    tolerance <- if (is.null(tolerance)) cfg$null_matching_tolerance else
        as.numeric(tolerance)
    seed <- if (is.null(seed)) cfg$seed else as.integer(seed)
    pos <- cache$positions(gene_set)
    if (length(pos) == 0) {
        stop("none of the gene set is in the cache", call. = FALSE)
    }
    floor <- adaptive_detection_floor(cache, gene_set)

    build <- function(fl) {
        MatchedNullBuilder$new(
            cache, detection_matrix = detection_matrix,
            detection_matrix_genes = detection_matrix_genes,
            det_floor = fl, seed = seed)
    }
    # Background genes left once the set's own genes are taken out.  The query
    # set is excluded from every draw by construction, so the candidates that
    # can actually stand in for it are the pool minus the set -- and it is that
    # number, not the pool size, that decides whether a matched null is
    # possible.
    available <- function(b) length(setdiff(b$pool, pos))

    builder <- tryCatch(build(floor), error = function(e) {
        # `MatchedNullBuilder` refuses a pool below 50 genes outright.  That
        # refusal is the same failure this block handles, one step earlier, so
        # it is reported as the brief's edge case rather than as the constructor
        # error it is raised as.
        stop(sprintf(
            paste0("Cannot construct matched null. Consider different gene ",
                   "set. At a detection floor of %g the background has too ",
                   "few genes to draw a replacement from (%s)."),
            floor, conditionMessage(e)), call. = FALSE)
    })

    n_available <- available(builder)
    floor_loosened <- FALSE
    if (n_available < MIN_BACKGROUND_GENES) {
        # Edge case one of the brief, in this framework's terms.  The brief
        # states it as a matching tolerance and asks for it to be loosened by
        # 10%; here the tolerance that decides pool membership is the detection
        # floor, so that is the knob that is turned.  The loosening is announced
        # rather than performed quietly, because a reader of the calibration
        # object has to be able to see that the null they are reading was drawn
        # from a widened pool.
        floor_loosened <- TRUE
        loosened <- floor * cfg$background_loosen_factor
        event("warning", "background_insufficient", n_available = n_available,
              required = MIN_BACKGROUND_GENES, det_floor = floor,
              loosened_to = loosened,
              action = sprintf("loosening detection floor by %.0f%%",
                               100 * (1 - cfg$background_loosen_factor)))
        builder <- tryCatch(build(loosened), error = function(e) {
            stop(sprintf(
                paste0("Cannot construct matched null. Consider different ",
                       "gene set. Loosening the detection floor from %g to %g ",
                       "left too few background genes to draw from (%s)."),
                floor, loosened, conditionMessage(e)), call. = FALSE)
        })
        n_available <- available(builder)
        if (n_available < MIN_BACKGROUND_GENES) {
            stop(sprintf(
                paste0("Cannot construct matched null. Consider different ",
                       "gene set. Only %d background genes can replace this ",
                       "%d-gene set, and loosening the detection floor from ",
                       "%g to %g did not reach %d."),
                n_available, length(pos), floor, loosened,
                MIN_BACKGROUND_GENES), call. = FALSE)
        }
        floor <- loosened
    }

    sets <- builder$sample(gene_set, n_sets, kind = kind, n_draws = n_draws)

    target_det <- mean(cache$detection[pos])
    drawn_det <- vapply(sets, function(s) {
        mean(cache$detection[match(s, cache$genes)])
    }, numeric(1))
    drawn_expr <- vapply(sets, function(s) {
        mean(cache$mean_expression[match(s, cache$genes)])
    }, numeric(1))
    target_expr <- mean(cache$mean_expression[pos])

    out <- list(
        null_genes = if (n_sets > 1) sets else sets[[1]],
        target_sparsity = 1 - target_det,
        null_sparsity = 1 - mean(drawn_det),
        target_detection = target_det,
        null_detection = mean(drawn_det),
        detection_ratio = if (target_det > 0) mean(drawn_det) / target_det else NA_real_,
        detection_ratio_sd = stats::sd(drawn_det / target_det),
        expression_ratio = if (target_expr > 0) mean(drawn_expr) / target_expr else NA_real_,
        det_floor = floor, n_sets = as.integer(n_sets), kind = kind)

    obs_det <- cache$detection[pos]
    null_det <- unlist(lapply(sets, function(s) cache$detection[match(s, cache$genes)]))
    ks <- suppressWarnings(stats::ks.test(obs_det, null_det))
    out$ks_statistic <- unname(ks$statistic)
    out$ks_pvalue <- ks$p.value

    query <- as.character(gene_set)
    overlaps <- vapply(sets, function(s) length(intersect(query, s)), integer(1))
    stats_last <- builder$last_draw_stats()
    out$n_from_query <- sum(overlaps)
    out$max_overlap <- if (length(overlaps)) max(overlaps) else 0L
    out$draw_relaxations <- stats_last$n_from_query
    out$duplicate_picks <- stats_last$n_duplicate
    out$pool_size <- length(builder$pool)
    # The floor the null was actually drawn at, and whether it had to be moved
    # to get there.  A reader of the calibration object has to be able to see
    # that the null in front of them came from a widened pool -- otherwise the
    # loosening is invisible in exactly the record that exists to expose it.
    out$pool_floor <- floor
    out$det_floor_loosened <- floor_loosened
    out$n_available_background <- n_available
    # The status is decided by the family's *typical* gap rather than by its
    # worst draw, for the same reason on this side as in the Python
    # implementation: a maximum over draws can only grow as more are asked for,
    # so deciding on it made one pool pass at 25 draws and fail at 100.  The
    # worst gap is still reported, next to the fraction within tolerance.
    gaps <- if (target_det > 0) abs(drawn_det / target_det - 1) else numeric(0)
    out$mean_detection_gap <- if (length(gaps)) mean(gaps) else NA_real_
    out$worst_detection_gap <- if (length(gaps)) max(gaps) else NA_real_
    out$frac_within_tolerance <- if (length(gaps)) mean(gaps <= tolerance) else NA_real_
    out$status <- if (!is.finite(out$mean_detection_gap)) {
        "UNDEFINED"
    } else if (out$mean_detection_gap <= tolerance && out$max_overlap == 0) {
        "PASS"
    } else {
        "FAIL"
    }
    out
}

#' Score a set and its matched null on the same cells
#'
#' @param cache A `RankCache`.
#' @param gene_set Character vector.
#' @param null_genes A drawn replacement set, or a list of them.
#' @param method `"AUCell"` or `"UCell"`.
#' @param axis Optional vector to correlate both scores against.
#' @param max_rank Rank ceiling.
#'
#' @return A list with `observed`, `null` (a matrix of per-cell scores, one
#'   column per drawn set), and, when `axis` is supplied, the correlations and
#'   the Monte-Carlo P value.
#'
#' @export
compute_calibrated_score <- function(cache, gene_set, null_genes = NULL,
                                     method = "AUCell", axis = NULL,
                                     max_rank = NULL) {
    score <- function(g) {
        if (identical(method, "UCell")) ucell(cache, g) else aucell(cache, g, max_rank)
    }
    observed <- score(gene_set)
    out <- list(observed = observed, method = method,
                gene_set = as.character(gene_set))
    if (!is.null(null_genes)) {
        if (!is.list(null_genes)) {
            null_genes <- list(null_genes)
        }
        null_scores <- vapply(null_genes, function(s) score(s), numeric(length(observed)))
        if (is.null(dim(null_scores))) {
            null_scores <- matrix(null_scores, ncol = 1)
        }
        out$null <- null_scores
        out$null_n <- ncol(null_scores)
    }
    if (!is.null(axis)) {
        out$observed_rho <- spearman(observed, axis)$rho
        if (!is.null(out$null)) {
            null_rho <- apply(out$null, 2, function(s) spearman(s, axis)$rho)
            out$null_rho <- null_rho
            # Two-sided, for the reason given in [empirical_p()]: a caller of
            # compute_calibrated_score asks whether the set is associated with
            # the axis, not whether it is depleted by it.  The one-sided values
            # are reported beside it so a directional reading needs no
            # recomputation, and so that nothing downstream has to guess which
            # question the default answered.
            out$summary <- null_summary(null_rho, out$observed_rho,
                                        tail = "two-sided")
            out$summary$p_lower <- empirical_p(null_rho, out$observed_rho,
                                               tail = "lower")
            out$summary$p_upper <- empirical_p(null_rho, out$observed_rho,
                                               tail = "upper")
        }
    }
    out
}
