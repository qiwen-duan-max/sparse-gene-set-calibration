"""Null gene sets for sparse single-cell data.

A gene-set score means nothing on its own.  What it means depends entirely on
the null it is compared against, and in sparse data the obvious nulls are the
wrong ones:

``random``
    Draw genes uniformly from the expressed background.  Because expression and
    detection are heavy-tailed, a random set is usually *less* detectable than
    the set under study, so the observed score looks high for reasons that have
    nothing to do with biology.  This is the null most published analyses use by
    default, explicitly or by omission.
``expression``
    Match each gene of the set to a replacement with a similar detection rate
    and mean expression.  Removes the gross composition bias of the random null.
``codetection``
    Expression matching controls the *marginals* but not the *joint*
    distribution: a co-regulated set whose genes are detected in the same cells
    does not behave like a set of independently drawn genes with the same
    marginals.  This null adds a match on the set's internal co-detection
    structure.

The three are nested, and :func:`compare_nulls` reports all of them at once so
the cost of each assumption is visible.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["MatchedNullBuilder", "compare_nulls", "adaptive_detection_floor"]


def adaptive_detection_floor(cache, gene_set, cap=0.01, frac=0.25, floor=1e-4):
    """A detection floor low enough that it does not bind for this gene set.

    ``MatchedNullBuilder`` excludes genes detected in fewer than ``det_floor`` of
    cells, so that the pool is not filled with genes that are almost never
    observed and matching degenerates.  A *fixed* floor does something else as
    well, and the something else is a bias: if the gene set's own genes are
    sparser than the floor, they are excluded from the pool of candidates that
    could replace them, and every replacement is denser than the gene it stands
    in for.  The null is then systematically richer than the observed set, and
    the test is conservative in exactly the regime the calibration is meant to
    address.

    In a simulated panel at a target detection of 0.01, a floor of 0.01 leaves
    22.6% of the background ineligible and makes the drawn null sets 19% denser
    than the set they replace.  This rule brings that to 1.6%.

    Parameters
    ----------
    cache : RankCache
    gene_set : sequence of str
    cap : float
        The largest floor this will return; the default 0.01 is the level that
        is appropriate for a well-detected set.
    frac : float
        The floor is this fraction of the set's 10th-percentile detection.
    floor : float
        Absolute lower bound.  A gene set containing genes that are never
        detected cannot be matched at all -- the pool has no member with zero
        detection to stand in for them -- and this is the point at which the
        function stops trying.

    Returns
    -------
    float
    """
    det = cache.detection
    pos = [cache._index[g] for g in gene_set if g in cache._index]
    if not pos:
        return floor
    q10 = float(np.quantile(det[pos], 0.10))
    return float(max(floor, min(cap, frac * q10)))


class MatchedNullBuilder:
    """Build expression- and co-detection-matched null gene sets.

    Parameters
    ----------
    cache : RankCache
        Supplies gene names, detection rates and mean expression.  Its gene list
        defines the background universe.
    detection_matrix : scipy.sparse matrix, optional
        Cells x genes binary matrix over the *same* background universe, used
        for co-detection matching.  Without it only expression matching is
        available.
    detection_matrix_genes : sequence of str, optional
        Gene names for the columns of ``detection_matrix``.
    n_bins : int
        Number of detection-rate strata.  Ten (deciles) is the default; the
        matching is done within a stratum and then refined by distance.
    near_frac : float
        Within a stratum, the fraction of closest candidates kept as the
        sampling pool for each gene.
    min_candidates : int
        Floor on the pool size, so that genes in sparse strata still have room.
    det_floor : float
        Genes detected in fewer than this fraction of cells are excluded from
        the background universe.  Without a floor the pool fills with genes that
        are almost never observed, and matching degenerates.
    """

    def __init__(self, cache, detection_matrix=None, detection_matrix_genes=None,
                 n_bins=10, near_frac=0.10, min_candidates=20, det_floor=0.02,
                 seed=2024):
        self.cache = cache
        self.n_bins = int(n_bins)
        self.near_frac = float(near_frac)
        self.min_candidates = int(min_candidates)
        self.det_floor = float(det_floor)
        self.rng = np.random.default_rng(seed)

        det = cache.detection
        self.pool = np.flatnonzero(det >= self.det_floor)
        if self.pool.size < 50:
            raise ValueError(
                f"only {self.pool.size} genes clear the detection floor "
                f"{det_floor}; lower det_floor or supply a denser matrix")

        # Stratum edges are quantiles of detection rate over the background, so
        # every stratum holds a comparable number of genes.
        self.edges = np.quantile(det[self.pool],
                                 np.linspace(0, 1, self.n_bins + 1)[1:-1])
        self.stratum = np.digitize(det, self.edges)
        self.universe = cache.genes[self.pool]

        self._B = None
        self._pool_pos = {int(g): i for i, g in enumerate(self.pool)}
        if detection_matrix is not None:
            if detection_matrix_genes is None:
                raise ValueError("detection_matrix_genes is required with "
                                 "detection_matrix")
            col = {g: i for i, g in enumerate(np.asarray(detection_matrix_genes))}
            missing = [g for g in self.universe if g not in col]
            if missing:
                raise ValueError(
                    f"{len(missing)} background genes are absent from the "
                    f"detection matrix, e.g. {missing[:5]}")
            cols = np.array([col[g] for g in self.universe], dtype=np.int64)
            self._B = sp.csr_matrix(detection_matrix)[:, cols].astype(np.float32)

        self._pool_cache = {}

    # ------------------------------------------------------------------
    def _expression_pool(self, gene_position):
        """Positions of the background genes that may replace one gene."""
        if gene_position in self._pool_cache:
            return self._pool_cache[gene_position]

        det = self.cache.detection
        mean = self.cache.mean_expression
        j = gene_position
        stratum = self.stratum[j]
        candidates = self.pool[self.stratum[self.pool] == stratum]
        if candidates.size < self.min_candidates:
            candidates = self.pool

        # Relative distance: a gene at 2% detection is matched on a 2x scale, a
        # gene at 60% on a 1.6x scale.  Absolute distance would let the pool
        # drift upward for sparse genes, which is exactly the regime of
        # interest.
        d = (np.abs(det[candidates] - det[j]) / max(det[j], 1e-6)
             + np.abs(mean[candidates] - mean[j]) / max(mean[j], 1e-6))
        d = np.where(candidates == j, np.inf, d)
        near = candidates[np.argsort(d)[:max(self.min_candidates,
                                             int(self.near_frac * candidates.size))]]
        self._pool_cache[gene_position] = near
        return near

    def expression_matched_set(self, gene_set, rng=None):
        """One expression-matched replacement set, same size as ``gene_set``.

        No replacement is drawn from the set being replaced.  Excluding only the
        gene it stands in for is not enough: the other members of the set are
        still eligible, and a null that shares genes with the observation is not
        a null -- its score is correlated with the observed score by
        construction, which makes every P value drawn from it conservative.

        Where the pool cannot supply distinct, non-member genes -- a small
        background, or a set drawn from the densest tail -- the draw relaxes in
        that order rather than failing, and reports how often it had to.
        """
        rng = rng or self.rng
        positions = [self.cache._index[g] for g in gene_set
                     if g in self.cache._index]
        if not positions:
            raise ValueError("no gene of the set is in the background universe")

        excl = np.zeros(self.cache.genes.size, dtype=bool)
        excl[positions] = True
        used = np.zeros(self.cache.genes.size, dtype=bool)

        picks, n_from_set, n_duplicate = [], 0, 0
        for j in positions:
            near = self._expression_pool(j)
            allowed = self._drawable(near, excl, used)
            if allowed.size == 0:
                # Nothing distinct and outside the set: give up the distinctness
                # first, since a repeated gene inflates the null's variance less
                # than a gene of the set biases its mean.
                allowed = self._drawable(near, excl, None)
                n_duplicate += 1 if allowed.size else 0
            if allowed.size == 0:
                allowed = near
                n_from_set += 1
                n_duplicate += 1
            pick = int(rng.choice(allowed))
            picks.append(pick)
            used[pick] = True
        self.last_draw_stats = dict(n_from_query=int(n_from_set),
                                    n_duplicate=int(n_duplicate),
                                    set_size=len(positions))
        return list(self.cache.genes[picks])

    @staticmethod
    def _drawable(near, excl, used):
        """Members of ``near`` that are neither in the set nor already used."""
        keep = ~excl[near]
        if used is not None:
            keep &= ~used[near]
        return near[keep]

    def random_set(self, k, rng=None, exclude=None):
        """Uniform draw from the background universe.

        ``exclude`` names genes that must not appear, which for a null means the
        members of the set being replaced: a uniform draw that can return them is
        not a null for that set.
        """
        rng = rng or self.rng
        pool = self.pool
        if exclude:
            drop = {self.cache._index[g] for g in exclude
                    if g in self.cache._index}
            if drop:
                pool = pool[~np.isin(pool, list(drop))]
        if pool.size < k:
            pool = self.pool
        picks = rng.choice(pool, size=k, replace=False)
        return list(self.cache.genes[picks])

    def expression_bin_set(self, gene_set, rng=None):
        """Match the *mean* detection rate only, ignoring per-gene structure.

        A deliberately coarser null, included because it is the natural
        compromise for small gene sets and because the difference between it and
        :meth:`expression_matched_set` bounds how much per-gene matching buys.
        It excludes the set's own genes, as the finer null does, so that the two
        differ in the matching they achieve and in nothing else.
        """
        rng = rng or self.rng
        positions = [self.cache._index[g] for g in gene_set
                     if g in self.cache._index]
        if not positions:
            raise ValueError("no gene of the set is in the background universe")
        target = float(np.mean(self.cache.detection[positions]))
        d = np.abs(self.cache.detection[self.pool] - target)
        near = self.pool[np.argsort(d)[:max(self.min_candidates,
                                            int(0.10 * self.pool.size))]]
        excl = np.zeros(self.cache.genes.size, dtype=bool)
        excl[positions] = True
        allowed = near[~excl[near]]
        if allowed.size < len(positions):
            allowed = near
        picks = rng.choice(allowed, size=min(len(positions), allowed.size),
                           replace=False)
        return list(self.cache.genes[picks])

    # ------------------------------------------------------------------
    # co-detection
    # ------------------------------------------------------------------
    def codetection(self, gene_set):
        """Mean pairwise co-detection similarity within a gene set.

        Values near zero mean the genes are detected independently; positive
        values mean they tend to be detected in the same cells.  This is the
        structure an expression-matched null does not preserve.

        The similarity block is computed from the detection matrix each call,
        restricted to the set's own genes.  The alternative -- one
        ``n_universe x n_universe`` gram matrix held for the lifetime of the
        builder -- is quadratic in the background, which on a real matrix is
        several hundred megabytes, and it is paid again per cell type.  The
        block is at most a few hundred columns wide, so computing it on demand
        is both smaller and, for the numbers a matched null actually draws,
        faster.
        """
        if self._B is None:
            raise RuntimeError(
                "co-detection matching needs detection_matrix at construction")
        cols = sorted({self._pool_pos[self.cache._index[g]]
                       for g in gene_set
                       if g in self.cache._index
                       and self.cache._index[g] in self._pool_pos})
        if len(cols) < 2:
            return np.nan
        sub = self._B[:, cols].astype(np.float32)
        gram = np.asarray((sub.T @ sub).todense())
        norm = np.sqrt(np.diag(gram))
        norm[norm == 0] = 1.0
        gram = gram / (norm[:, None] * norm[None, :])
        np.fill_diagonal(gram, np.nan)
        return float(np.nanmean(gram))

    def codetection_matched_set(self, gene_set, n_draws=200, rng=None):
        """Expression-matched, and additionally co-detection-matched.

        Draws ``n_draws`` expression-matched sets and returns the one whose mean
        pairwise co-detection is closest to that of ``gene_set``.  Matching on a
        single scalar summary of the joint distribution is a compromise -- it
        does not reproduce the full correlation matrix -- but it removes the
        first-order bias and is cheap enough to run inside a bootstrap.
        """
        rng = rng or self.rng
        target = self.codetection(gene_set)
        if not np.isfinite(target):
            return self.expression_matched_set(gene_set, rng)
        best, best_d = None, np.inf
        for _ in range(int(n_draws)):
            candidate = self.expression_matched_set(gene_set, rng)
            value = self.codetection(candidate)
            if not np.isfinite(value):
                continue
            d = abs(value - target)
            if d < best_d:
                best, best_d = candidate, d
        return best if best is not None else self.expression_matched_set(gene_set, rng)

    def sample(self, gene_set, n, kind="expression", rng=None, seed=None,
               n_draws=200):
        """Draw ``n`` null sets of the requested ``kind``.

        Parameters
        ----------
        kind : {'random', 'expression_bin', 'expression', 'codetection'}
        n_draws : int
            Candidate draws per returned set, used only by ``'codetection'``.
            Matching on a single scalar summary costs ``n_draws`` extra sets per
            set returned, so this is the knob to turn when the co-detection null
            dominates the runtime.
        """
        rng = rng or (np.random.default_rng(seed) if seed is not None else self.rng)
        out = []
        for _ in range(int(n)):
            if kind == "random":
                out.append(self.random_set(len(gene_set), rng,
                                           exclude=gene_set))
            elif kind == "expression_bin":
                out.append(self.expression_bin_set(gene_set, rng))
            elif kind == "expression":
                out.append(self.expression_matched_set(gene_set, rng))
            elif kind == "codetection":
                out.append(self.codetection_matched_set(gene_set, rng=rng,
                                                        n_draws=n_draws))
            else:
                raise ValueError(f"unknown null kind {kind!r}")
        return out


def compare_nulls(builder, gene_set, n=200, kinds=("random", "expression_bin",
                                                   "expression", "codetection"),
                  seed=0):
    """Report the composition of every null family for one gene set.

    Returns a :class:`pandas.DataFrame` with one row per null family holding the
    mean detection rate, mean expression and mean co-detection of the drawn
    sets, alongside the observed values.  A family whose columns differ from the
    observed row is not a valid null for this set, whatever its scores look
    like.
    """
    import pandas as pd

    cache = builder.cache
    pos = [cache._index[g] for g in gene_set if g in cache._index]
    rows = [dict(
        kind="observed", n_genes=len(pos),
        detection=float(np.mean(cache.detection[pos])),
        mean_expression=float(np.mean(cache.mean_expression[pos])),
        codetection=builder.codetection(gene_set))]

    rng = np.random.default_rng(seed)
    for kind in kinds:
        sets = builder.sample(gene_set, n, kind=kind, rng=rng)
        det, expr, codet = [], [], []
        for s in sets:
            p = [cache._index[g] for g in s if g in cache._index]
            if not p:
                continue
            det.append(np.mean(cache.detection[p]))
            expr.append(np.mean(cache.mean_expression[p]))
            codet.append(builder.codetection(s))
        rows.append(dict(kind=kind, n_genes=len(pos),
                         detection=float(np.nanmean(det)),
                         mean_expression=float(np.nanmean(expr)),
                         codetection=float(np.nanmean(codet))))
    return pd.DataFrame(rows)
