"""Gene-set scoring methods, implemented against the published definitions.

Two of the four methods are rank-based and can be evaluated from a
:class:`~sparsegs.rankcache.RankCache`; the other two are expression-based and
need the matrix.  All four are here so that a calibration result can be reported
as a property of *the analysis*, not of one particular scoring function.

Methods
-------
aucell
    Aibar et al., *Nat. Methods* 2017.  Area under the recovery curve over the
    top ``max_rank`` ranks, normalised by ``n_genes_in_set * max_rank``.
ucell
    Andreatta & Carmona, *Comput. Struct. Biotechnol. J.* 2021.  Mann-Whitney U
    of the set's capped ranks, normalised to ``[0, 1]``.
score_genes
    The scanpy default (Satija lab).  Mean expression of the set minus the mean
    of expression-bin-matched control genes.  Produces no null distribution.
module_score
    Seurat ``AddModuleScore``.  As above, with controls averaged per expression
    bin rather than pooled.
ssgsea
    Barbie et al., *Nature* 2009.  Weighted ECDF difference over the full
    ranking.  Included for completeness; needs the matrix, not the cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["aucell", "ucell", "score_genes", "module_score", "ssgsea",
           "score_all"]


# ----------------------------------------------------------------------
# rank-based methods
# ----------------------------------------------------------------------
def aucell(cache, genes, max_rank=None):
    """AUCell scores from a rank cache.

    Parameters
    ----------
    cache : RankCache
    genes : sequence of str
        Gene set.  Genes absent from the cache are dropped.
    max_rank : int, optional
        Rank ceiling.  Defaults to ``ceil(0.05 * cache.n_genes_total)``, the
        AUCell default ranking fraction.

    Returns
    -------
    numpy.ndarray
        One score per cell, in ``[0, 1]``.
    """
    present = cache.present(genes)
    if len(present) < 1:
        raise ValueError("none of the requested genes are in the cache")
    if max_rank is None:
        max_rank = int(np.ceil(0.05 * cache.n_genes_total))
    max_rank = int(max_rank)

    r = cache.ranks(present)
    contrib = np.where((r > 0) & (r <= max_rank), max_rank - r + 1, 0)
    return contrib.sum(axis=1) / (len(present) * max_rank)


def ucell(cache, genes, r_max=1500, normalisation="v2"):
    """UCell scores from a rank cache.

    Parameters
    ----------
    cache : RankCache
        The cache ceiling must be at least ``r_max``.
    genes : sequence of str
    r_max : int
        Rank cap.  The UCell documentation recommends setting this to the median
        number of detected genes per cell; the package default is 1500, which is
        appropriate for 10x data with typical depth but not for shallow data.
    normalisation : {'v2', 'v1'}
        ``'v2'`` uses ``U_max = k * r_max - k (k + 1) / 2``, the corrected
        maximum introduced in UCell v2.  ``'v1'`` uses the original paper's
        ``U_max = k * r_max``.

    Returns
    -------
    numpy.ndarray
        One score per cell, in ``[0, 1]`` up to a small negative floor.

    Notes
    -----
    The published cap is ``r' = r_max + 1`` for ranks beyond ``r_max``, so the
    largest value the U statistic can take is ``k (r_max + 1) - k (k + 1) / 2``,
    while the v2 normaliser is ``k * r_max - k (k + 1) / 2``.  The two differ by
    ``k``, so the score has a floor at ``-1 / (r_max - (k + 1) / 2)`` rather than
    exactly zero.  At the package default (``r_max = 1500``, ``k = 8``) the floor
    is ``-7e-4`` and invisible; it grows as ``r_max`` is lowered towards the
    median detected-gene count, and reaches ``-0.018`` at ``r_max = 60`` with
    ``k = 10``.  Scores are returned unclipped so that they reproduce the
    reference implementation exactly.
    """
    present = cache.present(genes)
    if len(present) < 1:
        raise ValueError("none of the requested genes are in the cache")
    r_max = int(r_max)
    if r_max > cache.ceiling:
        raise ValueError(
            f"cache ceiling {cache.ceiling} is below r_max {r_max}; rebuild the "
            f"cache with a ceiling of at least {r_max}")

    k = len(present)
    r = cache.ranks(present)
    # Ranks above the cap -- including the BEYOND sentinel -- become r_max + 1,
    # which is exactly the published cap.
    capped = np.where((r > 0) & (r <= r_max), r, r_max + 1)
    u_stat = capped.sum(axis=1) - k * (k + 1) / 2.0
    u_max = k * r_max if normalisation == "v1" else k * r_max - k * (k + 1) / 2.0
    return 1.0 - u_stat / u_max


# ----------------------------------------------------------------------
# expression-based methods
# ----------------------------------------------------------------------
def _expression_bins(mean_expression, n_bins):
    """Assign genes to equal-count bins of mean expression.

    Returns an integer bin label per gene, plus one boolean mask per bin.
    """
    ranks = pd.Series(mean_expression).rank(method="first")
    labels = pd.qcut(ranks, n_bins, labels=False).to_numpy()
    labels = np.where(np.isfinite(labels), labels, 0).astype(int)
    n_used = labels.max() + 1
    masks = [labels == b for b in range(n_used)]
    return labels, masks


def _mean_expression(X, positions):
    """Column means of a sparse matrix restricted to ``positions``."""
    sub = X[:, positions]
    return np.asarray(sub.mean(axis=1)).ravel()


def score_genes(X, genes, gene_set, ctrl_size=50, n_bins=25, seed=0):
    """scanpy's ``score_genes``: set mean minus expression-matched control mean.

    Parameters
    ----------
    X : scipy.sparse matrix
        Cells x genes expression matrix.
    genes : sequence of str
    gene_set : sequence of str
    ctrl_size : int
        Number of control genes drawn per gene of the set.
    n_bins : int
        Number of expression bins.
    seed : int

    Returns
    -------
    numpy.ndarray
        One score per cell.  Not bounded; higher means the set is more expressed
        than its expression-matched controls.
    """
    X = sp.csr_matrix(X)
    genes = np.asarray(genes)
    pos = {g: i for i, g in enumerate(genes)}
    target = np.array([pos[g] for g in gene_set if g in pos], dtype=np.int64)
    if target.size == 0:
        raise ValueError("none of the genes in gene_set are present")

    mean_expression = np.asarray(X.mean(axis=0)).ravel()
    labels, masks = _expression_bins(mean_expression, n_bins)
    in_set = np.zeros(len(genes), dtype=bool)
    in_set[target] = True

    rng = np.random.default_rng(seed)
    controls = []
    for t in target:
        candidates = np.flatnonzero(masks[labels[t]] & ~in_set)
        if candidates.size == 0:
            candidates = np.flatnonzero(~in_set)
        take = min(ctrl_size, candidates.size)
        controls.append(rng.choice(candidates, size=take, replace=False))
    control = np.concatenate(controls)

    return _mean_expression(X, target) - _mean_expression(X, control)


def module_score(X, genes, gene_set, ctrl_size=100, n_bins=24, seed=0):
    """Seurat's ``AddModuleScore``: per-bin control means, subtracted.

    The difference from :func:`score_genes` is that controls are averaged within
    each expression bin and each target gene has its own bin's control mean
    subtracted, rather than pooling every control gene into one average.
    """
    X = sp.csr_matrix(X)
    genes = np.asarray(genes)
    pos = {g: i for i, g in enumerate(genes)}
    target = np.array([pos[g] for g in gene_set if g in pos], dtype=np.int64)
    if target.size == 0:
        raise ValueError("none of the genes in gene_set are present")

    mean_expression = np.asarray(X.mean(axis=0)).ravel()
    labels, masks = _expression_bins(mean_expression, n_bins)
    in_set = np.zeros(len(genes), dtype=bool)
    in_set[target] = True

    rng = np.random.default_rng(seed)
    bin_control_mean = {}
    for b, mask in enumerate(masks):
        if not mask.any():
            continue
        candidates = np.flatnonzero(mask & ~in_set)
        if candidates.size == 0:
            candidates = np.flatnonzero(~in_set)
        take = min(ctrl_size, candidates.size)
        chosen = rng.choice(candidates, size=take, replace=False)
        bin_control_mean[b] = _mean_expression(X, chosen)

    score = np.zeros(X.shape[0], dtype=float)
    for t in target:
        score += (_mean_expression(X, np.array([t]))
                  - bin_control_mean.get(int(labels[t]), 0.0))
    return score / target.size


def ssgsea(X, genes, gene_set):
    """Weighted ECDF difference (Barbie et al. 2009) for one gene set.

    Implemented for completeness and used only where matrices are small enough
    that a full ranking pass is affordable.
    """
    X = sp.csr_matrix(X)
    genes = np.asarray(genes)
    pos = {g: i for i, g in enumerate(genes)}
    target = np.array([pos[g] for g in gene_set if g in pos], dtype=np.int64)
    if target.size == 0:
        raise ValueError("none of the genes in gene_set are present")

    dense = np.asarray(X.todense(), dtype=np.float32)
    n_cells, n_genes = dense.shape
    order = np.argsort(-dense, axis=1, kind="stable")
    rank_of = np.empty((n_cells, n_genes), dtype=np.int32)
    np.put_along_axis(rank_of, order, np.arange(1, n_genes + 1, dtype=np.int32)[None, :],
                      axis=1)

    in_set = np.zeros(n_genes, dtype=bool)
    in_set[target] = True
    out = np.empty(n_cells, dtype=float)
    for i in range(n_cells):
        ranks = np.sort(rank_of[i, target])
        values = np.abs(dense[i, target])
        weight = values / max(values.sum(), 1e-12)
        hit = np.cumsum(weight)
        miss_total = n_genes - target.size
        miss = np.zeros(target.size, dtype=float)
        # Number of non-set genes ranked above each set member, normalised.
        for j, r in enumerate(ranks):
            miss[j] = (r - 1 - j) / max(miss_total, 1)
        out[i] = hit.sum() / target.size - miss.sum() / target.size
    return out


# ----------------------------------------------------------------------
# convenience
# ----------------------------------------------------------------------
def score_all(cache, X, genes, gene_set, ucell_r_max=1500, seed=0):
    """Evaluate every method on one gene set, returning a tidy frame.

    Parameters
    ----------
    cache : RankCache
    X : scipy.sparse matrix
        Needed for the expression-based methods; pass ``None`` to skip them.
    genes : sequence of str
    gene_set : sequence of str
    ucell_r_max : int
    seed : int

    Returns
    -------
    pandas.DataFrame
        Columns ``method``, ``cell``, ``score``.
    """
    frames = []
    for name, values in (
            ("AUCell", aucell(cache, gene_set)),
            ("UCell", ucell(cache, gene_set, r_max=ucell_r_max))):
        frames.append(pd.DataFrame({"method": name,
                                    "cell": np.arange(len(values)),
                                    "score": values}))
    if X is not None:
        for name, fn in (("score_genes", score_genes),
                         ("module_score", module_score)):
            values = fn(X, genes, gene_set, seed=seed)
            frames.append(pd.DataFrame({"method": name,
                                        "cell": np.arange(len(values)),
                                        "score": values}))
    return pd.concat(frames, ignore_index=True)
