"""Head-to-head against the methods a reader will already be using.

A calibration claim is only interesting if the thing being calibrated is what
people actually run.  This module wraps the two families that matter:

**The established multivariate methods.**  ``decoupler`` (Badia-i-Mompel et al.,
*Bioinformatics Advances* 2022) is the current standard for "score a gene set
against a covariate" in single-cell data.  Its univariate and multivariate linear
models regress each gene of the set on the covariate and aggregate the
t-statistics, and they return a P value from a normal approximation that assumes
the genes' statistics are independent.  That assumption is what the sparse
regime violates, and the comparison is whether its P values are calibrated where
ours are.  The wrapper calls the real package; it is not a reimplementation.

**The established null strategies.**  ``size_matched`` and ``expression_bin``
draw replacement sets by set size and by mean expression, which is what a
careful analyst does without a framework.  They are here so that the step from
"matched at all" to "matched on the right quantity" is visible.

The framework's claim is deliberately not "our score is better".  It is that the
*null* is the thing that has to change, and that it can be attached to any of
these scores.  The comparison is therefore run with the same score wherever
possible, so that what differs is the null and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["have_decoupler", "decoupler_methods", "run_decoupler",
           "covariate_net", "COMPARATORS"]

#: The `decoupler` methods that take a continuous covariate, by the name the
#: package exposes them under.
_DECOUPLER = ("ulm", "mlm", "aucell", "zscore", "gsva", "waggr")

COMPARATORS = ("ulm", "mlm", "aucell_decoupler", "zscore", "gsva", "waggr")


def have_decoupler():
    """Whether the real ``decoupler`` package is importable in this environment."""
    try:
        import decoupler  # noqa: F401
        return True
    except Exception:
        return False


def decoupler_methods():
    """The framework's labels for the ``decoupler`` methods this build can call.

    Returns the labels :func:`run_decoupler` accepts, not the package's own
    names: AUCell is spelled ``aucell`` by the package and
    ``aucell_decoupler`` here, so returning the package's names would hand
    :func:`run_comparators` a label its own dispatch table does not carry.
    """
    if not have_decoupler():
        return []
    from decoupler import mt
    return [label for label, impl in COMPARATORS_BY_NAME if hasattr(mt, impl)]


def covariate_net(gene_set, source="set", weight=1.0):
    """A ``decoupler`` network linking one covariate to the genes of a set.

    ``decoupler`` expects the covariate to be a column of the data matrix and the
    network to name it as a source, so the frame returned here is what turns
    "regress this gene set on this axis" into a call the package understands.
    """
    return pd.DataFrame(dict(source=source, target=list(gene_set),
                             weight=float(weight)))


#: Maps the framework's label for each comparator onto the name ``decoupler``
#: exposes; they differ only for AUCell, which the package spells as we do.
COMPARATORS_BY_NAME = (("ulm", "ulm"), ("mlm", "mlm"),
                       ("aucell_decoupler", "aucell"), ("zscore", "zscore"),
                       ("gsva", "gsva"), ("waggr", "waggr"))

#: Methods that return a per-set statistic together with a P value of their own.
ANALYTIC = ("ulm", "mlm", "zscore", "waggr")

#: Methods that return one score *per cell* and no P value at all.  These are
#: scoring rules, not tests: whatever inference is reported on top of them is
#: supplied by the analyst, which is the situation this package exists to
#: address.  For these the wrapper reports the score's correlation with the axis
#: and a P value computed from that correlation in the usual way.
CELL_LEVEL = ("aucell", "gsva")


def run_decoupler(X, genes, gene_set, axis, method="ulm", source="set",
                  max_cells=None, seed=0):
    """Score one gene set with an installed ``decoupler`` method, per cell.

    ``decoupler``'s ``ulm`` and ``mlm`` are *per-observation* methods: they treat
    the genes as the sample and regress a gene-level statistic on set membership,
    so passing a cells x genes matrix returns one score **and one P value per
    cell**, not one number per gene set.  Their P values therefore speak to "is
    this cell's profile enriched for the set", a different question from "is the
    set's activity associated with this axis".  This wrapper keeps the two apart
    rather than quietly passing one off as the other:

    * ``per_cell`` -- the package's score for every cell it scored;
    * ``decoupler_frac_positive`` -- the share of cells the package's own P value
      calls significant, which is how its inference is normally consumed;
    * ``score``, ``p_value`` -- the correlation of that score with the axis and
      its P value, computed **here**, and labelled ``p_source`` accordingly.

    The regressor is the membership weight of each gene, so the matrix has to
    carry the genes that are *not* in the set as well; restricting it to the
    set's own columns makes the regressor constant and the method degenerate.
    This is the opposite of what a per-gene linear model would need, and getting
    it backwards is silent -- the package returns NaN and then fails inside its
    own FDR step.  ``max_cells`` subsamples cells when the dense frame would be
    too large.

    Parameters
    ----------
    X : scipy.sparse matrix or ndarray
        Cells x genes expression matrix over the gene universe, log-normalised as
        the package expects.
    genes : sequence of str
        Column names of ``X``.
    gene_set : sequence of str
    axis : array-like
        One value per cell.
    max_cells : int, optional
        Draw at most this many cells.  ``decoupler`` densifies internally, so a
        whole 50,000-cell panel does not fit; the sampled cells are the ones the
        comparison is then run on, and the same draw is used for every method.

    Returns
    -------
    dict
    """
    if not have_decoupler():
        raise ImportError("decoupler is not installed in this environment")
    from decoupler import mt

    from .analyses import _two_sided_from_spearman
    from .stats import spearman

    genes = np.asarray(genes)
    pos = {g: i for i, g in enumerate(genes)}
    requested = [g for g in gene_set if g in pos]
    n_missing = len(gene_set) - len(requested)
    if len(requested) < 5:                     # decoupler's own tmin
        return dict(score=np.nan, p_value=np.nan, method=method,
                    n_genes=len(requested), n_missing=n_missing,
                    p_source="framework", note="fewer than 5 genes; tmin")

    cells = np.arange(X.shape[0])
    if max_cells is not None and X.shape[0] > max_cells:
        cells = np.sort(np.random.default_rng(seed).choice(
            X.shape[0], size=int(max_cells), replace=False))

    Xs = X[cells]
    dense = np.asarray(Xs.todense(), dtype=np.float64) if sp.issparse(Xs) \
        else np.asarray(Xs, dtype=np.float64)
    # A gene with no counts anywhere has no residual variance and makes the
    # per-gene statistic undefined, which the package raises on rather than
    # skipping.  How many were dropped is part of the result.
    detected = (dense > 0).any(axis=0)
    n_dropped = int((~detected).sum())
    if detected.sum() < 5:
        return dict(score=np.nan, p_value=np.nan, method=method,
                    n_genes=0, n_dropped_zero=n_dropped,
                    p_source="framework",
                    note="fewer than 5 genes with any counts at all")
    frame = pd.DataFrame(dense[:, detected], columns=list(genes[detected]))
    frame = frame.set_axis([str(i) for i in range(frame.shape[0])], axis=0)

    present = [g for g in requested if g in set(genes[detected])]
    impl = dict(COMPARATORS_BY_NAME)[method]
    fn = getattr(mt, impl)

    score_df, pval_df = fn(frame, covariate_net(present, source=source),
                           verbose=False)
    score_df = pd.DataFrame(score_df)
    if source not in score_df.columns:
        raise RuntimeError(f"decoupler {impl} returned columns "
                           f"{list(score_df.columns)}, expected {source!r}")

    # `decoupler` drops every cell whose profile is empty across the panel -- in
    # sparse data that is the shallow end of the depth distribution, which is
    # precisely the population the calibration is about.  The drop is silent, so
    # the only way to keep the comparison honest is to read the surviving index
    # and report what was removed.
    kept = score_df.index.astype(int).to_numpy()
    axis_s = np.asarray(axis, dtype=float)[cells][kept]
    per_cell = np.asarray(score_df[source], dtype=float)
    n_dropped_empty = int(len(cells) - kept.size)

    rho = spearman(per_cell, axis_s)[0]
    out = dict(method=method, n_genes=len(present), n_missing=n_missing,
               n_dropped_zero=n_dropped, n_dropped_empty=n_dropped_empty,
               n_cells_scored=int(kept.size), n_cells_requested=int(len(cells)),
               decoupler_impl=impl, kind=("analytic" if impl in ANALYTIC
                                          else "score_only"),
               score=float(rho),
               p_value=float(_two_sided_from_spearman(rho, kept.size)),
               p_source="framework (correlation of the score with the axis)",
               per_cell=per_cell)
    if pval_df is not None:
        pd_ = pd.DataFrame(pval_df)
        if source in pd_.columns:
            out["decoupler_frac_positive"] = float(
                np.mean(np.asarray(pd_[source], dtype=float) < 0.05))
    return out


def run_comparators(X, genes, gene_set, axis, methods=None, source="set"):
    """Every available comparator on one gene set, as a tidy frame.

    One row per method, with the score it produced, the P value it reports on
    its own terms, and how many genes of the set it used.  A method that raised
    is reported with its error rather than dropped, because a comparison table
    that silently omits the method that failed is not a comparison.
    """
    methods = methods or decoupler_methods()
    rows = []
    for m in methods:
        try:
            out = run_decoupler(X, genes, gene_set, axis, method=m,
                                source=source)
        except Exception as exc:                          # pragma: no cover
            out = dict(score=np.nan, p_value=np.nan, method=m,
                       n_genes=0, error=f"{type(exc).__name__}: {exc}")
        # The per-cell vector belongs to the caller, not to a one-row-per-method
        # table; keeping it here would put an array inside a DataFrame cell.
        out.pop("per_cell", None)
        rows.append(out)
    return pd.DataFrame(rows)


def size_matched_null(cache, gene_set, n=200, tol=0.10, rng=None):
    """Replacement sets matched only on *mean detection*, not per gene.

    This is the compromise an analyst reaches for when the gene set is small:
    find genes whose detection rate is within ``tol`` of the set's average and
    draw from them.  It fixes the grossest composition bias and leaves the
    per-gene spread -- which in a real clock set spans two orders of magnitude --
    entirely uncontrolled.
    """
    rng = rng or np.random.default_rng(0)
    pos = [cache._index[g] for g in gene_set if g in cache._index]
    if not pos:
        raise ValueError("none of the gene set is in the cache")
    target = float(np.mean(cache.detection[pos]))
    det = cache.detection
    band = np.flatnonzero(np.abs(det - target) <= tol * max(target, 1e-6))
    band = band[~np.isin(band, pos)]
    if band.size < len(pos):
        band = np.argsort(np.abs(det - target))[: max(len(pos) * 5, 50)]
    out = []
    for _ in range(int(n)):
        out.append(list(cache.genes[rng.choice(band, size=len(pos),
                                               replace=False)]))
    return out
