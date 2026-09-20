"""The closed form behind the calibration: where the spurious signal comes from.

This module is the reason the rest of the package can be cheap.  The confound
that rank-based gene-set scores suffer in sparse data is not a phenomenon that
has to be discovered by simulation; it is an algebraic consequence of how the
ranking is defined, and it can be written down and evaluated per cell before any
score is computed.

The derivation
--------------
Let the matrix have ``G`` genes and let cell ``c`` have counts on ``D_c`` of them.
Write ``m`` for the rank ceiling (``m = ceil(f * G)``, ``f = 0.05`` for AUCell).

The ranking of cell ``c`` places the ``D_c`` detected genes at ranks ``1..D_c``.
The remaining ``G - D_c`` genes all have a count of zero and are ordered among
themselves by the tie-break, so each of them is equally likely to land at any
rank in ``D_c + 1 .. G``.

AUCell gives a gene at rank ``r`` the weight ``max(0, m - r + 1)``, and divides
the sum over the ``k`` genes of the set by ``k * m``.  So the *expected* weight of
a gene the cell never expressed is

    E[weight] = (1 / (G - D)) * sum_{t=1}^{m-D} t
              = (m - D)(m - D + 1) / (2 (G - D))          for D < m, else 0,

and the expected share of the score that such a gene contributes is that
quantity over ``m``:

    tau(D) = (m - D)(m - D + 1) / (2 m (G - D)).                    (1)

Write ``d_c(S)`` for the number of genes of the set detected in cell ``c``.  The
score splits into a detected part and a realised tie-break part,

    AUC_c = Det_c + TB_c,   with   E[TB_c] = ((k - d_c(S)) / k) * tau(D_c),  (2)

and the split is exact, cell by cell: ``score_decomposition`` returns both
halves, and a test asserts that their sum reproduces ``aucell`` exactly.  The
closed form is the tie-break's *expectation* under the uniform-order model, so
the residual ``TB_c - E[TB_c]`` is the realised tie-break centred on its mean;
the tie-break is a deterministic hash, which makes that residual a reproducible
property of the matrix rather than sampling noise (its measured size across the
calibration grid's configurations is recorded in
``results/decomposition_residual.json``).  ``Det_c``
depends on the gene set; the expected tie-break term is the same for every gene
set with the same detection profile, and it is a *decreasing function of depth*.
That is the
whole problem: any axis that correlates with sequencing depth acquires a
correlation with the score that owes nothing to biology, and the more the set
fails to be detected -- which is to say, the sparser the cell -- the larger the
coefficient on it.

Equation (2) also gives the quantity an expression-matched null has to
reproduce.  A gene detected in a fraction ``p`` of cells enters the top ``m``
either because it was counted or, failing that, by tie-break:

    P(in top m) = p + (1 - p) * (m - D) / (G - D),  averaged over D.  (3)

Matching on ``p`` alone leaves the second term free, and because it varies with
the cell's depth the mismatch is not a constant that cancels -- it is a term
that tracks the confounder.  :func:`adaptive_detection_floor` in
:mod:`sparsegs.nulls` exists to stop the pool from censoring sets whose ``p``
lies below a fixed floor, which biases (3) upward for exactly the sparse sets
the calibration is about.

What to do with it
------------------
:func:`tie_break_inclusion` and :func:`tie_break_level` evaluate (1) and (3).
:func:`preflight` turns them into the diagnostic to run *first*: correlate
``tau(D)`` with the axis of interest and see how much of the association the
tie-break will manufacture on its own.  :func:`decompose_score` measures (2) on
a real score, so a user can see how much of their own result is the second term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import spearman

__all__ = ["tie_break_inclusion", "tie_break_level", "effective_inclusion",
           "tie_break_share", "score_decomposition", "preflight",
           "confound_band"]


# ----------------------------------------------------------------------
# the closed form
# ----------------------------------------------------------------------
def tie_break_inclusion(n_detected, max_rank, n_genes):
    """P(a gene the cell never expressed lands inside the top ``max_rank``).

    Equation (3), first factor.  Evaluated per cell for a given depth; the
    quantity does not depend on the gene set at all, only on how many genes the
    cell managed to detect.

    Parameters
    ----------
    n_detected : array-like of int
        ``D_c`` per cell -- the number of genes with a non-zero count.
    max_rank : int
        ``m``.
    n_genes : int
        ``G``, the number of genes in the matrix, not in the cache.

    Returns
    -------
    numpy.ndarray
        Values in ``[0, 1]``; zero where the cell detects at least ``max_rank``
        genes, since then no slot is left for the tie-break.
    """
    D = np.asarray(n_detected, dtype=float)
    m = float(max_rank)
    G = float(n_genes)
    share = np.clip(m - D, 0.0, None) / np.maximum(G - D, 1.0)
    return np.clip(share, 0.0, 1.0)


def tie_break_level(n_detected, max_rank, n_genes):
    """``tau(D)`` from equation (1): the tie-break's mean contribution to the score.

    The expected share of an AUCell score contributed by one gene that the cell
    did not express.  Multiply by the fraction of the set that goes undetected
    and the product is the tie-break term of equation (2).
    """
    D = np.asarray(n_detected, dtype=float)
    m = float(max_rank)
    G = float(n_genes)
    gap = np.clip(m - D, 0.0, None)
    return gap * (gap + 1.0) / (2.0 * m * np.maximum(G - D, 1.0))


def effective_inclusion(detection, depth, max_rank, n_genes):
    """Equation (3): the probability a gene enters the top ``max_rank`` at all.

    Parameters
    ----------
    detection : array-like
        Per-gene detection rate ``p``.
    depth : array-like
        Per-cell ``D_c``.

    Returns
    -------
    numpy.ndarray
        One value per gene, averaged over the cells' depths.

    Notes
    -----
    This is the quantity an expression-matched null has to reproduce, and the
    reason matching on ``p`` alone is not enough: two genes with the same ``p``
    have different inclusion when they are expressed in cells of different
    depth, and depth is where the confound lives.
    """
    p = np.asarray(detection, dtype=float)[:, None]
    share = tie_break_inclusion(depth, max_rank, n_genes)[None, :]
    return (p + (1.0 - p) * share).mean(axis=1)


def tie_break_share(detection, depth, max_rank, n_genes):
    """The share of the inclusion rate that the tie-break supplies.

    Equation (3) says a gene of detection ``p`` enters the top ``max_rank`` with
    probability ``p + (1 - p) * E[share]``.  The first term is the gene being
    counted; the second is it being placed there anyway.  This returns the
    second as a fraction of the total, averaged over the genes of the set:

        (mean_inclusion - mean_p) / mean_inclusion.

    It is the cheapest honest answer to "is this score carried by expression?",
    and unlike a detection floor it is a property of the gene set *in this data*
    rather than of a number someone chose.  A set whose genes are detected in
    half the cells has a small share however sparse the matrix; the same matrix
    gives a large share to a set of genes detected in a tenth of them.

    Parameters
    ----------
    detection : array-like
        Per-gene detection rate ``p`` of the genes in the set.
    depth : array-like
        Per-cell ``D_c``.
    max_rank : int
    n_genes : int
        ``G``, the number of genes in the matrix.

    Returns
    -------
    float
        In ``[0, 1)``.  Zero when the cells are deep enough that the tie-break
        never reaches the ceiling; approaching one as the set stops being
        detected at all.

    Examples
    --------
    A set detected everywhere has no tie-break share, whatever the depth:

    >>> import numpy as np
    >>> depth = np.full(500, 400.0)
    >>> tie_break_share(np.ones(30), depth, 1000, 20000)
    0.0
    """
    p = np.asarray(detection, dtype=float)
    if p.size == 0:
        return float("nan")
    mean_p = float(np.mean(p))
    mean_inc = float(np.mean(effective_inclusion(p, depth, max_rank, n_genes)))
    if mean_inc <= 0:
        return float("nan")
    return float(np.clip((mean_inc - mean_p) / mean_inc, 0.0, 1.0))


# ----------------------------------------------------------------------
# measuring the decomposition on a real score
# ----------------------------------------------------------------------
def score_decomposition(cache, gene_set, detection_matrix, max_rank=None,
                        depth=None):
    """Split a real AUCell score into its detected and tie-broken parts.

    The realised split is exact: ``detected + tie_break`` reproduces the score
    cell by cell, to the floating-point gap, and the test suite asserts it.
    Equation (2)'s closed form is the *expectation* of ``tie_break`` under the
    uniform-order model, so ``tie_break_expected`` is that closed form and the
    difference between the two columns is the residual ``eps_c`` of the
    manuscript's equation (3) -- centred, reproducible (the tie-break hash is
    deterministic), and measured across the calibration grid in
    ``results/decomposition_residual.json``.

    Parameters
    ----------
    cache : RankCache
    gene_set : sequence of str
    detection_matrix : scipy.sparse matrix
        Cells x genes binary matrix over the cache's gene universe.
    max_rank : int, optional
    depth : array-like, optional
        ``D_c``.  Taken from ``detection_matrix`` if omitted.

    Returns
    -------
    pandas.DataFrame
        One row per cell: ``detected``, ``tie_break``, ``tie_break_expected``,
        ``total`` and ``depth``.  ``detected + tie_break`` reproduces
        :func:`sparsegs.aucell` exactly.
    """
    import scipy.sparse as sp

    present = cache.present(gene_set)
    if not present:
        raise ValueError("none of the gene set is in the cache")
    if max_rank is None:
        max_rank = int(np.ceil(0.05 * cache.n_genes_total))
    m = int(max_rank)
    k = len(present)

    r = cache.ranks(present)
    contrib = np.where((r > 0) & (r <= m), m - r + 1, 0).astype(float)

    B = sp.csr_matrix(detection_matrix)
    if B.shape[1] != cache.genes.size:
        raise ValueError(
            f"detection_matrix has {B.shape[1]} columns; expected one per cache "
            f"gene ({cache.genes.size})")
    cols = cache.positions(present)
    det_set = np.asarray(B[:, cols].todense(), dtype=bool)

    if depth is None:
        depth = np.asarray(B.sum(axis=1)).ravel()
    depth = np.asarray(depth, dtype=float)

    # A gene that was detected but ranked beyond the ceiling stores BEYOND, and
    # contributes nothing; it still belongs to the detected part, not to the
    # tie-break, so the counts are taken from the detection matrix and not from
    # the ranks.
    detected = np.where(det_set, contrib, 0.0).sum(axis=1) / (k * m)
    tie = np.where(det_set, 0.0, contrib).sum(axis=1) / (k * m)
    n_undetected = k - det_set.sum(axis=1)
    expected = n_undetected / k * tie_break_level(depth, m, cache.n_genes_total)

    return pd.DataFrame(dict(
        cell=np.arange(depth.size), depth=depth,
        detected=detected, tie_break=tie, tie_break_expected=expected,
        total=detected + tie))


# ----------------------------------------------------------------------
# the pre-flight diagnostic
# ----------------------------------------------------------------------
def preflight(depth, axis, max_rank, n_genes, detection=None, k=None,
              score=None, rank_frac=0.05):
    """Will the tie-break manufacture an association with this axis?

    Everything here comes from the depth vector and the axis.  No gene set is
    scored and no matrix is touched, which is the point: this is the check to
    run before committing to an analysis, and it is exact rather than
    approximate, because ``tau`` is a deterministic function of the depth.

    Parameters
    ----------
    depth : array-like of int
        ``D_c`` per cell.
    axis : array-like
        The continuous variable the score will be tested against.
    max_rank : int
    n_genes : int
    detection : array-like, optional
        Per-gene detection rates.  Supplying it adds the *amplitude* of the
        confound: the share of a typical set that goes undetected, which is the
        coefficient on ``tau`` in equation (2).
    k : int, optional
        Gene-set size, used with ``detection``.
    score : array-like, optional
        An already-computed score for the same cells.  Supplying it turns the
        structural answer into a realised one: the tie-break's contribution to
        the score is attenuated by however much sampling noise the term carries,
        and the attenuation is measured here rather than assumed.

    Returns
    -------
    dict
        ``tau_axis_rho`` is the correlation of the score's depth-derived
        component with the axis; it is exact, needs no score, and is the
        headline number.  ``implied_rho`` scales it by the undetected share of a
        set with the supplied detection profile, giving the correlation of the
        score's *conditional mean* with the axis.

        The realised correlation is smaller, because the tie-break contributes
        a random amount in each cell and the average of that noise sits in the
        score's variance without moving its mean.  When ``score`` is supplied,
        ``attenuation`` is ``sd(tau) / sd(score)`` and ``realised_implied_rho``
        is the product, which is what the observed correlation should land on if
        nothing else is going on.

        The other half of the score, ``Det_c``, also varies with depth, in the
        opposite direction: a deeper cell detects more of the set, and a
        detected gene outranks every undetected one.  Which term wins is an
        empirical question about the data, answered by
        :func:`score_decomposition`, not by this function.  ``verdict`` is
        ``'CLEAR'``, ``'CAUTION'`` or ``'CONFOUNDED'``.
    """
    depth = np.asarray(depth, dtype=float)
    axis = np.asarray(axis, dtype=float)
    tau = tie_break_level(depth, max_rank, n_genes)

    rho_tau_axis = spearman(tau, axis)[0]
    # Sample standard deviation, matching R's `sd()`.  `np.std` defaults to the
    # population form, so the two implementations reported `tau_sd` a factor of
    # sqrt(n/(n-1)) apart on the same input -- invisible in `attenuation`, where
    # the convention cancels between numerator and denominator, and visible in
    # the field a reader is given.
    sd_tau = float(np.std(tau, ddof=1)) if tau.size > 1 else float("nan")

    out = dict(
        n_cells=int(depth.size), max_rank=int(max_rank), n_genes=int(n_genes),
        median_detected=float(np.median(depth)),
        depth_ratio=float(max_rank / max(np.median(depth), 1.0)),
        tau_axis_rho=float(rho_tau_axis),
        tau_sd=sd_tau,
        tau_range=[float(tau.min()), float(tau.max())],
        # A cell deeper than the ceiling contributes no tie-break at all, so if
        # every cell is, ``tau`` is the constant zero and its correlation with
        # the axis does not exist rather than being small.  The verdict is
        # ``CLEAR`` either way -- a component that never varies cannot carry an
        # association -- but "rho is near zero" and "rho is not defined" are
        # different statements about the data, and the record should say which
        # one it is making.
        #
        # Constancy is tested by comparing the extremes and not by asking
        # whether the standard deviation is zero: ``np.std`` subtracts a mean,
        # and for a constant vector that leaves roundoff rather than nothing --
        # 3.5e-18 for a vector of 300 sevens, which reads as "varies".  R's
        # two-pass ``sd`` returns an exact zero there, so the two
        # implementations disagreed about the same vector.
        tau_constant=bool(tau.min() == tau.max()),
        depth_axis_rho=spearman(depth, axis)[0])

    if detection is not None and k is not None:
        p = np.asarray(detection, dtype=float)
        # Mean undetected share of a k-gene set of this detection profile --
        # the coefficient on tau in equation (2).
        mean_detected = k * float(np.mean(p))
        gamma = 1.0 - mean_detected / k
        inclusion = float(np.mean(effective_inclusion(p, depth, max_rank, n_genes)))
        out.update(detection_mean=float(np.mean(p)), gamma=float(gamma),
                   implied_rho=float(gamma * rho_tau_axis),
                   mean_inclusion=inclusion,
                   # The share of the inclusion rate that the tie-break supplies.
                   tie_break_share=tie_break_share(p, depth, max_rank, n_genes))
    else:
        out["implied_rho"] = float(rho_tau_axis)

    a = abs(out["tau_axis_rho"])
    # What chance alone produces at this sample size.  A fixed cut would call a
    # 400-cell dataset clear at |rho| = 0.06 and a 50,000-cell dataset confounded
    # at the same value, so the band is taken from the null distribution of the
    # correlation rather than from a constant.
    band = 1.96 / np.sqrt(max(depth.size - 3, 1))
    out["chance_band"] = float(band)

    if score is not None:
        s = np.asarray(score, dtype=float)
        sd_s = float(np.std(s, ddof=1)) if s.size > 1 else float("nan")
        attenuation = (sd_tau / sd_s) if sd_s > 0 else float("nan")
        out["attenuation"] = float(attenuation)
        out["realised_implied_rho"] = float(out["implied_rho"] * attenuation)
        out["observed_rho"] = float(spearman(s, axis)[0])
        out["observed_vs_implied"] = float(out["observed_rho"]
                                           - out["realised_implied_rho"])

    if not np.isfinite(a) or a <= band:
        out["verdict"] = "CLEAR"
    elif a < 0.15:
        out["verdict"] = "CAUTION"
    else:
        out["verdict"] = "CONFOUNDED"
    return out


def confound_band(null_rho, n_cells, level=0.95):
    """The range of correlations a score with no biology would still show.

    A null-calibrated P value answers "is this correlation larger than the null
    produces"; this answers the cruder question a reader asks first, "how large
    does a correlation have to be before the null cannot produce it at all".
    """
    from scipy import stats as st
    z = st.norm.ppf(0.5 + level / 2.0)
    half = z / np.sqrt(max(n_cells - 3, 1))
    return dict(level=float(level),
                low=float(np.tanh(np.arctanh(np.clip(null_rho, -0.999, 0.999)) - half)),
                high=float(np.tanh(np.arctanh(np.clip(null_rho, -0.999, 0.999)) + half)),
                n_cells=int(n_cells))
