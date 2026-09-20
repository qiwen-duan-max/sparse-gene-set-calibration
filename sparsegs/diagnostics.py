"""Pre-flight diagnostics: what a score can and cannot resolve, before you test it.

The diagnostics here are cheap, need only the expression matrix and the gene set,
and are meant to be run *before* any association test.  Their job is to predict
which of the failure modes documented in this package is about to bite.

The structural zero rate
------------------------
AUCell and UCell scores are sums over the top ``max_rank`` ranks.  If a gene set
of ``k`` genes is ranked independently of everything else, the probability that
none of its genes lands in the top fraction ``f`` is ``(1 - f)^k``, and the score
is exactly zero.  This is the *structural* zero rate, and it is a property of the
scoring rule and the size of the matrix, not of the biology.

Observed zero rates that sit far from the structural value are informative:
deviation upward means the genes are lost together (co-detection), deviation
downward means they are found together (co-regulation).  Either way the set does
not behave like the independent draw the usual null assumes.

Rank-fraction comparability
---------------------------
``max_rank = ceil(f * n_genes_in_matrix)`` makes every AUCell-family score a
function of how many genes happen to be in the matrix.  Two cohorts processed
with different gene panels therefore have scores on different scales even when
the biology is identical, and a cross-cohort comparison of score *values* is not
interpretable without a correction.  :func:`comparability_report` measures how
large that effect is for a given pair of matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["structural_zero_rate", "sparsity_report", "depth_report",
           "comparability_report", "verdict", "DEFAULT_THRESHOLDS",
           "SEVERITY_ORDER", "raise_severity"]

#: Thresholds for :func:`verdict`.  These are not asserted; they are set from the
#: simulation grid in this project, which measures how the discrepancy between
#: conventional and calibrated inference grows with the zero rate.  Treat them as
#: calibrated defaults and re-derive them for a new scoring rule.
#:
#: ``tie_break_share_*`` govern the criterion that is used when the cache carries
#: per-cell depth.  ``detection_floor`` and ``min_detected_frac`` are the fallback
#: for when it does not, and they are kept deliberately separate: the floor is a
#: fixed rate and binds differently on every dataset, which is the criticism this
#: package exists to make, so it is not allowed to silently become the criterion
#: where the exact one is available.

#: Severities in the order they are raised, so that a weak reason can never
#: lower a verdict a strong one has already set.  Alphabetical ``max()`` on
#: these strings does exactly that -- ``"MODERATE"`` sorts above ``"HIGH"`` --
#: which is what this replaced.
SEVERITY_ORDER = ("LOW", "MODERATE", "HIGH")


def raise_severity(current, level):
    """The higher of two severities."""
    if SEVERITY_ORDER.index(level) > SEVERITY_ORDER.index(current):
        return level
    return current


DEFAULT_THRESHOLDS = dict(
    zero_rate_low=0.20,       # below this, conventional inference is usually safe
    zero_rate_high=0.50,      # above this, the naive null is unreliable
    tie_break_share_moderate=0.50,   # most of the score is the tie-break
    tie_break_share_high=0.75,       # almost all of it is
    detection_floor=0.02,     # fixed-rate fallback; see adaptive_detection_floor
    min_detected_frac=0.50,   # fraction of the set that must clear the fallback
    zero_rate_excess_tolerance=0.05,  # deviation from the independent-draw rate
)


def structural_zero_rate(k, n_genes, rank_frac=0.05):
    """Probability that a `k`-gene set scores exactly zero under independence.

    Parameters
    ----------
    k : int
        Number of genes in the set that are present in the matrix.
    n_genes : int
        Number of genes in the matrix.
    rank_frac : float
        Ranking fraction of the score (AUCell's default is 0.05).

    Notes
    -----
    ``max_rank`` is ``ceil(rank_frac * n_genes)``, so the effective fraction is
    marginally above ``rank_frac`` for small matrices; that is accounted for.
    """
    max_rank = int(np.ceil(rank_frac * n_genes))
    return float((1.0 - max_rank / n_genes) ** k)


def sparsity_report(cache, gene_set, rank_frac=0.05, thresholds=None,
                    detection_floor=None, depth=None):
    """Everything the framework knows about one gene set before testing it.

    Parameters
    ----------
    cache : RankCache
    gene_set : sequence of str
    rank_frac : float
    thresholds : dict, optional
        Overrides for :data:`DEFAULT_THRESHOLDS`.
    detection_floor : float, optional
        Rate below which a gene counts as not detected.  Omitting it uses
        :func:`~sparsegs.adaptive_detection_floor`, which takes the floor from
        the set's own detection profile.  Passing a number restores the fixed
        floor, and the report says which was used, because the two answer a
        different question.
    depth : array-like, optional
        Per-cell ``D_c``.  Taken from the cache when it carries it, which makes
        ``tie_break_share`` available.

    Returns
    -------
    dict
        ``observed_zero_rate`` from the score itself, ``structural_zero_rate``
        under independence, their difference, per-gene detection summaries, the
        fraction of the set clearing the floor, and -- where depth is known --
        ``tie_break_share``, the share of the set's inclusion rate that comes
        from rank tie-breaking rather than from expression.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    present = cache.present(gene_set)
    if not present:
        raise ValueError("none of the gene set is in the cache")
    pos = cache.positions(present)

    scores = None
    from .scores import aucell
    scores = aucell(cache, present)
    observed_zero = float((scores <= 0).mean())
    structural = structural_zero_rate(len(present), cache.n_genes_total,
                                      rank_frac=rank_frac)

    det = cache.detection[pos]
    if detection_floor is None:
        from .nulls import adaptive_detection_floor
        floor = float(adaptive_detection_floor(cache, present))
        floor_source = "adaptive (from this set's detection profile)"
    else:
        floor = float(detection_floor)
        floor_source = "fixed (supplied by the caller)"
    above = det >= floor
    usable = float(above.mean())

    out = dict(
        n_requested=len(gene_set), n_present=len(present),
        n_missing=len(gene_set) - len(present),
        observed_zero_rate=observed_zero,
        structural_zero_rate=structural,
        zero_rate_excess=observed_zero - structural,
        mean_detection=float(det.mean()), min_detection=float(det.min()),
        max_detection=float(det.max()),
        median_detection=float(np.median(det)),
        detection_floor=floor, detection_floor_source=floor_source,
        frac_above_floor=usable,
        n_above_floor=int(above.sum()),
        genes_below_floor=list(np.asarray(present)[~above]),
        median_score=float(np.median(scores)),
        mean_score=float(scores.mean()),
    )

    # Where the cache knows the cells' depth, the question the floor was standing
    # in for can be answered exactly.
    if depth is None:
        depth = getattr(cache, "depth", None)
    if depth is not None:
        from .theory import effective_inclusion, tie_break_share
        max_rank = int(np.ceil(rank_frac * cache.n_genes_total))
        out["max_rank"] = max_rank
        out["median_depth"] = float(np.median(depth))
        out["mean_inclusion"] = float(
            np.mean(effective_inclusion(det, depth, max_rank,
                                        cache.n_genes_total)))
        out["tie_break_share"] = tie_break_share(det, depth, max_rank,
                                                 cache.n_genes_total)
        out["criterion"] = "tie_break_share"
    else:
        out["criterion"] = "fixed_detection_floor"
    return out


def depth_report(cache_or_matrix, rank_frac=0.05):
    """Sequencing depth of the cells, and the UCell ``r_max`` it implies.

    UCell's documentation recommends setting ``r_max`` to roughly the median
    number of detected genes per cell, with 1500 as a default for 10x data.  For
    shallow data that default caps far above the observed depth, which changes
    the score; this function reports the value the data actually imply.
    """
    import scipy.sparse as sp

    X = sp.csr_matrix(cache_or_matrix)
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    median_detected = float(np.median(detected))
    return dict(
        median_genes_per_cell=median_detected,
        p10_genes_per_cell=float(np.percentile(detected, 10)),
        p90_genes_per_cell=float(np.percentile(detected, 90)),
        n_genes_in_matrix=int(X.shape[1]),
        n_cells=int(X.shape[0]),
        aucell_max_rank=int(np.ceil(rank_frac * X.shape[1])),
        ucell_r_max_implied=int(round(median_detected)),
        ucell_default_is_appropriate=bool(1200 <= median_detected <= 2000),
    )


def comparability_report(matrices, rank_frac=0.05, k=8):
    """How far apart two or more matrices place the same gene set.

    Parameters
    ----------
    matrices : dict
        ``{label: (n_genes_in_matrix, n_cells)}`` -- only the gene count is
        needed to expose the scale dependence.
    rank_frac : float
    k : int
        Representative gene-set size, used for the structural zero rate.

    Returns
    -------
    pandas.DataFrame
        One row per matrix with its ``max_rank``, the effective ranking
        fraction, and the structural zero rate for a ``k``-gene set.  Compare the
        ``max_rank`` column across rows: if the gene panels differ, the same
        biological signal lands on different scores.
    """
    rows = []
    for label, (n_genes, n_cells) in matrices.items():
        max_rank = int(np.ceil(rank_frac * n_genes))
        rows.append(dict(
            dataset=label, n_genes=int(n_genes), n_cells=int(n_cells),
            max_rank=max_rank,
            effective_frac=max_rank / n_genes,
            structural_zero_rate=structural_zero_rate(k, n_genes, rank_frac)))
    out = pd.DataFrame(rows)
    if len(out) > 1:
        spread = out["max_rank"].max() / max(out["max_rank"].min(), 1)
        out.attrs["max_rank_ratio"] = float(spread)
        out.attrs["scale_comparable"] = bool(spread < 1.05)
    return out


def verdict(report, used_matched_null=False, outcome_driven_cutpoint=False,
            externally_replicated=None, thresholds=None, preflight=None):
    """Turn a diagnostic report into a recommendation.

    The verdict is deliberately conservative and mechanical: it says what the
    numbers support, not what the analyst hopes they support.

    Parameters
    ----------
    report : dict
        Output of :func:`sparsity_report`.
    used_matched_null : bool
        Whether the association was tested against an expression-matched null
        rather than a conventional one.
    outcome_driven_cutpoint : bool
        Whether a continuous score was dichotomised at a threshold chosen from
        the outcome.
    externally_replicated : bool or None
        ``None`` if replication was not attempted.
    thresholds : dict, optional
    preflight : dict, optional
        Output of :func:`sparsegs.preflight`, if the axis and the per-cell depth
        are known.  Its ``verdict`` is ``'CLEAR'``, ``'CAUTION'`` or
        ``'CONFOUNDED'``, and the last two are folded into this verdict: an axis
        correlated with sequencing depth manufactures the association through
        the tie-break alone, which no amount of sparsity on the gene set's side
        prevents and no matched null removes.  It is a separate input rather
        than something read out of ``report`` because it needs the axis, which
        the sparsity report never sees.

    Returns
    -------
    dict
        ``verdict`` in ``{'INTERPRETABLE', 'INTERPRETABLE_WITH_MATCHED_NULL',
        'NOT_IDENTIFIABLE'}``, the reasons, the severities behind them, and
        ``criterion`` -- which of ``'tie_break_share'`` and
        ``'fixed_detection_floor'`` produced the first of those reasons.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    z = report["observed_zero_rate"]
    reasons, severity = [], "LOW"

    # Two criteria ask "is the score carried by expression or by the ranking's
    # tie-break?".  The exact one needs the cells' depth; the fallback compares
    # the set against a fixed rate instead and is weaker for it, so the report
    # is made to say which one produced the verdict.
    share = report.get("tie_break_share")
    if share is not None and np.isfinite(share):
        criterion = "tie_break_share"
        if share >= th["tie_break_share_high"]:
            reasons.append(
                f"{share:.0%} of the set's chance of entering the score comes "
                f"from rank tie-breaking rather than from expression; the score "
                f"is largely a depth proxy")
            severity = "HIGH"
        elif share >= th["tie_break_share_moderate"]:
            reasons.append(
                f"{share:.0%} of the set's chance of entering the score comes "
                f"from rank tie-breaking; more of it is the ranking than the "
                f"biology")
            severity = raise_severity(severity, "MODERATE")
    else:
        criterion = "fixed_detection_floor"
        if report["frac_above_floor"] < th["min_detected_frac"]:
            reasons.append(
                f"{report['frac_above_floor']:.0%} of the set is detected in "
                f"more than {report['detection_floor']:.2%} of cells, the floor "
                f"in use; below it the score is carried by rank tie-breaking "
                f"rather than by expression")
            severity = "HIGH"
    if z >= th["zero_rate_high"]:
        reasons.append(
            f"{z:.1%} of cells score zero, at or above the level at which the "
            f"conventional null is unreliable")
        severity = "HIGH"
    elif z >= th["zero_rate_low"]:
        reasons.append(
            f"{z:.1%} of cells score zero; the score is informative for a "
            f"minority of cells only")
        severity = raise_severity(severity, "MODERATE")

    if abs(report["zero_rate_excess"]) > th["zero_rate_excess_tolerance"]:
        reasons.append(
            f"the observed zero rate deviates from the independent-draw "
            f"expectation by {report['zero_rate_excess']:+.1%}, so the set is "
            f"not behaving like an independent draw")
        severity = raise_severity(severity, "MODERATE")

    if not used_matched_null:
        reasons.append("the association was not tested against an "
                       "expression-matched null")
        severity = raise_severity(severity, "MODERATE")

    if outcome_driven_cutpoint:
        reasons.append("a cutpoint was chosen from the outcome, which inflates "
                       "the false-positive rate well above the nominal level")
        severity = "HIGH"

    if externally_replicated is False:
        reasons.append("the association did not replicate in an independent "
                       "cohort")
        severity = "HIGH"
    elif externally_replicated is None and severity != "LOW":
        reasons.append("no independent cohort was tested")

    # The axis's own confound with depth.  It is a reason and not a refusal: a
    # matched null is calibrated *under* this confound -- the replacement sets
    # carry the same tie-break behaviour, which is the framework's central
    # result -- so the confound does not make the recommended analysis
    # untrustworthy.  What it does is say why the conventional analysis is not a
    # substitute for it, which is the one thing `used_matched_null` alone does
    # not convey: "no matched null was used" reads as a missing step until the
    # axis is named as the reason it is not optional.
    pf = (preflight or {}).get("verdict") if isinstance(preflight, dict) else None
    if pf == "CONFOUNDED":
        reasons.append(
            "the axis is correlated with sequencing depth, so the score's "
            "depth-derived component alone reproduces the association")
        severity = raise_severity(severity, "MODERATE")
    elif pf == "CAUTION":
        reasons.append(
            "the axis is weakly correlated with sequencing depth; the "
            "tie-break alone could account for part of the association")
        severity = raise_severity(severity, "MODERATE")

    if not reasons:
        # A clean verdict still has to say why it is clean, and the R
        # implementation of the framework reports the same sentence, so the
        # two objects are the same shape as well as the same decision.
        reasons = ["no diagnostic threshold was crossed"]

    # A set with nothing wrong with it is only plain INTERPRETABLE if the
    # association was in fact tested against a matched null.  Without one the
    # best available reading is that the numbers are not interpretable yet,
    # however unremarkable they look; this is the same rule the R
    # implementation of the framework uses, and the two are tested against
    # each other on it.
    if severity == "HIGH":
        name = "NOT_IDENTIFIABLE"
    elif not used_matched_null:
        name = "INTERPRETABLE_WITH_MATCHED_NULL"
    else:
        name = "INTERPRETABLE"
    return dict(verdict=name, severity=severity, reasons=reasons,
                criterion=criterion)
