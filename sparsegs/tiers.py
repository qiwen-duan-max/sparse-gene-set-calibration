"""Three-tier validation, with the tiers made regime-matched.

The usual three-tier story -- technical, internal, external -- is told as if the
tiers were independent hurdles.  They are not, and the failure is specific: a
technical check run at one sparsity tells you nothing about the same method at
another.  A scoring rule that separates a spiked population cleanly at 5% zeros
can be indistinguishable from noise at 67% zeros, so "Tier 1 passed" is only
meaningful when Tier 1 was run in the regime Tier 2 and Tier 3 inhabit.

:func:`tier1_technical` therefore takes the observed sparsity as an input and
reports the sparsity it was evaluated at alongside the result, and
:func:`run_tiers` refuses to report a Tier 1 pass whose regime does not match the
data the later tiers were computed on.

Tier 2 asks whether the association survives a matched null in the discovery
cohort.  Tier 3 asks whether it is present in an independent cohort -- and since
scores are not comparable across cohorts with different gene panels, it compares
*direction and rank*, never raw score values.
"""

from __future__ import annotations

import numpy as np

from .stats import spearman, empirical_p, null_summary

__all__ = ["tier1_technical", "tier2_internal", "tier3_external", "run_tiers",
           "REGIME_TOLERANCE"]

#: How far the sparsity of a Tier 1 check may sit from the sparsity of the data
#: before the check is reported as not applicable.
REGIME_TOLERANCE = 0.10


def tier1_technical(target_zero_rate, n_cells=800, n_genes=8000, k=8,
                    effect=0.6, median_detected=None, detection=None,
                    seed=0, n_repeats=5):
    """Can the scoring rule recover a planted signal at this sparsity?

    The question is deliberately narrow: given a gene set of this size, in data
    this sparse, does the score separate a cell population carrying a planted
    programme from one that does not?  ``target_zero_rate`` is what fixes the
    regime, and it is usually the observed zero rate of the real gene set.

    A single sparsity cannot be dialled in directly -- it is an output of the
    detection rate, the set size and the matrix -- so the function searches the
    detection rate for the one that lands nearest the requested sparsity, then
    measures the score's ability to separate the planted group.

    Parameters
    ----------
    target_zero_rate : float
        The zero rate the check must be run at, typically taken from the real
        data with :func:`sparsegs.sparsity_report`.
    effect : float
        Log-scale effect planted on the target genes.
    n_repeats : int
        Independent datasets; the reported separation is the mean.

    Returns
    -------
    dict
        ``auc`` (mean separation of the planted programme), ``zero_rate``
        (attained), ``regime_match`` (whether it is close enough to
        ``target_zero_rate`` to speak to it), and ``pass``.
    """
    from .simulate import SimConfig, simulate
    from .scores import aucell
    from .rankcache import RankCache
    from .stats import auroc

    grid = np.array([0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.45])
    achieved, aucs = [], []
    for det in grid:
        zs, auc_run = [], []
        for rep in range(n_repeats):
            cfg = SimConfig(n_cells=n_cells, n_genes=n_genes, n_target=k,
                            median_detected=median_detected or int(0.36 * n_genes),
                            detection=float(det), effect=effect, coexpr=0.0,
                            depth_programme_loading=0.0, seed=seed + 101 * rep)
            sim = simulate(cfg)
            cache = RankCache.build(sim.X, sim.genes, ceiling=max(sim.max_rank, 30),
                                    seed=2024, chunk=500)
            score = aucell(cache, sim.target, max_rank=sim.max_rank)
            zs.append(_zero_rate(score))
            # The programme is continuous; the planted group is its upper half.
            auc_run.append(auroc(score, sim.programme > np.median(sim.programme)))
        achieved.append(float(np.mean(zs)))
        aucs.append(float(np.mean(auc_run)))

    achieved = np.array(achieved)
    aucs = np.array(aucs)
    i = int(np.argmin(np.abs(achieved - target_zero_rate)))
    return dict(
        auc=float(aucs[i]),
        zero_rate=float(achieved[i]),
        target_zero_rate=float(target_zero_rate),
        regime_gap=float(abs(achieved[i] - target_zero_rate)),
        regime_match=bool(abs(achieved[i] - target_zero_rate) <= REGIME_TOLERANCE),
        detection_rate=float(grid[i]),
        # The threshold is a convention, not a law: it says the score has some
        # ability to rank cells, not that it is fit for a particular claim.
        passed=bool(aucs[i] >= 0.65),
        curve=[dict(detection=float(d), zero_rate=float(z), auc=float(a))
               for d, z, a in zip(grid, achieved, aucs)],
    )


def _zero_rate(score):
    score = np.asarray(score, dtype=float)
    return float((score <= 0).mean())


def tier2_internal(score, axis, null_builder, gene_set, max_rank, n_null=200,
                   seed=0, tail="two-sided",
                   kinds=("random", "expression", "codetection")):
    """Does the association survive a matched null in the discovery cohort?

    Returns one entry per null family.  The families are nested -- co-detection
    matching refines expression matching, which refines random -- so a claim that
    holds against the strictest one holds against the others, and the interesting
    case is when they disagree.

    ``passed`` and ``passed_strictest`` are the verdicts of the expression and
    co-detection families, and each carries the name of the family it came from
    in ``passed_family`` / ``passed_strictest_family``.  Both are ``None`` when
    that family is not among ``kinds``: a family that was not drawn has no
    verdict, and reporting one would be a claim about a test that never ran.
    """
    from .scores import aucell

    observed = spearman(score, axis)[0]
    out = dict(observed_rho=observed, n_cells=int(len(score)))
    rng = np.random.default_rng(seed)
    for kind in kinds:
        sets = null_builder.sample(gene_set, n_null, kind=kind, rng=rng)
        null = np.array([spearman(aucell(null_builder.cache, s, max_rank=max_rank),
                                  axis)[0] for s in sets])
        summary = null_summary(null, observed, tail=tail)
        out[kind] = dict(p_value=summary["p_value"],
                         # The directional values travel with the two-sided one
                         # so that a reader can see which question the reported
                         # P value answered, and a directional claim does not
                         # have to be recomputed from the null.
                         p_lower=empirical_p(null, observed, tail="lower"),
                         p_upper=empirical_p(null, observed, tail="upper"),
                         null_median=summary["null_median"],
                         null_sd=summary["null_sd"],
                         percentile=summary["percentile"],
                         passed=bool(summary["p_value"] < 0.05))
    # The two headline verdicts are named for the family they come from, and a
    # verdict is only recorded when that family was actually drawn.  ``kinds``
    # is a free parameter, so a caller who asks for a subset has no strictest
    # family to be judged against: returning the expression family's verdict
    # under the name ``passed_strictest`` would be a claim about co-detection
    # matching that was never computed, which is the failure this whole
    # framework exists to stop making.
    for family, key in (("expression", "passed"),
                        ("codetection", "passed_strictest")):
        out[key] = bool(out[family]["passed"]) if family in out else None
        out[key + "_family"] = family if family in out else None
    return out


def tier3_external(discovery_rho, replication_rho, n_replication,
                   discovery_n=None):
    """Is the association present, with the same sign, in an independent cohort?

    Scores from two cohorts are not comparable when the gene panels differ --
    ``max_rank`` is a fraction of the panel, so an identical biological signal
    lands on different score values.  This function therefore compares signs and
    significance, never score magnitudes, and reports the rank correlation
    implied by the two effect sizes rather than any difference between them.

    Parameters
    ----------
    discovery_rho, replication_rho : float
        Spearman correlation of score against the axis in each cohort.
    n_replication : int
        Cells in the replication cohort, used for the significance check.
    """
    from .analyses import _two_sided_from_spearman

    p_rep = _two_sided_from_spearman(replication_rho, n_replication)
    same_sign = bool(np.sign(discovery_rho) == np.sign(replication_rho))
    significant = bool(np.isfinite(p_rep) and p_rep < 0.05)
    return dict(
        discovery_rho=float(discovery_rho),
        replication_rho=float(replication_rho),
        replication_p=float(p_rep),
        n_replication=int(n_replication),
        same_sign=same_sign,
        replication_significant=significant,
        # A same-sign, significant result in the replication cohort is the bar.
        passed=bool(same_sign and significant),
        note=("score values are not comparable across cohorts with different "
              "gene panels; only the direction and the rank association are "
              "compared here"))


def run_tiers(discovery, replication=None, tier1=None):
    """Assemble the three tiers and say what the combination supports.

    ``discovery`` and ``replication`` are dicts with an ``observed_rho`` and an
    ``n_cells``; ``tier1`` is the output of :func:`tier1_technical`.  The verdict
    is intentionally conservative: a Tier 1 check that was not run in the right
    regime cannot rescue a failed Tier 3, and is reported as not applicable
    rather than as passed.
    """
    t1 = tier1 or {}
    if t1 and not t1.get("regime_match", False):
        t1 = dict(t1, applicable=False,
                  reason=(f"Tier 1 was evaluated at a zero rate of "
                          f"{t1.get('zero_rate', float('nan')):.2f}, which is "
                          f"{t1.get('regime_gap', float('nan')):.2f} away from "
                          f"the data's {t1.get('target_zero_rate', float('nan')):.2f}; "
                          f"it does not speak to this gene set"))
    elif t1:
        t1 = dict(t1, applicable=True)

    t3 = None
    if replication is not None:
        t3 = tier3_external(discovery["observed_rho"], replication["observed_rho"],
                            replication["n_cells"])

    passed = []
    if t1.get("applicable") and t1.get("passed"):
        passed.append("TIER_1")
    if discovery.get("passed"):
        passed.append("TIER_2")
    if t3 and t3["passed"]:
        passed.append("TIER_3")

    if t3 is None:
        verdict = "INCOMPLETE_NO_EXTERNAL"
    elif t3["passed"] and discovery.get("passed"):
        verdict = "VALIDATED"
    elif discovery.get("passed"):
        verdict = "INTERNAL_ONLY"
    else:
        verdict = "NOT_SUPPORTED"

    return dict(tier1=t1, tier2=discovery, tier3=t3,
                tiers_passed=passed, verdict=verdict)
