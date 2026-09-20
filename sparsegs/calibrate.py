"""One-call calibration: the checklist an analyst runs before believing a score.

Everything in this package is available piece by piece, and the pieces are meant
to be composable.  This module is the assembled version: a single object that
holds a matrix, a gene set and a proposed axis, runs the whole diagnostic
sequence, and returns a verdict together with the numbers behind it.

The four steps, and why they are in this order
----------------------------------------------
1. **Sparsity.**  How much of the set is detectable, and what fraction of cells
   score zero.  A set that is undetected in most cells cannot carry a
   cell-level claim, and no downstream correction changes that.
2. **Null construction.**  Build a replacement set matched on detection,
   expression and co-detection, and report how well it matched.  A null that
   failed to match is not a null, and its P value is not evidence.
3. **Cutpoint calibration.**  If the analysis dichotomises the score, measure
   the false-positive rate of the rule rather than assuming the nominal level.
   The gap between an outcome-driven and a pre-specified cutpoint is routinely a
   factor of five or more.
4. **External replication.**  Sign and significance in an independent cohort.
   Because ``max_rank`` is a fraction of the gene panel, scores from two
   cohorts are not comparable as values; only direction and rank association
   are compared.

The verdict is deliberately mechanical.  It says what the numbers support, not
what the analyst hopes they support, and it prefers ``NOT_IDENTIFIABLE`` to a
qualified approval when the sparsity makes the question unanswerable from the
data at hand.

Usage
-----
>>> cal = Calibrator.from_matrix(X, genes, rank_frac=0.05, seed=42)
>>> out = cal.calibrate(CLOCK_REPRESSORS, axis=exhaustion)
>>> out["checklist"]["overall_verdict"]
'NOT_IDENTIFIABLE'
"""

from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp

from .analyses import _two_sided_from_spearman
from .nulls import MatchedNullBuilder, adaptive_detection_floor
from .rankcache import RankCache
from .scores import aucell, ucell
from .stats import (cutpoint_fpr, null_summary, optimum_cutpoint, spearman,
                    split_pvalue)
from .theory import preflight, score_decomposition

__all__ = ["DEFAULT_CONFIG", "construct_expression_matched_null",
           "compute_calibrated_score", "diagnostic_checklist",
           "validate_tier_framework", "Calibrator", "to_json",
           "MIN_BACKGROUND_GENES", "MIN_SCORE_SD", "SUSPICIOUS_AUC"]


#: Defaults for the whole framework.  They are the settings used in the
#: accompanying manuscript; change them and every reported rate changes with
#: them, which is why they are collected here rather than scattered as literals.
DEFAULT_CONFIG = dict(
    sparsity_threshold=0.50,       # above this zero rate, flag the set
    null_matching_tolerance=0.05,  # accepted relative gap in detection rate
    auc_significance_level=0.05,
    replication_cohorts=2,         # independent cohorts wanted for Tier 3
    cutpoint_permutations=1000,
    rank_frac=0.05,                # AUCell's default ranking fraction
    n_null=200,                    # matched sets drawn per family
    n_draws=25,                    # candidates per co-detection-matched set
    seed=42,
    background_loosen_factor=0.9,  # the "±10%" the brief's edge case one asks for
)

#: The fewest drawable background genes --- the pool outside the gene set itself
#: --- at which a matched null still means anything.  Below this, the draw is
#: choosing among so few candidates that "matched" stops describing it, and the
#: framework says so rather than returning a null that would be read as a test.
MIN_BACKGROUND_GENES = 10

#: A score whose spread across cells is below this cannot carry a correlation
#: with anything: the statistic is undefined, not merely imprecise.
MIN_SCORE_SD = 0.001

#: An AUC at or above this means the score separates the two groups almost
#: perfectly, which is a statement about the data rather than about the score.
SUSPICIOUS_AUC = 0.99


def _merge(config):
    out = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(out)
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        out.update(config)
    return out


def _check_score(score, label=None, context=""):
    """Refuse a score that has no spread to correlate with anything.

    Edge case three of the brief.  A set whose genes are never detected in any
    cell, or detected in every cell, produces the same number for every cell;
    the correlation with an axis is then 0/0 and the P value that comes out of
    it is an artefact of whichever convention the arithmetic library chose.
    Failing here, at the point the score is made, means the failure is attached
    to the gene set that caused it rather than appearing later as a NaN whose
    origin has to be traced back.
    """
    from .events import event

    sd = float(np.std(np.asarray(score, dtype=float)))
    if not np.isfinite(sd) or sd < MIN_SCORE_SD:
        event("error", "score_has_no_variance", label=label, context=context,
              sd=sd, threshold=MIN_SCORE_SD)
        raise ValueError(
            f"Gene set score has zero variance. Gene set may be all zeros or "
            f"all highly expressed; consider different genes. "
            f"(sd = {sd:.3g} < {MIN_SCORE_SD:g}"
            + (f", set {label!r}" if label else "")
            + (f", {context}" if context else "") + ")")
    return sd


def _check_auc(auc, label=None, context=""):
    """Flag a score that separates the groups almost perfectly.

    Edge case two of the brief.  An AUC this extreme is not evidence that the
    score is good; it is evidence that the score and the grouping share a
    source, and the brief names the three sources worth checking.  The warning
    is emitted rather than raised, because an AUC of 0.99 can also be real --
    a marker that marks what it is supposed to mark -- and a framework that
    refused to report it would be hiding the result rather than qualifying it.

    Both directions are checked.  The brief states the case as "AUC >= 0.99",
    but ``auroc`` here does not fold its argument about 0.5, and a score that
    lands every high-axis cell below every low-axis cell separates them just as
    perfectly as one that does the reverse.  Reading only the upper tail would
    have left exactly the confound the case exists to catch, undetected, in
    half the ways it can arise.  The record therefore carries both the AUC and
    the separation, so a reader can tell the two apart.
    """
    from .events import event

    if not np.isfinite(auc):
        return float(auc)
    separation = max(float(auc), 1.0 - float(auc))
    if separation < SUSPICIOUS_AUC:
        return float(auc)
    event("warning", "suspicious_score_separation", label=label, context=context,
          auc=float(auc), separation=separation, threshold=SUSPICIOUS_AUC,
          check=["batch_effects",
                 "outcome_leakage_into_cell_type_annotation",
                 "extreme_sparsity"])
    return float(auc)


# ----------------------------------------------------------------------
# step 2, as a standalone function
# ----------------------------------------------------------------------
def construct_expression_matched_null(cache, gene_set, n_sets=1,
                                      kind="expression", n_draws=25,
                                      tolerance=None, seed=None, ks=True,
                                      detection_matrix=None):
    """Draw replacement gene sets matched to ``gene_set``, and report the match.

    Parameters
    ----------
    cache : RankCache
    gene_set : sequence of str
    n_sets : int
        How many replacement sets to draw.  A P value needs a distribution, so
        one set is a demonstration and not a test.
    kind : {'random', 'expression_bin', 'expression', 'codetection'}
        The nesting is the point: each family controls something the previous
        one did not.  ``'codetection'`` needs ``detection_matrix``; without it
        there is no way to measure co-detection and the call raises rather than
        quietly falling back to a weaker family.
    detection_matrix : scipy.sparse matrix, optional
        Cells x ``cache.genes`` binary matrix.  Required for ``'codetection'``.
    tolerance : float, optional
        Accepted relative gap between the drawn set's mean detection and the
        observed set's.  Reported, and used for the ``status`` field; it does
        not constrain the draw, which is a nearest-neighbour match by
        construction.
    ks : bool
        Run a Kolmogorov-Smirnov test of the drawn sets' per-gene detection
        distribution against the observed set's.

    Returns
    -------
    dict
        ``null_genes`` (one list per drawn set), ``target_sparsity``,
        ``null_sparsity``, ``ks_statistic``, ``ks_pvalue``, ``status``, the
        per-set detection and expression ratios, and ``max_overlap`` /
        ``n_from_query`` -- how many genes the drawn sets share with the
        observation.  That last pair is measured rather than assumed: a null
        that reuses genes of the set it replaces has a score correlated with the
        observed one, and every P value drawn from it is conservative.

        ``status`` is ``'PASS'`` when the family's mean detection gap is within
        ``tolerance`` and no drawn set reuses a gene.  The gap is averaged over
        draws rather than maximised: a maximum can only grow with the number of
        draws, which made the same pool pass a small draw and fail a large one.
        ``mean_detection_gap``, ``worst_detection_gap`` and
        ``frac_within_tolerance`` are all reported, so a family that matches on
        average but has wild draws is visible as such.
    """
    from scipy import stats as st

    cfg = _merge(dict(tolerance=tolerance) if tolerance is not None else {})
    tol = cfg["null_matching_tolerance"] if tolerance is None else float(tolerance)

    pos = [cache._index[g] for g in gene_set if g in cache._index]
    if not pos:
        raise ValueError("none of the gene set is in the cache")

    # The detection floor is derived from the set rather than fixed: a fixed
    # floor above the set's own detection censors the pool from below and makes
    # every replacement denser than the gene it replaces.
    floor = adaptive_detection_floor(cache, gene_set)

    def _build(fl):
        return MatchedNullBuilder(cache, det_floor=fl,
                                  detection_matrix=detection_matrix,
                                  detection_matrix_genes=(cache.genes
                                                          if detection_matrix
                                                          is not None else None),
                                  seed=cfg["seed"] if seed is None else seed)

    def _available(builder):
        """Background genes left once the set's own genes are taken out.

        The query set is excluded from every draw by construction, so the
        candidates that can actually stand in for it are the pool minus the set
        -- and it is that number, not the pool size, that decides whether a
        matched null is possible.  A 400-gene set drawn from a 3,000-gene
        background has a large pool and almost nothing to draw from.
        """
        keep = np.setdiff1d(builder.pool, np.asarray(pos, dtype=int))
        return int(keep.size)

    try:
        builder = _build(floor)
    except ValueError as exc:
        # ``MatchedNullBuilder`` refuses a pool below 50 genes outright.  That
        # refusal is the same failure this block handles, one step earlier, so
        # it is reported as the brief's edge case rather than as the constructor
        # error it is raised as.
        raise ValueError(
            f"Cannot construct matched null. Consider different gene set. "
            f"At a detection floor of {floor:g} the background has too few "
            f"genes to draw a replacement from ({exc}).") from exc

    available = _available(builder)
    floor_loosened = False
    if available < MIN_BACKGROUND_GENES:
        # Edge case one of the brief, in this framework's terms.  The brief
        # states it as a matching tolerance and asks for it to be loosened by
        # 10%; here the tolerance that decides pool membership is the detection
        # floor, so that is the knob that is turned.  The loosening is announced
        # rather than performed quietly, because a reader of the calibration
        # object has to be able to see that the null they are reading was drawn
        # from a widened pool.
        floor_loosened = True
        loosened = float(floor) * float(cfg["background_loosen_factor"])
        from .events import event
        event("warning", "background_insufficient",
              n_available=available, required=MIN_BACKGROUND_GENES,
              det_floor=floor, loosened_to=loosened,
              action="loosening detection floor by "
                     f"{100 * (1 - cfg['background_loosen_factor']):.0f}%")
        try:
            builder = _build(loosened)
        except ValueError as exc:
            raise ValueError(
                f"Cannot construct matched null. Consider different gene set. "
                f"Loosening the detection floor from {floor:g} to {loosened:g} "
                f"left too few background genes to draw from ({exc}).") from exc
        available = _available(builder)
        if available < MIN_BACKGROUND_GENES:
            raise ValueError(
                f"Cannot construct matched null. Consider different gene set. "
                f"Only {available} background genes can replace this "
                f"{len(pos)}-gene set, and loosening the detection floor from "
                f"{floor:g} to {loosened:g} did not reach "
                f"{MIN_BACKGROUND_GENES}.")
        floor = loosened

    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    sets = builder.sample(gene_set, int(n_sets), kind=kind, rng=rng,
                          n_draws=n_draws)

    target_det = float(np.mean(cache.detection[pos]))
    drawn_det = np.array([np.mean(cache.detection[[cache._index[g] for g in s
                                                   if g in cache._index]])
                          for s in sets])
    drawn_expr = np.array([np.mean(cache.mean_expression[
        [cache._index[g] for g in s if g in cache._index]]) for s in sets])
    target_expr = float(np.mean(cache.mean_expression[pos]))

    out = dict(null_genes=sets if n_sets > 1 else sets[0],
                target_sparsity=1.0 - target_det,
                null_sparsity=1.0 - float(np.mean(drawn_det)),
                target_detection=target_det,
                null_detection=float(np.mean(drawn_det)),
                detection_ratio=float(np.mean(drawn_det) / target_det
                                      if target_det else np.nan),
                detection_ratio_sd=float(np.std(drawn_det / target_det))
                if target_det else np.nan,
                expression_ratio=float(np.mean(drawn_expr) / target_expr
                                       if target_expr else np.nan),
                det_floor=float(floor), n_sets=int(n_sets), kind=kind,
                # Whether the pool had to be widened to draw at all, and how
                # many genes were available to draw from.  A null built from a
                # loosened pool is not the same object as one built from the
                # pool the set's own detection implies, and a reader who cannot
                # tell them apart is reading a different test than they think.
                det_floor_loosened=bool(floor_loosened),
                n_available_background=available)

    if ks:
        d_stat, p_val = st.ks_2samp(cache.detection[pos],
                                    np.concatenate(
                                        [[cache.detection[cache._index[g]]
                                          for g in s if g in cache._index]
                                         for s in sets]))
        out.update(ks_statistic=float(d_stat), ks_pvalue=float(p_val))

    gaps = (np.abs(drawn_det / target_det - 1.0) if target_det
            else np.full(len(sets), np.nan))
    # The status is decided by the *family's* typical gap rather than by its
    # worst draw.  A maximum over draws can only grow as more are asked for, so
    # deciding on it made the same pool pass at 25 draws and fail at 100 -- the
    # diagnostic punished the better practice.  The worst gap is still reported
    # next to the fraction within tolerance, so a family with a few wild draws
    # is visible without being mistaken for one that does not match.
    mean_gap = float(np.nanmean(gaps)) if np.isfinite(gaps).any() else np.nan
    # A null that shares genes with the observation is not a null, so the overlap
    # is measured rather than assumed; the builder reports how often it had to
    # relax to place a replacement at all.
    query = set(gene_set)
    overlaps = [len(query & set(s)) for s in sets]
    stats = getattr(builder, "last_draw_stats", {})
    out["n_from_query"] = int(np.sum(overlaps))
    out["max_overlap"] = int(np.max(overlaps)) if overlaps else 0
    out["draw_relaxations"] = int(stats.get("n_from_query", 0))
    out["duplicate_picks"] = int(stats.get("n_duplicate", 0))
    out["pool_size"] = int(builder.pool.size)
    out["pool_floor"] = float(floor)
    out["mean_detection_gap"] = mean_gap
    out["worst_detection_gap"] = float(np.nanmax(gaps)) if np.isfinite(gaps).any() \
        else np.nan
    out["frac_within_tolerance"] = (float(np.mean(gaps <= tol))
                                    if np.isfinite(gaps).any() else np.nan)
    out["status"] = ("PASS" if np.isfinite(mean_gap) and mean_gap <= tol
                     and out["max_overlap"] == 0 else "FAIL")
    if not np.isfinite(mean_gap):
        out["status"] = "UNDEFINED"
        out["note"] = ("the gene set is not detected at all, so no replacement "
                       "can be matched to it")
    return out


# ----------------------------------------------------------------------
# the score, with its null alongside it
# ----------------------------------------------------------------------
def compute_calibrated_score(cache, gene_set, method="AUCell", null_genes=None,
                             axis=None, max_rank=None, ucell_r_max=None):
    """Score a set and, optionally, a matched null, on the same cells.

    The point of returning both on one frame is that the score is only
    interpretable next to the null: the difference between them is what the
    gene set adds over a set with the same composition.

    Parameters
    ----------
    cache : RankCache
    gene_set : sequence of str
    method : {'AUCell', 'UCell'}
    null_genes : sequence of str, optional
        A matched replacement set, as returned by
        :func:`construct_expression_matched_null`.
    axis : array-like, optional
        The proposed covariate; adds a correlation column pair.
    max_rank, ucell_r_max : int, optional

    Returns
    -------
    pandas.DataFrame
        One row per cell with ``score_target``, ``score_null`` (when supplied),
        ``depth`` and, when an axis is given, ``axis``.
    """
    import pandas as pd

    if max_rank is None:
        max_rank = int(np.ceil(0.05 * cache.n_genes_total))

    def _score(genes):
        if method == "AUCell":
            return aucell(cache, genes, max_rank=max_rank)
        if method == "UCell":
            r_max = ucell_r_max or cache.ceiling
            return ucell(cache, genes, r_max=r_max)
        raise ValueError(f"unknown method {method!r}")

    target = _score(gene_set)
    frame = pd.DataFrame(dict(score_target=target))
    if null_genes is not None:
        frame["score_null"] = _score(null_genes)
        frame["score_difference"] = frame["score_target"] - frame["score_null"]
    if axis is not None:
        frame["axis"] = np.asarray(axis, dtype=float)
    return frame


# ----------------------------------------------------------------------
# the checklist
# ----------------------------------------------------------------------
def diagnostic_checklist(cache, gene_set, axis=None, label=None, depth=None,
                         used_matched_null=None, cutpoint_method=None,
                         cutpoint_pre_specified=None, externally_replicated=None,
                         n_perm=None, config=None, seed=None):
    """The four-step diagnostic, as a single structured result.

    Parameters
    ----------
    cache : RankCache
    gene_set : sequence of str
    axis : array-like, optional
        The variable the score is to be tested against.  Without it, steps 3
        and 4 cannot run and are reported as not attempted.
    label : str, optional
        A name for the set, carried into the result.
    used_matched_null : bool, optional
        Whether the association was tested against a matched null.  Defaults to
        ``False``, which is what an unqualified claim amounts to.
    cutpoint_method : {None, 'optimum', 'median', 'pre_specified'}, optional
        If the analysis dichotomises the score, this is the rule it uses.
    cutpoint_pre_specified : float, optional
        Required for ``'pre_specified'``.
    externally_replicated : bool, optional
        Result of step 4, if it has been run elsewhere.
    n_perm : int, optional
        Permutations for the cutpoint calibration.
    config : dict, optional
        Overrides for :data:`DEFAULT_CONFIG`.

    Returns
    -------
    dict
        ``step1_sparsity``, ``step2_null_construction``,
        ``step3_false_positive_rate``, ``step4_external_replication``, plus
        ``overall_verdict`` and ``confidence``.

    Raises
    ------
    ValueError
        Two of the brief's edge cases stop the call rather than reporting a
        number that would be read as one.  A gene set whose score has no spread
        (edge case three) makes the correlation 0/0, and a background too thin
        to draw a replacement from (edge case one, via
        :func:`construct_expression_matched_null`) makes the null not a null.
        The third edge case --- a score that separates its groups almost
        perfectly --- is a warning and not a refusal, and travels in
        ``step3_false_positive_rate['score_auc']`` and in the log.

    Examples
    --------
    >>> import numpy as np, sparsegs as sg
    >>> X = np.abs(np.random.default_rng(0).normal(size=(60, 80)))
    >>> cache = sg.RankCache.build(X, [f"g{i}" for i in range(80)], ceiling=4)
    >>> out = sg.diagnostic_checklist(cache, [f"g{i}" for i in range(8)],
    ...                               axis=np.arange(60.0), n_perm=50,
    ...                               config={"n_null": 3})
    >>> out["step1_sparsity"]["severity"] in ("LOW", "MODERATE", "HIGH")
    True
    """
    cfg = _merge(config)
    seed = cfg["seed"] if seed is None else int(seed)
    n_perm = cfg["cutpoint_permutations"] if n_perm is None else int(n_perm)

    from .diagnostics import sparsity_report, verdict as _verdict

    diag = sparsity_report(cache, gene_set, rank_frac=cfg["rank_frac"])
    pos = [cache._index[g] for g in gene_set if g in cache._index]

    # ---- step 1: can this score carry a cell-level claim at all? ----
    z = diag["observed_zero_rate"]
    if z >= cfg["sparsity_threshold"]:
        sev, rec = "HIGH", "USE_MATCHED_NULL"
    elif z >= 0.20:
        sev, rec = "MODERATE", "USE_MATCHED_NULL"
    else:
        sev, rec = "LOW", "CONVENTIONAL_NULL_ACCEPTABLE"
    step1 = dict(percent_zero=float(100.0 * z),
                 structural_zero_rate=float(diag["structural_zero_rate"]),
                 zero_rate_excess=float(diag["zero_rate_excess"]),
                 n_present=int(diag["n_present"]), n_missing=int(diag["n_missing"]),
                 frac_above_floor=float(diag["frac_above_floor"]),
                 detection_floor=float(diag["detection_floor"]),
                 mean_detection=float(diag["mean_detection"]),
                 criterion=diag["criterion"],
                 severity=sev, recommendation=rec)
    # Where depth is known, the share of the set's inclusion that the tie-break
    # supplies is the exact version of what `frac_above_floor` approximates, so
    # it is carried into the checklist rather than left in the report.
    if "tie_break_share" in diag:
        step1["tie_break_share"] = float(diag["tie_break_share"])
        step1["mean_inclusion"] = float(diag["mean_inclusion"])

    # ---- step 2: is there a null that matches? ----
    built = construct_expression_matched_null(
        cache, gene_set, n_sets=cfg["n_null"], kind="expression", seed=seed)
    step2 = dict(null_pool_size=int(built["pool_size"]),
                 det_floor=built["det_floor"],
                 target_detection=built["target_detection"],
                 null_detection=built["null_detection"],
                 detection_ratio=built["detection_ratio"],
                 expression_ratio=built["expression_ratio"],
                 ks_test_pvalue=float(built.get("ks_pvalue", np.nan)),
                 ks_statistic=float(built.get("ks_statistic", np.nan)),
                 status=built["status"])

    # ---- step 3: what does the cutpoint rule actually cost? ----
    step3 = dict(attempted=False)
    if axis is not None and cutpoint_method is not None:
        score = aucell(cache, gene_set, max_rank=int(
            np.ceil(cfg["rank_frac"] * cache.n_genes_total)))
        _check_score(score, label=label, context="cutpoint calibration")
        outcome = np.asarray(axis) > float(np.median(axis))
        # Edge case two.  How well the score separates the two outcome groups is
        # the quantity the brief asks to be checked against 0.99, and it is
        # reported whether or not it crosses, so the record shows the number the
        # judgement was made on.  Checking it here rather than only in the tier
        # framework is what makes the two entry points agree: a caller who runs
        # the checklist and never calls `run_tiers` would otherwise be the one
        # caller the confound is hidden from.
        from .stats import auroc

        score_auc = _check_auc(auroc(score, outcome), label=label,
                               context="cutpoint calibration")
        fpr = cutpoint_fpr(score, outcome, n_perm=n_perm,
                           method=cutpoint_method,
                           pre_specified=cutpoint_pre_specified, seed=seed)
        base = cutpoint_fpr(score, outcome, n_perm=n_perm, method="median",
                            seed=seed)
        step3 = dict(attempted=True, method=cutpoint_method,
                     score_auc=score_auc,
                     optimum_cutpoint_FPR=float(fpr["fpr"]),
                     median_cutpoint_FPR=float(base["fpr"]),
                     pre_specified_cutpoint_FPR=(
                         float(fpr["fpr"]) if cutpoint_method == "pre_specified"
                         else np.nan),
                     observed_p=float(fpr["observed_p"]),
                     n_perm=int(fpr["n_perm"]),
                     recommendation=("USE_PRESPECIFIED"
                                     if fpr["fpr"] > 2 * cfg["auc_significance_level"]
                                     else "OK"))
    elif axis is not None:
        step3 = dict(attempted=False,
                     note="no cutpoint rule was declared; step 3 measures the "
                          "rule, and an undeclared rule is the outcome-driven one")

    # ---- step 4: does it hold anywhere else? ----
    step4 = dict(attempted=externally_replicated is not None,
                 replication_cohorts=int(externally_replicated is not None),
                 success=(None if externally_replicated is None
                          else bool(externally_replicated)),
                 status=("NOT_ATTEMPTED" if externally_replicated is None
                         else "PASS" if externally_replicated else "FAIL"))

    # ---- the pre-flight, then the verdict ----
    # The pre-flight check needs the per-cell detected counts and nothing else,
    # and it is computed *before* the verdict rather than attached afterwards,
    # because it is an input to the verdict.  It used to be attached only: the
    # checklist returned `preflight` and the verdict never read it, so a gene
    # set that was sparse-free in a matrix whose axis tracked sequencing depth
    # came back INTERPRETABLE -- the confound the framework exists to catch,
    # reported as a clean pass by the framework itself.  Found by scoring the
    # checklist against the regimes where the conventional test is
    # anticonservative, where it was flagging fewer of them than of the
    # well-behaved ones.
    pre = None
    if depth is None:
        # The cache carries D_c for every cell it was built from, so the
        # pre-flight is available without the caller supplying anything.  The R
        # implementation of this framework has always defaulted here; this one
        # only ever computed the pre-flight when `depth` was passed explicitly,
        # which meant the same call on the same data returned an object with
        # `preflight` in one language and without it in the other.
        depth = cache.depth
    if axis is not None and depth is not None:
        pre = preflight(depth, axis,
                        int(np.ceil(cfg["rank_frac"] * cache.n_genes_total)),
                        cache.n_genes_total,
                        detection=cache.detection[pos], k=len(pos))

    # ---- the verdict ----
    v = _verdict(diag, used_matched_null=bool(used_matched_null),
                 outcome_driven_cutpoint=(cutpoint_method == "optimum"),
                 externally_replicated=externally_replicated, preflight=pre)
    out = dict(
        label=label, n_cells=int(cache.n_cells), n_genes_total=int(cache.n_genes_total),
        max_rank=int(np.ceil(cfg["rank_frac"] * cache.n_genes_total)),
        step1_sparsity=step1, step2_null_construction=step2,
        step3_false_positive_rate=step3, step4_external_replication=step4,
        overall_verdict=v["verdict"], severity=v["severity"],
        confidence=("HIGH" if step1["severity"] == "HIGH" else "MODERATE"),
        reasons=v["reasons"], config=dict(cfg, seed=seed, n_perm=n_perm,
                                          n_null=cfg["n_null"]))
    if pre is not None:
        out["preflight"] = pre
    return out


# ----------------------------------------------------------------------
# the tiers
# ----------------------------------------------------------------------
def validate_tier_framework(discovery, replication=None, tier1=None,
                            cutpoint_specification="pre_specified"):
    """Assemble the three validation tiers into one result.

    Thin wrapper over :func:`sparsegs.run_tiers` that also carries the cutpoint
    specification through, so that the tier record states which rule was used
    rather than leaving it to the prose around it.
    """
    from .tiers import run_tiers

    out = run_tiers(discovery, replication=replication, tier1=tier1)
    out["cutpoint_specification"] = cutpoint_specification
    out["tiers"] = {
        "TIER_1": out.get("tier1"),
        "TIER_2": out.get("tier2"),
        "TIER_3": out.get("tier3"),
    }
    return out


# ----------------------------------------------------------------------
# the object
# ----------------------------------------------------------------------
class Calibrator:
    """A matrix, its rank cache, and the checklist that goes with them.

    Building the cache once and reusing it is what makes the checklist
    affordable: a sweep over hundreds of gene sets pays for the ranks once.

    Parameters
    ----------
    cache : RankCache
    detection_matrix : scipy.sparse matrix, optional
        Cells x cache-genes binary matrix, needed for co-detection matching.
    config : dict, optional
    """

    def __init__(self, cache, detection_matrix=None, config=None):
        self.cache = cache
        self.config = _merge(config)
        self._detection = detection_matrix

    # --------------------------------------------------------------
    @classmethod
    def from_matrix(cls, X, genes, detection_matrix=None, rank_frac=0.05,
                    ceiling=None, genes_of_interest=None, config=None,
                    seed=None, chunk=1000, verbose=False):
        """Build a calibrator from an expression matrix.

        ``ceiling`` defaults to the rank ceiling with headroom, because a cache
        built exactly at ``max_rank`` cannot serve a UCell score, which needs its
        own ``r_max``.
        """
        cfg = _merge(dict(config) if config else {})
        rank_frac = float(rank_frac)
        max_rank = int(np.ceil(rank_frac * X.shape[1]))
        ceiling = ceiling or int(min(X.shape[1] - 1, max(max_rank + 1, 1500)))
        cache = RankCache.build(
            X, genes, ceiling=ceiling, genes_of_interest=genes_of_interest,
            seed=cfg["seed"] if seed is None else int(seed), chunk=chunk,
            verbose=verbose)
        if detection_matrix is None:
            keep = np.array([list(np.asarray(genes)).index(g)
                             for g in cache.genes], dtype=np.int64)
            detection_matrix = sp.csr_matrix(sp.csr_matrix(X)[:, keep] > 0)
        return cls(cache, detection_matrix=detection_matrix, config=cfg)

    # --------------------------------------------------------------
    def matched_null(self, gene_set, n_sets=1, kind="expression", seed=None,
                     ks=True, detection_matrix=None):
        """See :func:`construct_expression_matched_null`."""
        if detection_matrix is None:
            detection_matrix = self._detection
        if kind == "codetection" and detection_matrix is None:
            raise RuntimeError(
                "co-detection matching needs a detection matrix; build the "
                "calibrator with one, or use kind='expression'")
        return construct_expression_matched_null(
            self.cache, gene_set, n_sets=n_sets, kind=kind,
            n_draws=self.config["n_draws"], seed=seed, ks=ks,
            detection_matrix=detection_matrix)

    def calibrated_score(self, gene_set, method="AUCell", null_genes=None,
                         axis=None, ucell_r_max=None):
        """See :func:`compute_calibrated_score`."""
        return compute_calibrated_score(
            self.cache, gene_set, method=method, null_genes=null_genes,
            axis=axis, ucell_r_max=ucell_r_max)

    def checklist(self, gene_set, axis=None, depth=None, **kwargs):
        """See :func:`diagnostic_checklist`."""
        kwargs.setdefault("config", self.config)
        if depth is None:
            # The cache records each cell's depth as it builds, so the exact
            # criterion is available without the detection matrix.  Fall back to
            # the matrix only for caches written before depth was recorded.
            depth = self.cache.depth
        if depth is None and self._detection is not None:
            depth = np.asarray(self._detection.sum(axis=1)).ravel()
        return diagnostic_checklist(self.cache, gene_set, axis=axis,
                                    depth=depth, **kwargs)

    def decomposition(self, gene_set):
        """See :func:`sparsegs.theory.score_decomposition`."""
        if self._detection is None:
            raise RuntimeError("no detection matrix was supplied")
        return score_decomposition(self.cache, gene_set, self._detection)

    # --------------------------------------------------------------
    def calibration_object(self, gene_set, axis=None, replication=None,
                           label=None, **kwargs):
        """The whole calibration as one JSON-serialisable dict.

        ``replication`` is a dict with ``cohort_name``, ``observed_rho`` and
        ``n_cells``; supplying it fills in Tier 3 and turns the result into a
        complete record of the analysis rather than of the discovery cohort
        alone.
        """
        checklist = self.checklist(gene_set, axis=axis, label=label, **kwargs)
        max_rank = checklist["max_rank"]
        score = aucell(self.cache, gene_set, max_rank=max_rank)

        # The three rules the attachment's object asks for, measured rather than
        # assumed.  `optimum` re-selects a cutpoint in every permutation, which
        # is what makes it anticonservative; `median` and a caller-supplied
        # pre-specified value do not, and the gap between the three is the
        # quantity worth reporting.
        fpr = dict(optimum_cutpoint_FPR=np.nan, median_split_FPR=np.nan,
                   pre_specified_cutpoint_FPR=np.nan)
        if axis is not None:
            n_perm = int(kwargs.get("n_perm")
                         or self.config["cutpoint_permutations"])
            outcome = np.asarray(axis, dtype=float) > float(np.median(axis))
            seed = int(self.config["seed"])
            fpr["optimum_cutpoint_FPR"] = float(
                cutpoint_fpr(score, outcome, n_perm=n_perm, method="optimum",
                             seed=seed)["fpr"])
            fpr["median_split_FPR"] = float(
                cutpoint_fpr(score, outcome, n_perm=n_perm, method="median",
                             seed=seed)["fpr"])
            pre = kwargs.get("cutpoint_pre_specified")
            if pre is not None:
                fpr["pre_specified_cutpoint_FPR"] = float(
                    cutpoint_fpr(score, outcome, n_perm=n_perm,
                                 method="pre_specified", pre_specified=pre,
                                 seed=seed)["fpr"])

        disc = dict(n_cells=int(score.size), observed_rho=np.nan,
                    association_pval=np.nan)
        if axis is not None:
            rho, _ = spearman(score, axis)
            high = np.asarray(axis, dtype=float) > float(np.median(axis))
            disc.update(
                score_auc=float(_auc(score, high)),
                association_rho=float(rho),
                association_pval=float(_two_sided_from_spearman(rho, score.size)))
        _check_score(score, label=label, context="discovery")
        if np.isfinite(disc.get("score_auc", np.nan)):
            _check_auc(disc["score_auc"], label=label, context="discovery")

        rep = dict(attempted=False)
        if replication is not None:
            rho_rep = float(replication["observed_rho"])
            n_rep = int(replication["n_cells"])
            p_rep = _two_sided_from_spearman(rho_rep, n_rep)
            same_sign = (np.sign(rho_rep) == np.sign(disc["association_rho"])
                         if np.isfinite(disc["association_rho"]) else None)
            rep = dict(attempted=True,
                       cohort_name=replication.get("cohort_name", "unnamed"),
                       score_auc=float(replication.get("score_auc", np.nan)),
                       association_rho=rho_rep,
                       association_pval=float(p_rep),
                       n_cells=n_rep,
                       same_sign=same_sign,
                       replication_success=bool(same_sign and p_rep < 0.05))

        tiers = validate_tier_framework(
            discovery=dict(
                observed_rho=disc["association_rho"],
                n_cells=disc["n_cells"],
                # Tier 2 asks whether the association survives a matched
                # null.  The score alone cannot answer that, so the tier is
                # passed in from the checklist's own step 2 rather than
                # assumed from the score's P value.
                passed=bool(checklist["step2_null_construction"]["status"]
                            == "PASS")),
            replication=(dict(observed_rho=rep["association_rho"],
                              n_cells=rep["n_cells"]) if rep["attempted"]
                         else None))

        return dict(
            gene_set=list(gene_set), label=label,
            discovery=disc, replication=rep, tiers=tiers,
            checklist=checklist,
            false_positive_rates=fpr,
            software=dict(package="sparsegs", version=_version()))

    # --------------------------------------------------------------
    def report(self, gene_set, path, title=None, axis=None, n_null=100,
               null_kind=None, dataset=None, cell_type=None,
               notes=None, **kwargs):
        """Write the self-contained HTML report for one gene set.

        Everything the report shows is computed here rather than passed in, so
        that the file is a record of one analysis rather than of whatever the
        caller happened to have to hand.  The matched nulls are drawn at
        ``n_null`` sets of ``null_kind``; the report's figure is a comparison, so
        the number only needs to be large enough to show a distribution.

        Parameters
        ----------
        gene_set : sequence of str
        path : str
            Output ``.html`` path.
        title : str, optional
        axis : array-like, optional
            If supplied, the association is tested and the cutpoint rules are
            calibrated; without it steps 3 and 4 cannot run.
        n_null : int
        null_kind : {None, 'random', 'expression_bin', 'expression',
            'codetection'}
            ``None`` takes the strongest family the object can support:
            co-detection where a detection matrix was supplied, expression
            otherwise.  The family actually used is the label on the figure.
        dataset, cell_type, notes : str, optional
            Provenance carried into the header.

        Returns
        -------
        str
            ``path``.
        """
        from .diagnostics import sparsity_report, verdict as _verdict
        from .report import render_report

        cfg = self.config
        max_rank = int(np.ceil(cfg["rank_frac"] * self.cache.n_genes_total))
        present = self.cache.present(gene_set)
        if not present:
            raise ValueError("none of the gene set is in the cache")
        if null_kind is None:
            null_kind = ("codetection" if self._detection is not None
                         else "expression")

        diag = sparsity_report(self.cache, gene_set,
                               rank_frac=cfg["rank_frac"])
        score = aucell(self.cache, present, max_rank=max_rank)

        observed = None
        nulls = None
        if axis is not None:
            observed = float(spearman(score, axis)[0])
            built = construct_expression_matched_null(
                self.cache, present, n_sets=int(n_null), kind=null_kind,
                n_draws=cfg["n_draws"], seed=cfg["seed"],
                detection_matrix=self._detection)
            sets = built["null_genes"]
            if not isinstance(sets[0], (list, tuple)):
                sets = [sets]
            drawn = np.array([spearman(aucell(self.cache, s, max_rank=max_rank),
                                       axis)[0] for s in sets])
            nulls = {null_kind: drawn[np.isfinite(drawn)]}

        cutpoint_rows = None
        if axis is not None:
            outcome = np.asarray(axis, dtype=float) > float(np.median(axis))
            cutpoint_rows = [
                (label, cutpoint_fpr(score, outcome, n_perm=cfg["cutpoint_permutations"],
                                     method=method, seed=cfg["seed"])["fpr"])
                for label, method in (("Outcome-chosen", "optimum"),
                                      ("Median split", "median"))]

        v = _verdict(diag, used_matched_null=nulls is not None and
                     built["status"] == "PASS",
                     outcome_driven_cutpoint=False)

        written = render_report(
            path, title=title or "Gene-set score calibration",
            gene_set=list(gene_set), dataset=dataset, cell_type=cell_type,
            diagnostics=diag, verdict=v, nulls=nulls, observed=observed,
            score=score, cutpoint_rows=cutpoint_rows, notes=notes)
        return str(written)


def _auc(score, groups):
    """AUROC of a score against a binary grouping, computed directly."""
    from .stats import auroc
    groups = np.asarray(groups).astype(bool)
    if groups.all() or (~groups).all():
        return np.nan
    return auroc(score, groups)


def _version():
    from . import __version__
    return __version__


def to_json(obj, path):
    """Write a calibration object to disk, with numpy scalars made JSON-safe."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_jsonable)
    return path


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
