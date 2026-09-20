"""Sparse gene-set scoring calibration.

Gene-set scores computed on sparse single-cell data are routinely interpreted
against a null that the data cannot support.  This package supplies the pieces
needed to replace that null with one the data can support, and to find out
before running anything whether it will matter.

The mechanism it is built around
-------------------------------
Every rank-based score (AUCell, UCell and relatives) asks where a gene sits
among *all* genes in a cell.  A cell that expresses fewer genes than the score's
rank ceiling has no count to place in the remaining positions, and the tie-break
fills them with genes the cell never expressed.  The expected share of the score
contributed by one gene the cell did not express is

    tau(D) = (m - D)(m - D + 1) / (2 m (G - D)),

with ``m`` the rank ceiling, ``G`` the panel size and ``D`` the number of genes
the cell detected -- a function of the cell's depth, not of the gene set.  Depth
tracks almost every biological axis one cares about, so that term is a
confound; :mod:`sparsegs.theory` derives it, :func:`~sparsegs.tie_break_level`
evaluates it, and :func:`~sparsegs.preflight` turns it into a check that can be
run on the depth vector alone, before any score exists.

What is here
------------
:class:`Calibrator` and :mod:`sparsegs.calibrate`
    The assembled surface: the four-step checklist, the matched null, and a
    JSON-serialisable calibration object.
:class:`RankCache`
    Ranks computed once and clipped at a ceiling, so that a calibration sweep
    over hundreds of gene sets is affordable.  ``build_streaming`` never
    materialises the matrix.
:func:`aucell`, :func:`ucell` and the rest
    Scoring rules over the cache.
:class:`MatchedNullBuilder`
    Three nested null families -- uniform, expression-matched, and
    co-detection-matched -- each reporting the composition it achieved.
:func:`sparsity_report`, :func:`verdict`
    Pre-flight diagnostics and the decision they support.
:mod:`sparsegs.analyses`
    The four conventional analyses the framework is calibrated against, so that
    all seven can be compared on one dataset.
:mod:`sparsegs.tiers`
    Technical, internal and external validation, with the tiers regime-matched.
:mod:`sparsegs.comparators`
    Head-to-head against ``decoupler``'s methods, called through the real
    package rather than reimplemented.
:mod:`sparsegs.report`
    A self-contained HTML report for one gene set.

Quick start
-----------
>>> import sparsegs as sg
>>> cal = sg.Calibrator.from_matrix(X, genes, rank_frac=0.05, seed=42)
>>> out = cal.calibration_object(CLOCK_REPRESSORS, axis=exhaustion)
>>> out["checklist"]["overall_verdict"]
'NOT_IDENTIFIABLE'

>>> out["checklist"]["step1_sparsity"]["percent_zero"]
67.3

>>> out["false_positive_rates"]["optimum_cutpoint_FPR"]
0.44

The verdict is meant to be read before the analysis rather than after it: step 1
is free, and it is the step that decides whether the remaining three can say
anything at all.

See :mod:`sparsegs.theory` for the derivation the calibration rests on,
:mod:`sparsegs.rankcache` for why the cache exists, :mod:`sparsegs.nulls` for
what the different nulls assume, and :mod:`sparsegs.diagnostics` for the
pre-flight checks.
"""

from .rankcache import RankCache
from .scores import aucell, ucell, score_genes, module_score, ssgsea, score_all
from .nulls import (MatchedNullBuilder, compare_nulls,
                    adaptive_detection_floor)
from .stats import (spearman, auroc, empirical_p, percentile_of, null_summary,
                    optimum_cutpoint, split_pvalue, cutpoint_fpr, dt50)
from .diagnostics import (structural_zero_rate, sparsity_report, depth_report,
                          comparability_report, verdict, DEFAULT_THRESHOLDS,
                          SEVERITY_ORDER, raise_severity)
from .theory import (tie_break_inclusion, tie_break_level, effective_inclusion,
                     tie_break_share, score_decomposition, preflight,
                     confound_band)
from . import (analyses, calibrate, comparators, convenience, diagnostics,
               events, nulls, rankcache, report, scores, stats, theory,
               tiebreak, tiers)
from .tiebreak import tie_break_keys, MASK32
from .calibrate import (DEFAULT_CONFIG, Calibrator,
                        construct_expression_matched_null,
                        compute_calibrated_score, diagnostic_checklist,
                        validate_tier_framework, to_json,
                        MIN_BACKGROUND_GENES, MIN_SCORE_SD, SUSPICIOUS_AUC)
from .events import set_log_level
from .convenience import (select_null_method, predict_fpr, FPR_MODEL,
                          CO_DETECTION_GAP_FLAG, CO_DETECTION_GAP_REFUSE,
                          DEPTH_LINKED_AXIS)
from .analyses import (naive_test, permutation_test, cutpoint_test,
                       all_conventional, ALPHA)
from .tiers import (tier1_technical, tier2_internal, tier3_external, run_tiers,
                    REGIME_TOLERANCE)
from .report import render_report

__version__ = "0.1.0"

__all__ = [
    # scoring
    "RankCache",
    "aucell", "ucell", "score_genes", "module_score", "ssgsea", "score_all",
    # nulls
    "MatchedNullBuilder", "compare_nulls", "adaptive_detection_floor",
    # statistics
    "spearman", "auroc", "empirical_p", "percentile_of", "null_summary",
    "optimum_cutpoint", "split_pvalue", "cutpoint_fpr", "dt50",
    # diagnostics
    "structural_zero_rate", "sparsity_report", "depth_report",
    "comparability_report", "verdict", "DEFAULT_THRESHOLDS",
    "SEVERITY_ORDER", "raise_severity",
    # the closed form the calibration is built on
    "tie_break_inclusion", "tie_break_level", "effective_inclusion",
    "tie_break_share", "score_decomposition", "preflight", "confound_band",
    # the tie-break itself, which the R port has to reproduce exactly
    "tie_break_keys", "MASK32",
    # the four conventional analyses the framework is calibrated against
    "naive_test", "permutation_test", "cutpoint_test", "all_conventional",
    "ALPHA",
    # validation tiers and the report
    "tier1_technical", "tier2_internal", "tier3_external", "run_tiers",
    "REGIME_TOLERANCE", "render_report",
    # the assembled calibration surface
    "DEFAULT_CONFIG", "Calibrator", "construct_expression_matched_null",
    "compute_calibrated_score", "diagnostic_checklist",
    "validate_tier_framework", "to_json",
    # the edge cases the framework refuses or warns on, and their thresholds
    "set_log_level", "MIN_BACKGROUND_GENES", "MIN_SCORE_SD", "SUSPICIOUS_AUC",
    # the decision table and the fitted rejection probability, as functions
    "select_null_method", "predict_fpr", "FPR_MODEL",
    "CO_DETECTION_GAP_FLAG", "CO_DETECTION_GAP_REFUSE", "DEPTH_LINKED_AXIS",
    # submodules
    "analyses", "calibrate", "comparators", "diagnostics", "events", "nulls",
    "rankcache", "report", "scores", "stats", "theory", "tiebreak", "tiers",
    "__version__",
]

