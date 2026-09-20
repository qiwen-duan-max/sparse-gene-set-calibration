"""Two conveniences that read like the paper's decision table.

``select_null_method`` returns the branch a set's measured quantities put it
on, with the thresholds it applied echoed in the result.  The manuscript's
decision table (Supplementary Table 5) is generated from the same constants
this module imports, so the function and the table cannot disagree.

``predict_fpr`` returns the conventional test's rejection probability under a
true null, given the magnitude of the implied product.  It is a logistic fit,
pooled over the calibration grids' null runs, and the record it returns
carries the fit's provenance so the number cannot be mistaken for anything
measured fresh.

Both are pure functions of their arguments: no matrix, no cache, no scoring.
They sit on top of the diagnostics, and they are honest about what they do
not know -- a quantity that was not supplied is reported as unevaluated, not
assumed quiet.

References
----------
The fit's source and the boundary rules' provenance are recorded in the
constants below, and tests assert that this module's constants still match
the files they were computed from.
"""

from __future__ import annotations

import numpy as np

from .calibrate import DEFAULT_CONFIG
from .diagnostics import DEFAULT_THRESHOLDS

__all__ = ["FPR_MODEL", "CO_DETECTION_GAP_FLAG", "CO_DETECTION_GAP_REFUSE",
           "DEPTH_LINKED_AXIS", "select_null_method", "predict_fpr"]

#: Logistic fit of the conventional cell-level test's rejection under a true
#: null on the magnitude of the implied product, pooled over the null runs of
#: grids A (calibration) and B (regime): n = 1,680, model "product alone" in
#: ``results/grid_A_B_joint_logistic.csv``.  The predictors were standardised
#: on the fit set, so the mean and SD travelled with the coefficients; a test
#: recomputes both from the grid files and compares.
FPR_MODEL = dict(
    intercept=-0.07245436534023197,
    coef_per_sd=4.166465012123036,
    mean_abs_implied=0.0512505673509389,
    sd_abs_implied=0.06159690289425014,
    n_runs=1680,
    grids=("A_calibration", "B_regime"),
    test="conventional cell-level Spearman, alpha=0.05, effect=0",
    source="results/grid_A_B_joint_logistic.csv, model 'product alone'",
    fitted_range=(0.0, 0.2969),
)

#: The co-detection gap rules the grid's boundary analysis uses
#: (``BOUNDARIES`` in ``experiments/make_numbers.py``): below the flag the
#: matched families are nominal on the same draws; above the refusal line, and
#: with a depth-linked axis, no family is safe.  A test pins these to the
#: analysis constants so the package and the grid rules cannot drift.
CO_DETECTION_GAP_FLAG = 0.005
CO_DETECTION_GAP_REFUSE = 0.020
DEPTH_LINKED_AXIS = 0.2

_ACTIONS = ("conventional_null_ok", "expression_matched_null",
            "refuse_not_identifiable", "refuse_no_null_family_safe")
#: Severity order for the escalation, matching the diagnostics module: a
#: refusal beats a flag, and a flag beats a recommendation.  The two refusals
#: carry the same severity -- they are both terminal -- and the step order
#: decides which label a set gets: the checklist evaluates sparsity before
#: null construction, so a set too sparse to test is reported as such even
#: when the boundary would also have refused it.
_SEVERITY = {"conventional_null_ok": 0, "expression_matched_null": 1,
             "refuse_not_identifiable": 2, "refuse_no_null_family_safe": 2}


def _escalate(current, proposed):
    return proposed if _SEVERITY[proposed] > _SEVERITY[current] else current


def select_null_method(tie_break_share=None, implied_rho_abs=None,
                       chance_band=None, codetection_gap=None,
                       axis_depth_rho=None, config=None, thresholds=None):
    """The checklist's decision, from quantities the caller already measured.

    Parameters
    ----------
    tie_break_share : float, optional
        Step 1's exact quantity: the share of the set's inclusion rate that
        comes from tie-breaking.
    implied_rho_abs : float, optional
        The magnitude of the pre-flight implied product.
    chance_band : float, optional
        The chance band to read the implied product against (the package's
        :func:`~sparsegs.preflight` returns it as ``chance_band``).  Without
        it the pre-flight step is reported as unevaluated.
    codetection_gap : float, optional
        The achieved co-detection gap of the best matched draw (negative when
        the draws are more co-detected than the query; the rules read the
        magnitude of the shortfall, so a negative gap never flags).
    axis_depth_rho : float, optional
        The correlation of the tested axis with the cells' detected-gene
        count.
    config, thresholds : dict, optional
        Overrides for the package defaults, as elsewhere in the framework.

    Returns
    -------
    dict
        ``recommended_action`` -- one of ``conventional_null_ok``,
        ``expression_matched_null``, ``refuse_not_identifiable``,
        ``refuse_no_null_family_safe`` -- with ``steps`` reporting what each
        supplied quantity said, ``thresholds_applied`` echoing the constants
        that were used, and ``notes`` carrying anything a reader of the
        output should know (the co-detection flag, the unevaluated steps,
        the cutpoint reminder when an action other than the conventional one
        is recommended).
    """
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    th = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    steps = {}
    notes = []
    action = "conventional_null_ok"

    # Pre-flight: the channel, from the depth vector alone.
    if implied_rho_abs is None:
        steps["preflight"] = "not_evaluated"
    elif chance_band is None:
        steps["preflight"] = "not_evaluated"
        notes.append("implied product supplied without a chance band; "
                     "preflight() computes the band")
    else:
        open_channel = implied_rho_abs > chance_band
        steps["preflight"] = ("channel_open" if open_channel
                              else "channel_closed")
        if open_channel:
            action = _escalate(action, "expression_matched_null")

    # Step 1: sparsity, by the exact share.
    if tie_break_share is None:
        steps["sparsity"] = "not_evaluated"
    elif tie_break_share >= th["tie_break_share_high"]:
        steps["sparsity"] = "tie_break_share_high"
        action = _escalate(action, "refuse_not_identifiable")
    elif tie_break_share >= th["tie_break_share_moderate"]:
        steps["sparsity"] = "tie_break_share_moderate"
        action = _escalate(action, "expression_matched_null")
    else:
        steps["sparsity"] = "below_moderate"

    # The boundary: where matching on composition stops being enough.  The
    # gap is a shortfall of the draws below the query, so only a positive
    # gap, and only on a depth-linked axis, can flag.
    if codetection_gap is None or axis_depth_rho is None:
        steps["codetection"] = "not_evaluated"
    else:
        steps["codetection"] = f"gap={codetection_gap:.4f}"
        if (codetection_gap > CO_DETECTION_GAP_REFUSE
                and axis_depth_rho > DEPTH_LINKED_AXIS):
            action = _escalate(action, "refuse_no_null_family_safe")
            notes.append("co-detection gap beyond the refusal line on a "
                         "depth-linked axis; no null family is safe here")
        elif (codetection_gap > CO_DETECTION_GAP_FLAG
                and axis_depth_rho > DEPTH_LINKED_AXIS):
            notes.append("co-detection gap beyond the flag line on a "
                         "depth-linked axis; prefer the expression-matched "
                         "family and report the achieved gap")

    if action != "conventional_null_ok":
        notes.append("if the analysis dichotomises the score, measure the "
                     "rule's false-positive rate by permutation rather than "
                     "assuming the nominal level")

    return dict(recommended_action=action,
                steps=steps,
                thresholds_applied=dict(
                    tie_break_share_moderate=th["tie_break_share_moderate"],
                    tie_break_share_high=th["tie_break_share_high"],
                    null_matching_tolerance=cfg["null_matching_tolerance"],
                    co_detection_gap_flag=CO_DETECTION_GAP_FLAG,
                    co_detection_gap_refuse=CO_DETECTION_GAP_REFUSE,
                    depth_linked_axis=DEPTH_LINKED_AXIS),
                notes=notes)


def predict_fpr(implied_rho_abs, model=None):
    """The conventional test's rejection probability under a true null.

    Parameters
    ----------
    implied_rho_abs : float
        The magnitude of the implied product: how strongly the score tracks
        depth times how strongly the axis does.  :func:`~sparsegs.preflight`
        computes it from the depth vector and the detection profile before
        any score exists.
    model : dict, optional
        Override the shipped fit (mainly for the cross-language test).

    Returns
    -------
    dict
        ``p_reject`` -- the fitted probability -- with ``within_fitted_range``
        saying whether the input sits inside the range the fit saw, and the
        fit's provenance echoed under ``model``.  The fit is an interpolation
        over the calibration grids' null runs; outside that range the number
        is an extrapolation and ``within_fitted_range`` says so rather than
        hiding it.
    """
    m = dict(FPR_MODEL, **(model or {}))
    z = (implied_rho_abs - m["mean_abs_implied"]) / m["sd_abs_implied"]
    logit = m["intercept"] + m["coef_per_sd"] * z
    p_reject = float(1.0 / (1.0 + np.exp(-logit)))
    lo, hi = m["fitted_range"]
    return dict(p_reject=p_reject,
                within_fitted_range=bool(lo <= implied_rho_abs <= hi),
                model=dict(source=m["source"], n_runs=m["n_runs"],
                           grids=list(m["grids"]), test=m["test"]))
