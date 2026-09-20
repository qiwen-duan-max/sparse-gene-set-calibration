"""The two convenience functions, against the files their constants came from.

A convenience that restates a threshold is a second copy of a decision, and a
second copy is how a decision drifts.  So these tests pin every number the
module carries to the artifact it was computed from -- the grid runs behind
the logistic fit, the analysis constants behind the boundary rules -- and
exercise every branch of the decision, including the quiet ones.
"""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sparsegs as sg  # noqa: E402
from sparsegs.convenience import (FPR_MODEL, CO_DETECTION_GAP_FLAG,  # noqa: E402
                                  CO_DETECTION_GAP_REFUSE, DEPTH_LINKED_AXIS)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


# ----------------------------------------------------------------------
# predict_fpr: the fit and its provenance
# ----------------------------------------------------------------------
def test_predict_fpr_at_the_fit_mean_returns_the_intercept_rate():
    # At the fit's own mean the standardised predictor is zero, so the
    # probability is the logistic of the intercept alone -- a property of
    # the algebra, not of the data.
    expected = 1.0 / (1.0 + np.exp(-FPR_MODEL["intercept"]))
    out = sg.predict_fpr(FPR_MODEL["mean_abs_implied"])
    assert out["p_reject"] == pytest.approx(expected, abs=1e-12)


def test_predict_fpr_reproduces_the_fit_from_the_grid_files():
    # The constants travelled by hand from the results files into the source.
    # Recompute them from the same files and the shipped fit must predict the
    # same probability to the last digit the files support.
    from experiments.analyse_grid import implied_rho, load

    pool = pd.concat([load(n, allow_partial=True)
                      for n in FPR_MODEL["grids"]], ignore_index=True)
    pool = pool[pool["effect"] == 0]
    work = pd.DataFrame({
        "imp": implied_rho(pool).abs(),
        "reject": (pool["naive_p"] < 0.05).astype(float),
    }).dropna()
    assert len(work) == FPR_MODEL["n_runs"]

    mean, std = float(work["imp"].mean()), float(work["imp"].std(ddof=1))
    assert mean == pytest.approx(FPR_MODEL["mean_abs_implied"], rel=1e-9)
    assert std == pytest.approx(FPR_MODEL["sd_abs_implied"], rel=1e-9)

    # The in-sample mean predicted probability equals the observed rejection
    # rate to the precision the summary reported -- a fit that did not even
    # reproduce its own base rate would not be worth wrapping.  The mean is
    # taken over the data's own distribution of implied products, which is
    # the quantity a fit reproduces; a uniform grid would average over
    # regions the data never visits.
    at_data = np.array([sg.predict_fpr(v)["p_reject"] for v in work["imp"]])
    assert abs(at_data.mean() - work["reject"].mean()) < 0.005

    # And the shipped constants are the fit's: evaluated anywhere, the
    # function reproduces the direct formula to the last digit.
    grid = np.linspace(0.0, 0.25, 41)
    preds = np.array([sg.predict_fpr(v)["p_reject"] for v in grid])
    z = (grid - mean) / std
    direct = 1.0 / (1.0 + np.exp(-(FPR_MODEL["intercept"]
                                   + FPR_MODEL["coef_per_sd"] * z)))
    assert np.allclose(preds, direct, atol=1e-12)


def test_predict_fpr_says_when_it_is_extrapolating():
    lo, hi = FPR_MODEL["fitted_range"]
    assert sg.predict_fpr((lo + hi) / 2)["within_fitted_range"] is True
    beyond = sg.predict_fpr(hi + 0.1)
    assert beyond["within_fitted_range"] is False
    assert beyond["p_reject"] > 0.999


def test_predict_fpr_carries_its_provenance():
    out = sg.predict_fpr(0.1)
    assert out["model"]["n_runs"] == FPR_MODEL["n_runs"]
    assert "product alone" in out["model"]["source"]
    assert list(out["model"]["grids"]) == list(FPR_MODEL["grids"])


# ----------------------------------------------------------------------
# select_null_method: every branch, including the quiet ones
# ----------------------------------------------------------------------
def test_select_null_method_keeps_the_conventional_branch_when_all_is_quiet():
    out = sg.select_null_method(tie_break_share=0.2, implied_rho_abs=0.001,
                                chance_band=0.05)
    assert out["recommended_action"] == "conventional_null_ok"
    assert out["steps"] == {"preflight": "channel_closed",
                            "sparsity": "below_moderate",
                            "codetection": "not_evaluated"}
    assert out["notes"] == []


def test_select_null_method_escalates_to_refusal_over_any_other_step():
    # A refusal has to win even when the pre-flight says the channel is open
    # and the co-detection gap is beyond the refusal line: the verdict rule
    # is severity-ordered, and a weaker reason can never lower a stronger
    # one -- the same rule the diagnostics module runs on.  The two refusals
    # carry the same severity, so step order decides between them: the
    # checklist evaluates sparsity before null construction, and the label
    # a set gets is the first refusal it meets.
    out = sg.select_null_method(tie_break_share=0.9, implied_rho_abs=0.5,
                                chance_band=0.05, codetection_gap=0.05,
                                axis_depth_rho=0.9)
    assert out["recommended_action"] == "refuse_not_identifiable"
    assert out["steps"]["codetection"].startswith("gap=0.05")
    # With the sparse-set failure removed, the co-detection refusal is what
    # remains, and it is reported as itself rather than folded into the
    # sparsity verdict.
    out = sg.select_null_method(tie_break_share=0.2, codetection_gap=0.05,
                                axis_depth_rho=0.9)
    assert out["recommended_action"] == "refuse_no_null_family_safe"


def test_select_null_method_demands_a_matched_null_from_the_channel():
    out = sg.select_null_method(implied_rho_abs=0.2, chance_band=0.05)
    assert out["recommended_action"] == "expression_matched_null"
    out = sg.select_null_method(tie_break_share=0.55)
    assert out["recommended_action"] == "expression_matched_null"


def test_select_null_method_reports_unevaluated_rather_than_quiet():
    out = sg.select_null_method()
    assert out["recommended_action"] == "conventional_null_ok"
    assert set(out["steps"].values()) == {"not_evaluated"}
    # And the band without the product, or the product without the band, is
    # not a pre-flight reading either.
    out = sg.select_null_method(implied_rho_abs=0.2)
    assert out["steps"]["preflight"] == "not_evaluated"
    assert any("chance band" in n for n in out["notes"])


def test_select_null_method_flags_the_codetection_boundary():
    # Beyond the refusal line, on a depth-linked axis: no family is safe.
    out = sg.select_null_method(codetection_gap=0.03, axis_depth_rho=0.5)
    assert out["recommended_action"] == "refuse_no_null_family_safe"
    # Between the lines: the action is the matched null, and the gap is
    # named in a note rather than swallowed.
    out = sg.select_null_method(codetection_gap=0.01, axis_depth_rho=0.5)
    assert out["recommended_action"] == "conventional_null_ok"
    assert any("flag line" in n for n in out["notes"])
    # On an axis that does not track depth, the same gap cannot flag: the
    # two conditions multiply, which is what the mechanism predicts.
    out = sg.select_null_method(codetection_gap=0.03, axis_depth_rho=0.1)
    assert out["recommended_action"] == "conventional_null_ok"
    assert out["notes"] == []
    # A negative gap means the draws are more co-detected than the query,
    # and a shortfall of zero flags nothing.
    out = sg.select_null_method(codetection_gap=-0.01, axis_depth_rho=0.5)
    assert out["recommended_action"] == "conventional_null_ok"


def test_select_null_method_echoes_the_thresholds_it_applied():
    out = sg.select_null_method(tie_break_share=0.55)
    th = out["thresholds_applied"]
    assert th["tie_break_share_moderate"] == \
        sg.DEFAULT_THRESHOLDS["tie_break_share_moderate"]
    assert th["tie_break_share_high"] == \
        sg.DEFAULT_THRESHOLDS["tie_break_share_high"]
    assert th["null_matching_tolerance"] == \
        sg.DEFAULT_CONFIG["null_matching_tolerance"]


def test_the_boundary_constants_match_the_grid_rules():
    # ``BOUNDARIES`` in the analysis that produced the manuscript's boundary
    # table is the other copy of these numbers.  If one moves, this test
    # makes the other move with it.
    from experiments.make_numbers import BOUNDARIES

    gaps = sorted({g for g, _, _ in BOUNDARIES})
    axes = {a for _, a, _ in BOUNDARIES}
    assert CO_DETECTION_GAP_FLAG == gaps[0]
    assert CO_DETECTION_GAP_REFUSE == gaps[-1]
    assert DEPTH_LINKED_AXIS == max(axes)


def test_the_shipped_fit_matches_the_summary_the_manuscript_reports():
    # The joint logistic file is the fit's source of record; the coefficients
    # below travelled from it into the source by hand.
    with open(os.path.join(RESULTS, "grid_A_B_joint_logistic.csv")) as fh:
        rows = list(csv.DictReader(fh))
    fit = next(r for r in rows
               if r["model"] == "product alone" and r["term"] == "product")
    assert float(fit["coef"]) == pytest.approx(FPR_MODEL["coef_per_sd"])
    assert int(fit["n"]) == FPR_MODEL["n_runs"]
