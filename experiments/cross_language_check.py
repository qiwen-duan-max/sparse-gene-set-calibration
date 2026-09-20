#!/usr/bin/env python
"""Check that the R and Python implementations decide the same thing.

The two packages are implementations of one framework, and the tie-break keys
are already tested against each other to the integer.  The verdict is the other
half of that contract: a user who runs the Python package and the R package on
the same diagnostic report has to get the same answer, or the framework has two
definitions of what counts as interpretable.

Only the decision is compared -- the verdict name, the severity, and how many
reasons were given.  The reason *text* is written for the reader of each
language and is allowed to differ.

The Monte-Carlo P value is compared too, on fixed null vectors.  A P value looks
like a number that cannot disagree between two implementations of the same three
lines, which is why the tail convention was written down three times in this
project and drifted: the core tested the lower tail, the R wrapper wrapped the
same lower tail, and the app folded the values about zero.  All three look
right in isolation.  Comparing the outputs is the only thing that catches it.

The R package is loaded from the source tree rather than from an installed
library, so the check runs against the working copy.

The framework's structured records are compared too, as text.  The claim that a
log from either implementation parses the same way is the kind of claim that
holds until the first float is rounded differently or the first empty field is
written out on one side only -- and neither of those shows up in a P value.

    python experiments/cross_language_check.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sparsegs as sg  # noqa: E402

R_PACKAGE = ROOT / "r-package" / "sparseGenSetCal"


def battery():
    """Diagnostic reports spanning every branch of the verdict rule.

    Each report carries the fields the rule reads and nothing else, so a
    disagreement can only come from the rule itself.
    """
    shape = dict(observed_zero_rate=0.05, zero_rate_excess=0.01,
                 tie_break_share=0.01, frac_above_floor=1.0,
                 detection_floor=0.005)
    cases = []
    for share in (0.01, 0.40, 0.60, 0.90):
        for zero in (0.05, 0.25, 0.60):
            for excess in (0.01, -0.30):
                r = dict(shape)
                r.update(tie_break_share=share, observed_zero_rate=zero,
                         zero_rate_excess=excess)
                cases.append(r)
    # the fallback criterion, with no depth to compute a share from
    for frac in (0.05, 0.40, 0.80):
        cases.append(dict(observed_zero_rate=0.10, zero_rate_excess=0.01,
                          frac_above_floor=frac, detection_floor=0.02))
    flags = [dict(), dict(used_matched_null=True),
             dict(outcome_driven_cutpoint=True),
             dict(externally_replicated=False),
             dict(externally_replicated=True),
             # The axis's own confound with depth, which the verdict folds in
             # alongside the gene set's sparsity.  Carried here because it is
             # the input a verdict can be given without any sparsity reason
             # firing, so it is the one whose effect is easiest to implement on
             # one side and forget on the other.
             dict(preflight={"verdict": "CLEAR"}),
             dict(used_matched_null=True, preflight={"verdict": "CAUTION"}),
             dict(used_matched_null=True, preflight={"verdict": "CONFOUNDED"})]
    return [(c, f) for c in cases for f in flags]


R_DRIVER = r"""
suppressPackageStartupMessages(
    pkgload::load_all("%(pkg)s", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
jobs <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
out <- lapply(jobs, function(job) {
    r <- job$report
    # fromJSON gives every scalar as a list of one; the verdict wants numbers.
    r <- lapply(r, function(v) if (is.list(v)) as.numeric(v[[1]]) else v)
    flags <- job$flags
    rep <- tryCatch(
        do.call(verdict, c(list(report = r), flags)),
        error = function(e) list(verdict = "ERROR", severity = "ERROR",
                                 reasons = list(conditionMessage(e))))
    list(verdict = rep$verdict, severity = rep$severity,
         n_reasons = length(rep$reasons))
})
jsonlite::write_json(out, args[2], auto_unbox = TRUE, digits = 10)
"""


R_DRIVER_P = r"""
suppressPackageStartupMessages(
    pkgload::load_all("%(pkg)s", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
jobs <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
out <- lapply(jobs, function(job) {
    nulls <- as.numeric(unlist(job$nulls))
    observed <- as.numeric(job$observed)
    tails <- unlist(job$tails)
    p <- vapply(tails, function(t) empirical_p(nulls, observed, tail = t),
                numeric(1))
    # The default is the thing that drifted, so it is compared too, and it is
    # called without a tail argument rather than with the expected one.
    s <- null_summary(nulls, observed)
    list(p = as.list(p),
         default_p = s$p_value,
         # Both defaults are called without a tail argument.  Checking only the
         # summary's default left this hole: the vector above always passes a
         # tail, so empirical_p's own default was never exercised, and the
         # check passed with the two defaults set differently.
         empirical_p_default = empirical_p(nulls, observed),
         percentile = s$percentile,
         null_sd = s$null_sd)
})
jsonlite::write_json(out, args[2], auto_unbox = TRUE, digits = 15)
"""

#: Emits `preflight()`'s numbers, which are pure arithmetic on the depth vector.
R_DRIVER_PF = r"""
suppressPackageStartupMessages(
    pkgload::load_all("%(pkg)s", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
jobs <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
out <- lapply(jobs, function(job) {
    pf <- preflight(as.numeric(unlist(job$depth)), as.numeric(unlist(job$axis)),
                    as.integer(job$max_rank), as.integer(job$n_genes))
    list(tau_axis_rho = pf$tau_axis_rho, tau_sd = pf$tau_sd,
         chance_band = pf$chance_band, implied_rho = pf$implied_rho,
         depth_axis_rho = pf$depth_axis_rho, median_detected = pf$median_detected,
         tau_constant = pf$tau_constant, verdict = pf$verdict)
})
jsonlite::write_json(out, args[2], auto_unbox = TRUE, digits = 15)
"""


#: Emits the framework's structured records as text, one per job.
R_DRIVER_R = r"""
suppressPackageStartupMessages(
    pkgload::load_all("%(pkg)s", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
jobs <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
# The battery carries records below the default level, and a record that is
# filtered out before it is built cannot be compared.  The level is opened all
# the way down: this is a check on the *format*, and the filter is a separate
# behaviour that both sides test in their own suites.
set_log_level("debug")
out <- lapply(jobs, function(job) {
    # fromJSON gives every scalar as a list of one, every null as NULL and every
    # array as a list of several; the framework takes a scalar as a scalar, a
    # dropped field as NULL and a vector as a vector -- which is also how the
    # Python side passes them, so neither side normalises before the call.
    fields <- lapply(job$fields, function(v) {
        if (is.null(v) || length(v) == 0L) return(NULL)
        if (length(v) == 1L) return(v[[1]])
        vapply(v, function(x) as.character(x[[1]]), character(1))
    })
    withCallingHandlers(
        do.call(event, c(list(job$level, job$name), fields)),
        sparsegs_event = function(w) invokeRestart("muffleWarning"))
})
set_log_level("warning")
jsonlite::write_json(out, args[2], auto_unbox = TRUE)
"""


#: The convenience pair: the decision table and the fitted rejection
#: probability, both pure functions whose constants were transcribed by hand
#: into the R port -- which is exactly the transcription a check like this
#: exists to police.
R_DRIVER_CV = r"""
suppressPackageStartupMessages(
    pkgload::load_all("%(pkg)s", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
jobs <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
# fromJSON wraps every scalar in a list of one; the functions take scalars,
# so each argument is unwrapped here -- and an argument the Python side left
# as null arrives as NULL, which is the same as absent for both functions'
# is.null() branches.
out <- lapply(jobs, function(job) {
    args <- lapply(job$args, function(v) {
        if (is.null(v) || length(v) == 0L) return(NULL)
        if (length(v) == 1L) return(v[[1]])
        v
    })
    if (job$kind == "select") {
        res <- do.call(select_null_method, args)
        # A length-1 character vector would unbox to a bare string and stop
        # being a list on the Python side, so the notes travel as a list.
        res$notes <- as.list(res$notes)
        res
    } else {
        res <- do.call(predict_fpr, args)
        res$model$grids <- as.list(res$model$grids)
        res
    }
})
# digits: the default four would round the fitted probability before the
# comparison ever saw it, which is how a transcription check silently passes.
jsonlite::write_json(out, args[2], auto_unbox = TRUE, digits = 15)
"""


def pvalue_battery():
    """Null vectors that separate the three tail conventions.

    A symmetric null is included to show the conventions agreeing, and an
    off-centre one to show them not agreeing: an off-centre null is what
    imperfect matching produces, so it is the case the framework exists for and
    the case a parity check must not skip.  Ties and non-finite entries are
    carried through because the two languages filter them in different places.
    """
    import numpy as np

    rng = np.random.default_rng(20260917)
    cases = []
    for nulls in (
        np.linspace(-1.0, 1.0, 201),                 # symmetric, centre 0
        np.linspace(-1.0, 1.0, 201) + 0.4,           # off centre
        np.r_[np.linspace(-0.2, 0.2, 51), rng.normal(0.0, 1.0, 150)],
        np.round(rng.normal(0.3, 0.05, 100), 2),     # heavy ties
        np.r_[np.linspace(0.0, 1.0, 40), np.nan, np.inf, -np.inf],
    ):
        for observed in (-2.0, -0.5, 0.0, 0.3, 0.9, 3.0):
            cases.append((nulls, observed))
    return cases


def preflight_battery():
    """Depth/axis pairs spanning every branch of the pre-flight's verdict.

    The pre-flight is the framework's judgement about the *axis* rather than
    about the gene set, and it is pure arithmetic on the depth vector -- which
    is exactly why it is worth comparing: arithmetic this simple looks like it
    cannot disagree between two languages, and the fields it reports are where
    a convention (which standard deviation, what to do with a constant vector)
    diverges silently.  Two were found here: the population-vs-sample standard
    deviation, and scipy raising where R's ``cor.test`` returns ``NA``.
    """
    rng = np.random.default_rng(20260917)
    cases = []
    for n in (60, 500, 4000):
        depth = rng.integers(10, 2000, size=n).astype(float)
        for axis in (rng.normal(size=n), depth.copy(),
                     depth + rng.normal(0.0, 300.0, size=n),
                     np.linspace(-1.0, 1.0, n)):
            cases.append(dict(depth=depth, axis=axis, max_rank=1000,
                              n_genes=20000))
    # Every cell above the ceiling: tau is the constant zero and the
    # correlation does not exist rather than being small.  The one case where
    # the two languages disagreed on whether to complain.
    cases.append(dict(depth=np.full(200, 5000.0), axis=rng.normal(size=200),
                      max_rank=100, n_genes=2000))
    # A constant axis, and a depth vector with a single value repeated: the
    # other two ways to reach an undefined rank correlation.
    cases.append(dict(depth=rng.integers(10, 2000, size=300).astype(float),
                      axis=np.full(300, 2.0), max_rank=1000, n_genes=20000))
    cases.append(dict(depth=np.full(300, 7.0), axis=rng.normal(size=300),
                      max_rank=1000, n_genes=20000))
    return cases


def record_battery():
    """Structured records, in the shapes where two languages would diverge.

    The framework's claim is that a log from either implementation parses the
    same way, and the places two languages would break that claim are exactly
    the ones that look trivial: how a float is rounded, how a boolean is
    spelled, whether a field with nothing in it is written out at all, and
    whether a field holding several values keeps its brackets.  None of these
    is visible in a P value, so none of them is caught by the batteries above.
    """
    return [
        ("warning", "demo_event",
         [("n", 3), ("ratio", 0.25), ("ok", True), ("why", "two words")]),
        ("warning", "score_has_no_variance",
         [("label", None), ("context", ""), ("sd", 0.0), ("threshold", 0.001)]),
        ("warning", "background_insufficient",
         [("n_available", 3), ("required", 10), ("det_floor", 0.02),
          ("loosened_to", 0.018),
          ("action", "loosening detection floor by 10%")]),
        ("warning", "suspicious_score_separation",
         [("label", "CLOCK"), ("context", "discovery"), ("auc", 0.995),
          ("threshold", 0.99),
          ("check", ["batch_effects",
                     "outcome_leakage_into_cell_type_annotation",
                     "extreme_sparsity"])]),
        # The reversed separation: an AUC below 0.5 is flagged on the same
        # argument as one above 0.99, and the record has to carry both numbers
        # so a reader can tell which way round it was.
        ("warning", "suspicious_score_separation",
         [("label", "reversed"), ("context", ""), ("auc", 0.002),
          ("separation", 0.998), ("threshold", 0.99),
          ("check", ["batch_effects",
                     "outcome_leakage_into_cell_type_annotation",
                     "extreme_sparsity"])]),
        # A full-precision float and a large integer: the two ways a number
        # arrives at six significant figures rather than as itself.
        ("info", "draw_complete", [("kept", 200), ("gap", 0.0123456789)]),
        ("debug", "pool_built", [("n_strata", 10), ("floor", 2e-06)]),
    ]


def convenience_battery():
    """Inputs covering every branch of the two convenience functions.

    The branches are the point: an escalation has to win over a quieter
    recommendation from any other step, an unevaluated quantity has to read
    as unevaluated rather than as quiet, and a negative co-detection gap --
    draws more co-detected than the query -- must never flag.  The predicted
    probabilities include both ends of the fitted range and one point beyond
    it, because the extrapolation flag is part of the contract.
    """
    return [
        dict(kind="select", args=dict()),
        dict(kind="select", args=dict(tie_break_share=0.8,
                                      implied_rho_abs=0.01, chance_band=0.05)),
        dict(kind="select", args=dict(tie_break_share=0.55)),
        dict(kind="select", args=dict(implied_rho_abs=0.2, chance_band=0.05)),
        dict(kind="select", args=dict(implied_rho_abs=0.01, chance_band=0.05,
                                      tie_break_share=0.2)),
        dict(kind="select", args=dict(codetection_gap=0.03,
                                      axis_depth_rho=0.5)),
        dict(kind="select", args=dict(codetection_gap=0.01,
                                      axis_depth_rho=0.5,
                                      implied_rho_abs=0.1, chance_band=0.05)),
        dict(kind="select", args=dict(codetection_gap=0.03,
                                      axis_depth_rho=0.1)),
        dict(kind="select", args=dict(codetection_gap=-0.01,
                                      axis_depth_rho=0.5)),
        dict(kind="select", args=dict(tie_break_share=0.8,
                                      codetection_gap=0.03,
                                      axis_depth_rho=0.5,
                                      implied_rho_abs=0.2, chance_band=0.05)),
        dict(kind="select", args=dict(implied_rho_abs=0.2)),
        dict(kind="predict", args=dict(implied_rho_abs=0.0)),
        dict(kind="predict", args=dict(implied_rho_abs=0.0512505673509389)),
        dict(kind="predict", args=dict(implied_rho_abs=0.2969)),
        dict(kind="predict", args=dict(implied_rho_abs=0.5)),
    ]


def main():
    if shutil.which("Rscript") is None:
        print("Rscript is not on PATH; the check cannot run here.")
        return 1
    jobs = battery()
    payload = [dict(report = r, flags = f) for r, f in jobs]
    tails = ["lower", "upper", "two-sided"]
    pcases = pvalue_battery()
    payload_p = [dict(nulls = [float(v) if np.isfinite(v) else None for v in n],
                      observed = float(o), tails = tails)
                 for n, o in pcases]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")
        (tmp / "driver.R").write_text(R_DRIVER % {"pkg": R_PACKAGE},
                                      encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(tmp / "driver.R"), str(tmp / "jobs.json"),
             str(tmp / "out.json")],
            capture_output = True, text = True, cwd = str(tmp))
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            return 1
        r_out = json.loads((tmp / "out.json").read_text(encoding="utf-8"))

        # jsonlite writes NaN and Inf as null; the Python side has to see the
        # same vector, so the non-finite entries are passed as null and put back
        # here rather than being dropped from one side only.
        (tmp / "jobs_p.json").write_text(json.dumps(payload_p), encoding="utf-8")
        (tmp / "driver_p.R").write_text(R_DRIVER_P % {"pkg": R_PACKAGE},
                                        encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(tmp / "driver_p.R"), str(tmp / "jobs_p.json"),
             str(tmp / "out_p.json")],
            capture_output = True, text = True, cwd = str(tmp))
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            return 2
        r_p = json.loads((tmp / "out_p.json").read_text(encoding="utf-8"))

    disagreements = []
    for (report, flags), rv in zip(jobs, r_out):
        pv = sg.verdict(report, **flags)
        if (rv["verdict"] != pv["verdict"] or rv["severity"] != pv["severity"]
                or rv["n_reasons"] != len(pv["reasons"])):
            disagreements.append((report, flags, pv, rv))

    print(f"{len(jobs)} reports through both implementations")
    if disagreements:
        print(f"{len(disagreements)} disagreements")
        for report, flags, pv, rv in disagreements[:10]:
            print(f"  report={report}")
            print(f"  flags ={flags}")
            print(f"    python: {pv['verdict']:<32} {pv['severity']:<9} "
                  f"{len(pv['reasons'])} reasons")
            print(f"    R     : {rv['verdict']:<32} {rv['severity']:<9} "
                  f"{rv['n_reasons']} reasons")
        return 1
    print("verdict, severity and reason count agree on every one")

    bad = []

    def differ(pv, rp, tol):
        """True when the two implementations disagree, NaN against null included."""
        # jsonlite writes a *scalar* NA as the string "NA" rather than as null
        # -- its `na = "null"` default applies to vectors, not to the unboxed
        # scalars `auto_unbox` produces -- so an absent number arrives as text
        # on one side and as NaN on the other.  Read here rather than papered
        # over in the R driver, so that what the driver emits stays a plain
        # serialisation of what the function returned.
        if rp is None or (isinstance(rp, str) and rp.strip() == "NA"):
            return not (isinstance(pv, float) and np.isnan(pv))
        if isinstance(pv, float) and np.isnan(pv):
            return True
        return abs(pv - rp) > tol

    for (nulls, observed), rv in zip(pcases, r_p):
        for tail in tails:
            if differ(sg.empirical_p(nulls, observed, tail=tail),
                      rv["p"][tail], 1e-12):
                bad.append((f"empirical_p[{tail}]", observed,
                            sg.empirical_p(nulls, observed, tail=tail),
                            rv["p"][tail]))
        summary = sg.null_summary(nulls, observed)
        for key, rp in (("p_value", rv["default_p"]),
                        ("percentile", rv["percentile"]),
                        ("null_sd", rv["null_sd"])):
            if differ(summary[key], rp, 1e-9):
                bad.append((f"null_summary.{key}", observed, summary[key], rp))
        if differ(sg.empirical_p(nulls, observed),
                  rv["empirical_p_default"], 1e-12):
            bad.append(("empirical_p[default]", observed,
                        sg.empirical_p(nulls, observed),
                        rv["empirical_p_default"]))
    print(f"{len(pcases)} null vectors x {len(tails)} tails + summary fields "
          f"through both implementations")
    if bad:
        print(f"{len(bad)} disagreements")
        for tail, observed, pv, rp in bad[:10]:
            print(f"  tail={tail:<20} observed={observed:+.3g}  "
                  f"python={pv!r}  R={rp!r}")
        return 3
    print("P value, its default, percentile and null SD agree to 1e-9")

    # ---- the pre-flight, which is the judgement about the axis ----
    pf_cases = preflight_battery()
    payload_pf = [dict(depth=[float(v) for v in c["depth"]],
                       axis=[float(v) for v in c["axis"]],
                       max_rank=c["max_rank"], n_genes=c["n_genes"])
                  for c in pf_cases]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "jobs_pf.json").write_text(json.dumps(payload_pf),
                                          encoding="utf-8")
        (tmp / "driver_pf.R").write_text(R_DRIVER_PF % {"pkg": R_PACKAGE},
                                         encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(tmp / "driver_pf.R"), str(tmp / "jobs_pf.json"),
             str(tmp / "out_pf.json")],
            capture_output=True, text=True, cwd=str(tmp))
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            return 6
        r_pf = json.loads((tmp / "out_pf.json").read_text(encoding="utf-8"))

    bad_pf = []
    for case, rv in zip(pf_cases, r_pf):
        pv = sg.preflight(case["depth"], case["axis"], case["max_rank"],
                          case["n_genes"])
        for key in ("tau_axis_rho", "tau_sd", "chance_band", "implied_rho",
                    "depth_axis_rho", "median_detected"):
            if differ(pv[key], rv[key], 1e-12):
                bad_pf.append((key, case, pv[key], rv[key]))
        if pv["verdict"] != rv["verdict"]:
            bad_pf.append(("verdict", case, pv["verdict"], rv["verdict"]))
        # A boolean does not tolerate a tolerance, and it is the field that says
        # *why* the correlation is absent -- the one a reader would otherwise
        # have to infer from a NaN.
        if bool(pv["tau_constant"]) != bool(rv["tau_constant"]):
            bad_pf.append(("tau_constant", case, pv["tau_constant"],
                           rv["tau_constant"]))
    print(f"{len(pf_cases)} pre-flights through both implementations")
    if bad_pf:
        print(f"{len(bad_pf)} disagreements")
        for key, case, pval, rval in bad_pf[:10]:
            print(f"  field={key:<18} n_cells={len(case['depth'])}  "
                  f"python={pval!r}  R={rval!r}")
        return 7
    print("tau_axis_rho, tau_sd, the band and the verdict agree to 1e-12")

    # ---- the log records, which have to agree as text ----
    rc = record_battery()
    payload_r = [dict(level=lv, name=nm, fields=dict(fs))
                 for lv, nm, fs in rc]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "jobs_r.json").write_text(json.dumps(payload_r), encoding="utf-8")
        (tmp / "driver_r.R").write_text(R_DRIVER_R % {"pkg": R_PACKAGE},
                                        encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(tmp / "driver_r.R"), str(tmp / "jobs_r.json"),
             str(tmp / "out_r.json")],
            capture_output=True, text=True, cwd=str(tmp))
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            return 4
        r_records = json.loads((tmp / "out_r.json").read_text(encoding="utf-8"))

    from sparsegs.events import event as _event
    # `event()` returns the record whether or not the logger would print it, so
    # raising the logger's level here silences this check's own output without
    # changing a character of what is compared.
    old_level = sg.events.LOGGER.level
    sg.events.LOGGER.setLevel("CRITICAL")
    try:
        py_records = [_event(lv, nm, **dict(fields))
                      for lv, nm, fields in rc]
    finally:
        sg.events.LOGGER.setLevel(old_level)

    bad_records = []
    for (level, name, fields), pv, rp in zip(rc, py_records, r_records):
        if pv != rp:
            bad_records.append((name, pv, rp))
    print(f"{len(rc)} structured records through both implementations")
    if bad_records:
        print(f"{len(bad_records)} disagreements")
        for name, pv, rp in bad_records:
            print(f"  event={name}")
            print(f"    python: {pv}")
            print(f"    R     : {rp}")
        return 5
    print("every record is the same text in both languages")

    # ---- the conveniences, whose constants were transcribed by hand ----
    cv_cases = convenience_battery()
    payload_cv = [dict(kind=c["kind"],
                       args={k: (None if v is None else v)
                             for k, v in c["args"].items()})
                  for c in cv_cases]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "jobs_cv.json").write_text(json.dumps(payload_cv),
                                          encoding="utf-8")
        (tmp / "driver_cv.R").write_text(R_DRIVER_CV % {"pkg": R_PACKAGE},
                                         encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(tmp / "driver_cv.R"), str(tmp / "jobs_cv.json"),
             str(tmp / "out_cv.json")],
            capture_output=True, text=True, cwd=str(tmp))
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            return 8
        r_cv = json.loads((tmp / "out_cv.json").read_text(encoding="utf-8"))

    bad_cv = []
    for case, rv in zip(cv_cases, r_cv):
        if case["kind"] == "select":
            pv = sg.select_null_method(**case["args"])
            if pv["recommended_action"] != rv["recommended_action"]:
                bad_cv.append(("action", case, pv["recommended_action"],
                               rv["recommended_action"]))
            for step in ("preflight", "sparsity", "codetection"):
                if pv["steps"].get(step) != (rv["steps"] or {}).get(step):
                    bad_cv.append((f"steps.{step}", case,
                                   pv["steps"].get(step),
                                   (rv["steps"] or {}).get(step)))
            if list(pv["notes"]) != list(rv.get("notes") or []):
                bad_cv.append(("notes", case, pv["notes"], rv.get("notes")))
            for key, val in pv["thresholds_applied"].items():
                rp = (rv.get("thresholds_applied") or {}).get(key)
                if differ(float(val), rp, 1e-12):
                    bad_cv.append((f"thresholds.{key}", case, val, rp))
        else:
            pv = sg.predict_fpr(**case["args"])
            if differ(pv["p_reject"], rv["p_reject"], 1e-12):
                bad_cv.append(("p_reject", case, pv["p_reject"],
                               rv["p_reject"]))
            if bool(pv["within_fitted_range"]) != bool(
                    rv["within_fitted_range"]):
                bad_cv.append(("within_fitted_range", case,
                               pv["within_fitted_range"],
                               rv["within_fitted_range"]))
            if pv["model"]["source"] != (rv.get("model") or {}).get("source"):
                bad_cv.append(("model.source", case, pv["model"]["source"],
                               (rv.get("model") or {}).get("source")))
    print(f"{len(cv_cases)} convenience calls through both implementations")
    if bad_cv:
        print(f"{len(bad_cv)} disagreements")
        for key, case, pval, rval in bad_cv[:10]:
            print(f"  field={key}\n    args={case['args']}"
                  f"\n    python={pval!r}\n    R     ={rval!r}")
        return 9
    print("the decision and the fitted probability agree to 1e-12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
