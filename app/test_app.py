#!/usr/bin/env python
"""Headless checks for the Streamlit companion app.

``streamlit.testing.v1.AppTest`` runs the app script the way the server does,
so these assertions are about the app as deployed rather than about its helper
functions in isolation.  What is checked:

* every built-in dataset renders without an exception, on both axes and with
  both gene-set choices -- the eight paths a visitor can take;
* the app's own step-1 numbers agree with the package called directly, so the
  display is not a second implementation of the diagnostics;
* each example lands on the verdict branch it was built to demonstrate, which
  is the one claim the app makes about its own datasets;
* the matrix parser reads what it says it reads.

    python app/test_app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import sparsegs as sg  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

# Importable because the app only calls main() when Streamlit's runtime is up,
# which it is not in this process.
import calibration_app as app  # noqa: E402

FAILURES = []


@pytest.fixture(autouse=True)
def _no_check_records_a_failure():
    """Make pytest read the record the checks write to.

    ``check`` appends to ``FAILURES`` and the script entry point below exits
    non-zero on it, but pytest only sees whether the test function raised.  Run
    under pytest without this fixture, a failing check prints ``FAIL`` and the
    session still reports a pass -- the loudest failure mode a test file can
    have, and the one this file was written to avoid.
    """
    FAILURES.clear()
    yield
    assert not FAILURES, f"checks reported as FAIL: {FAILURES}"


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {name}{('  ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(name)


def cheap(at):
    """Turn the two expensive sliders down; the assertions are not about them."""
    at.slider[1].set_value(5)     # matched null draws
    at.slider[2].set_value(100)   # cutpoint permutations
    return at


def run_app(**set_values):
    """Run the app and apply sidebar choices before the assertions."""
    at = AppTest.from_file(str(ROOT / "app" / "calibration_app.py"),
                           default_timeout=900)
    at.run()
    cheap(at)
    for widget, value in set_values.items():
        kind, index = widget.rsplit("_", 1)
        getattr(at, kind)[int(index)].set_value(value)
    at.run()
    return at


def test_paths():
    print("rendering paths")
    examples = ["1 · clean regime — deep cells, real signal",
                "2 · confounded — shallow cells, depth tracks the axis",
                "3 · co-detected — genes found and lost together"]
    for i, name in enumerate(examples):
        for axis in ["the programme (the planted axis)", "sequencing depth"]:
            for gene_set in ["the planted set", "a random background set"]:
                at = AppTest.from_file(str(ROOT / "app" / "calibration_app.py"),
                                       default_timeout=900)
                at.run()
                cheap(at)
                at.radio[0].set_value(name)
                at.radio[1].set_value(
                    next(o for o in at.radio[1].options
                         if o.startswith(gene_set)))
                at.radio[2].set_value(axis)
                at.run()
                label = f"example {i + 1} / {axis[:12]} / {gene_set[:8]}"
                check(label, not at.exception,
                      "; ".join(str(e.value)[:200] for e in at.exception))
                if at.exception:
                    continue
                check(f"{label}: verdict shown",
                      any(m.label == "Ceiling ÷ depth" for m in at.metric))


def test_numbers_match_package():
    """The displayed numbers are the package's numbers, not the app's."""
    print("the app reports what the package computes")
    at = run_app()
    shown = {m.label: m.value for m in at.metric}
    check("cells metric", shown.get("Cells") == "900", str(shown.get("Cells")))

    from sparsegs.simulate import SimConfig, simulate
    data = simulate(SimConfig(n_cells=900, n_genes=6000, n_target=40,
                              median_detected=1500, detection=0.30, effect=0.70,
                              coexpr=0.30, depth_programme_loading=0.0, seed=1))
    cache = sg.RankCache.build(data.X, data.genes, ceiling=300, seed=42)
    report = sg.sparsity_report(cache, data.target, rank_frac=0.05)
    check("tie-break share agrees",
          shown.get("Tie-break share of the set's inclusion")
          == f"{report['tie_break_share']:.3f}",
          f"{shown.get('Tie-break share of the set\'s inclusion')} vs "
          f"{report['tie_break_share']:.3f}")
    check("zero rate agrees",
          shown.get("Cells scoring exactly zero")
          == f"{report['observed_zero_rate']:.1%}",
          f"{shown.get('Cells scoring exactly zero')} vs "
          f"{report['observed_zero_rate']:.1%}")


def banner_verdict(at):
    """The verdict the app is showing, read off the banner only.

    Searching every markdown element would find the verdict names in the
    "Reading the verdict" legend, which lists all three of them and is always
    present -- an assertion written that way passes whatever the app decides.
    The banner is the one element sized at 1.35rem, and its text is the first
    run inside that span.
    """
    for m in at.markdown:
        found = re.search(r"1\.35rem[^>]*>([^<]+)<", m.value)
        if found:
            return found.group(1).strip()
    return None


def test_verdict_branches():
    """Each example reaches the branch the app says it demonstrates."""
    print("the examples reach their advertised branches")
    expected = {
        0: "INTERPRETABLE",
        1: "NOT_IDENTIFIABLE",
        2: "NOT_IDENTIFIABLE",
    }
    examples = ["1 · clean regime — deep cells, real signal",
                "2 · confounded — shallow cells, depth tracks the axis",
                "3 · co-detected — genes found and lost together"]
    seen = set()
    for i, name in enumerate(examples):
        at = AppTest.from_file(str(ROOT / "app" / "calibration_app.py"),
                               default_timeout=900)
        at.run()
        cheap(at)
        at.radio[0].set_value(name)
        at.run()
        if at.exception:
            check(f"example {i + 1} runs", False, str(at.exception[0].value)[:200])
            continue
        got = banner_verdict(at)
        seen.add(got)
        check(f"example {i + 1} -> {expected[i]}", got == expected[i],
              f"the banner says {got!r}")
    # And the three are not the same branch wearing three datasets.
    check("the three examples do not all decide the same thing",
          len(seen) > 1, f"all three gave {seen}")


def test_parser():
    print("input parsing")
    frame = pd.DataFrame(np.arange(12.0).reshape(3, 4),
                         index=["A", "B", "C"], columns=list("wxyz"))
    text = frame.to_csv()
    X, genes = app.parse_matrix(text, ",")
    check("matrix is cells by genes", X.shape == (4, 3), str(X.shape))
    check("gene names survive", list(genes) == ["A", "B", "C"], str(list(genes)))
    check("values transpose", X.toarray()[0, 0] == 0.0 and
          X.toarray()[1, 0] == 1.0, str(X.toarray()[:2, 0]))
    parsed = app.parse_gene_set("PER1, PER2\nPER3\n\n PER1 ")
    check("gene set splits on commas and newlines and de-duplicates",
          parsed == ["PER1", "PER2", "PER3"], str(parsed))
    # The upload path reaches csr_matrix through the app's own reader, so the
    # dtype the rank cache will see is worth pinning down here rather than at
    # the point where a user's file is already loaded.
    check("matrix is sparse and float32",
          sp.issparse(X) and X.dtype == np.float32, str(X.dtype))
    tab = app.parse_matrix(frame.to_csv(sep="\t"), "\t")
    check("tab-delimited files parse too", tab[0].shape == (4, 3), str(tab[0].shape))


if __name__ == "__main__":
    test_paths()
    test_numbers_match_package()
    test_verdict_branches()
    test_parser()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all app checks passed")
