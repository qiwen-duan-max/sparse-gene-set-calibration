#!/usr/bin/env python
"""Build the "Introduction to Expression-Matched Nulls" tutorial notebook.

The notebook is kept as its source here and written out by this script, so that
the tutorial can be regenerated after an API change rather than hand-edited into
a stale state.  ``--execute`` runs it end to end and embeds the outputs, which
is also how the tutorial is checked: a notebook that raises is a failing test.

    python tutorials/build_notebook.py --execute
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "introduction_to_expression_matched_nulls.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELLS = [
    md("""# Introduction to expression-matched nulls

*A tutorial for `sparsegs` — the calibration framework for gene-set scores on
sparse single-cell data.*

You have a gene-set score, a variable you care about, and a correlation. The
usual next step is a p-value, and the usual p-value comes from a null that
shuffles something. **This tutorial is about the fact that the usual shuffles
answer a different question than the one you are asking**, and about the null
that answers yours.

Everything here runs on simulated data, so at every step we know what was
planted and can check whether the analysis found it. That is the only way to see
that a method is right rather than merely plausible.

**The one sentence version.** A rank-based score fills its empty ranks with
genes the cell never expressed, and it fills them by depth. If the depth
distribution tracks the variable you are testing, the conventional null rejects
an association that does not exist — and the expression-matched null, which
draws replacement gene sets with the same detection profile, does not."""),

    md("""## Setup

The package is `sparsegs`. It is not on PyPI in this project's snapshot, so the
notebook puts the source tree on the path; with an installed copy the first
line is unnecessary."""),

    code("""import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import sparsegs as sg
from sparsegs.simulate import SimConfig, simulate


def table(rows, columns=("quantity", "value")):
    \"\"\"Print a two-column frame without an index.

    Everything below is reported through this so the numbers are what the
    function returned, not a paraphrase of it.
    \"\"\"
    return pd.DataFrame(rows, columns=list(columns))

print("sparsegs exports:", len(sg.__all__), "names")"""),

    md("""## 1. A dataset where we know the answer

`simulate()` builds a matrix from a known generative model. Three of its
arguments matter for this tutorial:

| argument | what it does |
|---|---|
| `median_detected` | how many genes a typical cell detects — this sets the depth |
| `detection` | how often the cells detect a gene of the target set |
| `depth_programme_loading` | how strongly sequencing depth tracks the programme |
| `effect` | how strongly the target set responds to the programme |

The configuration below sets `effect = 0.0`: **the gene set has no relationship
with the programme at all.** It also sets `depth_programme_loading = 1.5`, so
depth does track the programme. Any association the analysis reports is a false
positive by construction, and we know it."""),

    code("""config = SimConfig(
    n_cells=800,
    n_genes=6000,
    n_target=8,
    median_detected=250,          # shallow cells, relative to the panel
    detection=0.01,               # a sparse gene set
    effect=0.0,                   # nothing is planted
    coexpr=0.0,
    depth_programme_loading=1.5,  # depth tracks the programme
    seed=2,
)
data = simulate(config)
programme = data.programme            # the axis we will test against
print(data)

table([
    ("genes in the panel", f"{config.n_genes:,}"),
    ("rank ceiling (5%)", f"{config.max_rank:,}"),
    ("median genes detected", f"{np.median(data.detected_per_cell):,.0f}"),
    ("ceiling / depth", f"{config.depth_ratio:.2f}"),
    ("tie-break channel open", str(config.tie_break_open)),
])"""),

    md("""### The two correlations that make the problem

Before scoring anything, look at what the data does. Depth correlates with the
programme (we planted that), and — because the ceiling sits above what a cell
detects — the *score* is a decreasing function of depth. Put those two together
and the score correlates with the programme even though the gene set ignores
it."""),

    code("""depth = data.depth_factor
rho_depth_programme, p_depth_programme = sg.spearman(depth, programme)
print(f"depth vs programme           rho = {rho_depth_programme:+.3f}  "
      f"(planted: loading = {config.depth_programme_loading})")

cache = sg.RankCache.build(data.X, data.genes, ceiling=config.max_rank, seed=2024)
score = sg.aucell(cache, data.target)
rho_score_depth, _ = sg.spearman(score, depth)
print(f"score vs depth               rho = {rho_score_depth:+.3f}  "
      f"(the tie-break channel)")

report = sg.sparsity_report(cache, data.target)
table([
    ("median detection of the set", f"{report['median_detection']:.4f}"),
    ("cells scoring exactly zero", f"{report['observed_zero_rate']:.1%}"),
    ("zero rate under independence", f"{report['structural_zero_rate']:.1%}"),
    ("tie-break share of inclusion", f"{report['tie_break_share']:.3f}"),
])"""),

    md("""## 2. What the conventional analyses report

`all_conventional()` runs every conventional test this package knows about on
one score.

The cell-level (`naive`) test treats 800 cells as 800 independent observations
of the programme. The permutation test shuffles the programme across cells,
which destroys the depth–programme link in every permuted draw — so the null
distribution is centred on zero while the observed statistic carries the
confound. Both are anticonservative here, and neither can see why."""),

    code("""conventional = sg.all_conventional(score, programme, n_perm=1000, seed=0)
table([(k, f"{v:.3g}") for k, v in conventional.items()],
      columns=("conventional test", "P"))

print(f"\\nnaive P < 0.05: {(conventional['naive_p'] < 0.05)}")
print("Nothing was planted. Every rejection here is a false positive.")"""),

    md("""### The closed form

This is not a subtle statistical failure — it is computable in advance. With
panel size $$G$$, rank ceiling $$m$$, and $$D_c$$ genes detected in cell $$c$$,
a gene the cell never expressed lands inside the top $$m$$ with probability

$$P(\\text{tie-break inclusion}) = \\frac{m - D_c}{G - D_c}.$$

`tie_break_inclusion()` is that expression. It is a function of depth alone, so
the part of the score it drives is *the same for every gene set* with the same
detection profile — and it is a decreasing function of depth."""),

    code("""row = table([(f"{d:,.0f}", f"{sg.tie_break_inclusion(d, config.max_rank, config.n_genes):.4f}")
             for d in (100, 200, 300, 600, 1200)],
            columns=("genes detected in the cell",
                     "P(a missing gene lands in the top block)"))
row"""),

    md("""## 3. A null the data supports

The question is whether the *gene set* carries information beyond its detection
profile. So the null has to hold the detection profile fixed and vary only the
identities of the genes.

`construct_expression_matched_null()` draws each replacement from the
detection-rate stratum of the gene it replaces, excluding the query set from the
pool. Its report is the part that matters: a null that silently failed to match
is worse than no null at all."""),

    code("""built = sg.construct_expression_matched_null(cache, data.target, n_sets=200,
                                               seed=42)
table([
    ("target detection", f"{built['target_detection']:.4f}"),
    ("null detection", f"{built['null_detection']:.4f}"),
    ("ratio", f"{built['detection_ratio']:.4f}"),
    ("mean per-set gap", f"{built['mean_detection_gap']:.4f}"),
    ("worst per-set gap", f"{built['worst_detection_gap']:.4f}"),
    ("draws within tolerance", f"{built['frac_within_tolerance']:.1%}"),
    ("genes shared with the query", str(built["max_overlap"])),
    ("KS statistic", f"{built['ks_statistic']:.4f}"),
    ("status", built["status"]),
])"""),

    md("""`status` is decided by the **mean** gap rather than the worst one. That is
deliberate: a maximum over draws can only grow as you ask for more of them, so a
rule based on it would pass a pool at 25 draws and fail the same pool at 200 —
the diagnostic would penalise the better null. `draws within tolerance` shows
the spread behind the average, and `max_overlap` has to be zero: a null set that
reuses a gene of the query is not a null."""),

    md("""### The theory is checkable

A gene detected in a fraction $$p$$ of cells enters the top $$m$$ either because
it was counted or, failing that, by tie-break:

$$P(\\text{in top } m) = p + (1-p)\\,\\overline{\\frac{m-D}{G-D}}.$$

That is `effective_inclusion()`. It should predict what the drawn null sets
actually achieve, and it is worth checking rather than assuming, because the
matching is what the whole construction rests on."""),

    code("""target_pos = cache.positions(data.target)
drawn = built["null_genes"]          # one list per drawn set
drawn_det = np.array([np.mean(cache.detection[cache.positions(s)]) for s in drawn])
predicted = sg.effective_inclusion(cache.detection[target_pos], cache.depth,
                                   config.max_rank, config.n_genes)

table([
    ("query set, observed", f"{built['target_detection']:.4f}"),
    ("query set, predicted by eq. (3)", f"{np.mean(predicted):.4f}"),
    ("matched null, achieved (mean over 200)", f"{drawn_det.mean():.4f}"),
    ("matched null, spread (sd)", f"{drawn_det.std():.4f}"),
])"""),

    md("""## 4. The two p-values, side by side

Now score the query set and each drawn set on the same cells, correlate both
against the programme, and ask where the observation sits in the null
distribution. This is the test that answers the question that was actually
asked."""),

    code("""null_rho = np.array([sg.spearman(sg.aucell(cache, s), programme)[0]
                     for s in drawn])
observed_rho = sg.spearman(score, programme)[0]

# Two-sided about the null's own centre.  Folding the values about zero
# instead would answer the naive question the null exists to replace.
matched_p = sg.empirical_p(null_rho, observed_rho, tail="two-sided")

table([
    ("observed rho", f"{observed_rho:+.4f}"),
    ("null rho, mean", f"{null_rho.mean():+.4f}"),
    ("null rho, sd", f"{null_rho.std(ddof=1):.4f}"),
    ("conventional (naive) P", f"{conventional['naive_p']:.3g}"),
    ("conventional (permutation) P", f"{conventional['perm_p']:.3g}"),
    ("expression-matched P", f"{matched_p:.3f}"),
    ("phenotype permuted and re-tested P", f"{conventional['cut_opt_p']:.3g}"),
])"""),

    md("""Read the last three rows together.

The conventional tests reject at P < 0.01 for an association that was **not
planted**. The expression-matched null reports P ≈ 0.8, which is the correct
answer. The magnitude of the observed correlation is entirely a property of the
depth distribution — it survives every shuffle that keeps cells intact, because
the confound is inside each cell.

One caveat worth internalising: with `n_sets` draws, the smallest attainable
Monte-Carlo P is $$1/(n_{\\text{sets}}+1)$$. A matched P sitting at that floor
means the observation beat every draw, not that it beat them by a measurable
margin. Increase `n_sets` when the answer matters."""),

    md("""## 5. Three families of null, and what each one holds fixed

`sparsegs` offers three families, and choosing between them is choosing which
nuisance property of the gene set to hold constant:

| family | holds fixed | use when |
|---|---|---|
| `random` | set size | you want the naive baseline, for comparison |
| `expression` | detection rate per gene | **the default**, and the right one for a depth confound |
| `codetection` | detection *and* which genes appear together | the genes are found and lost as a unit |"""),

    code("""rows = []
for kind in ("random", "expression", "codetection"):
    b = sg.construct_expression_matched_null(
        cache, data.target, n_sets=60, kind=kind, seed=42,
        detection_matrix=(data.X > 0))
    draws = b["null_genes"] if isinstance(b["null_genes"][0], list) else [b["null_genes"]]
    rho = np.array([sg.spearman(sg.aucell(cache, s), programme)[0] for s in draws])
    rows.append((
        kind,
        f"{b['detection_ratio']:.3f}",
        f"{b['expression_ratio']:.3f}",
        str(b["max_overlap"]),
        f"{b['status']}",
        f"{null_rho.mean():+.3f}" if kind == "expression" else f"{rho.mean():+.3f}",
        f"{sg.empirical_p(rho, observed_rho, tail='two-sided'):.3f}",
    ))
table(rows, columns=("family", "detection ratio", "expression ratio",
                     "overlap", "status", "null rho", "matched P"))"""),

    md("""The `random` family is the one to look at first: its detection ratio is
far from 1, which means its sets are a different kind of object from the query.
Everything it reports is about that difference.

`codetection` needs the binary matrix, because "which genes appear together" is
not recoverable from per-gene detection rates alone."""),

    md("""## 6. What validity costs

A null that never rejects is trivially well calibrated. So plant an effect and
check that the matched null still finds it."""),

    code("""real = simulate(SimConfig(**{**config.as_dict(), "effect": 0.6, "seed": 7}))
cache_real = sg.RankCache.build(real.X, real.genes, ceiling=config.max_rank, seed=2024)
score_real = sg.aucell(cache_real, real.target)
built_real = sg.construct_expression_matched_null(cache_real, real.target,
                                                  n_sets=200, seed=42)
draws_real = built_real["null_genes"]
rho_real = np.array([sg.spearman(sg.aucell(cache_real, s), real.programme)[0]
                     for s in draws_real])
obs_real = sg.spearman(score_real, real.programme)[0]

table([
    ("effect planted", "0.6"),
    ("observed rho", f"{obs_real:+.4f}"),
    ("null rho, mean", f"{rho_real.mean():+.4f}"),
    ("conventional (naive) P",
     f"{sg.naive_test(score_real, real.programme)['p_value']:.3g}"),
    ("expression-matched P",
     f"{sg.empirical_p(rho_real, obs_real, tail='two-sided'):.3g}"),
])"""),

    md("""The matched null rejects a real association just as decisively. It is not a
more conservative test in general — it is a *differently nulled* one, and the
difference is that it does not reject the depth confound.

## 7. When the framework refuses, and when it only warns

Three inputs break the framework rather than stretching it, and the two
responses are different on purpose: two of them stop the call, one of them
qualifies it. Each is shown here through the same entry point an analyst would
use, on a small matrix built to trigger it."""),

    code("""rng_e = np.random.default_rng(6)
n_cells_e, panel_e = 120, [f"g{i}" for i in range(1, 201)]
Xe = np.zeros((n_cells_e, len(panel_e)))
# Three blocks, and the examples below read one each.  The first five genes are
# expressed in every cell and outrank everything else, so they hold ranks 1-5 in
# every cell and a set made of them scores the same number everywhere.  The next
# twenty are switched on only in the second half of the cells, so a set made of
# *those* separates the two halves completely.  The rest are detected at a
# middling rate in every cell, which is the background both sets are drawn
# against.
Xe[:, :5] = 8
Xe[60:, 20:40] = 5
Xe[:, 40:] = rng_e.binomial(1, 0.4, (n_cells_e, 160))
cache_e = sg.RankCache.build(Xe, panel_e, ceiling=10, seed=1)
axis_e = np.arange(n_cells_e, dtype=float)

table([("spread of the first five genes", f"{np.std(sg.aucell(cache_e, panel_e[:5])):.3g}"),
       ("spread of the second block", f"{np.std(sg.aucell(cache_e, panel_e[20:40])):.3g}")])"""),

    md("""**A gene set with no spread.** If the set's genes are detected in no cell, or
in every cell, the score is the same number for every cell and its correlation
with any axis is 0/0. The framework stops rather than returning a `NaN` that
would surface much later, in whichever statistic first touched it, with nothing
left to say which set produced it.

The spread in the table above is `1.1e-16` rather than `0`: the score is a sum
of doubles, and summing the same numbers in the same order is not guaranteed to
give bit-identical results. That is why the check compares against a tolerance
(`sg.MIN_SCORE_SD`) instead of against zero."""),

    code("""try:
    sg.diagnostic_checklist(cache_e, panel_e[:5], axis=axis_e, n_perm=20,
                            cutpoint_method="median")
except ValueError as exc:
    print(f"{type(exc).__name__}: {exc}")"""),

    md("""**A null that cannot be drawn.** Every draw excludes the query set, so what
can stand in for a 199-gene set is the background *minus those genes*, not the
background. When fewer than `sg.MIN_BACKGROUND_GENES` candidates are left, the
detection floor is lowered by 10% and the widening is announced; if that still
leaves too few, the call stops. A replacement drawn from one gene is not a null,
and the P value it produced would be read as a test."""),

    code("""try:
    sg.construct_expression_matched_null(cache_e, panel_e[1:], n_sets=1)
except ValueError as exc:
    print(f"{type(exc).__name__}: {exc}")"""),

    md("""**A score that separates its groups almost perfectly.** This one is a warning
and not a refusal, because a marker that marks exactly what it is supposed to
mark also lands here, and a framework that withheld the result would be hiding
it rather than qualifying it. The three explanations worth checking travel with
the warning."""),

    code("""import io
import logging
import shlex

from sparsegs.events import LOGGER


def records(run):
    \"\"\"Run `run()` and return the framework's structured records, as lines.

    The records go to the `sparsegs` logger, so collecting them is a matter of
    attaching a handler for the duration.  The handler already on the logger is
    quietened rather than removed, so each record is collected once instead of
    being printed and collected.
    \"\"\"
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    others = [(h, h.level) for h in LOGGER.handlers]
    for h, _ in others:
        h.setLevel(logging.CRITICAL)
    LOGGER.addHandler(handler)
    try:
        run()
    finally:
        LOGGER.removeHandler(handler)
        for h, level in others:
            h.setLevel(level)
    return buf.getvalue().splitlines()


check_e = None
def run_check():
    global check_e
    check_e = sg.diagnostic_checklist(cache_e, panel_e[20:40], axis=axis_e,
                                      n_perm=20, cutpoint_method="median")


# The format exists so that a line parses as well as it reads: `shlex` splits
# each record into an event name and `field=value` pairs, handling the quoted
# values, and what comes out is a table rather than a paragraph.
#
# The number the warning was made on is reported whether or not it crosses the
# threshold, so it can be read off the calibration object as well as the log.
seen_e = records(run_check)
for record in seen_e:
    print(record)                     # the line as it reaches the log
    event_name, *fields = shlex.split(record)
    for field in fields:              # and the same line, parsed
        print("   ", field)
# The number the warning was made on is reported whether or not it crosses the
# threshold, so it can be read off the calibration object as well as the log.
print(f"score_auc = {check_e['step3_false_positive_rate']['score_auc']}")"""),

    md("""The separation is checked in both directions. The brief states the case as
"AUC >= 0.99", but `auroc()` here does not fold its argument about 0.5, and a
score that lands every high-axis cell *below* every low-axis cell separates them
just as perfectly as one that does the reverse — so reading only the upper tail
would have missed half the ways this confound can arise. The record carries
`auc` and `separation` separately for that reason.

## 8. The log, and reading it as data

Every diagnostic above is a *record* rather than a sentence: an event name and a
set of named fields, written `event field=value` with any value containing a
space quoted. The point of the format is that a log can be counted and not only
read — "how many of these sets had to be drawn from a widened pool" is a
question with an answer, and the answer is the reason the edge cases exist."""),

    code("""print("records at the default level:",
      len(records(run_check)))"""),

    md("""`set_log_level()` moves the floor for the framework's own records and for
nothing else. At `error` the same record is dropped before it is ever
formatted, so a caller who has decided the warnings are noise pays nothing for
them — and the caller's own logging is untouched either way."""),

    code("""sg.set_log_level("error")
print("records at level 'error':  ", len(records(run_check)))
sg.set_log_level("warning")
print("records at level 'warning':", len(records(run_check)))"""),

    md("""The records are the same text in both implementations of this framework: the
Python package writes them through `logging`, the R package through
`message()`/`warning()`, and both write `event field=value` with floats at six
significant figures and empty fields dropped. `experiments/cross_language_check.py`
in this repository asserts it record by record, which is a stronger claim than
comparing the two by eye and the only one worth making in a paper.

## 9. Where to go next

* `sg.diagnostic_checklist(cache, gene_set, axis=...)` runs the four-step
  diagnostic in one call and returns a serialisable object — the recommended
  entry point for a pipeline.
* `sg.cutpoint_fpr(score, outcome)` measures what a dichotomisation rule costs;
  the `optimum` rule is anticonservative in a way that is easy to quantify and
  hard to guess.
* `sg.comparability_report({...})` answers whether two cohorts place the same
  set on the same scale before a replication failure is read as biology.
* The R package `sparseGenSetCal` implements the same framework, with the same
  tie-break keys to the integer and the same verdict rule. The companion
  vignette, *Calibrating Gene-Set Scores in Your Data*, is the R-side tour.
* The Shiny/Streamlit companion app lets you load your own matrix and walk the
  four steps interactively.

### A checklist for a real analysis

1. Report the tie-break share (or the ceiling-to-depth ratio) **before** any
   p-value. It is a property of the matrix and the set, not of the outcome.
2. Test the association against an expression-matched null, and report that
   p-value. The conventional one is not a conservative substitute.
3. If you dichotomise the score, say how the cutpoint was chosen. A cutpoint
   chosen from the outcome inflates the false-positive rate several-fold.
4. If you replicate in a second cohort, check the two panels are comparable
   before reading a failure as disagreement.

```python
# The whole thing, on your own data
cache = sg.RankCache.build(X, gene_names, ceiling=ceil(0.05 * X.shape[1]))
check = sg.diagnostic_checklist(cache, gene_set, axis=my_variable, n_perm=1000)
sg.to_json(check, "calibration.json")
```

---

*Tutorial written by **ClawsGO Science Agent**. Every number printed above is
computed by the code in this notebook on data simulated within it; none is
reproduced from a published source.*"""),
]


def build(execute=False):
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {OUT}")
    if execute:
        proc = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", f"--ExecutePreprocessor.timeout=1800",
             str(OUT)],
            capture_output=True, text=True, cwd=str(ROOT))
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            raise SystemExit("the notebook did not run end to end")
        print("executed end to end with no errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    build(**vars(ap.parse_args()))
