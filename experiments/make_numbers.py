#!/usr/bin/env python
"""Every number the manuscript quotes, read out of the files that produced it.

The manuscript writes no numeral by hand.  Each one is a LaTeX macro defined
here, so a figure that is re-run or a grid that is extended updates the text
along with it, and a reader who wants to check a number has one file to open.
The alternative -- transcribing from a log -- is how a paper ends up quoting a
value that was true of the run before last.

An input that is not on disk yet gets a visibly empty macro rather than a stale
number, so a draft never reads as finished when it is not.

    python experiments/make_numbers.py            # writes manuscript/numbers.tex
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from experiments.analyse_grid import (ALL_TESTS, CONVENTIONAL, MATCHED, channel_summary,  # noqa: E402
                                      implied_calibration, implied_response, load,
                                      reject_table, share_ceiling_correlation,
                                      share_response, trend_test, wilson)

RESULTS = PKG / "results"
ALPHA = 0.05

#: LaTeX names for the analyses, in reporting order.
NAMES = {"naive": "cellcor", "permutation": "permuted", "cutpoint_median": "cutmed",
         "cutpoint_optimal": "cutopt", "random": "rand", "expression": "expr",
         "codetection": "codet"}

#: The boundary rules the paper states, as (co-detection gap, |rho(axis, depth)|,
#: letters-only tag).  Defined once and used both to compute the table and to
#: emit the thresholds, so the prose and the table cannot disagree.
BOUNDARIES = ((0.005, 0.0, "TightAny"), (0.005, 0.2, "TightAxis"),
              (0.010, 0.2, "MidAxis"), (0.020, 0.2, "LooseAxis"))

#: The grids other than the calibration grid, as (results file stem, macro
#: prefix).  The prefix is the tag every macro from that grid carries.
OTHER_GRIDS = (("B_regime", "gridB"), ("C_power", "gridC"),
               ("D_size", "gridD"), ("E_donor", "gridE"))

#: Quantities the cross-cohort summary reports as fractions of sets but the
#: manuscript quotes as percentages.  Named rather than inferred, so that adding
#: a ``{:.0f}`` key to that block cannot silently multiply it by a hundred.
RATE_KEYS = frozenset({"sign_agreement", "strong_sign_agreement"})

#: The factor breakdowns of those grids, as (parameter name, letters-only key
#: used in the macro name).  A factor with a single level is not a breakdown and
#: is skipped wherever these are used.
FACTORS = (("effect", "Eff"), ("n_samples", "Don"), ("n_target", "Size"))


def design_levels(name, factor):
    """The levels a grid is defined at, read from the grid that defines it.

    A level macro is named by the *position* of its value -- ``\\gridEDonA`` is
    the fewest donors because ``A`` is first -- so a declaration made before the
    grid runs has to know the ordering, and the ordering is a property of the
    design.  Read from the results file instead, a half-finished grid would
    answer with a prefix of its levels and no way to tell that it had, and the
    letters would then mean something different in the draft than in the text
    that replaces it.  So the levels come from the grid's own generator, the one
    the driver runs, and a value that reaches the results without being in the
    design is an error rather than a new letter.
    """
    from experiments.run_grid import GRIDS
    return sorted({row[factor] for row in GRIDS[name][0]() if factor in row})


def sci(x):
    """A P value as TeX scientific notation, valid in text or in math mode.

    ``2.6e-124`` written straight into math mode typesets as the italic product
    of a variable e and minus 124, which reads as an expression rather than a
    probability.  Wrapping it in ``\\ensuremath`` makes the macro safe wherever
    the manuscript places it.
    """
    if not np.isfinite(x) or x <= 0:
        return r"\textit{(pending)}"
    exp = int(np.floor(np.log10(x)))
    mant = x / 10.0 ** exp
    return rf"\ensuremath{{{mant:.1f}\times10^{{{exp}}}}}"


def level_tag(levels, value):
    """A letters-only tag for one level of a factor, by its rank among them.

    TeX reads a control sequence as letters and stops at the first digit, so
    ``\\newcommand{\\conf00N}`` defines ``\\conf`` and leaves ``00N`` as body
    text, and ``\\conf00N`` in the manuscript resolves to ``\\conf`` followed by
    the literal characters.  Digits are not merely awkward in a macro name, they
    are impossible, so a level is named by its position -- the comment written
    above each macro records the value it stands for.
    """
    order = sorted(levels)
    i = order.index(value)
    if i >= 26:
        raise ValueError(f"{len(order)} levels will not fit an alphabet")
    return string.ascii_uppercase[i]


def list_text(values):
    """A list of factor levels as it is read out: ``0, 0.3 and 0.6``.

    Joined with a non-breaking space the list typesets as numerals in a row and
    reads as one number, and the sentences that quote these macros name no
    separator of their own -- "loading 0 0.3 0.6" is how a sweep of three
    settings comes to look like a single one.
    """
    parts = [f"{v:g}" for v in values]
    if len(parts) < 2:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


class Numbers:
    """Collects macros, and remembers which ones never got a value."""

    def __init__(self):
        self.rows = []
        self.pending = []
        self.expected = {}

    def declare(self, name, comment=""):
        """Register a macro the manuscript uses before its input exists.

        A macro that is declared and never filled is written out as a visible
        placeholder instead of being omitted, so the manuscript compiles with the
        gap showing rather than failing to compile or, worse, keeping the value
        from the run before last.
        """
        self.expected[name] = comment

    @staticmethod
    def _check(name):
        # A digit anywhere in the name silently defines a different macro than
        # the one the manuscript will reference, and the manuscript still
        # compiles -- it just prints the tail of the name as text.  Better to
        # fail here.
        if not name.isalpha():
            raise ValueError(
                f"macro name {name!r} is not letters-only; TeX control "
                f"sequences cannot contain digits or underscores")
        return name

    def add(self, name, value, fmt="{:.3f}", comment=""):
        self._check(name)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            self.rows.append((name, r"\textit{(pending)}", comment))
            self.pending.append(name)
        else:
            self.rows.append((name, fmt.format(value), comment))

    def raw(self, name, text, comment=""):
        self._check(name)
        self.rows.append((name, text, comment))

    def pct(self, name, value, comment=""):
        self.add(name, value, "{:.1f}", comment)

    def count(self, name, value, comment=""):
        self.add(name, value, "{:,}", comment)

    def write(self, path):
        defined = {name for name, _, _ in self.rows}
        for name, comment in self.expected.items():
            if name not in defined:
                self.rows.append((name, r"\textit{(pending)}",
                                  comment or "declared, input not yet computed"))
                self.pending.append(name)
        # Two inputs can describe the same quantity -- a benchmark reports the
        # number of sets it scored in its log and records them in its table --
        # and if both write the macro, the file defines it twice.  LaTeX treats
        # the second ``\newcommand`` as an error and keeps the first, so the
        # manuscript would quote one of two numbers chosen by the order the
        # blocks happen to run in, which is not a reason.  Whoever writes a
        # quantity twice has to say which input owns it.
        seen = {}
        for name, _, comment in self.rows:
            if name in seen:
                raise ValueError(
                    f"{name} is written twice, by {seen[name]!r} and by "
                    f"{comment!r}; only one of the two inputs can own a macro")
            seen[name] = comment
        # A ``%`` inside a macro body is not a percent sign.  TeX comments out
        # the rest of the line, taking the newline with it and swallowing the
        # next ``\newcommand`` as part of the current argument -- the build then
        # stops with "Runaway argument" pointing at the ``\input`` rather than
        # at the macro that caused it, and every macro defined below that line
        # is reported as undefined.  The value is not escaped here, because
        # escaping it would silently turn a rate into something the line that
        # produced it did not say; the writer refuses and names the macro.
        for name, value, _ in self.rows:
            if re.search(r"(?<!\\)%", value):
                raise ValueError(
                    f"macro {name} has an unescaped % in its body "
                    f"({value!r}); write it as a number and put the \\% in the "
                    f"manuscript, or escape it as \\%")
        lines = [
            "% Generated by experiments/make_numbers.py -- do not edit.",
            "%",
            "% Every numeral the manuscript quotes is defined here, from the files",
            "% the project produces.  A macro reading (pending) marks a quantity",
            "% whose input is not on disk yet; it is not a value to be trusted.",
            "",
        ]
        for name, value, comment in self.rows:
            if comment:
                lines.append(f"% {comment}")
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {path}  ({len(self.rows)} macros, "
              f"{len(self.pending)} pending)")


def rate_macros(n, prefix, entry, comment=""):
    """A rate and its Wilson interval, at three levels of rounding."""
    n.add(f"{prefix}Rate", entry["rate"], "{:.3f}", comment)
    n.add(f"{prefix}Lo", entry["ci_low"], "{:.3f}")
    n.add(f"{prefix}Hi", entry["ci_high"], "{:.3f}")
    n.add(f"{prefix}Pct", entry["rate"] * 100, "{:.1f}")


def write_tail_table(out="manuscript/tail_robustness_table.tex"):
    """The tail-convention table, written as LaTeX from the grid CSVs.

    The paper reports two-sided matched P values, so the table exists to show
    what the other two conventions would have said on the same null draws.  It
    is generated rather than transcribed for the same reason every other number
    in the paper is: the columns are one substitution apart from each other, and
    a table typed by hand would not survive the next re-run.
    """
    cal_path = RESULTS / "grid_A_B_tail_robustness.csv"
    pow_path = RESULTS / "grid_C_tail_robustness_by_effect.csv"
    if not cal_path.exists():
        print(f"  {cal_path.name} not on disk; the tail table is not written")
        return None

    def rows(path, index=None):
        """The CSV as a wide family x tail table of rates.

        The pivot is given a constant index rather than none: with no index,
        pandas returns a Series and the lookup below would silently change type
        between the calibration file and the power file.
        """
        frame = pd.read_csv(path)
        parsed = frame["test"].str.extract(
            r"^(?P<family>\w+) \((?P<tail>[\w-]+)\)$")
        frame = pd.concat([frame, parsed], axis=1)
        if frame["family"].isna().any():
            raise SystemExit(f"{path} has a test label that is not "
                             f"'family (tail)': "
                             f"{sorted(frame.loc[frame['family'].isna(), 'test'])}")
        frame["_all"] = "all"
        wide = frame.pivot_table(index=index or "_all",
                                 columns=["family", "tail"], values="rate",
                                 observed=True)
        return frame, wide

    frame, wide = rows(cal_path)
    key = wide.index[0]
    n_cal = int(frame["n"].max())
    families = [f for f in ("random", "expression", "codetection")
                if (f, "two-sided") in wide.columns]
    lines = [
        "% Generated by experiments/make_numbers.py -- do not edit.",
        "%",
        "% Rejection rates of every matched family read in each of the three",
        "% tails, from results/grid_A_B_tail_robustness.csv and",
        "% results/grid_C_tail_robustness_by_effect.csv.",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Effect & Matched family & Lower & Two-sided & Upper \\\\",
        "\\midrule",
        f"\\multicolumn{{5}}{{l}}{{\\emph{{Grids A and B, no planted effect "
        f"($n = {n_cal:,}$ runs)}} --- these are false-positive rates}} \\\\",
    ]
    for family in families:
        cells = " & ".join(
            f"{float(wide.loc[key, (family, t)]):.3f}"
            if (family, t) in wide.columns else "---"
            for t in ("lower", "two-sided", "upper"))
        lines.append(f" & {family}-matched & {cells} \\\\")
    if pow_path.exists():
        pf = pd.read_csv(pow_path)
        n_pow = int(pf["n"].max())
        lines += [
            "\\midrule",
            f"\\multicolumn{{5}}{{l}}{{\\emph{{Grid C, planted effect "
            f"($n = {n_pow:,}$ runs per effect)}} --- these are power}} \\\\",
        ]
        for effect, block in pf.groupby("effect"):
            first = True
            for family in families:
                sub = block[block["test"].str.startswith(family + " ")]
                cells = []
                for tail in ("lower", "two-sided", "upper"):
                    hit = sub[sub["test"] == f"{family} ({tail})"]
                    cells.append(f"{float(hit['rate'].iloc[0]):.3f}"
                                 if len(hit) else "---")
                label = f"${effect:g}$" if first else ""
                first = False
                lines.append(f"{label} & {family}-matched & "
                             + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The matched null in three conventions, computed from the same",
        "null draws in every case. The paper reports the two-sided column; the",
        "directional columns are given so that the choice can be checked rather",
        "than taken on trust. Reading them together is what separates the two",
        "ways a matched null can fail: a false-positive rate that depends on the",
        "tail is a defect of the test, while a power that the lower tail cannot",
        "reach is not a defect at all but the reason the convention has to be",
        "stated before the grid is run.}",
        "\\label{tab:tail}",
        "\\end{table}",
    ]
    path = PKG / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def _latex_set(name):
    """A gene-set name that can be read and will break across a line.

    MSigDB names are long and carry underscores, which are both a LaTeX escape
    and an opportunity: ``\\_\\allowbreak{}`` prints the underscore and lets the
    line break after it, so a name too wide for its column wraps at a natural
    boundary instead of running into the next column or off the page.
    """
    return name.replace("_", r"\_\allowbreak{}")


def write_generality_table(out="manuscript/generality_table.tex"):
    """The collections, and the named sets a reader checks the framework against.

    Two questions are answered here.  The first table asks whether the regime and
    the diagnostic hold in every MSigDB collection the benchmark draws on; the
    second names the sets a reader will think of first -- hypoxia, cell cycle,
    immune checkpoint, interferon -- because a methods claim that survives only
    in the aggregate is not one a reader can check against the biology they
    know.  Both are written from the benchmark tables by ``analyse_benchmark.py``
    with nothing chosen by hand: theme membership is by name (``THEMES`` there),
    so the second list can be reproduced from the collections themselves.
    """
    from experiments.analyse_benchmark import THEMES

    fams = []
    for cohort in ("gse176078", "gse161529"):
        p = RESULTS / f"benchmark_{cohort}_by_collection.csv"
        if p.exists():
            df = pd.read_csv(p)
            if len(df):
                df["cohort"] = cohort
                fams.append(df)
    themes1 = RESULTS / "benchmark_gse176078_by_theme.csv"
    themes2 = RESULTS / "benchmark_gse161529_by_theme.csv"
    if not fams or not themes1.exists():
        print(f"  benchmark_*_by_collection.csv or {themes1.name} not on disk; "
              f"the generality table is not written")
        return None

    ex1 = pd.read_csv(themes1)
    ex2 = pd.read_csv(themes2) if themes2.exists() else None
    second = (ex2.set_index("gene_set")["median_abs_rho_depth"] if ex2 is not None
              else None)

    head = [
        "% Generated by experiments/make_numbers.py -- do not edit.",
        "%",
        "% Collections and named gene sets, from",
        "% results/benchmark_{gse176078,gse161529}_{by_collection,by_theme}.csv.",
        "",
    ]
    lines = head + [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Collection & Sets & Median $k$ & Detection & Median $|\\rho|$ & "
        "Pairs $\\geq 0.3$ & Within-compartment $\\rho$ \\\\",
        "\\midrule",
    ]
    # One block per cohort under a group heading, rather than a cohort column:
    # the eye compares collections within a cohort, and a repeated label on
    # every row would fight that comparison.
    seen_first = False
    for df in fams:
        label = "GSE176078" if df["cohort"].iloc[0] == "gse176078" else "GSE161529"
        if seen_first:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for _, r in df.sort_values("median_abs_rho_depth",
                                   ascending=False).iterrows():
            lines.append(
                f"{r['label']} & {int(r['n_sets'])} & {r['median_k']:.0f} & "
                f"{100 * r['median_detection']:.1f}\\% & "
                f"{r['median_abs_rho_depth']:.3f} & "
                f"{100 * r['frac_ge_03']:.0f}\\% & {r['detection_vs_absrho']:+.2f} \\\\")
        seen_first = True
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Every collection the benchmark draws on, scored in all",
        "sixteen annotated compartments of GSE176078 and all seventeen of",
        "GSE161529, so a difference between rows within a block is a difference",
        "between sets rather than between the compartments that happen to carry",
        "them. \\emph{Collection} is the MSigDB collection the set names belong",
        "to, with the immunologic signatures grouped by elimination; \\emph{Sets}",
        "counts the sets retained after the usable-overlap filter, \\emph{Detection}",
        "is the median fraction of cells in which a set's genes are seen (per",
        "cent), \\emph{Median $|\\rho|$} the median absolute correlation of the",
        "score with the cell's detected-gene count over that collection's",
        "(compartment, set) pairs, and \\emph{Within-compartment $\\rho$} the",
        "median over compartments of the Spearman correlation between a set's",
        "median detection rate and its absolute score--depth correlation. That",
        "last column is the diagnostic's within-compartment performance, and it",
        "has the same sign in every collection, every compartment and every",
        "cohort.}",
        "\\label{tab:generality}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Theme & Gene set & $k$ & Detection & Median $|\\rho|$, 1st & "
        "Median $|\\rho|$, 2nd \\\\",
        "\\midrule",
    ]
    for theme, _keys in THEMES:
        block = ex1[ex1["theme"] == theme].sort_values(
            "median_abs_rho_depth", ascending=False)
        for i, (_, r) in enumerate(block.iterrows()):
            name = _latex_set(r["gene_set"])
            got = ("---" if second is None or r["gene_set"] not in second.index
                   else f"{second[r['gene_set']]:.3f}")
            label = theme if i == 0 else ""
            lines.append(
                f"{label} & \\multicolumn{{1}}{{p{{6.1cm}}}}{{\\raggedright "
                f"{name}}} & {int(r['k'])} & {100 * r['median_detection']:.1f}\\% & "
                f"{r['median_abs_rho_depth']:.3f} & {got} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The named sets in the four themes a reader is most likely to",
        "test the framework on, selected from the collections by name rather than",
        "chosen: every set whose name carries a hypoxia, cell-cycle, immune-",
        "checkpoint or interferon term is here. \\emph{Median $|\\rho|$} is the",
        "median absolute correlation of the score with the cell's detected-gene",
        "count over the compartments of each cohort: the first is GSE176078, the",
        "second GSE161529, whose cells detect fewer genes against a higher",
        "ceiling (Table~\\ref{tab:cohorts}), so the same arithmetic predicts",
        "more damage there. A dash is a set that one cohort did not retain.}",
        "\\label{tab:themes}",
        "\\end{table}",
    ]
    path = PKG / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def write_null_selector_table(out="manuscript/null_selector_table.tex"):
    """The framework's decision points as a table, from the code that applies them.

    A reader who wants to use the framework needs to know, at each step, which
    reading obliges what.  Restating the thresholds here would make a second
    copy that can drift from the software, so every number is imported from the
    implementation -- ``DEFAULT_CONFIG`` and ``DEFAULT_THRESHOLDS`` from the
    package, the co-detection gap rules from this module's own ``BOUNDARIES`` --
    and the two quoted rates are the grid's measured values, referenced by
    macro so the boundary table remains their only home.
    """
    from sparsegs.calibrate import DEFAULT_CONFIG
    from sparsegs.diagnostics import DEFAULT_THRESHOLDS

    mod = DEFAULT_THRESHOLDS["tie_break_share_moderate"]
    high = DEFAULT_THRESHOLDS["tie_break_share_high"]
    tol = DEFAULT_CONFIG["null_matching_tolerance"]
    perms = DEFAULT_CONFIG["cutpoint_permutations"]
    rc = DEFAULT_CONFIG["replication_cohorts"]
    gaps = sorted({g for g, _, _ in BOUNDARIES})
    axis = max(a for _, a, _ in BOUNDARIES)

    lines = [
        "% Generated by experiments/make_numbers.py -- do not edit.",
        "%",
        "% The framework's decision points.  Thresholds are imported from",
        "% sparsegs.calibrate.DEFAULT_CONFIG and sparsegs.diagnostics.",
        "% DEFAULT_THRESHOLDS; gap rules from BOUNDARIES in this module.",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.4cm}"
        ">{\\raggedright\\arraybackslash}p{4.0cm}"
        ">{\\raggedright\\arraybackslash}p{5.2cm}}",
        "\\toprule",
        "Measurement & Reading & What it means \\\\",
        "\\midrule",
        "Implied product $\\rho_s\\rho_a$ from the depth vector alone "
        "(pre-flight) & inside the 95\\% chance band & "
        "the channel is closed; the conventional test stands, and the "
        "diagnostic's output is the certificate that nothing needs doing \\\\",
        " & outside the band & the channel is open; a conventional $p$-value "
        "is not evidence, whatever it reads \\\\",
        "\\midrule",
        f"\\code{{tie\\_break\\_share}} (step 1) & $< {mod:.2f}$ & "
        "proceed to the null \\\\",
        f" & $\\geq {mod:.2f}$ & most of the score is tie-break; a matched "
        "null is required \\\\",
        f" & $\\geq {high:.2f}$ & \\code{{NOT\\_IDENTIFIABLE}}; report the "
        "refusal rather than a $p$-value \\\\",
        "\\midrule",
        "Drawn null's composition (step 2) & overlap 0, detection gap "
        f"$\\leq {tol:.2f}$ & the draw is a null; its $p$-value is readable "
        "\\\\",
        " & either fails & the draw is not a null; no $p$-value it produces "
        "is evidence \\\\",
        "\\midrule",
        f"Co-detection gap of the best draw, $|\\rho(\\mathrm{{axis}}, "
        f"\\mathrm{{depth}})| > {axis:.1f}$ & $\\leq {gaps[0]:.3f}$ & "
        "matched families nominal "
        f"(\\boundaryTightAnyOutRate) \\\\",
        f" & $> {gaps[0]:.3f}$ & the co-detection family rejects at "
        "\\boundaryTightAxisInRate{} on the same draws; prefer the "
        "expression-matched family and report the achieved gap \\\\",
        f" & $> {gaps[-1]:.3f}$ & no family is safe; the honest output is a "
        "refusal (\\boundaryLooseAxisInRate) \\\\",
        "\\midrule",
        "Cutpoint (step 3) & pre-specified & the nominal rate is readable "
        "\\\\",
        f" & chosen from the outcome & measure the rate by permutation "
        f"({perms:,} draws) rather than assuming it \\\\",
        "\\midrule",
        "Independent cohort (step 4) & sign and rank association & direction "
        f"only, never values; \\code{{replication\\_cohorts}} $= {rc}$ \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The framework's decision points, as the software applies "
        "them.  Every threshold is imported from the implementation rather "
        "than restated, so the table and the code cannot disagree; the three "
        "quoted rates are the grid's measured values from "
        "Table~\\ref{tab:boundary}.  A reading the framework cannot take --- "
        "a cache without depth, falling back on the fixed detection floor "
        "--- weakens the verdict and is flagged as such in the output.}",
        "\\label{tab:selector}",
        "\\end{table}",
    ]
    path = PKG / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def write_tools_table(out="manuscript/tools_table.tex"):
    """Where the tie-break channel sits in each ranking tool the paper names.

    The table exists to keep two kinds of statement apart.  AUCell's default
    ceiling and UCell's shipped 1{,}500 are quoted from those tools' own
    documentation, and the paper measures their consequences here; the rows
    for singscore, GSVA and the module score describe the published
    algorithms, and the last column says plainly that nothing about them was
    measured for this paper.  A comparison table that did not carry that
    distinction would read as a benchmark of tools the project never ran.
    """
    lines = [
        "% Generated by experiments/make_numbers.py -- do not edit.",
        "%",
        "% Tool audit.  AUCell's default ceiling is quoted from the released",
        "% Bioconductor manual (AUCell_calcAUC, aucMaxRank = ceiling(0.05 *",
        "% nrow(rankings))); UCell's from its documentation, as in the Results;",
        "% the remaining rows describe the published algorithms only.",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l>{\\raggedright\\arraybackslash}p{2.0cm}"
        ">{\\raggedright\\arraybackslash}p{3.1cm}"
        ">{\\raggedright\\arraybackslash}p{4.3cm}"
        ">{\\raggedright\\arraybackslash}p{2.5cm}}",
        "\\toprule",
        "Tool & Ranking & Ceiling & Where the channel sits & "
        "Measured here \\\\",
        "\\midrule",
        "AUCell \\citep{aibar2017} & every gene of the panel, within the "
        "cell & released default $\\lceil 0.05\\,G \\rceil$ & empty top "
        "ranks are filled by tie-break: the subject of this paper & "
        "yes --- the grids and the benchmark \\\\",
        "UCell \\citep{andreatta2021} & every gene, within the cell & "
        "ships 1{,}500 for 10x data; its guidance recommends the median "
        "detected & the same fill, at whatever ceiling the call passes & "
        "yes --- default against recommendation, per pair \\\\",
        "singscore \\citep{foroutan2018} & every gene, within the cell & "
        "no truncation & undetected genes tie at the bottom of the "
        "ranking; the fill does not arise, and the arbitrary order inside "
        "the tied block is a different exposure & no \\\\",
        "GSVA \\citep{hanzelmann2013} & every gene, kernel-weighted & "
        "no truncation & bottom-block ties as in singscore, smoothed by "
        "the kernel & no \\\\",
        "module score \\citep{tirosh2016} & none --- a mean of expression "
        "values & --- & no ranking; depth enters through normalisation & "
        "no \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Where the tie-break channel sits in the ranking tools the "
        "introduction names.  The \\emph{Ceiling} column quotes each tool's "
        "own default where one exists: AUCell's from the released "
        "Bioconductor manual, checked against the current release; UCell's "
        "from its documentation, as in the Results.  Rows marked \\emph{no} "
        "in the last column are statements about the published algorithms, "
        "read rather than measured; the table keeps that distinction visible "
        "instead of blurring it.}",
        "\\label{tab:tools}",
        "\\end{table}",
    ]
    path = PKG / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def write_grid_defs_table(out="manuscript/grid_defs_table.tex"):
    """Factor levels, run counts and seed ranges of the five grids.

    Every cell is computed from the grid's own JSONL records -- the same files
    ``analyse_grid.py`` reads -- so the table cannot drift from what was run.
    A factor counts as varied when the records disagree on it; the factors the
    records agree on are listed with the value they were held at, which makes
    the table a complete specification of every grid rather than a summary of
    the interesting columns.
    """
    # Design factors only.  ``median_detected`` is a per-run realised outcome,
    # not a setting: the simulator draws the detection profile and the grid
    # records what came out, so listing its levels would read as if the runs
    # had swept hundreds of detection regimes on purpose.
    factors = [
        ("depth_programme_loading", "programme--depth loading"),
        ("detection", "per-gene detection"),
        ("coexpr", "planted co-expression"),
        ("effect", "planted effect"),
        ("n_target", "target-set size"),
        ("n_samples", "donors"),
    ]

    def fmt_levels(field, vals):
        vals = sorted(vals)
        if field == "detection":
            shown = [f"{v * 100:g}\\%" for v in vals]
        elif field in ("n_target", "n_samples", "median_detected"):
            shown = [f"{v:g}" for v in vals]
        else:
            shown = [f"{v:g}" for v in vals]
        return ", ".join(shown)

    grid_files = sorted((PKG / "results").glob("grid_*.jsonl"))
    if not grid_files:
        print("grid records missing; grid-defs table not written")
        return None
    rows = []
    total_runs = 0
    for gf in grid_files:
        recs = [json.loads(line) for line in gf.read_text().splitlines() if line.strip()]
        total_runs += len(recs)
        first = recs[0]
        varied, fixed = [], []
        for field, label in factors:
            vals = {r.get(field, first["params"].get(field)) for r in recs}
            vals.discard(None)
            if len(vals) > 1:
                varied.append(f"{label}: {fmt_levels(field, vals)}")
            elif vals:
                fixed.append(f"{label} {fmt_levels(field, vals)}")
        configs = len({(r["depth_programme_loading"], r["detection"], r["coexpr"],
                        r["effect"], r.get("n_target"), r.get("n_samples"))
                       for r in recs})
        cells = sorted({r["n_cells"] for r in recs})
        genes = sorted({r["n_genes"] for r in recs})
        ceil_ = sorted({r["max_rank"] for r in recs})
        seeds = [r["seed"] for r in recs]
        cells_s = f"{cells[0]:,}" if len(cells) == 1 else f"{cells[0]:,}--{cells[-1]:,}"
        genes_s = f"{genes[0]:,}" if len(genes) == 1 else f"{genes[0]:,}--{genes[-1]:,}"
        ceil_s = f"{ceil_[0]:,}" if len(ceil_) == 1 else f"{ceil_[0]:,}--{ceil_[-1]:,}"
        name = first["grid"].split("_")[0]
        rows.append(
            f"{name} & {len(recs):,} & {configs} ({len(recs) // configs} each) & "
            f"{cells_s} & {genes_s} & {ceil_s} & "
            f"{'; '.join(varied) if varied else '---'} & "
            f"{'; '.join(fixed) if fixed else '---'} & "
            f"{min(seeds):,}--{max(seeds):,} \\\\"
        )

    lines = [
        "% Generated by experiments/make_numbers.py -- do not edit.",
        "%",
        "% Grid definitions, computed from results/grid_*.jsonl.  A factor is",
        "% listed as varied when the grid's own records disagree on it; the",
        "% fixed column carries the factors every run of that grid shares.",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrcccl>{\\raggedright\\arraybackslash}p{4.0cm}"
        ">{\\raggedright\\arraybackslash}p{3.0cm}r}",
        "\\toprule",
        "Grid & Runs & Configurations & Cells & $G$ & $m$ & "
        "Factors varied (levels) & Held fixed & Seeds \\\\",
        "\\midrule",
    ]
    lines.extend(rows)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The five simulation grids as their own records state them: "
        f"{total_runs:,} runs in all.  A factor is varied when the grid's "
        "records disagree on it, with the observed levels; the held-fixed "
        "column carries the factors every run of that grid shares.  Cells, "
        "panel size $G$, ceiling $m$ and the seed range are counted from the "
        "records, so the table is a complete specification of what was run "
        "rather than a summary of the columns that moved.  The seed base and "
        "the replication rule are stated in the Methods.}",
        "\\label{tab:griddefs}",
        "\\end{table}",
    ])
    path = PKG / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def write_cohorts_table(out="manuscript/cohorts_table.tex"):
    """The two real cohorts side by side, from the logs that produced them.

    A reader deciding whether their own data sit in the regime this paper
    describes needs the cohorts' shape, not their names: how many cells, how
    large a panel, what ceiling the fraction implies, what the cells detect.
    Every cell of the table is read from the benchmark logs and tables rather
    than typed, so a re-run with different filtering moves the table with it.
    """
    rows = []
    for stem, label in (("gse176078", "GSE176078"), ("gse161529", "GSE161529")):
        log = RESULTS / f"log_bench_{stem}.txt"
        path = RESULTS / f"benchmark_{stem}_all.csv"
        if not log.exists() or not path.exists():
            print(f"  {log.name} or {path.name} not on disk; the cohorts table "
                  f"is not written")
            return None
        head = log.read_text(encoding="utf-8")
        m = re.search(r"(\d+)\s+cells\s+x\s+(\d+)\s+genes", head)
        cells, genes = int(m.group(1)), int(m.group(2))
        mr = int(re.search(r"max_rank = (\d+)", head).group(1))
        df = pd.read_csv(path)
        per_compartment = df.groupby("cell_type").agg(
            detected=("median_detected", "median"),
            inclusion=("tie_break_inclusion", "first"))
        rows.append(
            f"{label} & {cells:,} & {genes:,} & {df['cell_type'].nunique()} & "
            f"{mr:,} & {per_compartment['detected'].median():.0f} & "
            f"{per_compartment['inclusion'].median():.3f} & "
            f"{df['gene_set'].nunique()} \\\\")

    lines = [
        "% Generated by experiments/make_numbers.py -- do not edit.",
        "%",
        "% The two benchmark cohorts, from results/log_bench_*.txt and the",
        "% benchmark tables themselves.",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Cohort & Cells & Genes & Compartments & Ceiling & Median detected & "
        "Inclusion & Sets scored \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The two real cohorts the benchmark runs on. \\emph{Ceiling}",
        "is the rank ceiling the 5\\% AUCell fraction implies for that panel;",
        "because it is a fraction of the panel, the same gene set lands on a",
        "different scale in each cohort, which is why scores are compared between",
        "cohorts only in direction and rank association.",
        "\\emph{Median detected} is the median over compartments of the median",
        "genes a cell detects there, \\emph{inclusion} the median over",
        "compartments of the tie-break inclusion probability, and \\emph{Sets",
        "scored} the sets retained after the usable-overlap filter.}",
        "\\label{tab:cohorts}",
        "\\end{table}",
    ]
    path = PKG / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def main(out="manuscript/numbers.tex", allow_partial=False):
    n = Numbers()

    # ---- the study's own dataset, for the constants in the mechanism ----
    # Declared here and filled from the benchmark's log below, because these are
    # the QC pipeline's output and not decisions: a re-run with different
    # filtering has to move every sentence that quotes the panel with it, and a
    # constant typed here would stay behind.  ``rankFrac`` is the exception --
    # it is AUCell's own default fraction, a property of the method rather than
    # of the matrix.
    n.declare("panelGenes", "genes in the GSE176078 panel used")
    n.raw("rankFrac", "0.05")
    n.declare("maxRank", "rank ceiling: the top block the score reads")

    # ---- grid A: calibration under the null --------------------------
    path = RESULTS / "grid_A_calibration.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    a = load("A_calibration", allow_partial=allow_partial)
    n.count("gridARuns", len(a))
    overall = reject_table(a)
    for label, _ in ALL_TESTS:
        if label in overall:
            rate_macros(n, NAMES[label], overall[label],
                        f"grid A, pooled: {label}")
            lo = overall[label]["ci_low"]
            hi = overall[label]["ci_high"]
            n.raw(f"{NAMES[label]}Calibrated",
                  "yes" if lo <= ALPHA <= hi else "no")

    # by confounding, which is the shape of the conventional failure
    n.raw("confLevels", list_text(
        sorted(a["depth_programme_loading"].unique())))
    levels = a["depth_programme_loading"].unique()
    for level, sub in a.groupby("depth_programme_loading"):
        tag = f"conf{level_tag(levels, level)}"
        table = reject_table(sub)
        n.count(f"{tag}N", len(sub), f"confounding {level:g}")
        # The level as a numeral, for table rows that label themselves rather
        # than sharing one legend line.
        n.raw(f"{tag}Level", f"{level:g}", f"programme-depth loading {level:g}")
        # What the outcome-chosen cutpoint costs over the median split, at this
        # level of confounding.  The paper states it as a range, so both ends
        # are derived rather than typed.
        if "cutpoint_median" in table and "cutpoint_optimal" in table:
            n.pct(f"{tag}CutCost",
                  (table["cutpoint_optimal"]["rate"]
                   - table["cutpoint_median"]["rate"]) * 100,
                  f"percentage points the outcome-chosen cutpoint costs over "
                  f"the median split at loading {level:g}")
        for label in ("naive", "cutpoint_optimal", "expression", "codetection"):
            if label in table:
                n.add(f"{tag}{NAMES[label].capitalize()}",
                      table[label]["rate"], "{:.3f}",
                      f"confounding {level:g}: {label}")

    cs = channel_summary(a)
    for regime in ("open", "closed"):
        n.count(f"channel{regime.capitalize()}N", cs[regime]["n"])
        for label in ("naive", "expression", "codetection", "random"):
            if label in cs[regime]:
                rate_macros(n, f"chan{regime.capitalize()}{NAMES[label].capitalize()}",
                            cs[regime][label],
                            f"channel {regime}: {label}")

    # Replicates per configuration, so that a statement about precision is
    # anchored to the grid rather than to whatever the run script was passed.
    # Only the *design* columns go in the key: ``median_detected`` and
    # ``depth_ratio`` are achieved by the simulation, not chosen, so including
    # them would split each configuration into one group per run and report a
    # replicate count of one.
    params_cols = [c for c in ("depth_programme_loading", "detection", "coexpr",
                               "n_target", "n_cells", "effect", "n_samples",
                               "n_genes", "rank_frac")
                   if c in a and a[c].nunique() >= 1]
    params_cols = [c for c in params_cols if c not in
                   ("median_detected", "depth_ratio", "tie_break_share")]
    if params_cols:
        per_cell = a.groupby(params_cols).size()
        n.count("gridAReplicates", int(per_cell.min()),
                f"replicates per grid-A configuration (range "
                f"{int(per_cell.min())}--{int(per_cell.max())})"
                if per_cell.nunique() > 1 else
                f"replicates per grid-A configuration")
        n.count("gridACells", int(len(per_cell)), "distinct grid-A configurations")

    # the response curve: does the diagnostic predict the failure
    sr = share_response(a, test="naive")
    if sr is not None:
        n.add("shareLowestRate", sr["rate"].iloc[0], "{:.3f}",
              "naive rejection in the lowest tie-break-share bin")
        n.add("shareHighestRate", sr["rate"].iloc[-1], "{:.3f}",
              "naive rejection in the highest tie-break-share bin")
        n.add("shareLowestBin", sr["tie_break_share_low"].iloc[0], "{:.3f}")
        n.add("shareHighestBin", sr["tie_break_share_high"].iloc[-1], "{:.3f}")
    sr_expr = share_response(a, test="expression")
    if sr_expr is not None:
        n.add("shareMatchedLowest", sr_expr["rate"].iloc[0], "{:.3f}")
        n.add("shareMatchedHighest", sr_expr["rate"].iloc[-1], "{:.3f}")

    # ---- the pre-flight product, which is what actually predicts --------
    # The response curve the study set out to test was against the tie-break
    # share.  It is reported too, and reported as a refutation: the quantity the
    # first version proposed points the wrong way.  Both are emitted so that the
    # manuscript cannot quote one without the other being available.
    for label, _ in ALL_TESTS:
        name = NAMES[label].capitalize()
        imp = implied_response(a, test=label)
        if imp is not None:
            n.add(f"implied{name}LowRate", imp["rate"].iloc[0], "{:.3f}",
                  f"naive-scale rejection in the lowest |implied rho| bin: {label}")
            n.add(f"implied{name}LowLo", imp["ci_low"].iloc[0], "{:.3f}")
            n.add(f"implied{name}LowHi", imp["ci_high"].iloc[0], "{:.3f}")
            n.add(f"implied{name}HighRate", imp["rate"].iloc[-1], "{:.3f}",
                  f"rejection in the highest |implied rho| bin: {label}")
            n.add(f"implied{name}HighLo", imp["ci_low"].iloc[-1], "{:.3f}")
            n.add(f"implied{name}HighHi", imp["ci_high"].iloc[-1], "{:.3f}")
            n.add(f"implied{name}LowBin", imp["implied_high"].iloc[0], "{:.3f}")
            n.add(f"implied{name}HighBin", imp["implied_high"].iloc[-1], "{:.3f}")
            n.count(f"implied{name}Bins", len(imp))
            # Every bin, so the text can quote the middle of the curve without
            # the number being typed in by hand.
            for i, (rate, lo, hi) in enumerate(
                    zip(imp["rate"], imp["ci_low"], imp["ci_high"])):
                b = string.ascii_uppercase[i]
                n.add(f"implied{name}Bin{b}", rate, "{:.3f}",
                      f"rejection in |implied rho| bin {i + 1} of {len(imp)}: {label}")
                n.add(f"implied{name}Bin{b}Lo", lo, "{:.3f}")
                n.add(f"implied{name}Bin{b}Hi", hi, "{:.3f}")
        for column, prefix in (("implied", "implied"), ("tie_break_share", "share")):
            tr = trend_test(a, test=label, column=column)
            if tr is not None:
                n.add(f"{prefix}{name}TrendZ", tr["z"], "{:+.2f}",
                      f"Cochran-Armitage trend in {label} rejection "
                      f"across bins of {column}")
                n.raw(f"{prefix}{name}TrendP", sci(tr["p"]),
                      f"Cochran-Armitage P, {label} against {column}")

    cal = implied_calibration(a)
    if cal is not None:
        n.add("impliedCorr", cal["pearson"], "{:.3f}",
              "correlation of the pre-flight product with the realised rho")
        n.add("impliedSpearman", cal["spearman"], "{:.3f}")
        n.add("impliedSlope", cal["slope"], "{:.3f}",
              "slope of realised rho on the pre-flight product")
        n.add("impliedMeanPredicted", cal["mean_implied"], "{:+.4f}")
        n.add("impliedMeanRealised", cal["mean_observed"], "{:+.4f}")
        n.count("impliedN", cal["n"])
        # The extreme bin's edge, so the text can say where the curve ends
        # without naming a number the analysis did not produce.
        imp0 = implied_response(a, test="naive")
        if imp0 is not None:
            n.add("impliedTopDecile", imp0["implied_median"].iloc[-1], "{:.3f}",
                  "median |implied rho| in the most confounded bin")

    # How nearly the share is a relabelling of the ceiling ratio, on grid A.  It
    # is the calibration grid's counterpart of the same number on the other
    # sweeps, and the pair is what shows the share's sign is not its own.
    sc_a = share_ceiling_correlation(a)
    if sc_a is not None:
        n.add("shareCeilRhoA", sc_a["rho"], "{:+.2f}",
              "grid A: the tie-break share against the ceiling ratio")

    # ---- the other grids, as they land --------------------------------
    # Declared first, so that the manuscript compiles with the gap visible
    # while the grid is still running rather than failing on an undefined macro.
    for tag in ("gridB", "gridC", "gridD", "gridE"):
        n.declare(f"{tag}Runs", f"{tag}: number of runs")
        for label in NAMES.values():
            n.declare(f"{tag}{label.capitalize()}", f"{tag}: {label}")
    # The per-level family is declared alongside the pooled rates, and for the
    # same reason: the manuscript names a factor level by its position, so the
    # donor-count macros are used by the text whether or not the donor grid has
    # finished.  A declaration covering only the pooled rates left those
    # undefined, which is the quieter of the two failures -- TeX reports the
    # control sequence it does not know and prints the rest of the name as
    # ordinary text, so the page shows a fragment of a macro name where a rate
    # belongs and a reader takes it for a word.
    for name, tag in OTHER_GRIDS:
        for factor, key in FACTORS:
            levels = design_levels(name, factor)
            if len(levels) < 2:
                continue
            n.declare(f"{tag}{key}s", f"{tag}: the values of {factor}")
            for value in levels:
                tagl = f"{tag}{key}{level_tag(levels, value)}"
                for label in ("naive", "expression", "codetection"):
                    n.declare(f"{tagl}{NAMES[label].capitalize()}",
                              f"{tag}, {factor} {value:g}: {label}")

    for name, tag in OTHER_GRIDS:
        p = RESULTS / f"grid_{name}.jsonl"
        if not p.exists():
            continue
        df = load(name, allow_partial=allow_partial)
        n.count(f"{tag}Runs", len(df))
        tab = reject_table(df)
        for label, _ in ALL_TESTS:
            if label in tab:
                n.add(f"{tag}{NAMES[label].capitalize()}", tab[label]["rate"],
                      "{:.3f}", f"{name}: {label}")
        # The power grid is the one where a rate is a detection rate and not a
        # false-positive rate, so it is broken out by the planted effect.
        for factor, key in FACTORS:
            if factor not in df or df[factor].nunique() < 2:
                continue
            # The design's levels, not the file's, so that the letters mean the
            # same thing in a half-run draft as in the finished text.  A value
            # the grid does not define has no position and therefore no letter,
            # which is a change to the design and has to be made in both places.
            flevels = design_levels(name, factor) or sorted(df[factor].unique())
            n.raw(f"{tag}{key}s", list_text(flevels))
            for level, sub in df.groupby(factor):
                if level not in flevels:
                    raise ValueError(
                        f"{name} has runs at {factor} = {level:g}, which the "
                        f"grid definition does not produce; the level macros "
                        f"are named by position and cannot be assigned until "
                        f"the grid and its generator agree")
                st = reject_table(sub)
                tagl = f"{tag}{key}{level_tag(flevels, level)}"
                for label in ("naive", "expression", "codetection"):
                    if label in st:
                        n.add(f"{tagl}{NAMES[label].capitalize()}",
                              st[label]["rate"], "{:.3f}",
                              f"{name}, {factor} {level:g}: {label}")
        cs = channel_summary(df)
        if cs:
            for regime in ("open", "closed"):
                n.count(f"{tag}Chan{regime.capitalize()}N", cs[regime]["n"])
                for label in ("naive", "expression"):
                    if label in cs[regime]:
                        n.add(f"{tag}Chan{regime.capitalize()}"
                              f"{NAMES[label].capitalize()}",
                              cs[regime][label]["rate"], "{:.3f}")
        # Every grid gets its own implied-rho trend, so that the response curve
        # is shown to replicate on a sweep the calibration grid did not run
        # rather than being an artefact of grid A's factor levels.
        for label, _ in ALL_TESTS:
            tr = trend_test(df, test=label, column="implied")
            if tr is not None:
                n.add(f"{tag}Implied{NAMES[label].capitalize()}Z", tr["z"], "{:+.2f}",
                      f"{name}: Cochran-Armitage trend in {label} across "
                      f"|implied rho| bins")
                n.raw(f"{tag}Implied{NAMES[label].capitalize()}P", sci(tr["p"]),
                      f"{name}: the same trend's P value")
            # The refuted candidate's trend on the same runs, reported beside
            # the product's: the share's sign is not stable from one sweep to
            # the next, which is a second reason to abandon it.
            ts = trend_test(df, test=label, column="tie_break_share")
            if ts is not None:
                n.add(f"{tag}Share{NAMES[label].capitalize()}Z", ts["z"], "{:+.2f}",
                      f"{name}: the same trend against the tie-break share")
                n.raw(f"{tag}Share{NAMES[label].capitalize()}P", sci(ts["p"]),
                      f"{name}: the share trend's P value")
        # How nearly the share is a relabelling of the ceiling ratio on this
        # sweep.  Where it is high, the share's marginal trend belongs to the
        # ceiling ratio, and the share's sign is not its own.
        sc = share_ceiling_correlation(df)
        if sc is not None:
            n.add(f"{tag}ShareCeilRho", sc["rho"], "{:+.2f}",
                  f"{name}: the tie-break share against the ceiling ratio")
            n.raw(f"{tag}ShareCeilP", sci(sc["p"]),
                  f"{name}: that correlation's P value")
        # The achieved range of the quantity the grid sweeps, when that
        # quantity is continuous rather than a set of named levels.  Only the
        # ends are quoted, so the macro stays short whatever the grid's size.
        for col, key in (("depth_ratio", "Ratio"), ("median_detected", "Detected"),
                         ("n_genes", "Genes")):
            if col in df and df[col].nunique() > 3:
                n.add(f"{tag}{key}Min", float(df[col].min()), "{:.2f}",
                      f"{name}: smallest {col} over the grid")
                n.add(f"{tag}{key}Max", float(df[col].max()), "{:.2f}",
                      f"{name}: largest {col} over the grid")
        # The cutpoint's floor, read in the least confounded quintile: the part
        # of its error that the confound does not explain.
        ir = implied_response(df, test="cutpoint_optimal", n_bins=5)
        if ir is not None:
            n.add(f"{tag}CutoptFloor", ir["rate"].iloc[0], "{:.3f}",
                  f"{name}: outcome-cutpoint rejection in the least confounded "
                  f"quintile, where the channel contributes least")

    # ---- the two candidates in one model ---------------------------------
    # The decisive form of the refutation.  Binned rates show that the share
    # moves the wrong way; a model with both candidates in it shows that the
    # share carries nothing once the product is present, and that on its own it
    # is significantly negative.  Read from the file the analysis writes, so the
    # manuscript cannot quote a fit that has not been run.
    for macro, comment in (("jointShareCoef", "share's coefficient beside the product"),
                           ("jointShareZ", "share's z beside the product"),
                           ("jointShareP", "share's P beside the product"),
                           ("jointProductCoef", "product's coefficient"),
                           ("jointProductZ", "product's z"),
                           ("jointProductP", "product's P"),
                           ("jointShareAloneCoef", "share alone: coefficient"),
                           ("jointShareAloneP", "share alone: P"),
                           ("jointN", "runs in the joint model")):
        n.declare(macro, comment)
    jl_path = RESULTS / "grid_A_B_joint_logistic.csv"
    if jl_path.exists():
        jl = pd.read_csv(jl_path)
        joint = jl[jl["model"] == "product and share"].set_index("term")
        alone = jl[jl["model"] == "share alone"].set_index("term")
        n.count("jointN", int(jl["n"].iloc[0]))
        if {"product", "share"} <= set(joint.index):
            n.add("jointShareCoef", joint.loc["share", "coef"], "{:+.3f}",
                  "share's coefficient with the product in the model")
            n.add("jointShareZ", joint.loc["share", "z"], "{:+.2f}",
                  "share's Wald z with the product in the model")
            n.raw("jointShareP", sci(joint.loc["share", "p"]),
                  "share's P with the product in the model")
            n.add("jointProductCoef", joint.loc["product", "coef"], "{:+.3f}",
                  "product's coefficient with the share in the model")
            n.add("jointProductZ", joint.loc["product", "z"], "{:+.2f}")
            n.raw("jointProductP", sci(joint.loc["product", "p"]))
        if "share" in alone.index:
            n.add("jointShareAloneCoef", alone.loc["share", "coef"], "{:+.3f}",
                  "share's coefficient on its own, which is negative")
            n.raw("jointShareAloneP", sci(alone.loc["share", "p"]),
                  "share's P on its own")

    # ---- the decomposition's residual, measured ----------------------
    # Equation (3) writes the score as detected part + closed-form
    # expectation + residual; the residual is what a reader has to see
    # measured before trusting the decomposition per cell.  The sweep runs in
    # the pipeline before this step, and every number below is its record --
    # none is typed here.
    res_path = RESULTS / "decomposition_residual.json"
    if res_path.exists():
        res_raw = json.loads(res_path.read_text())
        res = res_raw["overall"]
        recs = res_raw["records"]
        corrs = [r["eps_depth_corr"] for r in recs]
        n.count("residNDatasets", res["n_datasets"])
        n.count("residNCells", res["n_cells"])
        n.raw("residExactGap", sci(res["exact_gap_max"]),
              "max |detected + tie_break - score| over the sweep")
        n.add("residMean", float(np.average(
            [r["eps_mean"] for r in recs],
            weights=[r["n_cells"] for r in recs])), "{:+.4f}",
            "cell-weighted mean of the residual")
        n.add("residSdMedian", res["eps_sd_median"], "{:.3f}",
              "median over datasets of the residual SD")
        n.add("residSdMax", res["eps_sd_max"], "{:.3f}")
        n.add("residMaxAbs", res["eps_max_abs_over_datasets"], "{:.2f}",
              "largest |residual| over all cells")
        n.add("residTailAbs", max(r["eps_q999_abs"] for r in recs), "{:.2f}",
              "largest 99.9th-percentile |residual| over datasets")
        n.add("residCorrMin", min(corrs), "{:+.2f}",
              "min corr(residual, depth) over datasets")
        n.add("residCorrMax", max(corrs), "{:+.2f}")

    # The cost of choosing the cutpoint from the outcome, as the range the
    # manuscript quotes rather than the two endpoints picked by hand.
    costs = []
    for level, sub in a.groupby("depth_programme_loading"):
        t = reject_table(sub)
        if "cutpoint_median" in t and "cutpoint_optimal" in t:
            costs.append((t["cutpoint_optimal"]["rate"]
                          - t["cutpoint_median"]["rate"]) * 100)
    if costs:
        n.pct("cutCostMinPct", min(costs),
              "smallest gap between the outcome-chosen cutpoint and the median split")
        n.pct("cutCostMaxPct", max(costs),
              "largest such gap, across the confounding levels")

    # ---- null composition actually achieved ---------------------------
    for label in ("random", "expression", "codetection"):
        col = f"{label}_null_detection_ratio"
        if col in a:
            n.add(f"{NAMES[label]}DetRatio", float(a[col].mean()), "{:.3f}",
                  f"grid A: achieved detection ratio of the {label} family")
            n.add(f"{NAMES[label]}DetRatioSd", float(a[col].std()), "{:.3f}")

    # ---- the response curve's extremes, declared before the curve exists --
    for name, comment in (("shareLowestRate", "naive rejection, lowest share bin"),
                          ("shareHighestRate", "naive rejection, highest share bin"),
                          ("shareLowestBin", "lowest bin's upper edge"),
                          ("shareHighestBin", "highest bin's upper edge"),
                          ("shareMatchedLowest", "matched rejection, lowest bin"),
                          ("shareMatchedHighest", "matched rejection, highest bin")):
        n.declare(name, comment)

    # ---- the real-data benchmark --------------------------------------
    for name, comment in (("benchSets", "gene sets benchmarked"),
                          ("benchCompartments", "compartments benchmarked"),
                          ("benchMedianDetected", "median genes detected"),
                          ("benchDepthRatio", "median ceiling-to-depth ratio"),
                          ("benchTieBreak",
                           "median tie-break inclusion: the chance that a gene "
                           "the cell cannot detect is ranked inside the ceiling "
                           "anyway -- the regime's exposure, not its effect"),
                          ("benchRealisedShare",
                           "median realised tie-break share: of the score the "
                           "sets actually received, the fraction that came from "
                           "ranks the tie-break settled"),
                          ("benchChannelOpen", "compartments with the channel open"),
                          ("benchChannelClosed", "compartments with the channel shut"),
                          ("benchMedianDetectedCdT",
                           "median genes detected by CD8+ T cells, the compartment "
                           "the introduction quotes")):
        n.declare(name, comment)

    # The shape of the dataset the benchmark ran on.  Read from the benchmark's
    # own log rather than typed, because these are the QC pipeline's output and
    # a re-run with different filtering must move the text with it.
    log = RESULTS / "log_bench_gse176078.txt"
    if log.exists():
        head = log.read_text(encoding="utf-8")
        m = re.search(r"(\d+)\s+cells\s+x\s+(\d+)\s+genes", head)
        if m:
            n.count("benchCells", int(m.group(1)), "cells in the benchmark matrix")
            n.count("benchGenes", int(m.group(2)), "genes in the benchmark matrix")
            # The mechanism's constants come from the same line: the panel is
            # what the cells were scored against, so the closed form and the
            # benchmark cannot be about two different matrices.
            genes = int(m.group(2))
            n.raw("panelGenes", f"{genes:,}".replace(",", "{,}"),
                  "genes in the GSE176078 panel used")
        m = re.search(r"(\d+)\s+groups on '([^']+)'", head)
        if m:
            n.count("benchGroups", int(m.group(1)),
                    f"groups the benchmark split on ('{m.group(2)}')")
        m = re.search(r"max_rank = (\d+)", head)
        if m:
            rank = int(m.group(1))
            n.raw("benchMaxRank", f"{rank:,}",
                  "rank ceiling used by the benchmark")
            n.raw("maxRank", f"{rank:,}".replace(",", "{,}"),
                  "rank ceiling: the top block the score reads")
        # The number of sets scored is not read from the log even though the log
        # states it: the scored table records which sets were scored, and a
        # count taken from the summary of a run is the kind of number that can
        # be right about the run and wrong about the file beside it.
        # Collections kept, so that the Methods can name them accurately rather
        # than assert a count that a different MSigDB release would change.
        cols = re.findall(r"^\s+([\w.]+)\.v[\d.]+\.Hs\.symbols\.gmt: (\d+) sets kept",
                          head, re.M)
        if cols:
            n.count("benchCollections", len(cols), "MSigDB collections used")
            n.raw("benchCollectionNames", ", ".join(
                c.replace(".Hs", "").replace("_", " ") for c, _ in cols))

    bench_path = RESULTS / "benchmark_gse176078_all.csv"
    if bench_path.exists():
        b = pd.read_csv(bench_path)
        n.count("benchSets", b["gene_set"].nunique() if "gene_set" in b else len(b))
        if "cell_type" in b:
            n.count("benchCompartments", b["cell_type"].nunique())
        for col, tag in (("median_detected", "benchMedianDetected"),
                         ("depth_ratio", "benchDepthRatio"),
                         ("tie_break_inclusion", "benchTieBreak")):
            if col in b:
                n.add(tag, float(b[col].median()), "{:.2f}",
                      f"benchmark: median {col} over compartments")
                n.add(f"{tag}Min", float(b[col].min()), "{:.2f}")
                n.add(f"{tag}Max", float(b[col].max()), "{:.2f}")
        n.raw("benchChannelOpen", str(int((b.groupby("cell_type")["depth_ratio"]
                                           .median() > 1).sum())))
        n.raw("benchChannelClosed", str(int((b.groupby("cell_type")["depth_ratio"]
                                             .median() <= 1).sum())))
        if "cell_type" in b and "median_detected" in b:
            for ct, sub in b.groupby("cell_type"):
                if ct.startswith("CD8"):
                    n.count("benchMedianDetectedCdT", int(sub["median_detected"].median()),
                            f"median genes detected by {ct}")
                    break

    # The realised share comes from its own file: it needs the cache, so
    # `compartment_tie_break.py` computes it after the benchmark rather than the
    # benchmark recording it.  One value per compartment here, taken as the
    # median over the sets scored in it, so the macro answers "what did the
    # tie-break do to a typical score in a typical compartment".
    for name, comment in (
            ("benchRealisedShare",
             "benchmark: median realised tie-break share over the "
             "(compartment, set) pairs"),
            ("benchRealisedShareMin", "benchmark: lowest compartment median "
                                      "realised share"),
            ("benchRealisedShareMax", "benchmark: highest compartment median "
                                      "realised share")):
        n.declare(name, comment)

    tb_path = RESULTS / "compartment_tie_break_gse176078.csv"
    if tb_path.exists():
        tb = pd.read_csv(tb_path)
        if "score_tie_break_share" in tb:
            share = tb["score_tie_break_share"]
            n.add("benchRealisedShare", float(share.median()), "{:.2f}",
                  "benchmark: median realised tie-break share over "
                  f"{len(tb)} (compartment, set) pairs")
            if "cell_type" in tb:
                by_ct = tb.groupby("cell_type")["score_tie_break_share"].median()
                n.add("benchRealisedShareMin", float(by_ct.min()), "{:.2f}",
                      "benchmark: lowest compartment median realised share")
                n.add("benchRealisedShareMax", float(by_ct.max()), "{:.2f}",
                      "benchmark: highest compartment median realised share")

    # ---- the two levels of the real-data analysis ----------------------
    for name, comment in (
            ("benchLevelOneRho",
             "across compartments: Spearman(tie-break inclusion, median |rho "
             "with depth|)"),
            ("benchLevelOneAbsMin",
             "across compartments: the smallest median |rho| with depth"),
            ("benchLevelOneAbsMax",
             "across compartments: the largest median |rho| with depth"),
            ("benchShareVsAbsRho",
             "within compartment: median Spearman(realised tie-break share, "
             "|rho with depth|)"),
            ("benchDetVsAbsRho",
             "within compartment: median Spearman(median detection rate, |rho "
             "with depth|), the detector the paper's first version proposed"),
            ("benchFracAbsRhoGtThree",
             "fraction of (compartment, set) pairs with |rho_depth| >= 0.3"),
            ("benchFracAbsRhoGtFive",
             "fraction of (compartment, set) pairs with |rho_depth| >= 0.5"),
            ("benchUcellBelow", "fraction of pairs where UCell's default "
                                "normaliser moves the score away from its own "
                                "recommendation")):
        n.declare(name, comment)

    summary_path = RESULTS / "benchmark_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ds = summary.get("gse176078", {})
        n.add("benchLevelOneRho", ds.get("level1_spearman"), "{:+.2f}",
              "benchmark: level 1 across compartments")
        # The compartment-level AUC at the 0.3 line is deliberately not
        # reported, and the reason is not that it is low.  The summary carries
        # it as NaN: no compartment's median |rho| reaches 0.3, so the
        # classification has one class and the AUC is undefined rather than
        # unimpressive.  What the data supports is the ordering, so the spread
        # is quoted and the AUC is left out; a macro that could only ever read
        # a marker would fail the build for a contrast the data does not
        # contain, which is not the same thing as a number that went missing.
        ct_path = RESULTS / "benchmark_gse176078_by_cell_type.csv"
        if ct_path.exists():
            ct = pd.read_csv(ct_path)
            if "median_abs_rho_depth" in ct.columns and len(ct):
                n.add("benchLevelOneAbsMin",
                      float(ct["median_abs_rho_depth"].min()), "{:.2f}",
                      "benchmark: smallest compartment median |rho| with depth")
                n.add("benchLevelOneAbsMax",
                      float(ct["median_abs_rho_depth"].max()), "{:.2f}",
                      "benchmark: largest compartment median |rho| with depth")
        for col, tag in (("frac_abs_rho_depth_ge_0_3", "benchFracAbsRhoGtThree"),
                         ("frac_abs_rho_depth_ge_0_5", "benchFracAbsRhoGtFive")):
            n.pct(tag, 100 * ds[col] if col in ds else None,
                  f"benchmark: {col.replace('_', ' ')}")
        l2 = ds.get("level2_median", {}) or {}
        n.add("benchShareVsAbsRho",
              l2.get("rho_score_tie_break_share_vs_absrho"), "{:+.2f}",
              "benchmark: the realised share predicts a set's own depth "
              "correlation within a compartment")
        n.add("benchDetVsAbsRho",
              l2.get("rho_median_detection_vs_absrho"), "{:+.2f}",
              "benchmark: the detection rate, for comparison")
        # Generality: the same quantities per collection and for the named sets
        # a reader checks first.  Filled from the CSVs the analysis writes, so a
        # collection where the diagnostic inverts would move the text rather
        # than only the table.  Both cohorts are read here, by name: the range
        # macros span the two files, so a sign flip in either cohort moves the
        # sentence, and the second-cohort fill cannot silently depend on block
        # order.
        fams = []
        for cohort in ("gse176078", "gse161529"):
            p = RESULTS / f"benchmark_{cohort}_by_collection.csv"
            if p.exists():
                df = pd.read_csv(p)
                if len(df):
                    df["cohort"] = cohort
                    fams.append(df)
        if fams:
            both = pd.concat(fams, ignore_index=True)
            n.add("genDiagMin", float(both["detection_vs_absrho"].min()),
                  "{:+.2f}", "generality: smallest within-compartment "
                  "Spearman(detection, |rho|) across collections and cohorts")
            n.add("genDiagMax", float(both["detection_vs_absrho"].max()),
                  "{:+.2f}", "generality: largest such correlation")
            n.add("genAbsMin", float(both["median_abs_rho_depth"].min()),
                  "{:.2f}", "generality: smallest collection median |rho|")
            n.add("genAbsMax", float(both["median_abs_rho_depth"].max()),
                  "{:.2f}", "generality: largest collection median |rho|")
        ths = {}
        for cohort in ("gse176078", "gse161529"):
            p = RESULTS / f"benchmark_{cohort}_by_theme.csv"
            if p.exists():
                df = pd.read_csv(p)
                if len(df):
                    ths[cohort] = df
        if ths:
            n.count("genThemeSets",
                    int(pd.concat(ths.values(),
                                  ignore_index=True)["gene_set"].nunique()),
                    "generality: named sets in the reader-checked themes")
            # Macro names carry no digits, so the named sets are tagged by
            # word: PdOne, CtlaFour, IfNg.
            tags = {"HALLMARK_HYPOXIA": ("genHypoxiaRho", "genHypoxiaGe",
                                         "genHypoxiaSecond"),
                    "KEGG_CELL_CYCLE": ("genCellCycleRho", "genCellCycleGe",
                                        "genCellCycleSecond"),
                    "REACTOME_PD_1_SIGNALING": ("genPdOneRho", "genPdOneGe",
                                                "genPdOneSecond"),
                    "BIOCARTA_CTLA4_PATHWAY": ("genCtlaFourRho",
                                               "genCtlaFourGe",
                                               "genCtlaFourSecond"),
                    "HALLMARK_INTERFERON_GAMMA_RESPONSE": (
                        "genIfNgRho", "genIfNgGe", "genIfNgSecond")}
            first = "gse176078"
            second = "gse161529"
            for gene_set, (t_rho, t_ge, t_second) in tags.items():
                in_first = first in ths and gene_set in set(
                    ths[first]["gene_set"])
                in_second = second in ths and gene_set in set(
                    ths[second]["gene_set"])
                if in_first:
                    row = ths[first][ths[first]["gene_set"] == gene_set]
                    n.add(t_rho,
                          float(row["median_abs_rho_depth"].iloc[0]),
                          "{:.2f}", f"generality: {gene_set} median |rho|, "
                          "first cohort")
                    n.pct(t_ge, 100 * float(row["frac_ge_03"].iloc[0]),
                          f"generality: {gene_set}, compartments at "
                          "|rho| >= 0.3")
                # Declared even when the row is missing, so the sentence that
                # compares cohorts reads (pending) instead of an undefined
                # macro.  Filled only when the second cohort actually scored
                # the set.
                if in_first or in_second:
                    n.declare(t_second, f"generality: {gene_set} median "
                              "|rho|, second cohort")
                if in_second:
                    row = ths[second][ths[second]["gene_set"] == gene_set]
                    n.add(t_second,
                          float(row["median_abs_rho_depth"].iloc[0]),
                          "{:.2f}", f"generality: {gene_set} median |rho|, "
                          "second cohort")
        uc = ds.get("ucell", {}) or {}
        n.pct("benchUcellBelow",
              100 * uc["frac_rho_below_090"] if "frac_rho_below_090" in uc else None,
              "benchmark: fraction of pairs where UCell's default and its own "
              "recommendation disagree materially")

    # ---- the second panel ---------------------------------------------
    # The replication is reported as the second panel's own arithmetic rather
    # than as a repetition of the first panel's numbers, because the ceiling is
    # a fraction of the panel and the same gene set therefore lands on a
    # different score scale in each.  A sentence that quoted one panel's ceiling
    # for both would be wrong about one of them.
    #
    # Declared before the fill, so a run whose second benchmark has not finished
    # leaves the macros visibly pending rather than absent: the manuscript's
    # sentence about the second panel is then either a number from that run or a
    # hole a reader can see, and never a leftover from an earlier one.
    for name, comment in (
            ("benchCohortGenes", "second panel: genes in the assayed panel"),
            ("benchCohortMaxRank", "second panel: the rank ceiling"),
            ("benchCohortGroups", "second panel: annotated cell types scored"),
            ("benchCohortCells", "second panel: cells across those cell types"),
            ("benchCohortMedianCells",
             "second panel: median cells scored in a compartment"),
            ("benchCohortDetected",
             "second panel: median genes detected, median over compartments"),
            ("benchCohortInclusion",
             "second panel: median tie-break inclusion probability"),
            ("benchCohortRhoDepth",
             "second panel: median correlation of the score with depth"),
            ("benchCohortSets", "gene sets scored in both panels"),
            ("benchCohortRhoAcross",
             "across those sets: Spearman between panels of the depth-axis "
             "correlation"),
            ("benchCohortSign",
             "across those sets: share that keeps the sign of its depth-axis "
             "correlation"),
            ("benchCohortShift",
             "across those sets: median absolute change in the depth-axis "
             "correlation"),
            ("benchCohortStrongCut",
             "the |rho_axis| a set needs in the first panel to count as strong"),
            ("benchCohortStrongN", "sets above that cut in the first panel"),
            ("benchCohortStrongSign",
             "share of those sets keeping the sign of the association"),
            ("benchCohortStrongFirst",
             "their median |rho_axis| in the first panel"),
            ("benchCohortStrongSecond",
             "their median |rho_axis| in the second panel")):
        n.declare(name, comment)

    # The panel's shape comes from the second benchmark's log, for the same
    # reason the first panel's does: it is the QC pipeline's output, and a
    # re-run with different filtering has to move the text with it.
    log = RESULTS / "log_bench_gse161529.txt"
    if log.exists():
        head = log.read_text(encoding="utf-8")
        m = re.search(r"(\d+)\s+cells\s+x\s+(\d+)\s+genes", head)
        if m:
            n.count("benchCohortGenes", int(m.group(2)),
                    "second panel: genes in the assayed panel")
        m = re.search(r"max_rank = (\d+)", head)
        if m:
            n.raw("benchCohortMaxRank", f"{int(m.group(1)):,}",
                  "second panel: the rank ceiling")

    second = RESULTS / "benchmark_gse161529_all.csv"
    if second.exists():
        s = pd.read_csv(second)
        if "cell_type" in s:
            # The shape of the second panel, counted the same way the first
            # panel's is: compartments, and cells summed over compartments
            # rather than over rows, since ``n_cells`` repeats for every gene
            # set scored in that compartment.
            per_ct = s.groupby("cell_type")
            n.count("benchCohortGroups", int(per_ct.ngroups),
                    "second panel: annotated cell types scored")
            if "n_cells" in s:
                n.count("benchCohortCells",
                        int(per_ct["n_cells"].first().sum()),
                        "second panel: cells across those cell types")
                n.count("benchCohortMedianCells",
                        int(per_ct["n_cells"].first().median()),
                        "second panel: median cells scored in a compartment")
        for col, tag, fmt in (("median_detected", "benchCohortDetected", "{:.0f}"),
                              ("tie_break_inclusion", "benchCohortInclusion", "{:.3f}"),
                              ("rho_depth", "benchCohortRhoDepth", "{:+.2f}")):
            if col in s and "cell_type" in s:
                # One value per compartment first, then the median over them,
                # which is the same reduction the first panel's macros use.  The
                # pooled median over pairs would weight a compartment by how
                # many sets happened to be scored in it.
                by_ct = s.groupby("cell_type")[col].median()
                n.add(tag, float(by_ct.median()), fmt,
                      f"second panel: median {col} across compartments")

    if summary_path.exists():
        cl = json.loads(summary_path.read_text(encoding="utf-8")
                        ).get("cross_cohort", {}) or {}
        if "n_common_gene_sets" in cl:
            n.count("benchCohortSets", int(cl["n_common_gene_sets"]),
                    "gene sets scored in both panels")
        for key, tag, fmt in (
                ("spearman_rho_axis_across_cohorts", "benchCohortRhoAcross", "{:+.2f}"),
                # The two agreement rates are written as numbers and the sign
                # is put in the text, because a bare ``%`` inside a macro body
                # is not a percent sign: TeX comments out the rest of the line,
                # which takes the following ``\newcommand`` with it and ends
                # the file mid-argument.  The guard in ``write`` says so if it
                # happens.
                ("sign_agreement", "benchCohortSign", "{:.0f}"),
                ("median_abs_change", "benchCohortShift", "{:.2f}"),
                ("strong_axis_threshold", "benchCohortStrongCut", "{:.2f}"),
                ("strong_sign_agreement", "benchCohortStrongSign", "{:.0f}"),
                ("strong_median_abs_1", "benchCohortStrongFirst", "{:.2f}"),
                ("strong_median_abs_2", "benchCohortStrongSecond", "{:.2f}")):
            if key in cl:
                # ``RATE_KEYS`` are fractions in the summary and percentages in
                # the text, so the scaling is by name rather than by format:
                # another ``{:.0f}`` key added to the list above would otherwise
                # be multiplied by a hundred for looking like a rate.
                scale = 100.0 if key in RATE_KEYS else 1.0
                n.add(tag, scale * cl[key], fmt, f"cross-cohort: {key}")
        if "n_strong_in_1" in cl:
            n.count("benchCohortStrongN", int(cl["n_strong_in_1"]),
                    "sets above the strength cut in the first panel")

    # ---- the calibration audit ----------------------------------------
    audit_path = RESULTS / "audit_matched_null.csv"
    audit_raw = RESULTS / "audit_matched_null.jsonl"
    if audit_raw.exists():
        first = json.loads(audit_raw.open().readline())
        if "n_null" in first:
            n.count("auditDraws", int(first["n_null"]),
                    "matched draws per test in the audit")
        if "n_cells" in first:
            n.count("auditCells", int(first["n_cells"]),
                    "cells per simulated dataset in the audit")
        if "n_genes" in first:
            n.count("auditGenes", int(first["n_genes"]),
                    "genes in the audit's simulated panel")
        if "n_target" in first:
            n.count("auditTarget", int(first["n_target"]),
                    "genes in the audit's target set")
        if "median_detected" in first:
            n.count("auditDetected", int(first["median_detected"]),
                    "median genes detected per cell in the audit")
        if "depth_ratio" in first and "median_detected" in first:
            n.raw("auditDepthRatio",
                  f"{first['n_genes'] * first.get('rank_frac', 0.05) / first['median_detected']:.2f}",
                  "audit: ceiling-to-depth ratio, held at the CD8+ T value")
    if audit_path.exists():
        au = pd.read_csv(audit_path)
        for _, row in au.iterrows():
            tag = ("".join(w.capitalize() for w in row["audit"].split("_")))
            kind = "Real" if row["kind"] == "real_target" else "Pseudo"
            n.add(f"audit{tag}{kind}Fpr", row["fpr"], "{:.3f}",
                  f"audit {row['audit']} / {row['kind']}")
            n.add(f"audit{tag}{kind}Disp", row["dispersion_ratio"], "{:.2f}")
            n.count(f"audit{tag}{kind}N", int(row["n"]),
                    f"audit {row['audit']} / {row['kind']}: sets tested")
            # The interval travels with the rate.  One of these rates -- the
            # exchangeable sets in the sparse configuration -- is the highest of
            # its three and sits close enough to the nominal level that the
            # claim "within nominal" cannot be checked from the rate alone.
            k = int(round(float(row["fpr"]) * int(row["n"])))
            lo, hi = wilson(k, int(row["n"]))
            n.add(f"audit{tag}{kind}Lo", lo, "{:.3f}",
                  f"audit {row['audit']} / {row['kind']}: Wilson lower bound")
            n.add(f"audit{tag}{kind}Hi", hi, "{:.3f}", "")
            if "tie_break" in row:
                # Keyed by kind as well: the tie-break share is a property of
                # the dataset, so it is the same number on the real and pseudo
                # rows, and dropping ``kind`` would define the macro twice.
                n.add(f"audit{tag}{kind}TieBreak", row["tie_break"], "{:.3f}",
                      f"audit {row['audit']}: tie-break share")

    # The same audit as a ratio, which is the form the paper's claim takes: an
    # exchangeable set and a set carrying structure the null cannot reproduce,
    # under the same confounder.
    if audit_path.exists():
        au = pd.read_csv(audit_path)

        def _fpr(config, kind):
            m = (au["audit"] == config) & (au["kind"] == kind)
            return float(au.loc[m, "fpr"].iloc[0]) if m.any() else None

        n.add("auditExchangeableFpr", _fpr("confounded_coexpressed", "pseudo_target"),
              "{:.3f}", "audit: a draw from the null's own generator, co-expressed config")
        n.add("auditStructuredFpr", _fpr("confounded_coexpressed", "real_target"),
              "{:.3f}", "audit: the simulator's target set, same config")
        n.add("auditSparseTargetFpr", _fpr("confounded_sparse", "real_target"), "{:.3f}")
        n.add("auditSparsePseudoFpr", _fpr("confounded_sparse", "pseudo_target"), "{:.3f}")
        # The interval for that rate, under the same short name.  It is the one
        # rate in the audit that sits close enough to the nominal level for the
        # interval to decide whether the claim "within nominal" holds, so the
        # text quotes it rather than the rate alone.
        _m = (au["audit"] == "confounded_sparse") & (au["kind"] == "pseudo_target")
        if _m.any():
            _k = int(round(float(au.loc[_m, "fpr"].iloc[0]) * int(au.loc[_m, "n"].iloc[0])))
            _lo, _hi = wilson(_k, int(au.loc[_m, "n"].iloc[0]))
            n.add("auditSparsePseudoLo", _lo, "{:.3f}",
                  "audit sparse configuration, exchangeable sets: Wilson lower bound")
            n.add("auditSparsePseudoHi", _hi, "{:.3f}", "")
        n.add("auditControlTargetFpr", _fpr("control_unconfounded", "real_target"),
              "{:.3f}")
        n.add("auditControlPseudoFpr", _fpr("control_unconfounded", "pseudo_target"),
              "{:.3f}")
        # The contrast the audit is built to make, as one number per
        # configuration.  It is informative only where the sets carry structure
        # the draws cannot reproduce; where they do not, the two rates are the
        # same number to within what 24 and 192 sets can resolve, and reporting
        # the ratio is how that gets said rather than hidden.
        for label, config in (("Structured", "confounded_coexpressed"),
                              ("Sparse", "confounded_sparse")):
            real = _fpr(config, "real_target")
            pseudo = _fpr(config, "pseudo_target")
            if real is not None and pseudo:
                n.add(f"audit{label}Ratio", real / pseudo, "{:.1f}",
                      f"audit {config}: the simulator's targets against sets "
                      f"exchangeable with the draws")
        n.count("auditReps", int(au.loc[au["kind"] == "real_target", "n"].max()),
                "datasets per audit configuration")
        n.count("auditPseudo", int(au.loc[au["kind"] == "pseudo_target", "n"].max()),
                "exchangeable pseudo-targets per configuration")

    # ---- the co-expression split of the calibration grid ----------------
    # The audit says the residual belongs to sets carrying structure the null
    # cannot reproduce.  The grid says the same thing independently, through its
    # co-expression factor, and the two are worth reporting as one claim.
    if "coexpr" in a and a["coexpr"].nunique() > 1:
        n.raw("coexprLevels", list_text(sorted(a["coexpr"].unique())))
        clevels = a["coexpr"].unique()
        for level, sub in a.groupby("coexpr"):
            tag = f"coexpr{level_tag(clevels, level)}"
            table = reject_table(sub)
            n.count(f"{tag}N", len(sub), f"co-expression {level:g}")
            for label in ("naive", "expression", "codetection"):
                if label in table:
                    n.add(f"{tag}{NAMES[label].capitalize()}", table[label]["rate"],
                          "{:.3f}",
                          f"grid A at co-expression {level:g}: {label}")
                    lo, hi = table[label]["ci_low"], table[label]["ci_high"]
                    n.raw(f"{tag}{NAMES[label].capitalize()}Calibrated",
                          "yes" if lo <= ALPHA <= hi else "no",
                          f"co-expression {level:g}, {label}: "
                          f"Wilson interval covers the nominal level")

    # ---- the co-detection boundary table --------------------------------
    # Two quantities, neither of which uses the outcome, separate the
    # co-detection-matched null's successes from its failures.  The rule the
    # paper states is read off this table, so it is generated rather than typed.
    if "codetection_null_codetection_gap" in a:
        gap = a["codetection_null_codetection_gap"].abs()
        axis = a["programme_depth_rho"].abs()
        for gt, at, tag in BOUNDARIES:
            n.add(f"boundary{tag}Gap", gt, "{:.3f}",
                  f"boundary rule: co-detection gap threshold, {tag}")
            if at > 0:
                n.add(f"boundary{tag}Axis", at, "{:.1f}",
                      f"boundary rule: |rho(axis, depth)| threshold, {tag}")
            m = (gap > gt) & (axis > at)
            rule = (f"co-detection gap > {gt} and |rho(axis,depth)| > {at}"
                    if at > 0 else f"co-detection gap > {gt}")
            for flag, sub, suffix in ((True, a[m], "In"), (False, a[~m], "Out")):
                if len(sub) == 0:
                    continue
                e = reject_table(sub)["codetection"]
                n.count(f"boundary{tag}{suffix}N", e["n"],
                        f"runs satisfying: {rule}" if flag else "runs not satisfying it")
                n.add(f"boundary{tag}{suffix}Rate", e["rate"], "{:.3f}",
                      f"co-detection FPR where {rule}" if flag
                      else "co-detection FPR elsewhere")
                n.add(f"boundary{tag}{suffix}Lo", e["ci_low"], "{:.3f}")
                n.add(f"boundary{tag}{suffix}Hi", e["ci_high"], "{:.3f}")

    n.write(PKG / out)
    if n.pending:
        print("pending: " + ", ".join(n.pending[:12])
              + (" ..." if len(n.pending) > 12 else ""))
    # The generated tables are fragments rather than macros: their cells are
    # only meaningful next to each other, and each is a view of the results the
    # prose summarises.
    write_tail_table()
    write_cohorts_table()
    write_generality_table()
    write_null_selector_table()
    write_tools_table()
    write_grid_defs_table()
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="manuscript/numbers.tex")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write numbers from a sweep that is still running")
    main(**vars(ap.parse_args()))
