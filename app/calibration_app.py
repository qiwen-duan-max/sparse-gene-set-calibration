"""Interactive companion to the sparse gene-set calibration framework.

The app is the four-step diagnostic of the manuscript as a thing you can point
at data: it takes a matrix and a gene set, reports how much of the score the
cell actually measured, builds an expression-matched null, measures what the
cutpoint rule costs, and writes the whole calibration object out as JSON.

Three datasets are built in, each one a failure mode the framework is meant to
catch.  They are simulated, so the app knows what was planted in each and can
show the verdict next to the truth -- which is the only way to see that the
framework's answer is the right one rather than merely a plausible one.

``clean_regime``
    Deep cells, a well-detected 40-gene set, a real association.  The tie-break
    never enters the score.  The framework endorses the association, and
    unticking "tested against the matched null" in the sidebar moves the same
    data to ``INTERPRETABLE_WITH_MATCHED_NULL`` -- the verdict tracks what the
    analysis did, not only what the numbers are.
``confounded``
    Shallow cells, a sparse 8-gene set, no planted association at all, but
    depth tracks the axis.  The conventional correlation rejects at P < 1e-8
    and the matched null does not; the set is not refuted, it is
    unidentifiable.  This is the case the framework exists for.
``co_detected``
    A strongly co-expressed 20-gene set.  The genes are found and lost
    together, so the independence the conventional null assumes fails in the
    other direction, and the set is unusable for a cell-level claim.

Run it with::

    cd naturemethods && streamlit run app/calibration_app.py

A fourth option loads a matrix the user supplies: a delimited text file with
gene names in the first column, plus the gene set pasted in.  Nothing leaves the
session -- the file is read in memory and never written to disk.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scipy.sparse as sp  # noqa: E402

import sparsegs as sg  # noqa: E402
from sparsegs.simulate import SimConfig, simulate  # noqa: E402

# --- logging ---------------------------------------------------------------
# Structured rather than free text: every record carries the step it came from
# and the numbers behind it, so a session can be reconstructed from the log
# alone when a user reports that a verdict looked wrong.
LOG = logging.getLogger("sparsegs.app")
if not LOG.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)

# --- palette ---------------------------------------------------------------
# Okabe-Ito, as used in the manuscript figures: colour-blind safe, and the same
# two colours mean the same two things wherever they appear in the app.
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#8C8C8C"

VERDICT_COLOUR = {
    "INTERPRETABLE": GREEN,
    "INTERPRETABLE_WITH_MATCHED_NULL": ORANGE,
    "NOT_IDENTIFIABLE": "#CC3311",
}


# ---------------------------------------------------------------------------
# example datasets
# ---------------------------------------------------------------------------
EXAMPLES = {
    "1 · clean regime — deep cells, real signal": dict(
        config=SimConfig(n_cells=900, n_genes=6000, n_target=40,
                         median_detected=1500, detection=0.30, effect=0.70,
                         coexpr=0.30, depth_programme_loading=0.0, seed=1),
        headline="Deep cells and a 40-gene set that is detected often enough "
                 "that the rank tie-break never fills a slot.",
        truth="A real association was planted (effect = 0.70) and depth does "
              "not track the axis (loading = 0.0)."),
    "2 · confounded — shallow cells, depth tracks the axis": dict(
        config=SimConfig(n_cells=800, n_genes=6000, n_target=8,
                         median_detected=250, detection=0.01, effect=0.0,
                         coexpr=0.0, depth_programme_loading=1.50, seed=2),
        headline="The ceiling sits far above what the cells detect, and depth "
                 "tracks the axis being tested.",
        truth="Nothing was planted (effect = 0.0). The conventional test "
              "rejects at P < 1e-8 here, and every part of that rejection is "
              "the measurement: the score is a decreasing function of depth, "
              "and depth is a function of the axis."),
    "3 · co-detected — genes found and lost together": dict(
        config=SimConfig(n_cells=900, n_genes=6000, n_target=20,
                         median_detected=900, detection=0.10, effect=0.0,
                         coexpr=1.00, depth_programme_loading=0.0, seed=3),
        headline="The set behaves as one unit, so it departs from the "
                 "independent draw the conventional null assumes.",
        truth="Nothing was planted (effect = 0.0), and the tie-break is not "
              "the problem here: co-detection is."),
}


@st.cache_resource(show_spinner=False)
def build_example(name: str):
    """Simulate an example and cache the matrix, the cache and the truth."""
    spec = EXAMPLES[name]
    data = simulate(spec["config"])
    ceiling = int(np.ceil(spec["config"].rank_frac * data.X.shape[1]))
    cache = sg.RankCache.build(data.X, data.genes, ceiling=ceiling, seed=42)
    LOG.info("example built name=%s n_cells=%d n_genes=%d ceiling=%d "
             "median_depth=%.0f ratio=%.2f", name, data.X.shape[0],
             data.X.shape[1], ceiling, float(np.median(cache.depth)),
             ceiling / max(float(np.median(cache.depth)), 1.0))
    return data, cache, ceiling


@st.cache_resource(show_spinner=False)
def build_uploaded(key: str, X, genes, ceiling):
    """Cache the rank cache for an uploaded matrix, keyed by its signature."""
    return sg.RankCache.build(X, genes, ceiling=ceiling, seed=42)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
    ax.tick_params(colors="#333333", labelsize=9)
    ax.set_facecolor("white")


def fig_depth(cache, ceiling):
    """Depth distribution, with the rank ceiling drawn where it falls."""
    depth = np.asarray(cache.depth, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=140)
    ax.hist(depth, bins=40, color=BLUE, alpha=0.85, edgecolor="white",
            linewidth=0.4)
    ax.axvline(ceiling, color=ORANGE, lw=2)
    ax.set_xlabel("Genes detected in the cell", fontsize=9.5)
    ax.set_ylabel("Cells", fontsize=9.5)
    top = ax.get_ylim()[1]
    ax.annotate(f"rank ceiling {ceiling}", xy=(ceiling, top * 0.93),
                xytext=(6, 0), textcoords="offset points", color=ORANGE,
                fontsize=9, fontweight="bold", va="top")
    med = float(np.median(depth))
    ax.annotate(f"median {med:.0f}", xy=(med, top * 0.55),
                xytext=(-6, 0), textcoords="offset points", color=BLUE,
                fontsize=9, fontweight="bold", ha="right", va="top")
    _clean(ax)
    fig.tight_layout()
    return fig


def fig_decomposition(decomp, depth, gene_set_size):
    """Where the score comes from, in fifths of depth, over scoring cells."""
    total = np.asarray(decomp["total"], dtype=float)
    tie = np.asarray(decomp["tie_break"], dtype=float)
    depth = np.asarray(depth, dtype=float)

    scoring = total > 0
    if scoring.sum() < 25:
        return None
    share = tie[scoring] / total[scoring]
    d = depth[scoring]
    edges = np.quantile(d, np.linspace(0, 1, 6))
    fifth = np.clip(np.digitize(d, edges[1:-1]), 0, 4)

    def by_fifth(v):
        return np.array([v[fifth == i].mean() if (fifth == i).any() else np.nan
                         for i in range(5)])

    x, tb, meas = by_fifth(d), by_fifth(share), 1 - by_fifth(share)
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=140)
    ax.plot(x, meas, "-o", color=BLUE, lw=2, ms=5)
    ax.plot(x, tb, "-o", color=ORANGE, lw=2, ms=5)
    ax.set_ylim(0, 1.16)
    ax.set_xlabel("Genes detected in the cell (fifths, scoring cells only)",
                  fontsize=9.5)
    ax.set_ylabel("Share of the cell's score", fontsize=9.5)
    ax.text(x[0], tb[0], " tie-break fill", color=ORANGE, fontsize=9,
            fontweight="bold", va="bottom")
    ax.text(x[-1], meas[-1], "measured genes ", color=BLUE, fontsize=9,
            fontweight="bold", ha="right", va="bottom")
    _clean(ax)
    fig.tight_layout()
    return fig


def fig_null_correlation(observed, null_values, label):
    """The observed correlation against the distribution the null produced."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=140)
    ax.hist(null_values, bins=18, color=GREY, alpha=0.55, edgecolor="white",
            linewidth=0.4, label="expression-matched null")
    ax.axvline(observed, color=BLUE, lw=2.2)
    span = max(float(np.ptp(null_values)), 1e-6)
    ax.annotate(f"observed {observed:+.3f}", xy=(observed, ax.get_ylim()[1]),
                xytext=(6 if observed < np.median(null_values) else -6, -6),
                textcoords="offset points", color=BLUE, fontsize=9,
                fontweight="bold",
                ha="left" if observed < np.median(null_values) else "right",
                va="top")
    ax.set_xlabel(label, fontsize=9.5)
    ax.set_ylabel("Null draws", fontsize=9.5)
    ax.set_xlim(min(observed, null_values.min()) - 0.15 * span,
                max(observed, null_values.max()) + 0.15 * span)
    _clean(ax)
    fig.tight_layout()
    return fig


def fig_cutpoint(rows):
    """Permutation false-positive rate under each cutpoint rule."""
    fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=140)
    names = [r["rule"] for r in rows]
    fprs = [r["fpr"] for r in rows]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, fprs, height=0.5, color=[BLUE if n == "median"
                                           else ORANGE for n in names])
    for y, f in zip(ypos, fprs):
        ax.text(f + 0.008, y, f"{f:.3f}", va="center", fontsize=9,
                color="#333333", fontweight="bold")
    ax.axvline(0.05, color="#333333", lw=1.2, ls=(0, (4, 3)))
    ax.annotate("nominal 0.05", xy=(0.05, ypos[0] + 0.45), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("False-positive rate under the permuted outcome", fontsize=9.5)
    ax.set_xlim(0, max(max(fprs) * 1.25, 0.16))
    _clean(ax)
    fig.tight_layout()
    return fig


def svg_bytes(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    return buf.getvalue().encode("utf-8")


def show(fig, name, caption=None):
    """Render a figure and offer it as an SVG -- the manuscript format."""
    if fig is None:
        st.info("Not enough scoring cells to draw this figure.")
        return
    st.pyplot(fig, width="stretch")
    if caption:
        st.caption(caption)
    st.download_button("Download as SVG", svg_bytes(fig),
                       file_name=f"{name}.svg", mime="image/svg+xml",
                       key=f"dl_{name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def parse_matrix(text, sep):
    """A delimited matrix with gene names in the first column.

    Returns ``(X_cells_by_genes, genes)``; the file is written genes by cells,
    which is how single-cell matrices are usually exported from R.
    """
    frame = pd.read_csv(io.StringIO(text), sep=sep, index_col=0)
    genes = np.asarray(frame.index.astype(str))
    X = np.asarray(frame.T, dtype=np.float32)
    return sp.csr_matrix(X), genes


def parse_gene_set(text):
    out = []
    for chunk in text.replace(",", "\n").replace("\t", "\n").split("\n"):
        g = chunk.strip()
        if g:
            out.append(g)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def verdict_banner(verdict):
    colour = VERDICT_COLOUR.get(verdict["verdict"], GREY)
    st.markdown(
        f"<div style='border-left:5px solid {colour};padding:10px 16px;"
        f"background:#FAFAFA;border-radius:3px'>"
        f"<span style='font-size:1.35rem;font-weight:600;color:{colour}'>"
        f"{verdict['verdict']}</span>"
        f"<span style='color:#666;font-size:.95rem'> &nbsp;severity "
        f"{verdict['severity']} &nbsp;·&nbsp; criterion "
        f"{verdict.get('criterion', '—')}</span></div>",
        unsafe_allow_html=True)
    for reason in verdict["reasons"]:
        st.markdown(f"- {reason}")


def step1(cache, gene_set, ceiling, rank_frac):
    st.subheader("Step 1 · How much of the score was never measured")
    report = sg.sparsity_report(cache, gene_set, rank_frac=rank_frac)
    LOG.info("sparsity_report k=%d median_detection=%.4f zero_rate=%.4f "
             "tie_break_share=%s", report["n_present"], report["median_detection"],
             report["observed_zero_rate"], report.get("tie_break_share"))

    left, right = st.columns([1, 1])
    with left:
        st.pyplot(fig_depth(cache, ceiling), width="stretch")
    with right:
        median_depth = float(np.median(cache.depth))
        st.metric("Rank ceiling ÷ median depth",
                  f"{ceiling / max(median_depth, 1.0):.2f}",
                  help="Above 1 the cell cannot fill the top block with genes "
                       "it detected, and the tie-break supplies the rest.")
        st.metric("Tie-break share of the set's inclusion",
                  f"{report.get('tie_break_share', float('nan')):.3f}",
                  help="Share of the set's chance of entering the score that "
                       "comes from the ranking rather than from expression.")
        st.metric("Cells scoring exactly zero",
                  f"{report['observed_zero_rate']:.1%}",
                  help=f"Independence would give "
                       f"{report['structural_zero_rate']:.1%}.")

    frame = pd.DataFrame([
        ("genes in the matrix", f"{report['n_present']} of {report['n_requested']}"),
        ("median detection rate", f"{report['median_detection']:.3f}"),
        ("observed zero rate", f"{report['observed_zero_rate']:.3f}"),
        ("zero rate under independence", f"{report['structural_zero_rate']:.3f}"),
        ("deviation", f"{report['zero_rate_excess']:+.3f}"),
        ("max rank / median depth", f"{ceiling / max(median_depth, 1.0):.2f}"),
        ("criterion used", str(report["criterion"])),
    ], columns=["quantity", "value"])
    # Every value is already text: the column would otherwise mix numbers and
    # strings, which the Arrow serialiser behind st.dataframe rejects outright.
    st.dataframe(frame, hide_index=True, width="stretch")
    return report


def step2(cache, gene_set, n_null, seed, ceiling):
    st.subheader("Step 2 · A null the data supports")
    st.markdown(
        "The conventional null shuffles cells, which asks a different question. "
        "Here the replacement sets are drawn from the detection-rate stratum of "
        "each gene of the query, so the null score carries the same "
        "composition and only the identity of the genes changes.")
    # The three edge cases the framework refuses or warns on all arrive here.
    # Caught rather than left to Streamlit, which would print a traceback for
    # what is a statement about the data: a query covering most of the
    # background has nothing left to be replaced by, and the framework says so
    # instead of drawing a null from a handful of genes.
    try:
        built = sg.construct_expression_matched_null(cache, gene_set,
                                                     n_sets=n_null, seed=seed)
    except ValueError as exc:
        st.error(str(exc))
        st.markdown(
            "This is the framework declining rather than approximating. A "
            "replacement set drawn from a handful of genes is not a null, and "
            "the P value it produces would be read as a test. Narrow the gene "
            "set, or run the analysis on a denser panel.")
        LOG.warning("matched null refused: %s", exc)
        return None
    LOG.info("matched null status=%s target_detection=%.4f null_detection=%.4f "
             "max_overlap=%d relaxations=%d", built["status"],
             built["target_detection"], built["null_detection"],
             built["max_overlap"], built["draw_relaxations"])

    if built["det_floor_loosened"]:
        # The widening is surfaced, not buried: a null drawn from a pool that
        # had to be widened is a weaker claim than one that did not, and the
        # difference is invisible in the P value that comes out of it.
        st.warning(
            f"Only {built['n_available_background']} genes could replace this "
            f"set at the usual detection floor, so the floor was lowered to "
            f"{built['pool_floor']:.4g} and the null was drawn from the "
            "widened pool. The match is looser than the tolerance suggests.")
    if built["status"] != "PASS":
        st.warning(
            f"The null could not be matched: status {built['status']}. The "
            "query is so sparse that nothing in the background resembles it, "
            "which is a finding about the data rather than a failure of the "
            "function.")
    ok = st.columns(4)
    ok[0].metric("Target detection", f"{built['target_detection']:.3f}")
    ok[1].metric("Null detection", f"{built['null_detection']:.3f}")
    ok[2].metric("Mean per-set gap", f"{built['mean_detection_gap']:.3f}",
                 help="Tolerance is 0.05 and it is this average that decides "
                      "the status. Worst gap "
                      f"{built['worst_detection_gap']:.3f}, "
                      f"{built['frac_within_tolerance']:.0%} of draws within "
                      "tolerance.")
    ok[3].metric("Genes shared with the query", f"{built['max_overlap']}",
                 help="Has to be zero: a null that reuses a gene of the query "
                      "is the observation with extra steps.")

    frame = pd.DataFrame([
        ("sets drawn", f"{built['n_sets']}"),
        ("detection ratio (null / target)", f"{built['detection_ratio']:.3f}"),
        ("expression ratio (null / target)", f"{built['expression_ratio']:.3f}"),
        ("mean detection gap", f"{built['mean_detection_gap']:.4f}"),
        ("worst detection gap", f"{built['worst_detection_gap']:.4f}"),
        ("draws within tolerance", f"{built['frac_within_tolerance']:.1%}"),
        ("detection floor used", f"{built['det_floor']:.4f}"),
        ("draws that had to relax the pool", f"{built['draw_relaxations']}"),
        ("duplicate picks rejected", f"{built['duplicate_picks']}"),
        ("pool size", f"{built['pool_size']}"),
        ("KS statistic", f"{built.get('ks_statistic', float('nan')):.3f}"),
        ("KS p-value", f"{built.get('ks_pvalue', float('nan')):.3g}"),
    ], columns=["quantity", "value"])
    st.dataframe(frame, hide_index=True, width="stretch")

    # The detection profiles, side by side -- the matching is a claim about
    # these two distributions and is worth looking at rather than trusting.
    pos = [cache._index[g] for g in gene_set if g in cache._index]
    drawn = [cache.detection[cache._index[g]]
             for s in (built["null_genes"] if isinstance(built["null_genes"][0], list)
                       else [built["null_genes"]]) for g in s
             if g in cache._index]
    fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=140)
    bins = np.linspace(0, max(max(cache.detection[pos]), max(drawn)) * 1.05, 22)
    ax.hist(cache.detection[pos], bins=bins, color=BLUE, alpha=0.7,
            edgecolor="white", linewidth=0.4, label="query set")
    ax.hist(drawn, bins=bins, color=ORANGE, alpha=0.55, edgecolor="white",
            linewidth=0.4, label="matched null")
    ax.set_xlabel("Detection rate of the gene", fontsize=9.5)
    ax.set_ylabel("Genes", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    fig.tight_layout()
    show(fig, "matched_null_detection",
         "The two profiles are the quantity being matched; a visible "
         "disagreement here is what the status column is checking numerically.")
    return built


def step3(cache, gene_set, axis, axis_label, built, n_perm, seed, rank_frac):
    st.subheader("Step 3 · What the rule you were going to use actually does")
    ceiling = int(np.ceil(rank_frac * cache.n_genes_total))
    score = sg.aucell(cache, gene_set, max_rank=ceiling)
    observed, naive_p = sg.spearman(score, axis)
    LOG.info("observed correlation=%.4f naive_p=%.4g", observed, naive_p)

    draws = built["null_genes"]
    if draws and not isinstance(draws[0], (list, tuple)):
        draws = [draws]
    null_rho = np.array([sg.spearman(sg.aucell(cache, s, max_rank=ceiling),
                                     axis)[0] for s in draws])
    # Two-sided about the null's own centre, through the package rather than
    # folded about zero here: a matched null can sit slightly off zero when
    # matching is imperfect, and folding about zero would then ask whether the
    # association differs from zero -- the naive question, and the one the null
    # was drawn to replace.
    matched_p = sg.empirical_p(null_rho, observed, tail="two-sided")
    matched_p_upper = sg.empirical_p(null_rho, observed, tail="upper")
    z = ((observed - null_rho.mean()) / null_rho.std(ddof=1)
         if null_rho.std(ddof=1) > 0 else float("nan"))
    LOG.info("matched null rho_mean=%.4f sd=%.4f matched_p=%.4g upper_p=%.4g",
             float(null_rho.mean()), float(null_rho.std(ddof=1)), matched_p,
             matched_p_upper)

    show(fig_null_correlation(observed, null_rho,
                              f"Spearman ρ of the score with {axis_label}"),
         "null_correlation",
         "The grey distribution is what a set of the same composition achieves "
         "by chance. A claim about the query set is a claim about where the "
         "blue line sits relative to it, not about the p-value on its own.")

    left, middle, right = st.columns(3)
    left.metric("Observed ρ", f"{observed:+.3f}")
    middle.metric("Null ρ", f"{null_rho.mean():+.3f} ± {null_rho.std(ddof=1):.3f}")
    right.metric("Standard deviations from the null", f"{z:+.2f}")

    st.markdown(
        f"The conventional test reports "
        f"**P = {naive_p:.3g}** for this correlation, treating cells as "
        f"independent. Against the expression-matched null it is "
        f"**P = {matched_p:.3g}**. The two differ because they are testing "
        f"different null hypotheses, and only the second one is about the gene "
        f"set.")

    outcome = np.asarray(axis, dtype=float) > np.median(axis)
    rows = []
    for rule in ("median", "optimum"):
        res = sg.cutpoint_fpr(score, outcome, n_perm=n_perm, method=rule,
                              seed=seed)
        rows.append(dict(rule=rule, fpr=res["fpr"], observed_p=res["observed_p"],
                         cutpoint=res["observed_cutpoint"]))
        LOG.info("cutpoint_fpr rule=%s fpr=%.4f observed_p=%.4g", rule,
                 res["fpr"], res["observed_p"])
    show(fig_cutpoint(rows), "cutpoint_fpr",
         "Permuting the outcome and re-applying the whole rule, including the "
         "choice of cutpoint, is what measures the cost of choosing it.")

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Tip: with {len(null_rho)} null draws the smallest attainable "
        f"matched-null P is 1/{len(null_rho) + 1} = "
        f"{1 / (len(null_rho) + 1):.3f}, by the (k+1)/(n+1) correction. A "
        f"matched P sitting at that floor means the observation beat every "
        f"draw, not that it beat them by a measurable margin.")
    return rows, observed, naive_p, matched_p, null_rho


def step4(checklist, label):
    st.subheader("Step 4 · The whole checklist in one call")
    st.markdown(
        "`diagnostic_checklist()` is the function to keep in a pipeline: it "
        "runs all four steps on one set and returns a serialisable object.")
    # The criterion lives inside step 1 rather than at the top level: it says
    # which of the two sparsity criteria produced the verdict, which is a
    # property of the report and not of the checklist.
    verdict_banner(dict(
        verdict=checklist["overall_verdict"],
        severity=checklist["confidence"],
        reasons=checklist.get("reasons", []),
        criterion=checklist.get("step1_sparsity", {}).get("criterion", "—")))

    if checklist.get("step3_false_positive_rate", {}).get("attempted") is False:
        st.caption("Step 3 did not run: "
                   + str(checklist["step3_false_positive_rate"].get("note", "")))

    with st.expander("The full calibration object", expanded=False):
        st.json(json.loads(json.dumps(checklist, default=str)))

    payload = json.dumps(checklist, default=str, indent=2)
    st.download_button("Download the calibration object (JSON)",
                       payload.encode("utf-8"),
                       file_name=f"{label.replace(' ', '_')}_calibration.json",
                       mime="application/json")
    return payload


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _running_under_streamlit():
    """Whether this module is being executed as a Streamlit script.

    The app has to call ``main()`` at import when the server runs it, and has
    to *not* do so when the test suite imports it for its helpers.  Streamlit's
    own runtime check is the difference between the two, and it is cheaper than
    splitting the app from its helpers across two files.
    """
    try:
        from streamlit.runtime import exists
    except ImportError:  # pragma: no cover - renamed in some versions
        return False
    return bool(exists())


def main():
    st.set_page_config(page_title="Sparse gene-set calibration",
                       page_icon="📐", layout="wide")
    st.title("Calibrating gene-set scores")
    st.markdown(
        "Every rank-based score asks where a gene sits among *all* genes in a "
        "cell. When the cell detects fewer genes than the rank ceiling, the "
        "empty ranks are settled by the tie-break, and that part of the score "
        "is a function of sequencing depth. This app reports how much of a "
        "score was measured, builds a null the data can support, and measures "
        "what the cutpoint rule costs.")

    with st.sidebar:
        st.header("Data")
        choice = st.radio("Dataset", list(EXAMPLES) + ["4 · your own matrix"],
                          label_visibility="collapsed")

        upl = None
        if choice.startswith("4"):
            file = st.file_uploader("Matrix — genes in the first column",
                                    type=["csv", "tsv", "txt"])
            st.caption("Cells in the columns, genes in the rows, as exported "
                       "from R. Counts are what the functions expect.")
            genes_text = st.text_area("Gene set — one symbol per line",
                                      height=110)
            sep = st.selectbox("Delimiter", ["comma", "tab"]) if file else "comma"
            if file is not None and genes_text.strip():
                try:
                    X, genes = parse_matrix(file.getvalue().decode("utf-8"),
                                            "," if sep == "comma" else "\t")
                    upl = (X, genes, parse_gene_set(genes_text))
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    st.error(f"Could not read the matrix: {exc}")

        st.header("Parameters")
        rank_frac = st.slider("Rank ceiling as a fraction of the panel",
                              0.01, 0.20, 0.05, 0.01,
                              help="AUCell's default is 0.05.")
        n_null = st.slider("Matched null draws", 5, 100, 25, 5,
                           help="Used for the null distribution in step 3.")
        n_perm = st.slider("Cutpoint permutations", 100, 2000, 300, 100)
        seed = st.number_input("Seed", 0, 99999, 42, step=1)

    # --- assemble the dataset ---------------------------------------------
    if choice.startswith("4"):
        if upl is None:
            st.info("Upload a matrix and paste a gene set in the sidebar to "
                    "run the diagnostic on your own data.")
            st.markdown(
                "The three built-in datasets are simulated, so the app knows "
                "what was planted in each and can show the verdict next to "
                "the truth. Nothing is uploaded anywhere: the file is read in "
                "this session and never written to disk.")
            return
        X, genes, gene_set_req = upl
        key = f"{X.shape}-{X.nnz}-{hash(genes.tobytes())}-{rank_frac}"
        ceiling = int(np.ceil(rank_frac * len(genes)))
        cache = build_uploaded(key, X, genes, ceiling)
        gene_set = [g for g in gene_set_req if g in set(genes.tolist())]
        label = "uploaded data"
        truth = None
        axis = np.asarray(cache.depth, dtype=float)
        axis_label = "sequencing depth"
        if len(gene_set) < 3:
            st.error("Fewer than three genes of the set are in the matrix; "
                     "check that the symbols match.")
            return
        built_axis_choice = "Sequencing depth (the only axis available)"
    else:
        data, cache, ceiling = build_example(choice)
        spec = EXAMPLES[choice]
        st.info(spec["headline"])
        truth = spec["truth"]

        with st.sidebar:
            st.header("Gene set")
            which = st.radio(
                "Gene set",
                [f"the planted set ({spec['config'].n_target} genes)",
                 "a random background set of the same size"],
                label_visibility="collapsed")
            axis_choice = st.radio(
                "Test the score against",
                ["the programme (the planted axis)", "sequencing depth"],
                label_visibility="collapsed")

        if which.startswith("the planted"):
            gene_set = list(data.target)
            label = "planted set"
        else:
            rng = np.random.default_rng(seed)
            gene_set = list(rng.choice(
                [g for g in data.genes if not g.startswith("TGT")],
                size=spec["config"].n_target, replace=False))
            label = "background set"
        axis = (np.asarray(data.programme, dtype=float)
                if axis_choice.startswith("the programme")
                else np.asarray(cache.depth, dtype=float))
        axis_label = ("the programme" if axis_choice.startswith("the programme")
                      else "sequencing depth")

    median_depth = float(np.median(cache.depth))
    ratio = ceiling / max(median_depth, 1.0)

    head = st.columns(4)
    head[0].metric("Cells", f"{cache.depth.size:,}")
    head[1].metric("Genes in the panel", f"{cache.n_genes_total:,}")
    head[2].metric("Genes detected (median)", f"{median_depth:,.0f}")
    head[3].metric("Ceiling ÷ depth", f"{ratio:.2f}",
                   help="Above one, the tie-break enters the score.")

    cols = st.columns(4)
    branch = ("the tie-break regime" if ratio > 1 else
              "above the tie-break regime")
    cols[0].success(f"Regime: {branch}")
    cols[1].info(f"Gene set: {len(gene_set)} genes")
    cols[2].info(f"Axis: {axis_label}")
    cols[3].info(f"Seed: {int(seed)}")

    if truth:
        st.caption(f"**What was planted.** {truth}")

    # --- the diagnostic ----------------------------------------------------
    report = step1(cache, gene_set, ceiling, rank_frac)

    with st.sidebar:
        st.header("What the analysis did")
        # The verdict reads the analysis the user is describing, not the one
        # the app would have run.  Both flags start at the honest default: no
        # matched null, no cutpoint.
        used_null = st.checkbox("The association was tested against the "
                                "matched null", value=True)
        cut_choice = st.radio(
            "How the score entered the test",
            ["as a continuous variable",
             "dichotomised at the median",
             "dichotomised at a cutpoint chosen from the outcome"],
            label_visibility="collapsed")
        cutpoint_method = {"as a continuous variable": None,
                           "dichotomised at the median": "median",
                           "dichotomised at a cutpoint chosen from the "
                           "outcome": "optimum"}[cut_choice]

    st.divider()
    built = step2(cache, gene_set, int(n_null), int(seed), ceiling)
    if built is None:
        # Step 2 has already said why, in the terms the framework refused in.
        # Everything below this line reads the null it could not build.
        return
    st.divider()
    cutpoint_rows, observed, naive_p, matched_p, null_rho = step3(
        cache, gene_set, axis, axis_label, built, int(n_perm), int(seed),
        rank_frac)

    st.divider()
    # The checklist rebuilds the null and, when a cutpoint rule is declared,
    # scores the set -- so it can refuse for either of the reasons step 2 and
    # step 3 can.  Caught here for the same reason: a set with no spread is a
    # finding about the data, and a traceback is not how a finding is reported.
    try:
        check = sg.diagnostic_checklist(
            cache, gene_set, axis=axis, label=label, n_perm=int(n_perm),
            used_matched_null=used_null, cutpoint_method=cutpoint_method,
            externally_replicated=None, seed=int(seed))
    except ValueError as exc:
        st.error(str(exc))
        LOG.warning("checklist refused: %s", exc)
        return
    LOG.info("checklist verdict=%s severity=%s", check["overall_verdict"],
             check["confidence"])
    step4(check, label)

    with st.expander("Reading the verdict"):
        st.markdown(
            "- **INTERPRETABLE** — nothing crossed a threshold, and the "
            "association was tested against a matched null.\n"
            "- **INTERPRETABLE_WITH_MATCHED_NULL** — the association may be "
            "real, but every test of it has to be run against the matched "
            "null; the conventional one is not valid.\n"
            "- **NOT_IDENTIFIABLE** — the score cannot carry the cell-level "
            "claim being made about it, whatever the p-value was.\n\n"
            "A verdict of *not identifiable* is not a claim that the "
            "association is false. It is a claim that this matrix, with this "
            "gene set and this scoring rule, cannot tell you either way.")


if _running_under_streamlit():
    main()
