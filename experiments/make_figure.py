#!/usr/bin/env python
"""The companion figure: why the null has to be matched, in four panels.

The figure is the paper's argument in the order the paper makes it.

**a. The mechanism.**  Equation (1) drawn for the panel the study scores:
the probability that a rank the cell cannot fill is settled by the tie-break,
against the number of genes the cell detected.  The compartments of the real
dataset are marked at their own median depth, so the curve and the data are on
one axis.

**b. Where real data sits.**  The regime against what it did.  The x axis is
``max_rank / median detected`` -- the width of the top block relative to the
genes a cell detects, where a value above one means the block cannot be filled
by counts at all.  The y axis is the share of the score that came from ranks the
tie-break settled rather than from counts the cell observed, which is a
consequence of the regime and not a second reading of it.  A compartment to the
right of one has the channel open.

**c. Does the diagnostic predict the failure?**  The conventional test's
rejection rate under ``effect = 0``, in quantile bins of the pre-flight product
the paper argues for: the correlation of the score with depth times the
correlation of the tested axis with depth, both of which need no outcome.  The
row beneath shows the same runs binned by the quantity the paper first proposed
for that role, the tie-break share, and falls instead of rising: the refutation
is drawn rather than asserted.

**d. Where the boundary is.**  The same rates split by whether the channel is
open at all.  Pooling the two regimes averages a real effect with its absence,
so the split is the framework's own claim about when the problem exists.

Every number comes from files the project produces; nothing is transcribed.  A
missing input stops the script rather than leaving a panel out, because a
three-panel version of a four-panel figure is the kind of thing that gets
noticed after submission.

    python experiments/make_figure.py [--outdir figures]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from experiments.analyse_grid import load, reject_table, wilson  # noqa: E402

RESULTS = PKG / "results"

#: Okabe-Ito.  Every panel carries its own legend; the palettes are kept off each
#: other's colours so that nothing in the figure has to be disambiguated by
#: remembering which panel it came from.
BLUE, ORANGE, GREEN, PURPLE, GREY = ("#0072B2", "#D55E00", "#009E73",
                                     "#CC79A7", "#7F7F7F")
CONVENTIONAL_LABEL = {"naive": "cell-level correlation",
                      "permutation": "permuted outcome",
                      "cutpoint_median": "median split",
                      "cutpoint_optimal": "cutpoint chosen from outcome"}
MATCHED_LABEL = {"random": "size-matched null",
                 "expression": "expression-matched null",
                 "codetection": "co-detection-matched null"}

#: The scoring panel of the study, and the AUCell default fraction.  These are
#: properties of the dataset, not of this figure, and they are the numbers the
#: manuscript quotes.
PANEL_GENES = 24646
RANK_FRAC = 0.05
MAX_RANK = int(np.ceil(RANK_FRAC * PANEL_GENES))
ALPHA = 0.05

RC = {
    "font.size": 7.4,
    "axes.labelsize": 7.4,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    # The fonts are named rather than inherited, and named to the one matplotlib
    # ships with.  The host's default list decides what an unnamed figure is
    # typeset in, and on this machine that list leads with a CJK face, so every
    # Latin glyph in the figure came from it while the maths came from the
    # default maths font -- two typefaces in one panel, and a figure that would
    # change appearance on any other machine.  DejaVu Sans is present wherever
    # matplotlib is, so a reader who re-runs this gets the same page.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",     # keep text as text, so the SVG is editable
    "pdf.fonttype": 42,         # TrueType, which is what the journal accepts
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
}


def open_channel_curve(n_genes, max_rank, depths):
    """Equation (1): P(a missing gene is placed inside the top ``max_rank``)."""
    d = np.asarray(depths, dtype=float)
    return np.clip((max_rank - d) / np.maximum(n_genes - d, 1), 0, None)


def panel_a(ax, compartments):
    d = np.linspace(0, max(compartments["median_detected"].max() * 1.6, 2000),
                    400)
    ax.plot(d, open_channel_curve(PANEL_GENES, MAX_RANK, d), color=BLUE, lw=1.6)
    ax.axvline(MAX_RANK, color=ORANGE, lw=1.2, ls=(0, (4, 2.5)))

    # The real compartments, at their own median depth.  Jittered vertically
    # only, because the x position is the datum and must not be moved.
    ys = open_channel_curve(PANEL_GENES, MAX_RANK, compartments["median_detected"])
    ax.scatter(compartments["median_detected"], ys, s=11, color=GREEN,
               edgecolor="white", linewidth=0.4, zorder=5)
    med = float(np.median(compartments["median_detected"]))
    # Placed in the empty upper right rather than beside the point it names.
    # Above and to the left of the median compartment the only thing there is
    # the curve, which is descending through exactly that corner: the label sat
    # on the line, and a label the line runs through reads as two overlapping
    # marks rather than as a name.
    ax.annotate(f"median compartment\n{med:,.0f} genes detected",
                xy=(med, open_channel_curve(PANEL_GENES, MAX_RANK, [med])[0]),
                xytext=(0.28, 0.42), textcoords="axes fraction", color=GREEN,
                fontsize=6.6, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6))

    ax.set_xlabel("Genes detected in the cell")
    ax.set_ylabel("P(rank filled by tie-break)")
    ax.set_xlim(0, d.max())
    ax.set_ylim(0, max(0.09, float(np.nanmax(ys)) * 1.25))
    # Named once the limits are set, because the label is anchored inside them.
    # Written against a y of 0.62 -- which reads as a fraction of an axis and
    # was taken as a value on one that runs to 0.09 -- it was drawn above the
    # frame and appeared nowhere on the page.
    ax.annotate(f"rank ceiling {MAX_RANK:,}\n(5% of {PANEL_GENES:,} genes)",
                xy=(MAX_RANK, ax.get_ylim()[1] * 0.86), xytext=(6, 0),
                textcoords="offset points", color=ORANGE, fontsize=6.6,
                va="center")
    ax.set_title("a  The channel, in closed form and in the data", loc="left")


def panel_b(ax, compartments):
    """Regime against consequence, one point per compartment.

    The x axis is the regime -- how many ranks the top block holds relative to
    the genes a typical cell detects -- and the y axis is what the regime did:
    the share of the score that came from ranks the tie-break settled rather
    than from counts the cell observed.  The y quantity is not the inclusion
    probability, which is a function of the depth alone and therefore a
    relabelling of the x axis; it is the realised split of the score in that
    compartment, so the panel shows a consequence rather than the same regime
    twice.
    """
    x = compartments["median_depth_ratio"].to_numpy()
    y = compartments["median_score_tie_break_share"].to_numpy()
    size = np.clip(compartments["n_cells"].to_numpy() / 26, 8, 90)
    ax.axvline(1.0, color=GREY, lw=0.9, ls=(0, (4, 2.5)))

    open_ = x > 1.0
    ax.scatter(x[open_], y[open_], s=size[open_], color=ORANGE,
               edgecolor="white", linewidth=0.5, label="channel open")
    ax.scatter(x[~open_], y[~open_], s=size[~open_], color=BLUE,
               edgecolor="white", linewidth=0.5, label="channel closed")

    ax.set_xlabel("Rank ceiling ÷ median genes detected")
    ax.set_ylabel("Tie-break share of the score")
    ax.set_yscale("symlog", linthresh=0.01)
    # The axis limits are set before the labels, not after: the labels are
    # placed in display coordinates, so the transform they are placed with has
    # to be the final one.  Setting the limits afterwards moved every label off
    # the point it had just been fitted to.
    #
    # The ceiling above the largest point is the marker's radius: a point on the
    # top edge is a point the reader cannot see.
    ax.set_ylim(bottom=0, top=float(y.max()) * 1.28)
    # A share of the score cannot be negative, and symlog pads the axis below
    # zero by default, which puts a tick on the page that no datum can reach.
    #
    # The x limits reserve a gutter, because the shut compartments span the
    # whole range of the axis and a label column placed anywhere inside it lands
    # on top of the points -- and the names are what the panel is for.  The
    # reserve is a fraction of the data's own range rather than a constant, so a
    # different cohort with different ratios keeps the same clearance.
    span = max(float(x.max() - x.min()), 1e-6)
    width = span / 0.63                     # 32% of the axis to the left, 5%
    ax.set_xlim(float(x.min()) - 0.32 * width,
                float(x.min()) + 0.68 * width)
    _label_compartments(ax, compartments)
    # Lower right, because the shut compartments are the low-share ones and the
    # open ones are the high-share ones, so the panel's empty quarter is the
    # bottom right whichever way the data fall.  The labels occupy both margins
    # and the upper left, which is where the legend would otherwise sit.
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4,
              borderpad=0.2)
    ax.set_title("b  Where the compartments sit", loc="left")


def _spread(desired, gap, lo, hi):
    """Positions near ``desired``, at least ``gap`` apart, inside ``[lo, hi]``.

    Labels are placed in display space rather than in the data, because the y
    axis is symlog and the gap that matters is the printed one.  Each label
    starts where its point is and is pushed off the ones below it; the backward
    pass keeps the stack inside the axes.  When the labels cannot all fit at
    the requested gap they are spread evenly, which is the honest failure: the
    panel then reads as crowded rather than silently dropping points.
    """
    n = len(desired)
    if n == 0:
        return desired
    if (n - 1) * gap > hi - lo:
        return np.linspace(lo, hi, n)
    pos = np.array(desired, dtype=float)
    pos[0] = max(pos[0], lo)
    for i in range(1, n):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos[-1] = min(pos[-1], hi)
    for i in range(n - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - gap)
    pos[0] = max(pos[0], lo)
    for i in range(1, n):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    return pos


def _label_compartments(ax, compartments):
    """Every compartment named, without the labels overlapping each other.

    The depth ratios cluster into two tight groups -- the compartments whose
    channel is shut below the ceiling line and the ones whose channel is open
    above it -- and within a group they differ by less than the width of a
    label.  Two things follow.  Offsets anchored to each label's own point
    cannot work, because labels at the same height but different x still run
    into one another however far they are pushed apart vertically.  And
    labelling a chosen subset would silently hide the compartments left out.

    So each group is set as a column inside the axes on its own side, aligned
    on a common edge so that a vertical spread is sufficient, and placed in
    whichever half of the axes the group's own points are *not* in, so that no
    leader line crosses the cloud it belongs to and no column lands on the
    tick labels.  The placement is done twice: the first pass adds artists the
    constrained layout engine then sizes the axes around, and the second
    measures the layout that will actually be drawn.
    """
    for _ in range(2):
        for artist in list(ax.texts):
            artist.remove()
        _place_compartment_labels(ax, compartments)
        ax.figure.canvas.draw()


def _place_compartment_labels(ax, compartments):
    x = compartments["median_depth_ratio"].to_numpy()
    y = compartments["median_score_tie_break_share"].to_numpy()
    fig = ax.figure
    # One line of type, as a fraction of the axes height, so the spacing is
    # right whatever size the layout engine settles on.
    height_pt = max(ax.bbox.height * 72.0 / fig.dpi, 1.0)
    gap = 7.6 / height_pt
    for shut in (True, False):
        idx = np.where((x < 1.0) if shut else (x >= 1.0))[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(y[idx])]
        px = ax.transData.transform(np.column_stack([x[idx], y[idx]]))
        fy = ax.transAxes.inverted().transform(px)[:, 1]
        # The column grows away from the points, and only as far past the
        # midline as the labels need, so it stays as far from them as it can.
        span = (len(idx) - 1) * gap
        if float(np.median(fy)) < 0.5:
            hi = 0.985
            lo = min(0.50, hi - span)
        else:
            lo = 0.015
            hi = max(0.485, lo + span)
        target = _spread(fy, gap, lo, hi)
        column = 0.022 if shut else 0.978
        for i, t in zip(idx, target):
            ax.annotate(
                str(compartments["cell_type"].iloc[i])[:16],
                xy=(x[i], y[i]), xycoords="data",
                xytext=(column, t), textcoords=ax.transAxes,
                ha="left" if shut else "right", va="center",
                fontsize=6.2, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#c8c8c8", lw=0.45,
                                shrinkA=0.5, shrinkB=1.5))


def implied_rho(df):
    """The product the paper argues forecasts the damage.

    A score's spurious association with an axis is the product of two
    correlations, each measured against the depth vector: how much the score
    tracks depth, and how much the axis does.  Classical attenuation of a
    confounded association gives exactly this form, and both factors are
    measurable before the outcome is examined.
    """
    need = ("score_depth_rho", "programme_depth_rho")
    if any(c not in df for c in need):
        return None
    return df["score_depth_rho"] * df["programme_depth_rho"]


def response_curve(df, column, measure="implied", bins=6):
    """Rejection rate of one test in quantile bins of one candidate diagnostic.

    ``measure`` is either the product above or the tie-break share the paper
    proposed first and then refuted.  Both are computed from the matrix and the
    gene set, so either could serve as a pre-flight quantity; which one does is
    the question the two curves answer.  Bins run on the magnitude of the
    product, because the test it forecasts is two-sided.
    """
    if measure == "implied":
        x = implied_rho(df)
        x = None if x is None else x.abs()
    else:
        x = df["tie_break_share"]
    work = pd.DataFrame({"x": x, "y": df[column]}).dropna()
    work["_bin"] = pd.qcut(work["x"], bins, duplicates="drop")
    rows = []
    for level, sub in work.groupby("_bin", observed=True):
        k = int((sub["y"] < ALPHA).sum())
        lo, hi = wilson(k, len(sub))
        rows.append(dict(x=float(level.mid), n=len(sub), rate=k / len(sub),
                         lo=lo, hi=hi))
    return pd.DataFrame(rows)


def panel_c(ax, frames):
    """The quantity that forecasts the failure, on runs where there is none."""
    pooled = pd.concat([f.assign(_src=name) for name, f in frames.items()],
                       ignore_index=True)
    pooled = pooled[pooled["effect"] == 0]
    curves = [("naive", "naive_p", "cell-level correlation", "-", BLUE, -6, 7),
              ("expression", "expression_p", "expression-matched null",
               (0, (3.2, 1.6)), ORANGE, -6, 8)]
    for key, column, label, ls, colour, dx, dy in curves:
        curve = response_curve(pooled, column, "implied")
        ax.plot(curve["x"], curve["rate"], ls=ls, marker="o", ms=3.2, lw=1.4,
                color=colour, zorder=3)
        ax.fill_between(curve["x"], curve["lo"], curve["hi"], color=colour,
                        alpha=0.16, lw=0, zorder=1)
        last = curve.iloc[-1]
        ax.annotate(label, xy=(last["x"], last["rate"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="right", va="center", fontsize=6.6, color=colour)
    ax.axhline(ALPHA, color="#333333", lw=0.9, ls=(0, (4, 2.5)))
    # The nominal line is left unlabelled in both c panels and described in the
    # caption: every position on the line is already taken by one interval band
    # or the other, and a label placed clear of one collides with the next.
    ax.set_xlabel(r"Pre-flight estimate of the spurious association "
                  r"$|\rho(\mathrm{score},D)\,\rho(\mathrm{axis},D)|$")
    ax.set_ylabel("Rejection\nrate at P < 0.05")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("c  The pre-flight estimate predicts the failure", loc="left")


def panel_c_share(ax, grid_a):
    """The candidate the study proposed first, on the same runs, refuted.

    Drawn below panel c on the same vertical scale, so that the reader sees the
    two curves disagree rather than being told that they do.
    """
    share = response_curve(grid_a, "naive_p", "share")
    ax.plot(share["x"], share["rate"], "-o", ms=3.0, lw=1.3, color=GREY, zorder=3)
    ax.fill_between(share["x"], share["lo"], share["hi"], color=GREY, alpha=0.16,
                    lw=0, zorder=1)
    ax.axhline(ALPHA, color="#333333", lw=0.9, ls=(0, (4, 2.5)))
    ax.set_xlabel("Tie-break share of the score")
    ax.set_ylabel("Rejection\nrate at P < 0.05")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("binned instead by the tie-break share, on the same runs",
                 loc="left", fontsize=7.2)


def panel_d(ax, df):
    """The same rates, split by whether there is a channel to be fooled by."""
    groups = [("channel open", df[df["depth_ratio"] > 1.0], PURPLE),
              ("channel closed", df[df["depth_ratio"] <= 1.0], GREY)]
    order = [("naive", "naive_p", False), ("permutation", "perm_p", False),
             ("cutpoint_median", "cut_median_p", False),
             ("cutpoint_optimal", "cut_opt_p", False),
             ("random", "random_p", True),
             ("expression", "expression_p", True),
             ("codetection", "codetection_p", True)]
    ypos = np.arange(len(order))[::-1]
    for offset, (label, sub, colour) in zip((0.17, -0.17), groups):
        rates = reject_table(sub)
        for y, (key, column, _) in zip(ypos, order):
            if key not in rates:
                continue
            r = rates[key]
            ax.plot([r["ci_low"], r["ci_high"]], [y + offset] * 2, color=colour,
                    lw=1.0, alpha=0.75, solid_capstyle="butt")
            ax.plot([r["rate"]], [y + offset], "o", ms=3.4, color=colour)
    ax.axvline(ALPHA, color="#333333", lw=0.9, ls=(0, (4, 2.5)))
    ax.set_yticks(ypos)
    ax.set_yticklabels([CONVENTIONAL_LABEL.get(k, MATCHED_LABEL.get(k, k))
                        for k, _, _ in order])
    ax.set_xlabel("Rejection rate at P < 0.05  (Wilson 95% CI)")
    ax.set_xlim(0, None)
    handles = [Line2D([], [], color=c, lw=1.4, marker="o", ms=3.4, label=l)
               for l, _, c in groups]
    ax.legend(handles=handles, frameon=False, loc="lower right",
              handletextpad=0.5, borderpad=0.2)
    ax.set_title("d  Conventional, matched, and the regime boundary", loc="left")


def main(outdir="figures"):
    grids = {}
    for name in ("A_calibration", "B_regime"):
        path = RESULTS / f"grid_{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}; run run_grid.py first")
        grids[name] = load(name)
    null = pd.concat([f[f["effect"] == 0] for f in grids.values()],
                     ignore_index=True)
    if null.empty:
        raise SystemExit("no effect = 0 runs in the grids; panel c is undefined")

    bench_path = RESULTS / "benchmark_gse176078_all.csv"
    if not bench_path.exists():
        raise SystemExit(
            f"missing {bench_path}; run experiments/real_benchmark.py first")
    bench = pd.read_csv(bench_path)
    # ``tie_break_inclusion`` is deliberately not required: it is the regime
    # quantity, and both panels that use the benchmark now take the regime from
    # the depth ratio and the consequence from elsewhere.  Requiring a column
    # nothing reads makes a missing-input message that points at the wrong file.
    needed = {"cell_type", "median_detected", "depth_ratio", "n_cells"}
    missing = needed - set(bench.columns)
    if missing:
        raise SystemExit(f"{bench_path} lacks {sorted(missing)}")
    by_type = bench.groupby("cell_type", observed=True)
    compartments = by_type.agg(
        n_cells=("n_cells", "max"),
        median_detected=("median_detected", "median"),
        median_depth_ratio=("depth_ratio", "median"),
    ).reset_index()

    # Panel b's y axis is the realised split of the score, which the benchmark
    # table does not carry: it needs the cache, so it comes from the companion
    # script.  Without it the panel would plot the inclusion probability against
    # the depth ratio, which is the same regime twice.
    tie_path = RESULTS / "compartment_tie_break_gse176078.csv"
    if not tie_path.exists():
        raise SystemExit(
            f"missing {tie_path}; run experiments/compartment_tie_break.py "
            f"after the benchmark, or panel b has no y axis")
    tie = pd.read_csv(tie_path)
    missing = {"cell_type", "score_tie_break_share"} - set(tie.columns)
    if missing:
        raise SystemExit(f"{tie_path} lacks {sorted(missing)}")
    share = (tie.groupby("cell_type", observed=True)["score_tie_break_share"]
             .median().rename("median_score_tie_break_share").reset_index())
    before = len(compartments)
    compartments = compartments.merge(share, on="cell_type", how="left")
    if compartments["median_score_tie_break_share"].isna().any():
        lost = compartments.loc[
            compartments["median_score_tie_break_share"].isna(), "cell_type"]
        raise SystemExit(f"{tie_path} has no row for {list(lost)}; the panel "
                         f"would silently drop {len(lost)} of {before}")
    compartments = compartments.sort_values("median_depth_ratio")

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.09, 5.3), layout="constrained")
        gs = fig.add_gridspec(2, 2)
        # Panel c carries two curves that answer the same question with opposite
        # verdicts, so it gets two stacked axes rather than an inset: overlaying
        # the refuted candidate on the accepted one put a box across the very
        # curve the panel exists to show.
        gs_c = gs[1, 0].subgridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.20)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs_c[0]), fig.add_subplot(gs_c[1]),
                fig.add_subplot(gs[1, 1])]
        panel_a(axes[0], compartments)
        panel_b(axes[1], compartments)
        panel_c(axes[2], grids)
        panel_c_share(axes[3], grids["A_calibration"])
        panel_d(axes[4], null)

    out = PKG / outdir
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("svg", "pdf", "png"):
        path = out / f"figure1_calibration.{ext}"
        # Written inside the context, because ``pdf.fonttype`` and
        # ``svg.fonttype`` are read when the file is written rather than when the
        # figure is drawn: outside it the PDF comes back in Type 3 fonts, which
        # the journal rejects, and the SVG comes back with its text converted to
        # outlines, which is the opposite of what the setting asks for.
        with plt.rc_context(RC):
            fig.savefig(path, format=ext, bbox_inches="tight",
                        dpi=600 if ext == "png" else None)
        written.append(path)
    plt.close(fig)

    # A figure that cannot be described numerically has not been checked, so the
    # quantities it draws are printed alongside it.
    curve = response_curve(null, "naive_p", "implied")
    match = response_curve(null, "expression_p", "implied")
    share = response_curve(grids["A_calibration"], "naive_p", "share")
    print(f"compartments plotted: {len(compartments)}")
    print(f"channel open in {int((compartments['median_depth_ratio'] > 1).sum())}"
          f" of {len(compartments)}")
    print(f"naive rejection, lowest pre-flight product {curve['rate'].iloc[0]:.3f} "
          f"-> highest {curve['rate'].iloc[-1]:.3f}")
    print(f"matched rejection, lowest pre-flight product "
          f"{match['rate'].iloc[0]:.3f} -> highest {match['rate'].iloc[-1]:.3f}"
          f"  (endpoints alone mislead: the curve is not monotone)")
    print(f"naive rejection, lowest tie-break share {share['rate'].iloc[0]:.3f} "
          f"-> highest {share['rate'].iloc[-1]:.3f}  (the refuted candidate)")
    for path in written:
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures",
                    help="where to write the figure, relative to the package")
    main(**vars(ap.parse_args()))
