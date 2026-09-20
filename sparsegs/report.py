"""Self-contained HTML calibration report.

The report is meant to be read *before* an association is believed, so it leads
with the verdict and the evidence for it, and puts the numbers underneath.  It
is written as a single HTML file with inline SVG: no external assets, nothing to
fetch, and it renders identically wherever it is opened.

The figures are drawn directly rather than through a plotting library so that
the styling stays under control and the output stays vector.  A CJK-capable font
stack is declared throughout, since a report about a real dataset routinely
carries gene and cell-type names outside Latin-1.
"""

from __future__ import annotations

import html
import os
import string

import numpy as np

__all__ = ["render_report", "score_histogram_svg", "null_comparison_svg",
           "threshold_svg"]

#: Neutral base with a single accent ramp, so that severity reads at a glance.
PALETTE = dict(
    ink="#1c1c1e", muted="#6b6b70", rule="#d8d8dc", paper="#ffffff",
    wash="#f6f6f7", ok="#2f6f4f", warn="#9a6b1f", bad="#9c2b2b",
    accent="#2b4c7e", null="#b0b0b6",
)

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", '
        '"Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", '
        '"Microsoft YaHei", Helvetica, Arial, sans-serif')


def _esc(value):
    return html.escape(str(value))


#: The stylesheet, as a template rather than an f-string.  CSS is mostly braces,
#: and an f-string would need every one of them doubled -- a transcription rule
#: that is easy to get wrong once and impossible to see when you do, since the
#: file still looks like CSS.  ``$name`` substitution leaves the braces alone.
_CSS = string.Template("""
body{font-family:$font;color:$ink;background:$wash;margin:0;
     padding:32px 20px;line-height:1.55}
.wrap{max-width:760px;margin:0 auto;background:$paper;padding:40px 44px 48px;
      border:1px solid $rule;border-radius:6px}
h1{font-size:22px;font-weight:600;margin:0 0 4px;letter-spacing:-0.01em}
.sub{color:$muted;font-size:13px;margin-bottom:28px}
h2{font-size:13px;font-weight:600;text-transform:uppercase;
   letter-spacing:0.06em;color:$muted;margin:34px 0 12px}
.verdict{background:$wash;padding:16px 20px;border-radius:4px;margin:0 0 8px}
.verdict-name{font-size:17px;font-weight:600;letter-spacing:-0.01em}
.verdict-sev{color:$muted;font-size:12px;margin-bottom:8px}
.reasons{margin:0;padding-left:18px;font-size:13.5px;color:$ink}
.reasons li{margin:3px 0}
table{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 4px}
caption{caption-side:top;text-align:left;color:$muted;font-size:12px;
        padding-bottom:6px}
th{text-align:left;font-weight:600;border-bottom:1px solid $rule;
   padding:6px 10px 6px 0;font-size:12px;text-transform:uppercase;
   letter-spacing:0.04em;color:$muted}
td{padding:6px 10px 6px 0;border-bottom:1px solid $rule;
   font-variant-numeric:tabular-nums}
.genes{font-size:12.5px;color:$muted;word-break:break-word}
figure{margin:10px 0 2px}
figcaption{font-size:11.5px;color:$muted;margin-top:6px}
.note{font-size:12.5px;color:$muted;border-top:1px solid $rule;
      margin-top:34px;padding-top:14px}
""")


def _css():
    return _CSS.substitute(
        font=FONT, **{k: v for k, v in PALETTE.items()})


def _num(value, digits=3):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def score_histogram_svg(score, width=560, height=170, n_bins=40, label="Score"):
    """Histogram of the score, with the exact-zero mass called out.

    The zero spike is the point of the figure, so it is drawn in the alert
    colour and annotated rather than left to be inferred from a tall bar.
    """
    score = np.asarray(score, dtype=float)
    score = score[np.isfinite(score)]
    if score.size == 0:
        return ""
    zero_frac = float((score <= 0).mean())
    pos = score[score > 0]
    pad_l, pad_r, pad_t, pad_b = 44, 14, 16, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    if pos.size:
        hi = float(np.percentile(score, 99.5))
        lo = float(min(score.min(), 0.0))
        counts, edges = np.histogram(pos, bins=n_bins, range=(0, max(hi, 1e-9)))
    else:
        counts, edges = np.array([0]), np.array([0.0, 1.0])
    top = max(int(counts.max()), 1)
    bar_w = plot_w / max(len(counts), 1)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px" role="img" '
             f'aria-label="distribution of {_esc(label)}">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" '
                 f'fill="{PALETTE["paper"]}"/>')
    # baseline
    y0 = pad_t + plot_h
    parts.append(f'<line x1="{pad_l}" y1="{y0}" x2="{pad_l + plot_w}" y2="{y0}" '
                 f'stroke="{PALETTE["rule"]}" stroke-width="1"/>')

    for i, c in enumerate(counts):
        h = plot_h * (c / top)
        x = pad_l + i * bar_w
        parts.append(f'<rect x="{x + 0.5:.1f}" y="{y0 - h:.1f}" '
                     f'width="{max(bar_w - 1, 0.6):.1f}" height="{h:.1f}" '
                     f'fill="{PALETTE["accent"]}" opacity="0.85"/>')

    # the zero bar, drawn to the left of the axis origin
    zh = plot_h * (float((score <= 0).sum()) / max(top, 1))
    parts.append(f'<rect x="{pad_l - 26}" y="{y0 - zh:.1f}" width="20" '
                 f'height="{zh:.1f}" fill="{PALETTE["bad"]}"/>')
    parts.append(f'<text x="{pad_l - 16}" y="{y0 - zh - 5:.1f}" '
                 f'text-anchor="middle" font-size="10.5" fill="{PALETTE["bad"]}" '
                 f'font-family="{FONT}">{zero_frac:.0%}</text>')
    parts.append(f'<text x="{pad_l - 16}" y="{y0 + 13}" text-anchor="middle" '
                 f'font-size="10" fill="{PALETTE["muted"]}" '
                 f'font-family="{FONT}">zero</text>')

    for frac in (0.0, 0.5, 1.0):
        x = pad_l + frac * plot_w
        v = lo + frac * (hi - lo) if pos.size else frac
        parts.append(f'<text x="{x:.1f}" y="{y0 + 22}" text-anchor="middle" '
                     f'font-size="10" fill="{PALETTE["muted"]}" '
                     f'font-family="{FONT}">{v:.2f}</text>')
    parts.append(f'<text x="{pad_l}" y="{pad_t - 5}" font-size="10.5" '
                 f'fill="{PALETTE["muted"]}" font-family="{FONT}">'
                 f'{_esc(label)}; {zero_frac:.1%} of cells score exactly zero'
                 f'</text>')
    parts.append("</svg>")
    return "".join(parts)


def null_comparison_svg(observed, nulls, width=560, height=190,
                        label="Score association"):
    """Observed statistic against each null family, as dot strips.

    One row per family, with the null draws as small marks and the observed
    value as a rule.  Overlap between rows is the whole story, so the rows share
    one axis.
    """
    nulls = {k: np.asarray(v, dtype=float) for k, v in nulls.items()}
    nulls = {k: v[np.isfinite(v)] for k, v in nulls.items()}
    nulls = {k: v for k, v in nulls.items() if v.size}
    if not nulls or not np.isfinite(observed):
        return ""

    lo = min([observed] + [v.min() for v in nulls.values()])
    hi = max([observed] + [v.max() for v in nulls.values()])
    if hi - lo < 1e-9:
        lo, hi = lo - 0.1, hi + 0.1
    span = hi - lo
    lo -= 0.06 * span
    hi += 0.06 * span

    pad_l, pad_r, pad_t, pad_b = 96, 20, 26, 34
    plot_w = width - pad_l - pad_r
    rows = list(nulls.items())
    row_h = max((height - pad_t - pad_b) / max(len(rows), 1), 18)

    def x_of(v):
        return pad_l + (v - lo) / (hi - lo) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px" role="img" '
             f'aria-label="observed {_esc(label)} against matched nulls">',
             f'<rect width="{width}" height="{height}" fill="{PALETTE["paper"]}"/>']

    for i, (name, vals) in enumerate(rows):
        cy = pad_t + (i + 0.5) * row_h
        parts.append(f'<text x="{pad_l - 12}" y="{cy + 3.5:.1f}" '
                     f'text-anchor="end" font-size="11" fill="{PALETTE["ink"]}" '
                     f'font-family="{FONT}">{_esc(name)}</text>')
        step = max(1, vals.size // 220)
        for v in vals[::step]:
            parts.append(f'<circle cx="{x_of(float(v)):.1f}" cy="{cy:.1f}" r="2" '
                         f'fill="{PALETTE["null"]}"/>')
        med = float(np.median(vals))
        parts.append(f'<line x1="{x_of(med):.1f}" y1="{cy - 7:.1f}" '
                     f'x2="{x_of(med):.1f}" y2="{cy + 7:.1f}" '
                     f'stroke="{PALETTE["muted"]}" stroke-width="1.5"/>')

    ox = x_of(float(observed))
    parts.append(f'<line x1="{ox:.1f}" y1="{pad_t - 12}" x2="{ox:.1f}" '
                 f'y2="{pad_t + len(rows) * row_h + 4:.1f}" '
                 f'stroke="{PALETTE["accent"]}" stroke-width="2"/>')
    parts.append(f'<text x="{ox:.1f}" y="{pad_t - 15}" text-anchor="middle" '
                 f'font-size="10.5" fill="{PALETTE["accent"]}" '
                 f'font-family="{FONT}">observed {observed:+.3f}</text>')
    y_axis = pad_t + len(rows) * row_h + 20
    for frac in (0.0, 0.5, 1.0):
        v = lo + frac * (hi - lo)
        parts.append(f'<text x="{x_of(v):.1f}" y="{y_axis:.1f}" '
                     f'text-anchor="middle" font-size="10" '
                     f'fill="{PALETTE["muted"]}" font-family="{FONT}">{v:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def threshold_svg(rows, width=560, height=150):
    """False-positive rate by cutpoint rule, against the nominal level.

    ``rows`` is a list of ``(label, fpr)``.
    """
    rows = [(l, float(f)) for l, f in rows if np.isfinite(f)]
    if not rows:
        return ""
    pad_l, pad_t, pad_b = 150, 20, 30
    # The value label sits to the right of the bar, so the plot width has to stop
    # short of the full width by enough to hold it.  A bar at the maximum would
    # otherwise push its own label off the canvas, which is exactly the row the
    # reader is looking for.
    label_w = 34
    plot_w = width - pad_l - label_w - 8
    row_h = max((height - pad_t - pad_b) / len(rows), 20)
    hi = max(max(f for _, f in rows), 0.10)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px" role="img" '
             f'aria-label="false-positive rate by cutpoint rule">',
             f'<rect width="{width}" height="{height}" fill="{PALETTE["paper"]}"/>']
    nominal_x = pad_l + (0.05 / hi) * plot_w
    parts.append(f'<line x1="{nominal_x:.1f}" y1="{pad_t - 8}" '
                 f'x2="{nominal_x:.1f}" y2="{pad_t + len(rows) * row_h:.1f}" '
                 f'stroke="{PALETTE["ok"]}" stroke-width="1.2" '
                 f'stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{nominal_x:.1f}" y="{pad_t - 12}" text-anchor="middle" '
                 f'font-size="10" fill="{PALETTE["ok"]}" '
                 f'font-family="{FONT}">nominal 0.05</text>')

    for i, (lab, fpr) in enumerate(rows):
        cy = pad_t + (i + 0.5) * row_h
        w = (fpr / hi) * plot_w
        colour = PALETTE["ok"] if fpr <= 0.075 else (
            PALETTE["warn"] if fpr <= 0.20 else PALETTE["bad"])
        parts.append(f'<rect x="{pad_l}" y="{cy - row_h * 0.28:.1f}" '
                     f'width="{w:.1f}" height="{row_h * 0.56:.1f}" '
                     f'fill="{colour}" opacity="0.85" rx="1.5"/>')
        parts.append(f'<text x="{pad_l - 12}" y="{cy + 3.5:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{PALETTE["ink"]}" '
                     f'font-family="{FONT}">{_esc(lab)}</text>')
        parts.append(f'<text x="{pad_l + w + 7:.1f}" y="{cy + 3.5:.1f}" '
                     f'font-size="10.5" fill="{PALETTE["muted"]}" '
                     f'font-family="{FONT}">{fpr:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def _verdict_block(verdict):
    name = verdict.get("verdict", "UNKNOWN")
    colour = {"INTERPRETABLE": PALETTE["ok"],
              "INTERPRETABLE_WITH_MATCHED_NULL": PALETTE["warn"],
              "NOT_IDENTIFIABLE": PALETTE["bad"]}.get(name, PALETTE["muted"])
    items = "".join(f"<li>{_esc(r)}</li>" for r in verdict.get("reasons", []))
    # Which of the two criteria decided, stated rather than left to be inferred
    # from the reasons: the exact one needs the cells' depth, and its absence is
    # a reason to hold the verdict more loosely.
    by = verdict.get("criterion")
    criterion = ""
    if by == "tie_break_share":
        criterion = ("<div class='verdict-sev'>decided by the tie-break share, "
                     "which is exact for this matrix</div>")
    elif by == "fixed_detection_floor":
        criterion = ("<div class='verdict-sev'>decided by a fixed detection "
                     "floor, because this cache does not record the cells' "
                     "depth</div>")
    return (f'<div class="verdict" style="border-left:4px solid {colour}">'
            f'<div class="verdict-name" style="color:{colour}">'
            f'{_esc(name.replace("_", " ").title())}</div>'
            f'<div class="verdict-sev">severity {_esc(verdict.get("severity", "?"))}</div>'
            f'{criterion}'
            f'<ul class="reasons">{items}</ul></div>')


def _table(caption, rows, headers):
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
        for r in rows)
    return (f'<table><caption>{_esc(caption)}</caption>'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def render_report(path, *, title, gene_set, dataset=None, cell_type=None,
                  diagnostics=None, verdict=None, nulls=None, observed=None,
                  score=None, cutpoint_rows=None, tiers=None, notes=None):
    """Write a self-contained calibration report.

    Parameters
    ----------
    path : str
        Output ``.html`` path.
    title : str
    gene_set : sequence of str
        The set as supplied, before any genes missing from the matrix were
        dropped; the report lists what was lost.
    diagnostics : dict
        Output of :func:`sparsegs.sparsity_report`.
    verdict : dict
        Output of :func:`sparsegs.verdict`.
    nulls : dict
        ``{family: array of null statistics}``, for the comparison figure.
    observed : float
        The observed statistic the nulls are compared against.
    score : array-like
        Per-cell scores, for the histogram.
    cutpoint_rows : list of (label, fpr)
    tiers : dict
        Output of :func:`sparsegs.run_tiers`.
    """
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             f"<title>{_esc(title)}</title>", "<style>",
             _css(), "</style></head><body><div class='wrap'>"]

    parts.append(f"<h1>{_esc(title)}</h1>")
    where = ", ".join(str(x) for x in (dataset, cell_type) if x)
    parts.append(f"<div class='sub'>{_esc(where)}</div>")

    if verdict:
        parts.append("<h2>Verdict</h2>")
        parts.append(_verdict_block(verdict))

    if diagnostics:
        d = diagnostics
        parts.append("<h2>Gene set</h2>")
        parts.append(
            f"<div class='genes'>{len(gene_set)} genes supplied, "
            f"{d.get('n_present', 0)} present in the matrix"
            + (f"; {d['n_missing']} dropped"
               if d.get("n_missing") else "")
            + (": " + _esc(", ".join(d.get("genes_below_floor", [])))
               if d.get("genes_below_floor") else "") + "</div>")
        rows = [
            ("Cells scoring exactly zero", f"{d['observed_zero_rate']:.1%}"),
            ("Expected under independent ranks", f"{d['structural_zero_rate']:.1%}"),
            ("Difference", f"{d['zero_rate_excess']:+.1%}"),
            ("Median detection rate", _num(d["median_detection"], 4)),
            ("Lowest detection rate", _num(d["min_detection"], 4)),
            ("Detection floor in use",
             f"{_num(d['detection_floor'], 4)} "
             f"({d.get('detection_floor_source', 'fixed')})"),
            ("Clearing the detection floor",
             f"{d['frac_above_floor']:.0%} ({d['n_above_floor']} genes)"),
            ("Median score", _num(d.get("median_score"), 4)),
        ]
        # The exact criterion, where the cache knows the cells' depth.  It is
        # listed after the floor rather than instead of it so that a reader can
        # see both, and the verdict line below says which one decided.
        if "tie_break_share" in d:
            rows.extend([
                ("Tie-break share of inclusion",
                 f"{d['tie_break_share']:.1%}"),
                ("Mean inclusion probability", _num(d["mean_inclusion"], 4)),
                ("Median genes per cell", _num(d.get("median_depth"), 0)),
                ("Rank ceiling (max_rank)", _num(d.get("max_rank"), 0)),
            ])
        parts.append(_table("Sparsity and detection", rows, ["Quantity", "Value"]))

    if score is not None:
        parts.append("<h2>Score distribution</h2>")
        parts.append("<figure>" + score_histogram_svg(score) +
                     "<figcaption>The bar left of the axis is the mass at "
                     "exactly zero, which is not on the score scale.</figcaption>"
                     "</figure>")

    if nulls and observed is not None:
        parts.append("<h2>Association against matched nulls</h2>")
        parts.append("<figure>" + null_comparison_svg(observed, nulls) +
                     "<figcaption>Grey marks are null draws, the short rule is "
                     "their median, the long line is the observed value."
                     "</figcaption></figure>")

    if cutpoint_rows:
        parts.append("<h2>Cutpoint false-positive rate</h2>")
        parts.append("<figure>" + threshold_svg(cutpoint_rows) +
                     "<figcaption>Rate at which each rule declares an "
                     "association under a permuted outcome.</figcaption></figure>")

    if tiers:
        parts.append("<h2>Validation tiers</h2>")
        rows = []
        t1 = tiers.get("tier1") or {}
        if t1:
            rows.append(("Tier 1, technical",
                         "not applicable" if not t1.get("applicable", True)
                         else f"AUC {_num(t1.get('auc'), 3)} at zero rate "
                              f"{_num(t1.get('zero_rate'), 2)}",
                         "pass" if t1.get("passed") and t1.get("applicable", True)
                         else ("n/a" if not t1.get("applicable", True) else "fail")))
        t2 = tiers.get("tier2") or {}
        if t2:
            rows.append(("Tier 2, internal",
                         f"rho {_num(t2.get('observed_rho'), 3)}; "
                         f"P {_num((t2.get('expression') or {}).get('p_value'), 4)}",
                         "pass" if t2.get("passed") else "fail"))
        t3 = tiers.get("tier3")
        if t3:
            rows.append(("Tier 3, external",
                         f"rho {_num(t3.get('replication_rho'), 3)} in "
                         f"{t3.get('n_replication', '?')} cells"
                         + ("" if t3.get("same_sign") else "; sign reversed"),
                         "pass" if t3.get("passed") else "fail"))
        parts.append(_table("Tiers", rows, ["Tier", "Evidence", "Outcome"]))
        parts.append(f"<div class='genes'>Overall: "
                     f"<strong>{_esc(tiers.get('verdict', ''))}</strong></div>")

    parts.append("<div class='note'>")
    parts.append(
        "Generated by <strong>ClawsGO Science Agent</strong> with the "
        "<code>sparsegs</code> calibration framework. All statistics are "
        "computed from the supplied matrix; no value in this report is "
        "reproduced from a published source.")
    if notes:
        parts.append("<br>" + _esc(notes))
    parts.append("</div>")

    parts.append("</div></body></html>")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    return path
