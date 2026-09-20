"""Turn the real-data benchmark tables into the numbers the paper reports.

The benchmark records a score's correlation with sequencing depth for every
(cell type, gene set) pair in a cohort.  This script asks whether that
correlation was predictable *before* the score was taken, and it separates the
question into the two levels at which the answer lives:

Level 1, the cell type
    ``tie_break_inclusion`` is a function of the cell type's depth and the
    matrix's gene panel alone.  It is the probability that a gene the cell does
    not express is nonetheless ranked inside ``max_rank`` by the tie-break, and
    it is identical for every gene set scored in that cell type.  It therefore
    predicts the *level* around which that cell type's depth correlations sit.

Level 2, the gene set
    Within a cell type, what separates a depth-driven gene set from a robust one
    is its detection profile: a set whose genes are rarely observed has more of
    its score coming from tie-broken empty ranks.  The diagnostic is the set's
    own detection rate.

A diagnostic that works only between cell types would be useless for a
researcher choosing a gene set, and one that works only within a cell type would
miss the cohort-level warning.  Both are reported.

Usage
-----
    python experiments/analyse_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

import sparsegs as sg  # noqa: E402

RESULTS = os.path.join(PKG, "results")

#: Correlation magnitudes at which a score is called depth-driven.  Three levels
#: rather than one, because the useful threshold depends on what the score is
#: being used for and the shape of the trade-off is the informative part.
DEPTH_LEVELS = (0.2, 0.3, 0.5)


def _auc(score, positive):
    """Rank-based AUC, identical to the Mann-Whitney statistic."""
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    ok = np.isfinite(score)
    score, positive = score[ok], positive[ok]
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load(dataset):
    path = os.path.join(RESULTS, f"benchmark_{dataset}_all.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run real_benchmark.py first")
    df = pd.read_csv(path)
    nulls_path = os.path.join(RESULTS, f"benchmark_{dataset}_nulls.csv")
    nulls = pd.read_csv(nulls_path) if os.path.exists(nulls_path) else None
    return df, nulls


def merge_realised_share(df, dataset):
    """Add the realised tie-break share of each set, where it has been computed.

    ``score_decomposition`` needs the cache rather than the summary columns, so
    the benchmark table does not carry it; ``compartment_tie_break.py`` computes
    it for exactly the (compartment, set) pairs the benchmark scored, and the
    row counts are checked here so that a partial join cannot quietly shrink the
    analysis to the pairs that happen to be in both.
    """
    path = os.path.join(RESULTS, f"compartment_tie_break_{dataset}.csv")
    if not os.path.exists(path):
        print(f"  [{dataset}] {os.path.basename(path)} absent: the realised "
              f"share is not available, and level 2 falls back to the "
              f"detection profile alone")
        return df
    share = pd.read_csv(path)[["cell_type", "gene_set",
                              "score_tie_break_share", "mean_tie_break"]]
    merged = df.merge(share, on=["cell_type", "gene_set"], how="left")
    got = int(merged["score_tie_break_share"].notna().sum())
    if got < len(df):
        print(f"  [{dataset}] realised share present for {got} of {len(df)} "
              f"pairs; the rest are dropped from the level-2 comparison "
              f"rather than entered with a missing predictor")
    return merged


# ----------------------------------------------------------------------
# level 1: the cell type
# ----------------------------------------------------------------------
def cell_type_level(df):
    """Does the cell type's tie-break exposure predict its depth dependence?

    One row per cell type, so the sample is small by construction and the
    reported correlation is a property of the cohort, not a fitted model.
    """
    g = df.groupby("cell_type")
    tab = pd.DataFrame(dict(
        n_cells=g["n_cells"].first(),
        n_sets=g.size(),
        median_detected=g["median_detected"].first(),
        max_rank=g["max_rank"].first(),
        depth_ratio=g["depth_ratio"].first(),
        tie_break_inclusion=g["tie_break_inclusion"].first(),
        median_abs_rho_depth=g["rho_depth"].apply(lambda s: float(np.nanmedian(np.abs(s)))),
        mean_rho_depth=g["rho_depth"].mean(),
        median_rho_depth=g["rho_depth"].median(),
        frac_depth_gt_03=g["rho_depth"].apply(lambda s: float((np.abs(s) >= 0.3).mean())),
        median_zero_rate=g["observed_zero_rate"].median(),
        median_zero_excess=g["zero_rate_excess"].median(),
    )).reset_index()
    # The realised share, where it has been computed.  Inclusion is what the
    # regime exposes every set to; the share is what the sets in this
    # compartment did with it, and the two are reported side by side so that a
    # compartment where they disagree is visible rather than averaged away.
    if "score_tie_break_share" in df.columns:
        tab["median_score_tie_break_share"] = (
            g["score_tie_break_share"].median().to_numpy())
    tab = tab.sort_values("tie_break_inclusion")

    x = tab["tie_break_inclusion"].to_numpy()
    y = tab["median_abs_rho_depth"].to_numpy()
    tab.attrs["spearman_inclusion_vs_absrho"] = float(sg.spearman(x, y)[0])
    tab.attrs["auc_inclusion_for_absrho_ge_0.3"] = _auc(x, y >= 0.3)
    return tab


# ----------------------------------------------------------------------
# level 2: the gene set within a cell type
# ----------------------------------------------------------------------
def gene_set_level(df):
    """Does a set's detection profile predict its own depth dependence?

    The correlation is computed within each cell type and then pooled, so the
    cell-type-level confound -- which is shared by every set in that cell type --
    cannot manufacture the result.

    ``score_tie_break_share`` is the realised quantity: of the score this set
    actually received in this compartment, the fraction that came from ranks the
    tie-break settled.  It is the mechanism's consequence rather than its
    exposure, so it should beat the detection profile at predicting the set's
    own depth correlation -- and if it does not, the framework's account of why
    sparse sets track depth is wrong in a way the regime arithmetic alone cannot
    show.
    """
    # Predictors that rise with depth dependence, and those that fall with it.
    # Each is (label, source column); the labels are what the report names and
    # are kept stable so that the tables stay comparable across runs.
    rises = [("median_detection", "median_detection"),
             ("mean_detection", "mean_detection"),
             ("frac_above_floor", "frac_above_floor"),
             ("score_tie_break_share", "score_tie_break_share")]
    falls = [("zero_excess", "zero_rate_excess"),
             ("zero_rate", "observed_zero_rate")]
    predictors = [(lab, col) for lab, col in rises if col in df.columns]
    predictors += [(lab, col) for lab, col in falls if col in df.columns]
    predictors.append(("k", "k"))

    rows = []
    for ct, sub in df.groupby("cell_type"):
        if len(sub) < 30:
            continue
        r = dict(cell_type=ct, n_sets=len(sub))
        for name, col in predictors:
            v = sub[col].to_numpy(dtype=float)
            ok = np.isfinite(v)
            target = np.abs(sub["rho_depth"].to_numpy(dtype=float))[ok]
            if ok.sum() < 30 or not np.isfinite(target).any():
                r[f"rho_{name}_vs_absrho"] = float("nan")
                r[f"auc_{name}_for_absrho_ge_0.3"] = float("nan")
                continue
            r[f"rho_{name}_vs_absrho"] = float(sg.spearman(v[ok], target)[0])
            r[f"auc_{name}_for_absrho_ge_0.3"] = _auc(
                -v[ok] if name in falls else v[ok], target >= 0.3)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# the UCell normaliser
# ----------------------------------------------------------------------
def ucell_level(df):
    """How much does UCell's own default normaliser change the answer?

    UCell's documentation recommends ``r_max`` near the median detected genes per
    cell, with 1500 as the default for 10x data.  Where the two disagree is
    exactly where the package's stated default is not the package's own advice,
    and the correlation between the two resulting scores says how much the choice
    matters in practice.
    """
    rho = df["rho_ucell_default_implied"].to_numpy(dtype=float)
    ok = np.isfinite(rho)
    appropriate = df["ucell_default_is_appropriate"].to_numpy(dtype=bool)
    out = dict(
        n_pairs=int(ok.sum()),
        frac_default_appropriate=float(appropriate.mean()),
        median_rho_default_implied=float(np.median(rho[ok])),
        # The threshold is in the name, so the name cannot be written the way it
        # reads: an identifier may not contain a decimal point, and
        # ``frac_rho_below_0.99`` is not one -- it parses as a name followed by a
        # float literal, which is a syntax error.  Underscores it is.
        frac_rho_below_099=float((rho[ok] < 0.99).mean()),
        frac_rho_below_090=float((rho[ok] < 0.90).mean()),
    )
    for label, mask in [("appropriate", appropriate), ("inappropriate", ~appropriate)]:
        v = rho[ok & mask]
        out[f"median_rho_{label}"] = float(np.median(v)) if v.size else float("nan")
        out[f"frac_below_099_{label}"] = (
            float((v < 0.99).mean()) if v.size else float("nan"))
    # AUC of the depth diagnostic for flagging the pairs where the two disagree.
    inc = df["tie_break_inclusion"].to_numpy(dtype=float)
    out["auc_inclusion_for_rho_below_099"] = _auc(inc[ok], rho[ok] < 0.99)
    return out


# ----------------------------------------------------------------------
# matched nulls on real gene sets
# ----------------------------------------------------------------------
def null_level(nulls):
    """Rejection rate of each null family on real gene sets, by target.

    The comparison that matters is against the conventional tests, which are not
    run here: what the matched nulls say about the *observed* correlation is
    reported alongside the rate at which a conventional |rho| >= 0.3 screen would
    have called the same set associated at all.
    """
    if nulls is None or nulls.empty:
        return {}
    out = {"n_pairs": int(len(nulls))}
    for target, rho_col in (("depth", "rho_depth"), ("axis", "rho_axis")):
        if rho_col in nulls:
            rho = np.abs(nulls[rho_col].to_numpy(dtype=float))
            out[f"frac_abs_rho_ge_0.3_{target}"] = float(
                np.nanmean(rho >= 0.3))
        for kind in ("random", "expression", "codetection"):
            c = f"{kind}_{target}_reject"
            if c in nulls:
                out[f"reject_{target}_{kind}"] = float(
                    nulls[c].astype(float).mean())
                if rho_col in nulls:
                    out[f"auc_{kind}_reject_vs_absrho_{target}"] = _auc(
                        np.abs(nulls[rho_col].to_numpy(dtype=float)),
                        nulls[c].astype(bool).to_numpy())
    return out


# ----------------------------------------------------------------------
# generality: collections, and the sets a reader will check first
# ----------------------------------------------------------------------
#: The MSigDB collections the benchmark draws on, keyed by the prefix a set name
#: carries.  The immunologic signatures are named by their source study rather
#: than by their collection, so they have no single prefix and are recognised by
#: elimination -- a set that is one of these prefixes is in that collection, and
#: anything else came from the immunologic collection.
COLLECTIONS = (("HALLMARK", "Hallmark"), ("KEGG", "KEGG"),
               ("REACTOME", "Reactome"), ("BIOCARTA", "BioCarta"),
               ("WP", "WikiPathways"))
IMMUNOLOGIC = "Immunologic"

#: The themes a reader tests the framework against, as (label, substrings).  A
#: set belongs to a theme when its name carries one of the substrings, so the
#: table cannot become a selection of the examples that happened to suit the
#: argument: someone holding the collection can reproduce the list, and a theme
#: whose sets are all benign would say so.
THEMES = (("Hypoxia", ("HYPOXIA",)),
          ("Cell cycle", ("CELL_CYCLE", "E2F", "G2M", "G2_M")),
          ("Immune checkpoint", ("PD_1", "PDCD1", "CTLA4")),
          ("Interferon", ("INTERFERON",)))


def collection_of(name):
    """The collection a gene set came from, from the name it carries."""
    head = name.split("_")[0]
    for prefix, label in COLLECTIONS:
        if head == prefix:
            return label
    return IMMUNOLOGIC


def feature_table(df, group):
    """Damage and detection per group of gene sets, within compartments.

    Every quantity is computed inside a compartment and then summarised, because
    the compartment sets the ceiling-to-depth ratio for all of its sets at once:
    a difference between collections computed on pooled pairs could be a
    difference between the compartments that happen to carry them, and every
    collection here is scored in every compartment, so it need not be.
    """
    rows = []
    for label, sub in df.groupby(group):
        absrho = np.abs(sub["rho_depth"].to_numpy(dtype=float))
        ok = np.isfinite(absrho)
        per_compartment = []
        for _, g in sub.groupby("cell_type"):
            if len(g) < 30:
                continue
            # ``median_detection`` is the predictor the paper already reports a
            # within-compartment correlation for; using the same column here
            # keeps the per-collection numbers comparable with the pooled one.
            v = g["median_detection"].to_numpy(dtype=float)
            w = np.abs(g["rho_depth"].to_numpy(dtype=float))
            m = np.isfinite(v) & np.isfinite(w)
            if m.sum() >= 30:
                per_compartment.append(float(sg.spearman(v[m], w[m])[0]))
        rows.append(dict(
            label=label,
            n_sets=int(sub["gene_set"].nunique()),
            n_pairs=int(len(sub)),
            median_k=float(sub.groupby("gene_set")["k"].first().median()),
            # A set's detection rate differs between compartments -- the same
            # genes are seen in a larger fraction of cells where the cells are
            # deeper -- so this is the median over the set's compartments and
            # then over the sets, rather than the value in whichever compartment
            # happened to come first.
            median_detection=float(
                sub.groupby("gene_set")["median_detection"].median().median()),
            median_abs_rho_depth=float(np.nanmedian(absrho[ok])),
            frac_ge_03=float(np.nanmean(absrho[ok] >= 0.3)),
            frac_ge_05=float(np.nanmean(absrho[ok] >= 0.5)),
            detection_vs_absrho=float(np.median(per_compartment)) if per_compartment else float("nan"),
            n_compartments=len(per_compartment)))
    return pd.DataFrame(rows).sort_values("median_abs_rho_depth", ascending=False)


def theme_sets(df, cohort):
    """One row per named set in the themes, with the compartment median damage.

    These are the sets a reader of a methods paper checks first -- the ones whose
    biology they know -- so they are reported individually rather than only
    inside their collection's average.
    """
    rows = []
    names = df["gene_set"].unique()
    for theme, keys in THEMES:
        for name in sorted(n for n in names if any(k in n for k in keys)):
            sub = df[df["gene_set"] == name]
            absrho = np.abs(sub["rho_depth"].to_numpy(dtype=float))
            rows.append(dict(
                cohort=cohort, theme=theme, gene_set=name, k=int(sub["k"].iloc[0]),
                median_detection=float(sub["median_detection"].median()),
                median_abs_rho_depth=float(np.nanmedian(absrho)),
                frac_ge_03=float(np.nanmean(absrho >= 0.3)),
                n_compartments=int(len(sub))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# cross-cohort comparability
# ----------------------------------------------------------------------
#: The association a set has to carry in the *first* cohort for its sign in the
#: second to be worth counting.  Below it the two estimates are both small
#: enough that the sign is decided by sampling noise, and a sign agreement
#: pooled over every set is therefore a statement about noise: on these data it
#: reads 57% overall and 83% among sets above this line.  The cut is a choice,
#: which is why it is reported alongside the number it selects.
STRONG_AXIS = 0.2


def cohort_level(d1, d2, name1, name2):
    """What changes when the same gene sets are scored on a different panel?

    ``max_rank`` is a fraction of the panel, so an identical gene set lands on a
    different score scale in each cohort.  This reports the panel arithmetic and
    then the empirical consequence: how much the score's association with the
    dominant axis moves between cohorts for sets present in both.

    Direction and magnitude are reported separately, because on real data they
    behave differently and a single correlation would hide it.  Most gene sets
    have no association with a compartment's dominant axis in either cohort, so
    the pooled Spearman -- and the pooled sign agreement -- are dominated by
    pairs of small numbers whose signs are noise.  The restricted figures say
    what a replication can actually ask for: whether a set that is strong in one
    cohort is still pointing the same way in the other.
    """
    mr1 = int(d1["max_rank"].iloc[0])
    mr2 = int(d2["max_rank"].iloc[0])
    out = dict(
        max_rank_1=mr1, max_rank_2=mr2, max_rank_ratio=mr2 / mr1,
        median_detected_1=float(d1["median_detected"].median()),
        median_detected_2=float(d2["median_detected"].median()),
        tie_break_inclusion_1=float(d1["tie_break_inclusion"].median()),
        tie_break_inclusion_2=float(d2["tie_break_inclusion"].median()),
        median_rho_depth_1=float(np.nanmedian(d1["rho_depth"])),
        median_rho_depth_2=float(np.nanmedian(d2["rho_depth"])),
    )
    if "rho_axis" not in d1 or "rho_axis" not in d2:
        return out
    a = d1.groupby("gene_set")["rho_axis"].median()
    b = d2.groupby("gene_set")["rho_axis"].median()
    common = a.index.intersection(b.index)
    out["n_common_gene_sets"] = int(len(common))
    if len(common) >= 20:
        va, vb = a[common].to_numpy(), b[common].to_numpy()
        ok = np.isfinite(va) & np.isfinite(vb)
        out["spearman_rho_axis_across_cohorts"] = float(sg.spearman(va[ok], vb[ok])[0])
        out["sign_agreement"] = float((np.sign(va[ok]) == np.sign(vb[ok])).mean())
        m = np.abs(va[ok]) >= STRONG_AXIS
        if m.sum() >= 20:
            out["strong_axis_threshold"] = float(STRONG_AXIS)
            out["n_strong_in_1"] = int(m.sum())
            out["strong_sign_agreement"] = float(
                (np.sign(va[ok][m]) == np.sign(vb[ok][m])).mean())
            out["strong_median_abs_1"] = float(np.median(np.abs(va[ok][m])))
            out["strong_median_abs_2"] = float(np.median(np.abs(vb[ok][m])))
        out["median_abs_change"] = float(np.median(np.abs(vb[ok] - va[ok])))
    return out


# ----------------------------------------------------------------------
def main(datasets=("gse176078", "gse161529")):
    report = {}
    frames = {}
    for ds in datasets:
        path = os.path.join(RESULTS, f"benchmark_{ds}_all.csv")
        if not os.path.exists(path):
            print(f"[{ds}] not present, skipped")
            continue
        df, nulls = load(ds)
        df = merge_realised_share(df, ds)
        frames[ds] = df
        print(f"\n=== {ds} ===  {len(df)} (cell type, gene set) pairs, "
              f"{df['cell_type'].nunique()} cell types, "
              f"{df['gene_set'].nunique()} gene sets")

        tab = cell_type_level(df)
        print("\n-- level 1: cell type --")
        cols = ["cell_type", "n_cells", "median_detected", "depth_ratio",
                "tie_break_inclusion", "median_abs_rho_depth",
                "frac_depth_gt_03", "median_zero_rate"]
        if "median_score_tie_break_share" in tab.columns:
            cols.insert(5, "median_score_tie_break_share")
        print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"  Spearman(tie-break inclusion, median |rho_depth|) = "
              f"{tab.attrs['spearman_inclusion_vs_absrho']:+.3f}")
        # Printed rather than omitted, because the AUC is undefined here for a
        # reason worth seeing: the 0.3 line is a per-pair threshold, and at the
        # compartment level every median falls on the same side of it.  A
        # silently absent line reads as an analysis that did not run; this one
        # says the contrast is not in the data, and quotes the spread instead.
        auc = tab.attrs["auc_inclusion_for_absrho_ge_0.3"]
        med = tab["median_abs_rho_depth"]
        if np.isfinite(auc):
            print(f"  AUC for median |rho_depth| >= 0.3 = {auc:.3f}")
        else:
            print(f"  AUC for median |rho_depth| >= 0.3 = undefined: no "
                  f"compartment's median reaches it "
                  f"(range {med.min():.3f} to {med.max():.3f}), so the "
                  f"classification has one class")

        gl = gene_set_level(df)
        print("\n-- level 2: gene set within cell type --")
        agg = gl.drop(columns=["cell_type"]).median()
        print(agg.to_string(float_format=lambda v: f"{v:+.3f}"))

        uc = ucell_level(df)
        print("\n-- UCell normaliser --")
        for k, v in uc.items():
            print(f"  {k:38s} {v:.4f}" if isinstance(v, float) else f"  {k:38s} {v}")

        nl = null_level(nulls)
        if nl:
            print("\n-- matched nulls on real gene sets --")
            for k, v in nl.items():
                print(f"  {k:38s} {v:.4f}" if isinstance(v, float) else f"  {k:38s} {v}")

        # Generality: does the regime, and the diagnostic, hold in every
        # collection the benchmark draws on, and in the named sets a reader
        # checks first?  Printed rather than summarised because a collection
        # where the diagnostic inverts is a finding, not an outlier.
        print("\n-- generality: collections --")
        fam = feature_table(df, df["gene_set"].map(collection_of))
        print(fam.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n-- generality: named sets in the themes a reader checks --")
        ex = theme_sets(df, ds)
        for theme, sub in ex.groupby("theme"):
            print(f"  [{theme}]")
            print(sub.drop(columns=["cohort", "theme"]).to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))
        if len(fam) > 1:
            rho = sg.spearman(fam["median_k"], fam["median_abs_rho_depth"])[0]
            print(f"  across collections: Spearman(median set size, median "
                  f"|rho| with depth) = {rho:+.2f} on {len(fam)} points, "
                  f"reported as a pattern rather than a test")
        ex.to_csv(os.path.join(RESULTS, f"benchmark_{ds}_by_theme.csv"), index=False)
        fam.to_csv(os.path.join(RESULTS, f"benchmark_{ds}_by_collection.csv"),
                   index=False)

        report[ds] = dict(
            n_pairs=int(len(df)),
            n_cell_types=int(df["cell_type"].nunique()),
            n_gene_sets=int(df["gene_set"].nunique()),
            max_rank=int(df["max_rank"].iloc[0]),
            median_abs_rho_depth=float(np.nanmedian(np.abs(df["rho_depth"]))),
            frac_abs_rho_depth_ge_0_3=float((np.abs(df["rho_depth"]) >= 0.3).mean()),
            frac_abs_rho_depth_ge_0_5=float((np.abs(df["rho_depth"]) >= 0.5).mean()),
            level1=tab.drop(columns=["cell_type"]).to_dict(orient="list"),
            level1_cell_types=tab["cell_type"].tolist(),
            level1_spearman=tab.attrs["spearman_inclusion_vs_absrho"],
            level1_auc=tab.attrs["auc_inclusion_for_absrho_ge_0.3"],
            collections=fam.to_dict(orient="records"),
            themes=ex.drop(columns=["cohort"]).to_dict(orient="records"),
            level2=gl.to_dict(orient="list"),
            level2_median={k: (float(v) if np.isfinite(v) else None)
                           for k, v in agg.items()},
            ucell=uc, nulls=nl,
            depth_levels={str(l): float((np.abs(df["rho_depth"]) >= l).mean())
                          for l in DEPTH_LEVELS},
        )
        tab.to_csv(os.path.join(RESULTS, f"benchmark_{ds}_by_cell_type.csv"),
                   index=False)
        gl.to_csv(os.path.join(RESULTS, f"benchmark_{ds}_by_gene_set.csv"),
                  index=False)

    if "gse176078" in frames and "gse161529" in frames:
        print("\n=== cross-cohort ===")
        cl = cohort_level(frames["gse176078"], frames["gse161529"],
                          "gse176078", "gse161529")
        for k, v in cl.items():
            print(f"  {k:38s} {v:.4f}" if isinstance(v, float) else f"  {k:38s} {v}")
        report["cross_cohort"] = cl

    out = os.path.join(RESULTS, "benchmark_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {out}")
    return report


if __name__ == "__main__":
    main()
