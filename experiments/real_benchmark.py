"""Real-data breadth benchmark: does the mechanism hold outside one dataset?

The simulation grid establishes the mechanism under controlled conditions.  This
script asks whether the same structure is present in real data at scale, and
whether the cheap diagnostics predict it before any outcome is looked at.

Design
------
For every (cell type, gene set) pair in a dataset it records

* the score's correlation with sequencing depth, ``rho_depth`` -- how much of the
  score is technical;
* the score's correlation with the cell type's dominant transcriptional axis,
  ``rho_axis`` -- the association a researcher would actually report;
* the partial correlation of score and axis given depth -- what is left once the
  technical component is removed;
* the detection profile of the set and the tie-break inclusion probability it
  implies, which is what the diagnostics are built from.

The claim under test is that ``rho_depth`` is predictable from the detection
profile alone.  That turns a post-hoc caveat into a pre-flight check, and the
script reports the operating characteristic of that prediction.

Memory
------
Nothing here is persisted between runs.  A rank is computed within one cell
across genes, so it does not depend on which other cells are in the matrix --
but a cache over all 55,003 cells is 55,003 x 24,646 integers, which is 2.7 GB
at int16 and 5.4 GB at int32, and the container this runs in has 8 GB shared
with other work.  The cache is therefore built **per cell type**, on the cells
that cell type's analysis actually uses.  Four thousand cells is 134 MB rather
than 2.7 GB, the ranks are identical, and the cost is a few seconds per cell
type.

The one thing that must not be done per cell type is computing the ranks over a
reduced gene panel: a rank is a position among *all* genes, so the block always
carries the full panel and the cache keeps only the columns any retained gene set
can query.

Usage
-----
    python experiments/real_benchmark.py --dataset gse176078
    python experiments/real_benchmark.py --dataset gse161529 --label-key leiden_1.0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

import sparsegs as sg  # noqa: E402

OUT_DIR = os.path.join(PKG, "results")
GENESET_DIR = os.path.join(ROOT, "data", "genesets")

#: Gene-set collections, with a cap on how many to take from each so that one
#: large collection does not dominate the benchmark.
COLLECTIONS = [
    ("h.all.v2024.1.Hs.symbols.gmt", None),
    ("c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt", None),
    ("c2.cp.biocarta.v2024.1.Hs.symbols.gmt", None),
    ("c2.cp.reactome.v2024.1.Hs.symbols.gmt", 250),
    ("c2.cp.wikipathways.v2024.1.Hs.symbols.gmt", 200),
    ("c7.immunesigdb.v2024.1.Hs.symbols.gmt", 400),
]

MIN_OVERLAP = 5
MAX_OVERLAP = 500

DATASETS = {
    "gse176078": dict(
        path=os.path.join(ROOT, "work", "gse176078_annotated.h5ad"),
        label_key="cell_type",
        depth_key="n_genes_by_counts",
        # The annotated object carries log-normalised values, so the ranks can
        # be taken from it directly.  A per-cell monotone transform cannot
        # reorder genes within a cell, so scoring a log-normalised matrix and
        # scoring the counts it came from give the same ranks.
        normalised=True,
        scale=None,
        panel_note="24,646 genes; max_rank = 1233 at the 5% AUCell fraction"),
    "gse161529": dict(
        # The transcribed object carries no `var` columns at all, so the axis
        # could not be built and the whole confound half of this cohort's
        # benchmark was silently absent -- reported as `+nan`, which reads as a
        # number that came out undefined rather than as a step that never ran.
        # The highly-variable genes are computed here instead, from the same
        # library-size-scaled log-normalisation the scores are taken from, and
        # stored beside the original counts rather than replacing them.
        path=os.path.join(ROOT, "work", "gse161529_hvg.h5ad"),
        label_key="leiden_1.0",
        depth_key="n_genes_by_counts",
        # Raw counts, so they are normalised here.  The gene panel differs from
        # the discovery cohort, which is the point of the comparison.
        normalised=False,
        scale=1e4,
        panel_note="33,538 genes; max_rank = 1677 at the 5% AUCell fraction"),
}


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------
def load_gene_sets(genes):
    """Read the MSigDB collections, keeping sets with a usable overlap."""
    universe = set(genes)
    out = {}
    for fname, cap in COLLECTIONS:
        path = os.path.join(GENESET_DIR, fname)
        if not os.path.exists(path):
            print(f"  missing {fname}, skipped")
            continue
        opener = gzip.open if path.endswith(".gz") else open
        rows = []
        with opener(path, "rt") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name, members = parts[0], [g for g in parts[2:] if g]
                keep = [g for g in members if g in universe]
                if MIN_OVERLAP <= len(keep) <= MAX_OVERLAP:
                    rows.append((name, keep))
        if cap and len(rows) > cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(rows), size=cap, replace=False)
            rows = [rows[i] for i in sorted(idx)]
        for name, keep in rows:
            out.setdefault(name, keep)
        print(f"  {fname}: {len(rows)} sets kept (cap {cap})")
    return out


def normalise(X, scale):
    """Library-size scaling followed by ``log1p``, in place on the sparse data."""
    tot = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    tot[tot <= 0] = 1.0
    X = sp.diags(scale / tot) @ X
    X = sp.csr_matrix(X)
    X.data = np.log1p(X.data)
    return X


def fetch(a, cells, spec):
    """Pull one cell type's rows and return what the analysis needs.

    Returns ``(X_norm_dense, detection_bool_sparse, depth)``.  ``cells`` are row
    positions in the backed file; the file is sliced rather than read whole, so
    the footprint is the cell type, not the cohort.
    """
    X = sp.csr_matrix(a.X[cells]).astype(np.float32)
    if not spec["normalised"]:
        X = normalise(X, spec["scale"])
    det = (X > 0)
    det.eliminate_zeros()
    Xn = np.asarray(X.todense(), dtype=np.float32)
    return Xn, det, X.getnnz(axis=1).astype(float)


# ----------------------------------------------------------------------
# the two axes a score is correlated against
# ----------------------------------------------------------------------
def dominant_axis(Xn, hvg, seed=0):
    """First principal component of a cell type's expression over its HVGs.

    This is the axis a researcher would use without further justification --
    "cells along the dominant gradient" -- so it is the right stand-in for the
    biological axis a gene-set score is usually correlated against.  The sign is
    fixed by making the component positively loaded on its strongest gene.
    """
    from sklearn.decomposition import PCA

    sub = Xn[:, hvg]
    sub = sub - sub.mean(axis=0, keepdims=True)
    pca = PCA(n_components=1, random_state=seed)
    comp = pca.fit_transform(sub).ravel()
    if pca.components_[0][np.argmax(np.abs(pca.components_[0]))] < 0:
        comp = -comp
    return comp, float(pca.explained_variance_ratio_[0])


def tie_break_inclusion(depth, max_rank, n_genes):
    """Mean probability that a never-detected gene is ranked inside ``max_rank``.

    A gene's rank is decided among all genes, and a cell that expresses fewer
    than ``max_rank`` genes has no count for the rest -- those positions are
    filled by whatever the tie-break puts there.  A gene absent from ``D`` of
    ``G`` genes therefore lands inside the top ``max_rank`` in a fraction
    ``(max_rank - D) / (G - D)`` of cells, averaged over the cells.  This is the
    quantity that makes a sparsity-biased gene set score higher than its
    expression justifies, and it is computable before any score is taken.
    """
    depth = np.asarray(depth, dtype=float)
    open_slots = np.clip(max_rank - depth, 0, None)
    return float(np.mean(open_slots / np.maximum(n_genes - depth, 1.0)))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z.

    Implemented on ranks, so that it is the partial correlation of the ranked
    variables -- which is what a nonparametric reading of "controlling for depth"
    means in practice.
    """
    import scipy.stats as st

    rx, ry, rz = st.rankdata(x), st.rankdata(y), st.rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom <= 0:
        return float("nan")
    return float((rxy - rxz * ryz) / denom)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def run(dataset, n_null_sets=20, n_null=100, n_draws=25, max_cells_per_type=4000,
        seed=0, cell_types=None, label_key=None, block=500):
    t_start = time.time()
    spec = dict(DATASETS[dataset])
    if label_key:
        spec["label_key"] = label_key
    a = ad.read_h5ad(spec["path"], backed="r")
    genes = np.asarray(a.var_names).astype(str)
    n_genes = genes.size
    max_rank = int(np.ceil(0.05 * n_genes))
    ceiling = max(max_rank, 1500)
    print(f"[{dataset}] {a.shape[0]} cells x {n_genes} genes, "
          f"max_rank = {max_rank}, ceiling = {ceiling}", flush=True)
    print(f"[{dataset}] {spec['panel_note']}", flush=True)

    labels = np.asarray(a.obs[spec["label_key"]].astype(str))
    depth_obs = np.asarray(a.obs[spec["depth_key"]], dtype=float)
    hvg = (np.flatnonzero(np.asarray(a.var["highly_variable"]))
           if "highly_variable" in a.var else None)
    print(f"[{dataset}] {len(set(labels))} groups on "
          f"{spec['label_key']!r}, hvg = "
          f"{'none' if hvg is None else hvg.size}", flush=True)

    gene_sets = load_gene_sets(genes)
    universe = sorted(set().union(*gene_sets.values())) if gene_sets else []
    pos_of = {g: i for i, g in enumerate(genes)}
    keep = np.array([pos_of[g] for g in universe], dtype=np.int64)
    print(f"[{dataset}] {len(gene_sets)} gene sets, {keep.size} genes to retain "
          f"({100 * keep.size / n_genes:.0f}% of the panel)", flush=True)

    types = sorted(set(labels)) if cell_types is None else cell_types
    rng = np.random.default_rng(seed)
    rows, null_rows = [], []

    for ct in types:
        all_cells = np.flatnonzero(labels == ct)
        if all_cells.size < 200:
            print(f"  [{ct}] only {all_cells.size} cells, skipped", flush=True)
            continue
        if all_cells.size > max_cells_per_type:
            cells = np.sort(rng.choice(all_cells, size=max_cells_per_type,
                                       replace=False))
        else:
            cells = all_cells

        t0 = time.time()
        Xn, det, depth = fetch(a, cells, spec)
        # `n_genes_by_counts` was computed before this object was subset to its
        # gene panel, so it counts genes the score can never rank.  The panel's
        # own non-zero count is the quantity the tie-break denominator needs, and
        # the gap is recorded rather than assumed away.  It is typically one or
        # two genes; it is not assumed to be, which is why it is checked.
        depth_gap = float(np.median(depth_obs[cells] - depth))
        if abs(depth_gap) > 0.02 * np.median(depth):
            raise RuntimeError(
                f"[{ct}] detected-gene count disagrees with "
                f"obs[{spec['depth_key']!r}] by a median of {depth_gap:.1f} "
                f"against a depth of {np.median(depth):.0f}; the depth statistic "
                f"and the score are not describing the same gene panel")
        sub = sg.RankCache.build_streaming(
            ((lo, min(lo + block, Xn.shape[0]), Xn[lo:lo + block])
             for lo in range(0, Xn.shape[0], block)),
            n_cells=Xn.shape[0], n_genes=n_genes, genes=genes,
            ceiling=ceiling, keep=keep, seed=seed)
        B_sub = sp.csr_matrix(det)
        del Xn, det
        t_cache = time.time() - t0

        axis, evr, axis_depth_rho = None, np.nan, np.nan
        if hvg is not None and hvg.size >= 50:
            Xaxis, _, _ = fetch(a, cells, spec)
            axis, evr = dominant_axis(Xaxis, hvg)
            del Xaxis
            axis_depth_rho = sg.spearman(axis, depth)[0]

        include = tie_break_inclusion(depth, max_rank, n_genes)
        med_depth = float(np.median(depth))
        # A step that was never run is not a number that came out undefined.
        # The axis is skipped when the object carries no highly-variable genes,
        # and printing the NaN it leaves behind as `+nan` reads as a correlation
        # that was computed and failed -- which is how a whole missing half of
        # the benchmark went unnoticed in the log for an hour of wall time.
        rho_note = (f"{axis_depth_rho:+.3f}" if np.isfinite(axis_depth_rho)
                    else "not attempted (no highly-variable genes)")
        print(f"  [{ct}] {cells.size} cells (of {all_cells.size}), "
              f"median detected {med_depth:.0f} "
              f"(max_rank/median = {max_rank / max(med_depth, 1):.2f}), "
              f"tie-break inclusion {include:.4f}, "
              f"axis-depth rho {rho_note}, cache {t_cache:.0f}s",
              flush=True)

        r_implied = int(round(med_depth))
        t0 = time.time()
        for name, members in gene_sets.items():
            present = sub.present(members)
            if len(present) < MIN_OVERLAP:
                continue
            # UCell's normaliser needs r_max > (k + 1) / 2, and the cache has to
            # hold the cap.  The implied r_max can fall below half the set size
            # for a large set in a shallow cell type.
            r_imp = int(min(max(r_implied, len(present) + 1), sub.ceiling))
            score = sg.aucell(sub, present, max_rank=max_rank)
            diag = sg.sparsity_report(sub, present, rank_frac=0.05,
                                      thresholds=None)
            u_def = sg.ucell(sub, present, r_max=min(1500, sub.ceiling))
            u_imp = sg.ucell(sub, present, r_max=r_imp)
            rec = dict(
                dataset=dataset, cell_type=ct, gene_set=name, k=len(present),
                n_cells=int(cells.size), max_rank=max_rank, ceiling=ceiling,
                median_detected=med_depth,
                depth_ratio=float(max_rank / max(med_depth, 1)),
                tie_break_inclusion=include,
                median_depth_gap=depth_gap,
                axis_evr=evr, axis_depth_rho=axis_depth_rho,
                score_mean=float(score.mean()), score_sd=float(np.std(score)),
                ucell_default_mean=float(u_def.mean()),
                ucell_implied_mean=float(u_imp.mean()),
                ucell_r_implied=r_imp,
                rho_ucell_default_implied=sg.spearman(u_def, u_imp)[0],
                rho_aucell_ucell=sg.spearman(score, u_def)[0],
                rho_depth=sg.spearman(score, depth)[0],
                rho_depth_ucell=sg.spearman(u_def, depth)[0],
                observed_zero_rate=diag["observed_zero_rate"],
                structural_zero_rate=diag["structural_zero_rate"],
                zero_rate_excess=diag["zero_rate_excess"],
                mean_detection=diag["mean_detection"],
                min_detection=diag["min_detection"],
                median_detection=diag["median_detection"],
                frac_above_floor=diag["frac_above_floor"],
                ucell_default_is_appropriate=bool(1200 <= med_depth <= 2000),
            )
            if axis is not None:
                rec["rho_axis"] = sg.spearman(score, axis)[0]
                rec["partial_axis_given_depth"] = partial_spearman(
                    score, axis, depth)
            rows.append(rec)
        print(f"  [{ct}] {len(gene_sets)} sets scored in {time.time() - t0:.0f}s "
              f"({time.time() - t_start:.0f}s total)", flush=True)

        # --- matched nulls on a stratified subsample ----------------
        table = pd.DataFrame([r for r in rows if r["cell_type"] == ct])
        if table.empty:
            del sub, B_sub
            continue
        table = table.sort_values("frac_above_floor")
        picks = np.linspace(0, len(table) - 1,
                            min(n_null_sets, len(table))).astype(int)
        picked_names = [table.iloc[i]["gene_set"] for i in picks]
        # One builder serves every picked set, so it takes the lowest of their
        # floors.  A lower floor only widens the candidate pool, so this cannot
        # censor any of them from below -- which a floor fixed at any single
        # value could.
        floor = min(sg.adaptive_detection_floor(sub, gene_sets[n])
                    for n in picked_names)
        builder = sg.MatchedNullBuilder(
            sub, detection_matrix=B_sub, detection_matrix_genes=genes,
            det_floor=floor, seed=seed)
        nrng = np.random.default_rng(seed + 1)
        for name in picked_names:
            present = sub.present(gene_sets[name])
            score = sg.aucell(sub, present, max_rank=max_rank)
            rho_obs = sg.spearman(score, depth)[0]
            rho_axis_obs = (sg.spearman(score, axis)[0]
                            if axis is not None else np.nan)
            rec = dict(dataset=dataset, cell_type=ct, gene_set=name,
                       k=len(present), n_cells=int(cells.size),
                       max_rank=max_rank, rho_depth=rho_obs,
                       rho_axis=rho_axis_obs,
                       observed_codetection=builder.codetection(present))
            for kind in ("random", "expression", "codetection"):
                sets = builder.sample(present, n_null, kind=kind, rng=nrng,
                                      n_draws=n_draws)
                d = np.array([sg.spearman(sg.aucell(sub, s, max_rank=max_rank),
                                          depth)[0] for s in sets])
                d = d[np.isfinite(d)]
                rec[f"{kind}_null_depth_median"] = (
                    float(np.median(d)) if d.size else np.nan)
                rec[f"{kind}_null_depth_p"] = (
                    sg.empirical_p(d, rho_obs, tail="two-sided")
                    if d.size else np.nan)
                rec[f"{kind}_depth_reject"] = bool(
                    d.size and rec[f"{kind}_null_depth_p"] < 0.05)
                if axis is not None:
                    a_ = np.array([sg.spearman(sg.aucell(sub, s, max_rank=max_rank),
                                               axis)[0] for s in sets])
                    a_ = a_[np.isfinite(a_)]
                    rec[f"{kind}_null_axis_median"] = (
                        float(np.median(a_)) if a_.size else np.nan)
                    rec[f"{kind}_axis_p"] = (
                        sg.empirical_p(a_, rho_axis_obs, tail="two-sided")
                        if a_.size else np.nan)
                    rec[f"{kind}_axis_reject"] = bool(
                        a_.size and rec[f"{kind}_axis_p"] < 0.05)
            null_rows.append(rec)
        print(f"    matched nulls done for {len(picks)} sets "
              f"({time.time() - t_start:.0f}s total)", flush=True)
        del sub, B_sub

    os.makedirs(OUT_DIR, exist_ok=True)
    main_df = pd.DataFrame(rows)
    if main_df.empty:
        # An empty frame here means every label was wrong or every group was too
        # small; writing it would overwrite a good earlier run with nothing.
        raise SystemExit(
            f"[{dataset}] no rows produced -- check the label key "
            f"{spec['label_key']!r} against {sorted(set(labels))[:20]}")
    main_path = os.path.join(OUT_DIR, f"benchmark_{dataset}_all.csv")
    main_df.to_csv(main_path, index=False)
    print(f"wrote {main_path}  ({main_df.shape})", flush=True)

    if null_rows:
        null_df = pd.DataFrame(null_rows)
        null_path = os.path.join(OUT_DIR, f"benchmark_{dataset}_nulls.csv")
        null_df.to_csv(null_path, index=False)
        print(f"wrote {null_path}  ({null_df.shape})", flush=True)
    return main_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gse176078", choices=sorted(DATASETS))
    ap.add_argument("--null-sets", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-draws", type=int, default=25,
                    help="candidate draws per returned co-detection null")
    ap.add_argument("--max-cells", type=int, default=4000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--label-key", default=None)
    ap.add_argument("--cell-types", nargs="*", default=None)
    args = ap.parse_args()
    run(args.dataset, n_null_sets=args.null_sets, n_null=args.n_null,
        n_draws=args.n_draws, max_cells_per_type=args.max_cells,
        cell_types=args.cell_types, label_key=args.label_key, block=args.block)
