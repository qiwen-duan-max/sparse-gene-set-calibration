#!/usr/bin/env python
"""The realised tie-break share of the score, per compartment and gene set.

The benchmark records, for every (cell type, gene set) pair, the *probability*
that a gene the cell cannot detect is nevertheless ranked inside the top block
-- ``tie_break_inclusion``.  That quantity is a property of the compartment's
depth and the matrix's panel alone: it is the same number for every gene set
scored in that compartment, so it describes the regime rather than what the
regime did to any particular score.

The figure's compartment panel needs the other quantity: of the score each set
actually received in each compartment, how much came from ranks settled by the
tie-break rather than by an observed count.  That is the realised split of
equation (2), it differs from set to set, and it is the one a reader can act on.
It is not in the benchmark table because computing it needs the cache, not only
the summary columns, so it is computed here.

The cell subsample is replayed rather than redrawn.  The benchmark draws it from
``numpy.random.default_rng(seed)`` in sorted compartment order, so the same
sequence is reproduced by walking the compartments the same way; the count is
checked against the benchmark table for every compartment, and a mismatch stops
the script rather than silently describing a different sample.

    python experiments/compartment_tie_break.py --dataset gse176078
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

import anndata as ad  # noqa: E402

import sparsegs as sg  # noqa: E402
from real_benchmark import (DATASETS, fetch, load_gene_sets,  # noqa: E402
                            tie_break_inclusion)

RESULTS = os.path.join(PKG, "results")
MIN_CELLS = 200                 # the benchmark's own floor, kept the same


def main(dataset="gse176078", block=500, seed=0, max_cells=4000,
         outdir=RESULTS):
    bench_path = os.path.join(outdir, f"benchmark_{dataset}_all.csv")
    if not os.path.exists(bench_path):
        raise SystemExit(
            f"missing {bench_path}; run experiments/real_benchmark.py first")
    table = pd.read_csv(bench_path)
    missing = {"cell_type", "gene_set", "n_cells"} - set(table.columns)
    if missing:
        raise SystemExit(f"{bench_path} lacks {sorted(missing)}")

    spec = dict(DATASETS[dataset])
    a = ad.read_h5ad(spec["path"], backed="r")
    genes = np.asarray(a.var_names).astype(str)
    n_genes = genes.size
    max_rank = int(np.ceil(0.05 * n_genes))
    ceiling = max(max_rank, 1500)
    labels = np.asarray(a.obs[spec["label_key"]].astype(str))
    gene_sets = load_gene_sets(genes)
    pos_of = {g: i for i, g in enumerate(genes)}
    universe = sorted(set().union(*gene_sets.values()))
    keep = np.array([pos_of[g] for g in universe], dtype=np.int64)
    print(f"[{dataset}] {a.shape[0]} cells x {n_genes} genes, "
          f"max_rank = {max_rank}, {len(gene_sets)} sets", flush=True)

    # What the benchmark actually scored, so the two tables describe the same
    # pairs.  A pair in one and not the other is a join that quietly drops rows
    # from the figure.
    #
    # Both sides are keyed as text, and the reason is not stylistic.  The two
    # datasets are annotated differently -- one by a published cell-type column,
    # the other by Leiden clusters -- so the benchmark table's ``cell_type``
    # column holds names in one file and integers in the other.  pandas infers
    # the dtype on read, so comparing the table's values against the labels
    # (which are cast to text here) matched nothing at all for the clustered
    # cohort: every compartment was skipped as unscored, and the script
    # finished with no rows and the single line "no compartment produced a
    # decomposition".  Text on both sides is what the comparison -- and the
    # merge in ``analyse_benchmark.py``, which reads the file written below --
    # can rely on either way.
    wanted = {}
    for ct, gs in zip(table["cell_type"], table["gene_set"]):
        wanted.setdefault(str(ct), set()).add(gs)
    expected_n = {str(k): v for k, v in
                  table.groupby("cell_type", observed=True)["n_cells"].max().items()}

    rng = np.random.default_rng(seed)
    rows = []
    for ct in sorted(set(labels)):
        all_cells = np.flatnonzero(labels == ct)
        if all_cells.size < MIN_CELLS:
            continue                      # the benchmark skipped it too
        if all_cells.size > max_cells:
            cells = np.sort(rng.choice(all_cells, size=max_cells,
                                       replace=False))
        else:
            cells = all_cells
        if ct not in wanted:
            continue                      # the benchmark scored nothing here
        if ct in expected_n and int(expected_n[ct]) != int(cells.size):
            raise SystemExit(
                f"[{ct}] replayed {cells.size} cells against the benchmark's "
                f"{int(expected_n[ct])}; the subsample did not reproduce, so "
                f"the shares would describe a different sample")

        t0 = time.time()
        Xn, det, depth = fetch(a, cells, spec)
        cache = sg.RankCache.build_streaming(
            ((lo, min(lo + block, Xn.shape[0]), Xn[lo:lo + block])
             for lo in range(0, Xn.shape[0], block)),
            n_cells=Xn.shape[0], n_genes=n_genes, genes=genes,
            ceiling=ceiling, keep=keep, seed=seed)
        # ``score_decomposition`` reads its matrix in the cache's own column
        # order, and ``fetch`` returns the whole panel, so the alignment happens
        # here rather than by name inside the call: ``keep`` is the cache's gene
        # universe in the order the cache holds it.  Slicing loses nothing --
        # every set scored below is a subset of that universe -- and it is the
        # panel-wide depth that has to stay whole, which is why ``depth`` is
        # still the one ``fetch`` returned.
        B = sp.csr_matrix(det)[:, keep]
        del Xn, det
        med_depth = float(np.median(depth))
        ratio = float(max_rank / max(med_depth, 1))
        # Recomputed from the same depth vector the benchmark used, then checked
        # against the benchmark's own column.  The two scripts must agree on the
        # regime before either can be said to disagree about the consequence.
        inclusion = tie_break_inclusion(depth, max_rank, n_genes)
        if "tie_break_inclusion" in table.columns:
            # Compared as text for the same reason the keys above are: the
            # compartment labels of the two datasets do not share a dtype, and
            # an integer column tested against a string label selects nothing
            # rather than raising -- ``iloc[0]`` on an empty selection was the
            # only thing that said so.
            theirs = float(table.loc[table["cell_type"].astype(str) == ct,
                                     "tie_break_inclusion"].iloc[0])
            if abs(theirs - inclusion) > 1e-9:
                raise SystemExit(
                    f"[{ct}] tie-break inclusion {inclusion:.6f} does not "
                    f"reproduce the benchmark's {theirs:.6f}")

        for name in sorted(wanted[ct]):
            members = gene_sets.get(name)
            if members is None:
                continue
            present = cache.present(members)
            if not present:
                continue
            split = sg.score_decomposition(cache, present, B,
                                           max_rank=max_rank, depth=depth)
            total = float(split["total"].mean())
            share = float(split["tie_break"].mean() / total) if total > 0 else np.nan
            rows.append(dict(
                dataset=dataset, cell_type=ct, gene_set=name, k=len(present),
                n_cells=int(cells.size), max_rank=max_rank,
                median_detected=med_depth, depth_ratio=ratio,
                tie_break_inclusion=inclusion,
                mean_tie_break=float(split["tie_break"].mean()),
                mean_detected_part=float(split["detected"].mean()),
                mean_total=total, score_tie_break_share=share))
        print(f"  [{ct}] {cells.size} cells, {len(wanted[ct])} sets, "
              f"{time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no compartment produced a decomposition")
    path = os.path.join(outdir, f"compartment_tie_break_{dataset}.csv")
    out.to_csv(path, index=False)

    by_ct = out.groupby("cell_type", observed=True)["score_tie_break_share"]
    print(f"\nwrote {path}  ({len(out)} rows, {out['cell_type'].nunique()} "
          f"compartments)")
    print("realised tie-break share of the score, median across sets")
    for ct, value in by_ct.median().sort_values(ascending=False).items():
        sub = out[out["cell_type"] == ct]
        print(f"  {ct:28s} {value:.4f}   (max {sub['score_tie_break_share'].max():.4f},"
              f" depth ratio {sub['depth_ratio'].iloc[0]:.2f})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="gse176078")
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cells", type=int, default=4000)
    ap.add_argument("--outdir", default=RESULTS)
    main(**vars(ap.parse_args()))
