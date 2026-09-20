"""Generate the cross-language fixtures the R tests compare against.

The R package is a port, so "it produces plausible numbers" is not the bar --
it has to produce the *same* numbers as the reference implementation on the same
matrix.  That needs a matrix both languages can read, and the scores the Python
package computes on it.

    python tests/testthat/make_fixtures.py

The matrix is written as a dense CSV of the non-zero entries only, with the
dimensions in the header comment, so the fixture stays small enough to check
into the repository.

The counts are integers on purpose, and not for realism alone.  A rank is
decided by a comparison, so for the two languages to agree *exactly* they have
to hold the same values, and an integer count is the same number in R's double
and in NumPy's float32 while a rounded decimal is not: writing a gamma draw to
six decimals would hand R a value that differs from Python's in the seventh
digit, and the two would then disagree about the ranking of any two genes that
fall in the same decimal bin.  Counts also put exact ties everywhere -- a cell
has many genes at a count of one -- which is precisely the regime the
deterministic tie-break exists for, so the fixture exercises it rather than
tolerating it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
# .../sparseGenSetCal/tests/testthat -> the Python package lives four levels up.
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "tests"))

FIXTURES = os.path.join(HERE, "fixtures")
N_CELLS, N_GENES, SEED = 250, 900, 7
GENE_SET_SIZE = 12
CEILING = 60
RANK_FRAC = 0.05
#: The tie-break seed.  The R tests build with this one and compare against the
#: ranks and keys below, so it is part of the fixture and not a free choice.
TIE_SEED = 2024


def count_matrix(n_cells, n_genes, median_detected, seed):
    """Integer counts with a realistic gene-activity gradient.

    Mirrors what ``sparsegs.simulate`` produces, at the size a fixture can
    afford: per-gene activities spanning three orders of magnitude, per-cell
    depth factors, and Poisson counts on the product.
    """
    rng = np.random.default_rng(seed)
    activity = np.exp(rng.normal(0, 1.5, size=n_genes))
    activity /= np.median(activity)
    depth = np.exp(rng.normal(0, 0.45, size=n_cells))
    shape = np.outer(depth, activity)

    # One scalar governs the regime; solve for the one that puts the median
    # cell at the requested detection count.  Bisection on the log scale,
    # because the counts span orders of magnitude.
    def detected_at(log_scale):
        rate = shape * np.exp(log_scale)
        return np.median((1.0 - np.exp(-rate)).sum(axis=1))

    lo, hi = -30.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if detected_at(mid) < median_detected:
            lo = mid
        else:
            hi = mid
    rate = np.clip(shape * np.exp(0.5 * (lo + hi)), 0, 30.0)
    values = rng.poisson(rate).astype(np.float32)
    return sp.csr_matrix(values)


def main():
    import sparsegs as sg

    os.makedirs(FIXTURES, exist_ok=True)
    X = count_matrix(N_CELLS, N_GENES, median_detected=180, seed=SEED)
    genes = [f"G{i:05d}" for i in range(N_GENES)]
    gene_set = genes[:GENE_SET_SIZE]

    cache = sg.RankCache.build(X, genes, ceiling=CEILING, seed=TIE_SEED,
                               chunk=200)
    max_rank = int(np.ceil(RANK_FRAC * N_GENES))
    score = sg.aucell(cache, gene_set, max_rank=max_rank)
    diag = sg.sparsity_report(cache, gene_set, rank_frac=RANK_FRAC)
    r_max = int(np.median(cache.depth))
    uscore = sg.ucell(cache, gene_set, r_max=min(r_max, CEILING))

    # The matrix, as coordinate triples.  Dense would be 250 x 900 doubles.
    # Integer values, so that R's double and Python's float32 hold the same
    # number and a tie in one is a tie in the other.
    coo = X.tocoo()
    with open(os.path.join(FIXTURES, "matrix.csv"), "w") as fh:
        detected = int((X > 0).sum())
        fh.write(f"# n_cells={N_CELLS} n_genes={N_GENES} seed={SEED} "
                 f"median_detected=180 integer_counts=1 "
                 f"nonzeros={coo.nnz} detected={detected}\n")
        fh.write("row,col,value\n")
        for i, j, v in zip(coo.row, coo.col, coo.data):
            fh.write(f"{i},{j},{int(v)}\n")

    with open(os.path.join(FIXTURES, "aucell_reference.csv"), "w") as fh:
        fh.write(f"# n_cells={N_CELLS} n_genes={N_GENES} ceiling={CEILING} "
                 f"max_rank={max_rank} rank_frac={RANK_FRAC} "
                 f"gene_set={','.join(gene_set)}\n")
        fh.write("cell,aucell,ucell\n")
        for i in range(N_CELLS):
            fh.write(f"{i},{score[i]:.10f},{uscore[i]:.10f}\n")

    with open(os.path.join(FIXTURES, "diagnostics_reference.csv"), "w") as fh:
        fh.write("quantity,value\n")
        for key in ("observed_zero_rate", "structural_zero_rate",
                    "zero_rate_excess", "mean_detection", "median_detection",
                    "min_detection", "max_detection", "tie_break_share",
                    "mean_inclusion", "median_depth"):
            fh.write(f"{key},{float(diag[key]):.10f}\n")

    # --- the tie-break itself ---------------------------------------
    # Written out separately from the scores, because when the two languages
    # disagree this is the file that says which of the two did something
    # different: an equal key on the same coordinates means the hash is right
    # and the disagreement is downstream, and an unequal one localises it to
    # the arithmetic in R/tiebreak.R.
    probed = np.array([0, 1, 2, 17, 63, 128, 249], dtype=np.int64)
    gene_probe = np.array([0, 1, 5, 41, 512, 899], dtype=np.int64)
    keys = sg.tie_break_keys(probed, N_GENES, seed=TIE_SEED)
    with open(os.path.join(FIXTURES, "keys_reference.csv"), "w") as fh:
        fh.write(f"# n_genes={N_GENES} seed={TIE_SEED}\n")
        fh.write("cell,gene,key\n")
        for a, cell in enumerate(probed):
            for b, gene in enumerate(gene_probe):
                fh.write(f"{cell},{gene},{int(keys[a, b])}\n")

    # The ranks the reference implementation settled on, for a sample of cells:
    # the tie-break only matters through which gene takes which rank, so this
    # is the end-to-end check that it decided the same way in both languages.
    # `clipped` is indexed by gene and holds the rank, so the pairs come out of
    # its non-zeros, not out of its columns.
    probe_cells = list(range(0, N_CELLS, 11))
    clipped = np.asarray(cache.clipped, dtype=np.int64)
    with open(os.path.join(FIXTURES, "ranks_reference.csv"), "w") as fh:
        fh.write(f"# n_cells={N_CELLS} n_genes={N_GENES} ceiling={CEILING} "
                 f"seed={TIE_SEED}\n")
        fh.write("cell,rank,gene\n")
        written = 0
        for cell in probe_cells:
            row = clipped[cell]
            genes_hit = np.nonzero(row)[0]
            for gene, rank in sorted(zip(genes_hit, row[genes_hit]),
                                     key=lambda pair: pair[1]):
                fh.write(f"{cell},{int(rank)},{int(gene)}\n")
                written += 1

    print(f"wrote fixtures to {FIXTURES}: matrix {coo.nnz} non-zeros, "
          f"max_rank {max_rank}, ucell r_max {min(r_max, CEILING)}, "
          f"{written} ranks over {len(probe_cells)} cells")


if __name__ == "__main__":
    main()
