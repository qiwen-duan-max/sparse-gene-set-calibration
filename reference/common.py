"""The reference AUCell engine the test suite compares ``sparsegs`` against.

Extracted verbatim from the project's legacy scoring scripts (the module name
is kept as ``common`` because ``tests/test_core.py`` imports it under that
name): a naive Python reimplementation of Aibar et al. (2017) whose random
tie-breaking is exactly the confound the package exists to quantify.  Only the
scoring engine travels here -- the legacy module's gene-set definitions,
plotting style and machine-specific paths were outside the tests' reach and
stay out of the repository.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 2024


def _chunk_ranks(X, lo, hi, rng):
    """Descending expression ranks for a block of cells (1 = most expressed).

    Ties are broken by a minute random perturbation -- equivalent to the random
    tie-breaking the R implementation of AUCell performs, which the legacy
    engine inherited.
    """
    block = X[lo:hi]
    n = block.shape[0]
    n_genes = X.shape[1]
    dens = np.asarray(block.todense(), dtype=np.float32)
    dens += rng.random((n, n_genes), dtype=np.float32) * 1e-6
    order = np.argsort(-dens, axis=1, kind="stable")
    ranks = np.empty((n, n_genes), dtype=np.int32)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, n_genes + 1, dtype=np.int32)[None, :]
    return ranks


def _auc_from_ranks(ranks, gene_idx, max_rank):
    """Area under the recovery curve (Aibar et al. 2017, Nat Methods).

    Definition: the recovery curve f(k) is the fraction of the set among the
    top-k ranked genes, and the AUC is the integral of f over k in [0, maxRank]
    divided by maxRank.  Exchanging the order of summation, this is

        AUC = sum_i max(0, maxRank - r_i + 1) / (n_set * maxRank)

    -- each gene contributes from its own rank position up to maxRank.  Do not
    rewrite it as sum (r_i - r_{i-1}) * (n - i + 1) / n / maxRank: that
    quantity measures a gene's span along the rank axis, and for a random set
    it grows with r_i (observed in practice to saturate at 1.0), so it points
    in exactly the opposite direction.
    """
    sub = np.sort(ranks[:, gene_idx], axis=1)          # (n_cells, n_set) ascending
    n_set = sub.shape[1]
    contrib = np.where(sub <= max_rank, max_rank - sub + 1, 0)
    return np.clip(contrib.sum(axis=1) / (n_set * max_rank), 0.0, 1.0)


def score_matrix(counts, var_names, sets_dict, max_rank_frac=0.05, chunk=2000, seed=SEED):
    """AUCell scores for an arbitrary sparse matrix (e.g. spot x gene spatial
    data); returns a DataFrame with one ``AUC_<set>`` column per set."""
    X = sp.csr_matrix(counts)
    gidx = {g: i for i, g in enumerate(var_names)}
    max_rank = int(np.ceil(max_rank_frac * len(var_names)))
    rng = np.random.default_rng(seed)
    out = {}
    names = [(n, np.array([gidx[g] for g in gs if g in gidx]))
             for n, gs in sets_dict.items()]
    names = [(n, ix) for n, ix in names if len(ix) >= 3]
    for n, _ in names:
        out[f"AUC_{n}"] = np.empty(X.shape[0], dtype=np.float32)
    for lo in range(0, X.shape[0], chunk):
        hi = min(lo + chunk, X.shape[0])
        ranks = _chunk_ranks(X, lo, hi, rng)
        for n, ix in names:
            out[f"AUC_{n}"][lo:hi] = _auc_from_ranks(ranks, ix, max_rank)
        del ranks
    return pd.DataFrame(out, index=None)
