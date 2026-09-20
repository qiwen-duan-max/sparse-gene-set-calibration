"""Deterministic tie-breaking for the rank cache.

Most of the entries of a single-cell expression matrix are zero, so most of the
comparisons a rank-based score makes are between two values that are equal.  The
order the zeros end up in is not a detail: it decides which undetected genes
occupy the ranks inside the ceiling, and hence what fraction of a gene set's
score was never measured at all.  That is the subject of :mod:`sparsegs.theory`.

The reference AUCell implementation breaks the ties by adding a uniform random
perturbation to the expression values before ranking.  Drawing that perturbation
from the language's own random number generator has two consequences a
calibration tool cannot live with:

* the ranks depend on how the matrix was blocked.  A generator produces a
  stream, and one block of 500 cells consumes a different part of it than two
  blocks of 250, so the same matrix and the same seed give different scores
  depending on the ``chunk`` argument;
* the ranks depend on which language is running, because R's Mersenne-Twister
  and NumPy's PCG64 do not produce the same numbers.  A port can then only be
  tested for statistical similarity, never for equality.

This module replaces the stream with a *key*: a pure function of the cell index,
the gene index and the seed, with no generator state and no dependence on how
the work is divided up.  Ranking is by ``(expression descending, key ascending)``,
so a tie is settled by the key and everything else is settled by the data.

The key is MurmurHash3's 32-bit finaliser over a linear mix of the coordinates.
It uses 32-bit wrapping arithmetic and shifts only, both of which are exact in
IEEE doubles, so :mod:`sparsegs` and the R package ``sparseGenSetCal`` compute
the same integers and the two implementations rank every cell identically.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tie_break_keys", "ranked_columns", "MASK32"]

#: 2**32 - 1, the modulus every key operation works in.
MASK32 = 0xFFFFFFFF

# Odd multipliers with good bit dispersion.  The first two are the golden-ratio
# and MurmurHash3 constants; the third separates the seeds.
_C_CELL = 0x9E3779B1
_C_GENE = 0x85EBCA6B
_C_SEED = 0xC2B2AE35


def _fmix32(h):
    """MurmurHash3's finaliser, on a ``uint32`` array.

    Multiplication wraps modulo ``2**32``; NumPy does that silently for unsigned
    integers, which is exactly the behaviour wanted here.
    """
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x846CA68B)
    h ^= h >> np.uint32(16)
    return h


def tie_break_keys(cells, n_genes, seed=2024):
    """Tie-break keys for a block of cells against a whole panel of genes.

    Parameters
    ----------
    cells : sequence of int
        Row indices of the block in the *original* matrix.  They have to be the
        original ones: using the position within the block would reintroduce the
        dependence on the blocking that this module exists to remove.
    n_genes : int
        Number of genes in the panel.
    seed : int
        Seed.

    Returns
    -------
    numpy.ndarray
        ``uint32`` array of shape ``(len(cells), n_genes)``.  The keys are
        uniformly distributed over the range and independent across coordinates,
        so comparing two of them is a fair coin.
    """
    cells = np.asarray(cells, dtype=np.uint32).reshape(-1, 1)
    genes = np.arange(n_genes, dtype=np.uint32).reshape(1, -1)
    # The seed term is folded in Python integers before the cast, so it wraps on
    # purpose instead of raising NumPy's scalar-overflow warning.
    seed_term = np.uint32((int(seed) & MASK32) * _C_SEED & MASK32)
    h = (cells * np.uint32(_C_CELL) + genes * np.uint32(_C_GENE) + seed_term)
    return _fmix32(h)


def ranked_columns(block, keys, ceiling, row_budget=1 << 28):
    """Gene columns occupying ranks ``1..ceiling`` in each row of a block.

    The total order is ``(expression descending, key ascending)``, built as a
    single 64-bit integer so that it is a *total* order and not a comparison the
    selection algorithm is free to complete:

    ``composite = ((2**32 - 1 - bits(value)) << 32) | key``

    The bit pattern of a positive ``float32`` increases with the value, so
    complementing it gives a number that increases as the value falls, and the
    key rides in the low half to settle exact equality.  A zero has a bit
    pattern of zero, so every undetected gene gets a composite above every
    detected one automatically, ordered among themselves by key -- which is
    equation (1)'s uniform random order, and the reason no special case is
    needed for the undetected fill.

    This matters more than it looks.  Count data is full of exact ties -- a cell
    with a thousand detected genes has many of them at a count of one -- so
    "which of these tied genes gets the last rank inside the ceiling" is a
    question asked in essentially every cell.  Leaving it to ``argpartition``,
    which is free to break ties however it likes, would make the membership of
    the ranked set an implementation detail of the selection algorithm.

    Parameters
    ----------
    block : numpy.ndarray
        Dense ``(n_cells, n_genes)`` block of expression values.  Values at or
        below zero count as undetected; NaN is not supported.
    keys : numpy.ndarray
        ``uint32`` array of the same shape, from :func:`tie_break_keys`.
    ceiling : int
        Rank ceiling ``m``.
    row_budget : int
        Bytes of temporary per pass.  The composite is eight bytes per entry, so
        a wide panel is ranked a band of rows at a time to keep the peak down.

    Returns
    -------
    numpy.ndarray
        ``int32`` array of shape ``(n_cells, ceiling)``, the gene column of each
        rank in order.
    """
    from .rankcache import BEYOND

    n, n_genes = block.shape
    ceiling = int(ceiling)
    if ceiling < 1 or ceiling > n_genes:
        raise ValueError(f"ceiling must lie in [1, {n_genes}], got {ceiling}")
    if keys.shape != block.shape:
        raise ValueError("keys must have the same shape as the block")

    out = np.full((n, ceiling), BEYOND, dtype=np.int32)
    rows_per = max(1, int(row_budget // max(8 * n_genes, 1)))
    for lo in range(0, n, rows_per):
        hi = min(lo + rows_per, n)
        out[lo:hi] = _ranked_band(np.ascontiguousarray(block[lo:hi]),
                                  keys[lo:hi], ceiling)
    return out


def _ranked_band(block, keys, ceiling):
    """One band of rows, small enough to hold the composite in memory."""
    bits = block.view(np.uint32)
    composite = (np.uint32(MASK32) - bits).astype(np.uint64)
    composite <<= np.uint64(32)
    composite |= keys

    cand = np.argpartition(composite, ceiling - 1, axis=1)[:, :ceiling]
    order = np.argsort(np.take_along_axis(composite, cand, axis=1), axis=1)
    return np.take_along_axis(cand, order, axis=1).astype(np.int32)
