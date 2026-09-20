"""Clipped-rank cache: the scoring engine that makes calibration tractable.

Every rank-based gene-set score (AUCell, UCell, and relatives) is a function of
each gene's rank *within a cell*, and in practice only of the ranks below a
ceiling.  Building the full rank matrix is O(n_cells x n_genes x log n_genes) and
storing it costs 4 bytes per entry, which makes a calibration sweep over hundreds
of gene sets and thousands of null draws impossible on ordinary hardware.

This module builds the ranks once and keeps only what scoring needs: for each
cell and each gene of interest, the exact rank if that rank is at or below a
ceiling, and a sentinel zero otherwise.  Ranks above the ceiling contribute
nothing to any score implemented here, so the compression is lossless.

The cost is then O(n_cells x n_genes) once, instead of once per gene set.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .tiebreak import ranked_columns, tie_break_keys

__all__ = ["RankCache"]

#: Sentinel stored for a gene whose rank exceeds the ceiling.  Zero is safe
#: because ranks are one-based, so no real rank can collide with it.
BEYOND = 0


class RankCache:
    """Per-cell gene ranks, clipped at a rank ceiling.

    Parameters
    ----------
    clipped : numpy.ndarray
        Integer array of shape ``(n_cells, n_genes)``.  ``clipped[i, j]`` is the
        one-based rank of gene ``j`` in cell ``i`` when that rank is at most
        ``ceiling``, and :data:`BEYOND` otherwise.
    genes : numpy.ndarray
        Gene names, aligned with the columns of ``clipped``.
    n_genes_total : int
        Number of genes in the expression matrix the ranks were computed from,
        including genes that were not retained in ``clipped``.  Rank-based
        scores depend on this number, so it is part of the cache's identity.
    ceiling : int
        Rank ceiling used when building the cache.
    detection : numpy.ndarray
        Per-gene detection rate (fraction of cells with a non-zero count) over
        the cells the cache was built from, aligned with ``genes``.
    mean_expression : numpy.ndarray
        Per-gene mean expression over the same cells, aligned with ``genes``.
    depth : numpy.ndarray or None
        ``D_c``: the number of genes each cell detected, counted over the *whole*
        panel rather than over the retained columns.  It is the argument of the
        closed form in :mod:`sparsegs.theory`, so the cache carries it rather
        than making every caller recompute it from a matrix it may no longer
        have.  ``None`` for caches built before this was recorded.

    Notes
    -----
    Ties are broken by a deterministic key: genes are ranked by
    ``(expression descending, key ascending)``, where the key is a hash of the
    cell index, the gene index and the seed.  Nothing about the ranks depends on
    how the cells were blocked, so a cache built in one chunk is identical to
    one built in a hundred; and because the key is written in 32-bit integer
    arithmetic rather than drawn from a generator, the R package
    ``sparseGenSetCal`` reproduces it exactly.  The uniform order the keys give
    the undetected genes is what makes equation (1) of :mod:`sparsegs.theory` an
    expectation over the tie-break, as the reference AUCell perturbation does,
    without inheriting that perturbation's dependence on the generator.
    """

    __slots__ = ("clipped", "genes", "n_genes_total", "ceiling",
                 "detection", "mean_expression", "depth", "_index")

    def __init__(self, clipped, genes, n_genes_total, ceiling, detection,
                 mean_expression, depth=None):
        self.clipped = clipped
        self.genes = np.asarray(genes)
        self.n_genes_total = int(n_genes_total)
        self.ceiling = int(ceiling)
        self.detection = np.asarray(detection, dtype=float)
        self.mean_expression = np.asarray(mean_expression, dtype=float)
        self.depth = None if depth is None else np.asarray(depth, dtype=np.int32)
        self._index = {g: i for i, g in enumerate(self.genes)}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, X, genes, ceiling, genes_of_interest=None, seed=2024,
              chunk=1500, verbose=False):
        """Compute a :class:`RankCache` from a sparse expression matrix.

        Parameters
        ----------
        X : scipy.sparse matrix or array-like
            Cells x genes expression matrix (log-normalised values, not
            counts).  Entries must be finite and non-negative: the rank order
            is defined on the bit pattern of a non-negative ``float32``, so
            NaN, infinite or negative values would sort on the wrong side of
            every detected gene, and they are rejected here rather than
            silently mis-ranked.
        genes : sequence of str
            Gene names aligned with the columns of ``X``.
        ceiling : int
            Largest rank that must be stored exactly.
        genes_of_interest : sequence of str, optional
            Restrict the cache to these genes.  Defaults to every gene, which is
            rarely what you want: the point of the cache is to drop the columns
            no gene set will ever query.
        seed : int
            Seed of the tie-breaking key.  The ranks are a function of the
            matrix and this number alone.
        chunk : int
            Number of cells processed per block.  Lower it if memory is tight.
            It changes how long the build takes and how much memory it needs,
            and nothing else: the ranks are the same at every chunk size.
        verbose : bool
            Print progress.

        Returns
        -------
        RankCache
        """
        X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
        genes = np.asarray(genes)
        n_cells, n_genes = X.shape
        if len(genes) != n_genes:
            raise ValueError(
                f"genes has length {len(genes)} but X has {n_genes} columns")
        if X.data.size:
            if not np.isfinite(X.data).all():
                raise ValueError(
                    "X contains NaN or infinite values; the rank cache needs "
                    "finite, non-negative expression values")
            if X.data.min() < 0:
                raise ValueError(
                    "X contains negative values; the rank order is defined "
                    "for non-negative values only -- pass log-normalised data")

        if genes_of_interest is None:
            keep = np.arange(n_genes)
        else:
            pos = {g: i for i, g in enumerate(genes)}
            keep = np.array([pos[g] for g in genes_of_interest
                             if g in pos], dtype=np.int64)
            if keep.size == 0:
                raise ValueError("none of genes_of_interest are present in X")

        ceiling = int(ceiling)
        if ceiling < 1 or ceiling >= n_genes:
            raise ValueError(
                f"ceiling must lie in [1, {n_genes - 1}], got {ceiling}")

        detection = np.asarray((X > 0).sum(axis=0)).ravel() / n_cells
        mean_expression = np.asarray(X.mean(axis=0)).ravel()
        # Depth is the rank ceiling's counterpart and costs one pass over the
        # same data the detection rates already need.
        depth = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.int32)

        def blocks():
            for lo in range(0, n_cells, chunk):
                hi = min(lo + chunk, n_cells)
                yield lo, hi, X[lo:hi]

        return cls.build_streaming(
            blocks(), n_cells=n_cells, n_genes=n_genes, genes=genes,
            ceiling=ceiling, keep=keep, seed=seed, verbose=verbose,
            detection=detection, mean_expression=mean_expression, depth=depth)

    @classmethod
    def build_streaming(cls, blocks, n_cells, n_genes, genes, ceiling, keep=None,
                        seed=2024, detection=None, mean_expression=None,
                        on_binary=None, depth=None, verbose=False):
        """Build a cache from an iterator of row blocks, never holding the matrix.

        A 55,000-cell matrix is unremarkable and a half-million-cell one is not
        rare; materialising either to compute ranks is wasteful, and on a shared
        machine it is the difference between finishing and being killed.  This
        takes the matrix one row block at a time instead.

        Parameters
        ----------
        blocks : iterable of (lo, hi, X_block)
            ``X_block`` is a sparse ``(hi - lo, n_genes)`` matrix.  The blocks
            must cover every row exactly once and arrive in any order.
        n_cells, n_genes : int
            Dimensions of the matrix the blocks came from.
        genes : sequence of str
            Gene names for all ``n_genes`` columns.
        ceiling : int
        keep : array-like of int, optional
            Columns to retain.  Defaults to all of them.
        detection, mean_expression : array-like, optional
            Precomputed per-gene values.  Computed from the blocks when omitted,
            which needs a second pass over them, so pass them if you have them.
        on_binary : callable, optional
            Called as ``on_binary(lo, hi, binary_block)`` for each block, where
            ``binary_block`` is the sparse detection pattern.  Lets a caller
            accumulate the detection matrix in the same pass.
        depth : array-like, optional
            Precomputed per-cell ``D_c``.  Accumulated from the blocks when
            omitted, which the block loop does anyway at no extra cost.

        Returns
        -------
        RankCache
        """
        genes = np.asarray(genes)
        keep = np.arange(n_genes) if keep is None else np.asarray(keep)
        ceiling = int(ceiling)
        if ceiling < 1 or ceiling >= n_genes:
            raise ValueError(
                f"ceiling must lie in [1, {n_genes - 1}], got {ceiling}")
        if keep.size == 0:
            raise ValueError("keep selects no genes")

        # Ranks are stored at the narrowest width that holds the ceiling.  A
        # ceiling below 32767 is the usable case for every panel in practice,
        # and halving the array is what decides whether a whole-panel cache fits
        # in memory at all.
        rank_dtype = np.int16 if ceiling < 32767 else np.int32
        clipped = np.zeros((n_cells, keep.size), dtype=rank_dtype)
        ranks = np.arange(1, ceiling + 1, dtype=rank_dtype)
        buffer = np.empty((0, n_genes), dtype=rank_dtype)

        det_counts = np.zeros(n_genes, dtype=np.int64) if detection is None else None
        expr_sum = np.zeros(n_genes, dtype=np.float64) if detection is None else None
        depth_counts = np.zeros(n_cells, dtype=np.int64) if depth is None else None

        all_genes = keep.size == n_genes
        report_every = max(n_cells // 5, 1)
        for lo, hi, X_block in blocks:
            n = hi - lo
            if n <= 0:
                continue
            if sp.issparse(X_block):
                if X_block.shape[1] != n_genes:
                    raise ValueError(
                        f"block {lo}:{hi} has {X_block.shape[1]} columns, expected "
                        f"{n_genes}")
                if det_counts is not None:
                    det_counts += np.asarray((X_block > 0).sum(axis=0)).ravel()
                    expr_sum += np.asarray(X_block.sum(axis=0)).ravel()
                block = np.asarray(X_block.todense(), dtype=np.float32)
                binary = (X_block > 0)
            else:
                # A dense block is converted rather than viewed only because the
                # ranking works in single precision; nothing writes into it.
                block = np.asarray(X_block, dtype=np.float32)
                if block.shape[1] != n_genes:
                    raise ValueError(
                        f"block {lo}:{hi} has {block.shape[1]} columns, expected "
                        f"{n_genes}")
                if det_counts is not None:
                    det_counts += (block > 0).sum(axis=0)
                    expr_sum += block.sum(axis=0, dtype=np.float64)
                binary = (block > 0)

            if depth_counts is not None:
                depth_counts[lo:hi] = np.asarray(binary.sum(axis=1)).ravel()

            # The entry checks of `build` cannot see a matrix that arrives as a
            # block iterator, so every block is checked as it streams instead.
            # A NaN would take rank 1 and a negative value would sort below
            # every detected gene -- both silently -- so they are refused here.
            if not np.isfinite(block).all() or block.min() < 0:
                raise ValueError(
                    f"block {lo}:{hi} contains NaN, infinite or negative "
                    "values; the rank cache needs finite, non-negative "
                    "expression values")

            # Ties are settled by a key that depends on the cell's index in the
            # original matrix rather than on its position in this block, so the
            # ranks do not move when `blocks` is re-cut.  See sparsegs.tiebreak.
            keys = tie_break_keys(np.arange(lo, hi), n_genes, seed)
            ranked_genes = ranked_columns(block, keys, ceiling)

            if buffer.shape[0] < n:
                buffer = np.empty((n, n_genes), dtype=rank_dtype)
            buf = buffer[:n]
            buf.fill(BEYOND)
            np.put_along_axis(buf, ranked_genes, ranks[None, :], axis=1)
            clipped[lo:hi] = buf if all_genes else buf[:, keep]
            del block, keys, ranked_genes, buf

            if on_binary is not None:
                on_binary(lo, hi, binary)
            if verbose and (lo // report_every) != (hi // report_every):
                print(f"      rank cache {hi}/{n_cells}", flush=True)

        if detection is None:
            detection = det_counts / max(n_cells, 1)
            mean_expression = expr_sum / max(n_cells, 1)
        detection = np.asarray(detection, dtype=float)
        mean_expression = np.asarray(mean_expression, dtype=float)
        if depth is None and depth_counts is not None:
            depth = depth_counts

        return cls(clipped=clipped, genes=genes[keep], n_genes_total=n_genes,
                   ceiling=ceiling, detection=detection[keep],
                   mean_expression=mean_expression[keep], depth=depth)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    @property
    def n_cells(self):
        return self.clipped.shape[0]

    @property
    def n_genes(self):
        return self.clipped.shape[1]

    def positions(self, names):
        """Column positions of ``names``, raising KeyError on absent genes."""
        missing = [g for g in names if g not in self._index]
        if missing:
            raise KeyError(f"genes absent from the cache: {missing}")
        return np.array([self._index[g] for g in names], dtype=np.int64)

    def present(self, names):
        """Subset of ``names`` that the cache holds, order preserved."""
        return [g for g in names if g in self._index]

    def ranks(self, names):
        """Clipped rank matrix for ``names``, shape ``(n_cells, len(names))``."""
        return self.clipped[:, self.positions(names)]

    def subset_cells(self, mask, detection=None, mean_expression=None):
        """Restrict the cache to a subset of cells.

        A rank is computed within one cell across genes, so it does not depend on
        which other cells are in the matrix, and the clipped rank matrix can be
        sliced directly.  This is what makes a per-cell-type analysis of a large
        matrix cheap: build one cache over every cell, then slice.

        Detection rate and mean expression *do* depend on the cell set.  Pass the
        recomputed per-gene vectors to get a cache that behaves as if it had been
        built from the subset alone; omitting them carries the whole-matrix
        values over, which is wrong whenever the cell types differ in depth.

        Parameters
        ----------
        mask : array-like
            Boolean mask over cells, or integer indices.
        detection, mean_expression : array-like, optional
            Per-gene values over the subset, aligned with :attr:`genes` or with
            the matrix's full gene list (a subset is taken from the latter).

        Returns
        -------
        RankCache
        """
        mask = np.asarray(mask)
        if mask.dtype != bool:
            idx = np.zeros(self.clipped.shape[0], dtype=bool)
            idx[np.asarray(mask, dtype=np.int64)] = True
            mask = idx
        if mask.shape[0] != self.clipped.shape[0]:
            raise ValueError(f"mask has length {mask.shape[0]} but the cache holds "
                             f"{self.clipped.shape[0]} cells")

        def align(values, name):
            if values is None:
                return None
            values = np.asarray(values, dtype=float)
            if values.size == self.genes.size:
                return values
            raise ValueError(f"{name} has length {values.size}; pass it aligned "
                             f"with the cache's {self.genes.size} genes")

        det = align(detection, "detection")
        mean = align(mean_expression, "mean_expression")
        return RankCache(
            clipped=self.clipped[mask],
            genes=self.genes,
            n_genes_total=self.n_genes_total,
            ceiling=self.ceiling,
            detection=self.detection if det is None else det,
            mean_expression=self.mean_expression if mean is None else mean,
            depth=None if self.depth is None else self.depth[mask])

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """Write the cache to a compressed ``.npz`` file."""
        np.savez_compressed(
            path, clipped=self.clipped, genes=self.genes,
            n_genes_total=np.array(self.n_genes_total),
            ceiling=np.array(self.ceiling), detection=self.detection,
            mean_expression=self.mean_expression,
            depth=np.array([]) if self.depth is None else self.depth)

    @classmethod
    def load(cls, path):
        """Read a cache written by :meth:`save`."""
        with np.load(path) as z:
            depth = z["depth"] if "depth" in z.files else None
            if depth is not None and depth.size == 0:
                depth = None
            return cls(clipped=z["clipped"], genes=z["genes"],
                       n_genes_total=int(z["n_genes_total"]),
                       ceiling=int(z["ceiling"]), detection=z["detection"],
                       mean_expression=z["mean_expression"], depth=depth)

    def __repr__(self):
        return (f"RankCache(n_cells={self.n_cells}, n_genes={self.n_genes}, "
                f"ceiling={self.ceiling}, matrix_genes={self.n_genes_total}, "
                f"depth={'recorded' if self.depth is not None else 'absent'})")
