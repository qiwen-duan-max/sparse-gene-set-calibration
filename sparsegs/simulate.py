"""Ground-truth simulation for sparse gene-set scoring.

What the simulator has to get right
-----------------------------------
It is tempting to model a sparse gene set as a set of lowly expressed genes.
That misses the mechanism that actually drives gene-set scores in single-cell
data.

AUCell-family scores ask whether a gene is among the top ``max_rank = f * G``
ranks of a cell, with ``f = 0.05`` and ``G`` the number of genes in the matrix.
In a shallow cell, however, far fewer than ``max_rank`` genes are detected at
all.  At a median of 883 detected genes against ``max_rank = 1233`` -- the
figures in the CD8+ T cell compartment of GSE176078 -- every detected gene is
inside the top block, and the remaining slots down to rank 1233 are filled by
genes with a count of zero, ordered by the random tie-break.

So a gene that is *never* expressed still enters the top ``max_rank`` with
probability ``(max_rank - D) / (G - D)``, where ``D`` is the number of genes
detected in that cell.  The measured consequence in GSE176078: CIART is detected
in 0.2% of cells but counts towards the score in 1.7% of them, a factor of 7.6,
while PER1, detected in 13.2%, counts in 14.5%, a factor of 1.1.  The score is
part expression and part tie-breaking noise.

That noise term is a function of ``D``, and ``D`` is sequencing depth, and depth
tracks almost every biological axis one cares about.  This is the confounder the
framework exists to neutralise, and the simulator builds it explicitly.

Design
------
Three things are generated jointly:

1. a depth axis that depends on the latent programme ``z``;
2. a target gene set whose expression depends on depth but, under the null, not
   on ``z`` beyond that;
3. a background of co-expressed modules, so that composition-matched replacement
   sets exist in realistic numbers.

Under the null the biological effect is exactly zero, so any declaration of
association is a false positive *with respect to the claim being made* -- "this
set does something the composition does not".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp

__all__ = ["SimConfig", "SimData", "simulate", "evaluate", "effective_inclusion"]


@dataclass
class SimConfig:
    """Parameters of one simulated dataset.

    Parameters
    ----------
    n_cells, n_genes : int
        Cells and genes.  ``max_rank`` follows as ``ceil(rank_frac * n_genes)``.
    n_target : int
        Size of the gene set under study.
    median_detected : int
        Median number of genes detected per cell.  Together with ``n_genes``
        this sets the regime, and the regime is what matters: the tie-breaking
        channel is open when ``median_detected < max_rank``.
    detection : float
        Detection rate of a target gene in a cell of typical depth.  This is the
        sparsity knob.  In the CD8+ T cell compartment of GSE176078 the clock
        genes span 0.002 to 0.24.
    effect : float
        Log-scale effect of the programme on the target genes.  ``0.0`` is the
        null.
    coexpr : float
        Loading of a target-specific module factor.  Creates co-detection
        structure that expression matching alone does not reproduce.
    depth_programme_loading : float
        How strongly depth tracks the programme -- the confounder.  Setting it
        to zero is the control: the conventional test should be well calibrated
        there and badly calibrated otherwise.
    n_samples : int
        Donors.  Sample-level random effects sit on both depth and programme, so
        a cell-level test of a sample-level claim is anticonservative through
        pseudo-replication as well.
    """

    n_cells: int = 1200
    n_genes: int = 12000
    n_target: int = 8
    median_detected: int = 430
    detection: float = 0.05
    target_activity_spread: float = 1.2
    effect: float = 0.0
    coexpr: float = 0.0
    depth_programme_loading: float = 0.5
    n_samples: int = 4
    rank_frac: float = 0.05
    bg_activity_spread: float = 1.5
    bg_module_size: int = 25
    bg_module_load: float = 0.35
    seed: int = 0

    def as_dict(self):
        return asdict(self)

    @property
    def max_rank(self):
        return int(np.ceil(self.rank_frac * self.n_genes))

    @property
    def depth_ratio(self):
        """``max_rank`` divided by the median detected count.

        Above one, the cell does not fill the top block with detected genes and
        the remainder is settled by the tie-break among zeros.  GSE176078 sits at
        1233/883 = 1.40; that is the default here.
        """
        return self.max_rank / max(self.median_detected, 1)

    @property
    def tie_break_open(self):
        return self.depth_ratio > 1.0


@dataclass
class SimData:
    """One simulated dataset, with its ground truth attached."""

    X: sp.csr_matrix
    genes: np.ndarray
    target: list
    programme: np.ndarray
    depth_factor: np.ndarray
    detected_per_cell: np.ndarray
    sample: np.ndarray
    config: SimConfig = None
    achieved_detection: float = float("nan")
    max_rank: int = 0

    @property
    def retention_fraction(self):
        """Share of top-`max_rank` slots that go to genes with a zero count."""
        D = self.detected_per_cell
        n_zero = self.config.n_genes - D
        return np.mean(np.clip(self.max_rank - D, 0, None) / np.maximum(n_zero, 1))

    def __repr__(self):
        c = self.config
        return (f"SimData(n={c.n_cells}, G={c.n_genes}, k={c.n_target}, "
                f"det={self.achieved_detection:.3f}, D_med="
                f"{np.median(self.detected_per_cell):.0f}, max_rank={self.max_rank}, "
                f"effect={c.effect}, conf={c.depth_programme_loading})")


def effective_inclusion(gene_lambda, depth_factor, max_rank, n_genes):
    """Probability a gene enters the top ``max_rank``, detection aside.

    Computes ``P(detected) + P(undetected) * (max_rank - D) / (G - D)`` averaged
    over cells.  This is the quantity an expression-matched null has to
    reproduce, and the reason matching on detection rate alone comes close but
    not all the way: the second term depends on the cell's total detected count.
    """
    gene_lambda = np.asarray(gene_lambda, dtype=float)
    depth_factor = np.asarray(depth_factor, dtype=float)
    p_det = 1.0 - np.exp(-np.outer(depth_factor, gene_lambda))     # (cells, genes)
    D = p_det.sum(axis=1)                                          # expected D
    share = np.clip(max_rank - D, 0, None) / np.maximum(n_genes - D, 1)
    return (p_det + (1.0 - p_det) * share[:, None]).mean(axis=0)


def _solve_scale(activity, depth_factor, target_median, n_full, lo=1e-8, hi=1e4,
                 n_sub=2000, seed=0, iters=40):
    """Find a multiplier on the gene rates giving a target median detected count.

    ``activity`` is a sample of background gene activities normalised so that
    the median is one; ``depth_factor`` is per cell.  The detected count is
    ``sum_j 1 - exp(-c * activity_j * depth_i)``, monotone in ``c``, so a
    bisection converges quickly.

    The count is evaluated on a subsample of the activity distribution, which is
    the expensive part, and rescaled to ``n_full`` genes: summing over genes is a
    sum over draws from one distribution, so the subsample total is
    ``n_full / n_sub`` times the full total in expectation.
    """
    if activity.size > n_sub:
        activity = np.random.default_rng(seed).choice(activity, size=n_sub,
                                                      replace=False)
    target_sub = target_median * activity.size / max(n_full, 1)

    def detected(c):
        per_gene = 1.0 - np.exp(-c * np.outer(depth_factor, activity))
        return float(np.median(per_gene.sum(axis=1)))

    if detected(hi) < target_sub:
        return hi
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if detected(mid) < target_sub:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def simulate(config: SimConfig) -> SimData:
    """Generate one dataset from ``config``."""
    c = config
    rng = np.random.default_rng(c.seed)

    n_bg = c.n_genes - c.n_target
    if n_bg < 200:
        raise ValueError("need at least 200 background genes")

    genes = np.array([f"BG{i:05d}" for i in range(n_bg)]
                     + [f"TGT{i:03d}" for i in range(c.n_target)])
    target = [f"TGT{i:03d}" for i in range(c.n_target)]

    # --- sample structure -------------------------------------------
    n_samples = max(1, int(c.n_samples))
    sample = np.repeat(np.arange(n_samples),
                       int(np.ceil(c.n_cells / n_samples)))[:c.n_cells]
    sample_depth = rng.normal(0, 0.30, size=n_samples)[sample]
    sample_prog = rng.normal(0, 1.0, size=n_samples)[sample]

    # --- programme and depth axis -----------------------------------
    z = sample_prog + rng.normal(0, 0.6, size=c.n_cells)
    z = (z - z.mean()) / z.std()
    log_depth = (0.35 * rng.normal(0, 1, size=c.n_cells) + sample_depth
                 + c.depth_programme_loading * 0.6 * z)
    depth_factor = np.exp(log_depth - log_depth.mean())

    # --- background activities and their global scale ---------------
    bg_activity = np.exp(rng.normal(0, c.bg_activity_spread, size=n_bg))
    bg_activity /= np.median(bg_activity)
    scale = _solve_scale(bg_activity, depth_factor, c.median_detected, n_bg)

    # --- target gene rates ------------------------------------------
    # Choose the rate that gives the requested detection at typical depth, then
    # spread the individual genes around it: a real gene set is not uniformly
    # detectable, and the spread is what decides how much of the set carries
    # signal.  The clock repressor arm in GSE176078 spans 0.2% (CIART) to 13.2%
    # (PER1), so a spread of 1.2 on the log scale is conservative.
    med_depth = float(np.median(depth_factor))
    centre = -np.log1p(-min(max(c.detection, 1e-5), 0.95)) / med_depth
    offset = rng.normal(0, c.target_activity_spread, size=c.n_target)
    offset -= np.median(offset)
    target_lambda = centre * np.exp(offset)                 # (n_target,)

    # --- module structure -------------------------------------------
    n_modules = max(2, n_bg // max(c.bg_module_size, 1))
    module_of = rng.integers(0, n_modules, size=n_bg)
    module_factor = rng.normal(0, 1, size=(c.n_cells, n_modules))
    target_module = rng.normal(0, 1, size=c.n_cells)

    # --- rates and counts -------------------------------------------
    max_rank = int(np.ceil(c.rank_frac * c.n_genes))
    log_scale_bg = np.log(scale * bg_activity)
    log_target_lambda = np.log(target_lambda)[None, :]      # (1, n_target)
    blocks = []
    chunk = 400
    for lo in range(0, c.n_cells, chunk):
        hi = min(lo + chunk, c.n_cells)
        sl = slice(lo, hi)
        log_rate_bg = (log_scale_bg[None, :] + log_depth[sl, None]
                       + c.bg_module_load * module_factor[sl][:, module_of])
        bg = rng.poisson(np.exp(np.clip(log_rate_bg, -14, 4))).astype(np.float32)

        log_rate_t = (log_target_lambda + log_depth[sl, None]
                      + c.effect * z[sl, None]
                      + c.coexpr * target_module[sl, None])
        tg = rng.poisson(np.exp(np.clip(log_rate_t, -14, 4))).astype(np.float32)

        blocks.append(sp.csr_matrix(np.hstack([bg, tg])))
        del log_rate_bg, bg, log_rate_t, tg
    X = sp.vstack(blocks).tocsr()

    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    achieved = float((X[:, n_bg:] > 0).mean())
    return SimData(X=X, genes=genes, target=target, programme=z,
                   depth_factor=depth_factor, detected_per_cell=detected,
                   sample=sample, config=c, achieved_detection=achieved,
                   max_rank=max_rank)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def evaluate(sim, n_null=100, n_perm=1000, seed=0, n_draws=200,
             kinds=("random", "expression", "codetection")):
    """Run every test on one simulated dataset and report what each concludes.

    Seven analyses of one score: four conventional ones from
    :mod:`sparsegs.analyses`, and one per matched-null family.  All are judged at
    ``alpha = 0.05``, and under ``effect = 0`` every rejection is a false
    positive with respect to the claim being made -- "this set does something the
    composition does not".

    Returns
    -------
    dict
        Configuration, diagnostics, and per-test P values and rejection flags.
    """
    from . import RankCache, MatchedNullBuilder, aucell, spearman, null_summary
    from .stats import empirical_p
    from . import sparsity_report
    from .nulls import adaptive_detection_floor
    from .analyses import all_conventional
    from .theory import score_decomposition

    X, genes = sim.X, sim.genes
    cache = RankCache.build(X, genes, ceiling=max(sim.max_rank, 30), seed=2024,
                            chunk=500)
    # The floor is derived from the target set rather than fixed.  A fixed floor
    # above the set's own detection censors the candidate pool from below and
    # makes every replacement denser than the gene it replaces -- a bias that is
    # worst in the sparse regime this grid exists to characterise.
    builder = MatchedNullBuilder(
        cache, detection_matrix=(X > 0), detection_matrix_genes=genes,
        det_floor=adaptive_detection_floor(cache, sim.target), seed=seed)
    z = sim.programme

    score = aucell(cache, sim.target, max_rank=sim.max_rank)
    observed_rho, naive_p = spearman(score, z)
    diag = sparsity_report(cache, sim.target, rank_frac=sim.config.rank_frac)
    depth = sim.detected_per_cell

    out = dict(sim.config.as_dict())
    out.update(
        achieved_detection=sim.achieved_detection,
        median_detected=float(np.median(depth)),
        max_rank=sim.max_rank,
        depth_ratio=float(sim.max_rank / max(np.median(depth), 1)),
        retention_fraction=float(sim.retention_fraction),
        observed_rho=observed_rho,
        observed_zero_rate=diag["observed_zero_rate"],
        structural_zero_rate=diag["structural_zero_rate"],
        zero_rate_excess=diag["zero_rate_excess"],
        frac_above_floor=diag["frac_above_floor"],
        mean_detection=diag["mean_detection"],
        # The diagnostic the framework proposes: the share of the set's chance of
        # entering the score that comes from rank tie-breaking.  Carried through
        # so the grid can be asked whether it predicts the failure, rather than
        # only whether the failure happens.
        tie_break_share=float(diag.get("tie_break_share", np.nan)),
        mean_inclusion=float(diag.get("mean_inclusion", np.nan)),
        score_sd=float(np.std(score)),
        # The quantity the diagnostics are supposed to track: how much of the
        # score's variation is the depth axis rather than the gene set.
        score_depth_rho=spearman(score, depth)[0],
        programme_depth_rho=spearman(z, depth)[0])

    # The same share measured on the realised score rather than on the inclusion
    # rate: `detected` and `tie_break` are the two terms of equation (2) applied
    # to this set in this matrix, and their ratio is what the score is made of.
    try:
        split = score_decomposition(cache, sim.target, (X > 0),
                                    max_rank=sim.max_rank, depth=depth)
        total = float(split["total"].mean())
        out["score_tie_break_share"] = (
            float(split["tie_break"].mean() / total) if total > 0 else np.nan)
        out["score_detected_share"] = (
            float(split["detected"].mean() / total) if total > 0 else np.nan)
    except Exception:                                  # pragma: no cover
        out["score_tie_break_share"] = np.nan
        out["score_detected_share"] = np.nan

    # --- the four conventional analyses -----------------------------
    out.update(all_conventional(score, z, n_perm=n_perm, seed=seed + 7))

    # --- the matched nulls ------------------------------------------
    # Each family is reported with the composition it actually achieved, not
    # only with the P value it produced.  A null that failed to match is not a
    # null, and the failure is invisible in the P value alone -- so the achieved
    # detection, expression and co-detection of the drawn sets are carried
    # through and compared against the observed set's own values.
    rng = np.random.default_rng(seed + 1)
    pos = [cache._index[g] for g in sim.target if g in cache._index]
    obs_det = float(np.mean(cache.detection[pos]))
    obs_expr = float(np.mean(cache.mean_expression[pos]))
    obs_cod = builder.codetection(sim.target)
    for kind in kinds:
        null_sets = builder.sample(sim.target, n_null, kind=kind, rng=rng,
                                   n_draws=n_draws)
        null_rho = np.array([spearman(aucell(cache, s, max_rank=sim.max_rank), z)[0]
                             for s in null_sets])
        # Two-sided, matching the default the framework ships
        # (``tiers.tier2_internal``).  The tail is not a free parameter of the
        # experiment: a one-sided test asks a narrower question than the
        # conventional analyses it is being compared against, and the
        # confounder can move the association either way, so a lower-tail
        # P value would understate every matched rejection rate in the grid.
        # The other two tails are written out beside it so that the analysis
        # can show the conclusions do not hinge on this choice.
        summary = null_summary(null_rho, observed_rho, tail="two-sided")
        out[f"{kind}_p"] = summary["p_value"]
        out[f"{kind}_p_lower"] = empirical_p(null_rho, observed_rho, tail="lower")
        out[f"{kind}_p_upper"] = empirical_p(null_rho, observed_rho, tail="upper")
        out[f"{kind}_reject"] = bool(summary["p_value"] < 0.05)
        out[f"{kind}_null_median"] = summary["null_median"]
        out[f"{kind}_null_sd"] = summary["null_sd"]
        out[f"{kind}_percentile"] = summary["percentile"]
        # A replacement set can be sparse enough that its score never varies, in
        # which case the correlation is undefined.  How often that happens is a
        # property of the sparsity and is worth carrying through.
        out[f"{kind}_null_undefined"] = int(np.sum(~np.isfinite(null_rho)))

        det = np.array([np.mean(cache.detection[[cache._index[g] for g in s
                                                 if g in cache._index]])
                        for s in null_sets])
        expr = np.array([np.mean(cache.mean_expression[[cache._index[g] for g in s
                                                        if g in cache._index]])
                         for s in null_sets])
        cod = np.array([builder.codetection(s) for s in null_sets])
        out[f"{kind}_null_detection"] = float(np.nanmean(det))
        out[f"{kind}_null_expression"] = float(np.nanmean(expr))
        out[f"{kind}_null_codetection"] = float(np.nanmean(cod))
        out[f"{kind}_null_detection_ratio"] = float(
            np.nanmean(det) / obs_det if obs_det else np.nan)
        out[f"{kind}_null_expression_ratio"] = float(
            np.nanmean(expr) / obs_expr if obs_expr else np.nan)
        out[f"{kind}_null_codetection_gap"] = float(np.nanmean(cod) - obs_cod)
    out["observed_detection"] = obs_det
    out["observed_expression"] = obs_expr
    out["observed_codetection"] = obs_cod
    return out
