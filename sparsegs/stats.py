"""Association statistics and the permutation calibration of cutpoints.

Three statistics carry the framework, and each answers a different question:

``spearman``
    Does the score move with a continuous programme?  The default way a
    gene-set score is turned into a claim.
``auroc``
    Does the score separate two states the analyst defined?  Reported against
    0.5 by convention -- a convention this package exists to question.
``cutpoint_fpr``
    If the analyst picks the best split of the score and reports its P value,
    what is the actual false-positive rate?  Not the nominal one.

All three are rank statistics, so they are invariant to any monotone rescaling
of the score.  That is deliberate: it means a result cannot be changed by
switching scoring methods that differ only in scale, which makes the null, and
not the scoring function, the thing under test.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st

__all__ = ["spearman", "auroc", "empirical_p", "percentile_of", "null_summary",
           "optimum_cutpoint", "split_pvalue", "cutpoint_fpr", "dt50"]


# ----------------------------------------------------------------------
# association
# ----------------------------------------------------------------------
def spearman(x, y):
    """Spearman rank correlation, returning ``(rho, p)``.

    Ties are handled by the mid-rank convention of :func:`scipy.stats.spearmanr`.
    Non-finite pairs are dropped.  A vector with no spread has no rank
    correlation to report, and ``nan`` is returned for it without a warning:
    the framework asks this question of vectors that are constant whenever every
    cell is deeper than the score's ceiling, which is an ordinary property of
    deep data rather than a mistake in the call.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return np.nan, np.nan
    xs, ys = x[keep], y[keep]
    # Asked of scipy this returns nan *and* raises `ConstantInputWarning`.  The
    # check is made here instead, because the R implementation of the framework
    # has always returned `NA` in silence: a caller who has turned warnings into
    # errors got an exception from one implementation and a number from the
    # other on the same input, which is the kind of divergence this project
    # exists to find in other people's code.
    if xs.min() == xs.max() or ys.min() == ys.max():
        return np.nan, np.nan
    rho, p = st.spearmanr(xs, ys)
    return float(rho), float(p)


def auroc(score, positive):
    """Area under the ROC curve of ``score`` against a binary label.

    Uses the Mann-Whitney form, which is exact under ties and avoids building
    the ROC curve itself::

        AUC = (R_pos - n_pos (n_pos + 1) / 2) / (n_pos * n_neg)

    where ``R_pos`` is the sum of the mid-ranks of the positive class.
    """
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive).astype(bool)
    keep = np.isfinite(score)
    score, positive = score[keep], positive[keep]
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = st.rankdata(score)
    r_pos = ranks[positive].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------------------------------------------------
# null comparison
# ----------------------------------------------------------------------
def empirical_p(null_values, observed, tail="two-sided"):
    """Monte-Carlo P value with the ``(k + 1) / (n + 1)`` correction.

    The correction matters here: without it a statistic that beats every one of
    2,000 draws is reported as ``P = 0``, which is not a number that exists.

    ``tail`` defaults to ``"two-sided"`` because the default is the one callers
    forget to override.  A one-sided default silently answers a narrower
    question than the caller asked, and in this framework it did: the matched
    nulls were being tested in the lower tail while the conventional analyses
    they are compared against reject in either direction, which understated
    every matched rejection rate.  Pass the tail explicitly when a directional
    hypothesis is what is wanted.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    n = null_values.size
    if n == 0:
        return np.nan
    if tail == "lower":
        k = int((null_values <= observed).sum())
    elif tail == "upper":
        k = int((null_values >= observed).sum())
    elif tail == "two-sided":
        centre = np.median(null_values)
        k = int((np.abs(null_values - centre) >= abs(observed - centre)).sum())
    else:
        raise ValueError(f"unknown tail {tail!r}")
    return float((k + 1) / (n + 1))


def percentile_of(null_values, observed):
    """Fraction of the null at or below ``observed``, as a percentage."""
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0:
        return np.nan
    return float(100.0 * (null_values <= observed).mean())


def null_summary(null_values, observed, tail="two-sided"):
    """One row of everything worth reporting about a null comparison.

    The two-sided default matches [empirical_p()]; see there for why the tail
    is not a parameter to leave implicit.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    q = np.percentile(null_values, [2.5, 5, 25, 50, 75, 95, 97.5]) \
        if null_values.size else np.full(7, np.nan)
    return dict(
        observed=float(observed),
        null_n=int(null_values.size),
        null_mean=float(null_values.mean()) if null_values.size else np.nan,
        null_sd=float(null_values.std(ddof=1)) if null_values.size > 1 else np.nan,
        null_p2_5=float(q[0]), null_p5=float(q[1]), null_p25=float(q[2]),
        null_median=float(q[3]), null_p75=float(q[4]), null_p95=float(q[5]),
        null_p97_5=float(q[6]),
        percentile=percentile_of(null_values, observed),
        p_value=empirical_p(null_values, observed, tail=tail),
        # How far the observed value sits from the null centre, in null SDs.
        # Reported alongside the percentile because a heavy-tailed null can put
        # an observation at an extreme percentile without moving it far in
        # absolute terms.
        z_score=float((observed - null_values.mean()) / null_values.std(ddof=1))
        if null_values.size > 1 and null_values.std(ddof=1) > 0 else np.nan)


# ----------------------------------------------------------------------
# cutpoints
# ----------------------------------------------------------------------
def _chi2_split(values, groups):
    """Pearson chi-square statistic of a 2 x 2 table, computed directly."""
    a = float(values[groups].sum())
    b = float(groups.sum() - a)
    c = float(values[~groups].sum())
    d = float((~groups).sum() - c)
    n = a + b + c + d
    if n == 0:
        return 0.0
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom <= 0:
        return 0.0
    return float(n * (a * d - b * c) ** 2 / denom)


def optimum_cutpoint(score, outcome, min_frac=0.10, n_grid=99):
    """Cutpoint of ``score`` that best separates the two outcome classes.

    Scans quantiles of the score between ``min_frac`` and ``1 - min_frac`` and
    returns the one maximising the chi-square statistic of the resulting 2 x 2
    table.  This is what "we split patients at the optimal threshold" does.

    Returns
    -------
    float
        The cutpoint, or ``nan`` if no split is admissible.
    """
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome).astype(bool)
    keep = np.isfinite(score)
    score, outcome = score[keep], outcome[keep]
    if score.size < 20 or outcome.all() or (~outcome).all():
        return np.nan

    grid = np.quantile(score, np.linspace(min_frac, 1 - min_frac, n_grid))
    grid = np.unique(grid)
    best, best_stat = np.nan, -np.inf
    for cut in grid:
        groups = score > cut
        if groups.sum() < 5 or (~groups).sum() < 5:
            continue
        stat = _chi2_split(outcome, groups)
        if stat > best_stat:
            best, best_stat = float(cut), stat
    return best


def split_pvalue(score, outcome, cutpoint, above=True):
    """P value for a fixed split of ``score`` against a binary outcome."""
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome).astype(bool)
    keep = np.isfinite(score)
    score, outcome = score[keep], outcome[keep]
    groups = score > cutpoint if above else score <= cutpoint
    if groups.sum() < 2 or (~groups).sum() < 2:
        return np.nan
    stat = _chi2_split(outcome, groups)
    return float(st.chi2.sf(stat, 1))


def cutpoint_fpr(score, outcome, n_perm=1000, method="optimum",
                 pre_specified=None, alpha=0.05, seed=0):
    """False-positive rate of a cutpoint rule under a true null.

    The outcome is permuted ``n_perm`` times, the rule is re-applied each time,
    and the fraction of permutations declared significant at ``alpha`` is
    returned.  Under a valid rule this should be ``alpha``; the whole point is
    that an outcome-driven cutpoint is not a valid rule.

    Parameters
    ----------
    score, outcome : array-like
    n_perm : int
    method : {'optimum', 'median', 'pre_specified'}
        ``'optimum'`` re-selects the best cutpoint in every permutation.
        ``'median'`` uses the median of the score, which does not depend on the
        outcome.  ``'pre_specified'`` uses ``pre_specified``, a fixed cutpoint
        chosen without reference to the outcome at hand.
    alpha : float
    seed : int

    Returns
    -------
    dict
        ``fpr``, ``n_perm``, ``method``, plus the observed split P value and
        cutpoint for convenience.
    """
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome).astype(bool)
    keep = np.isfinite(score)
    score, outcome = score[keep], outcome[keep]
    if score.size < 20 or outcome.all() or (~outcome).all():
        return dict(fpr=np.nan, n_perm=0, method=method, observed_p=np.nan,
                    observed_cutpoint=np.nan)

    if method == "median":
        cut = float(np.median(score))
    elif method == "pre_specified":
        if pre_specified is None:
            raise ValueError("pre_specified cutpoint is required for that method")
        cut = float(pre_specified)
    elif method == "optimum":
        cut = float(optimum_cutpoint(score, outcome))
    else:
        raise ValueError(f"unknown method {method!r}")
    observed_p = split_pvalue(score, outcome, cut) if np.isfinite(cut) else np.nan

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n_perm)):
        permuted = rng.permutation(outcome)
        if method == "optimum":
            c = optimum_cutpoint(score, permuted)
        else:
            c = cut
        if not np.isfinite(c):
            continue
        p = split_pvalue(score, permuted, c)
        if np.isfinite(p) and p < alpha:
            hits += 1
    return dict(fpr=hits / float(n_perm), n_perm=int(n_perm), method=method,
                observed_p=float(observed_p) if np.isfinite(observed_p) else np.nan,
                observed_cutpoint=cut)


# ----------------------------------------------------------------------
# ordering
# ----------------------------------------------------------------------
def dt50(score, pseudotime, n_bins=10):
    """Difference in median pseudotime between the top and bottom score deciles.

    A common way to claim that one programme "changes earlier" than another: bin
    cells by the score, take the median pseudotime in each bin, and compare the
    bin where the score has fallen halfway to its floor with the corresponding
    bin for a reference programme.

    This function returns the raw quantity ``t50(score)``; the ordering claim
    needs a second programme and is computed by subtracting two calls.  Cells
    with non-finite pseudotime are dropped.
    """
    score = np.asarray(score, dtype=float)
    pseudotime = np.asarray(pseudotime, dtype=float)
    keep = np.isfinite(score) & np.isfinite(pseudotime)
    score, pseudotime = score[keep], pseudotime[keep]
    if score.size < n_bins * 5:
        return np.nan
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12
    medians = np.empty(n_bins, dtype=float)
    for b in range(n_bins):
        sel = (score >= edges[b]) & (score < edges[b + 1])
        medians[b] = np.median(pseudotime[sel]) if sel.sum() else np.nan
    if not np.isfinite(medians).any():
        return np.nan
    first, last = medians[0], medians[-1]
    half = (first + last) / 2.0
    # Walk in from the last bin; the crossing is the first bin whose median
    # reaches the halfway level.
    crossed = np.flatnonzero(medians <= half) if last < first \
        else np.flatnonzero(medians >= half)
    if crossed.size == 0:
        return float(medians[-1] - medians[0])
    b = int(crossed[0])
    if b == 0:
        return float(medians[0] - medians[0])
    lo, hi = medians[b - 1], medians[b]
    if hi == lo:
        return float(b)
    frac = (half - lo) / (hi - lo)
    return float(b - 1 + frac)
