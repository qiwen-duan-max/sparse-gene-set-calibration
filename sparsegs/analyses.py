"""The four analyses a score of this kind gets subjected to.

Each takes the *same* score vector and the *same* latent programme, and each can
declare an association.  They fail for different reasons, and the point of the
framework is to say which failure is in play.

``naive``
    Pearson/Spearman correlation of score against programme, P value from the
    cell count.  Treats cells as independent replicates.  Wrong when cells are
    nested in donors, and wrong again when the score carries a depth component
    that the programme also carries.
``permutation``
    The same correlation, with the programme permuted across cells.  This is the
    most common "we did a permutation test" in single-cell work.  It fixes the
    distributional assumption but not the structural one: permuting the
    programme severs its link to depth, so the permutation null is the
    correlation one would see if depth and programme were unrelated -- which is
    exactly the association under test.
``cutpoint_median``
    Dichotomise the programme at its median, compare scores between halves.
    Legitimate when the cut is pre-specified, and it does not fix the
    composition problem.
``cutpoint_optimal``
    Choose the cutpoint that maximises the split statistic, then report the
    P value of that split.  The selection is invisible in the reported number;
    the false-positive rate rises well above nominal.

The matched nulls are in :mod:`sparsegs.nulls`; this module supplies the three
conventional alternatives so that all of them can be compared on one dataset.
"""

from __future__ import annotations

import numpy as np

from .stats import spearman, optimum_cutpoint, split_pvalue

__all__ = ["naive_test", "permutation_test", "cutpoint_test", "all_conventional",
           "ALPHA"]

#: Nominal level every test in this module is judged against.
ALPHA = 0.05


def _two_sided_from_spearman(rho, n):
    """P value for a correlation under the independence assumption."""
    if not np.isfinite(rho) or n < 4:
        return float("nan")
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    from scipy import stats as st
    return float(2 * st.t.sf(abs(t), n - 2))


def naive_test(score, programme):
    """Cell-level correlation, P value computed as if cells were independent."""
    rho, _ = spearman(score, programme)
    return dict(rho=rho, p_value=_two_sided_from_spearman(rho, len(score)),
                reject=bool(np.isfinite(rho)
                            and _two_sided_from_spearman(rho, len(score)) < ALPHA))


def permutation_test(score, programme, n_perm=1000, seed=0):
    """Correlation against a permutation null on the programme."""
    rng = np.random.default_rng(seed)
    rho, _ = spearman(score, programme)
    null = np.array([spearman(score, rng.permutation(programme))[0]
                     for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(rho):
        return dict(rho=rho, p_value=float("nan"), reject=False)
    # Two-sided, with the (k+1)/(n+1) correction.
    p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (null.size + 1)
    return dict(rho=rho, p_value=float(p), reject=bool(p < ALPHA))


def cutpoint_test(score, programme, method="median", min_frac=0.10):
    """Dichotomise the *score* and test it against a binary programme call.

    ``method='median'`` cuts the score at its own median -- a rule that cannot
    see the outcome, so the resulting P value is calibrated.
    ``method='optimal'`` searches the score for the cut that best separates the
    two programme groups and reports that split's P value, without accounting
    for the search.  This is the analysis that inflates the false-positive rate.

    ``programme`` is dichotomised at its median to give the outcome; the
    dichotomy is not what is under test here, the cutpoint rule is.
    """
    outcome = np.asarray(programme) > float(np.median(programme))
    if outcome.sum() < 3 or (~outcome).sum() < 3:
        return dict(cut=float("nan"), p_value=float("nan"), reject=False,
                    frac_high=float(outcome.mean()))
    if method == "median":
        cut = float(np.median(score))
    else:
        cut = float(optimum_cutpoint(score, outcome, min_frac=min_frac))
    p = split_pvalue(score, outcome, cut)
    return dict(cut=cut, p_value=float(p),
                reject=bool(np.isfinite(p) and p < ALPHA),
                frac_high=float(outcome.mean()))


def all_conventional(score, programme, n_perm=1000, seed=0):
    """Every conventional analysis, on one score, as a flat dict."""
    out = {}
    n = naive_test(score, programme)
    out.update(naive_rho=n["rho"], naive_p=n["p_value"], naive_reject=n["reject"])
    p = permutation_test(score, programme, n_perm=n_perm, seed=seed)
    out.update(perm_p=p["p_value"], perm_reject=p["reject"])
    m = cutpoint_test(score, programme, method="median")
    out.update(cut_median_p=m["p_value"], cut_median_reject=m["reject"])
    o = cutpoint_test(score, programme, method="optimal")
    out.update(cut_opt_p=o["p_value"], cut_opt_reject=o["reject"])
    return out
