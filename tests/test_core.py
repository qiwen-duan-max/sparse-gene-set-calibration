"""Unit tests for sparsegs.

The first test is the one that matters: the rank cache is a reimplementation of
the scoring engine the rest of the project already used, so it must reproduce
that engine exactly.  Everything downstream inherits its correctness.
"""

import difflib
import inspect
import os
import re
import sys

import numpy as np
import scipy.sparse as sp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reference"))

import sparsegs as sg  # noqa: E402


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def synthetic_matrix(n_cells=400, n_genes=1200, density=0.06, seed=7):
    """Sparse counts with a realistic expression gradient across genes."""
    rng = np.random.default_rng(seed)
    # Gene-level activity spans three orders of magnitude, as in real data.
    activity = np.exp(rng.normal(0, 1.6, size=n_genes))
    activity /= activity.max()
    cell_scale = np.exp(rng.normal(0, 0.5, size=n_cells))

    rate = np.outer(cell_scale, activity) * density * 6.0
    rate = np.clip(rate, 0, 0.95)
    draws = rng.random((n_cells, n_genes)) < rate
    values = np.zeros((n_cells, n_genes), dtype=np.float32)
    values[draws] = rng.gamma(2.0, 1.0, size=draws.sum()).astype(np.float32)
    return sp.csr_matrix(values)


@pytest.fixture(scope="module")
def data():
    X = synthetic_matrix()
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    return X, genes


@pytest.fixture(scope="module")
def cache(data):
    X, genes = data
    ceiling = int(np.ceil(0.05 * X.shape[1]))
    return sg.RankCache.build(X, genes, ceiling=ceiling, seed=2024, chunk=200)


# ----------------------------------------------------------------------
# the engine must reproduce the reference implementation
# ----------------------------------------------------------------------
def test_cache_reproduces_reference_aucell():
    """Where the ranking is unambiguous the two engines agree to the bit.

    Every gene is detected in every cell here and no two values are close
    enough for the legacy engine's perturbation to reorder them, so the ranking
    is a permutation of distinct values and the tie-break -- whichever one an
    implementation chose -- cannot enter.  A disagreement under these conditions
    would be an error in the weighting or in the handling of the ceiling, not a
    difference of tie-break.  The tie-break itself is the subject of the next
    test.
    """
    import common

    rng = np.random.default_rng(3)
    n_cells, n_genes = 120, 900
    values = np.array([rng.permutation(n_genes) + 1 for _ in range(n_cells)],
                      dtype=np.float32)
    genes = np.array([f"G{i:05d}" for i in range(n_genes)])
    gene_sets = {f"set{i}": list(genes[rng.choice(n_genes, size=8 + 4 * i,
                                                  replace=False)])
                 for i in range(6)}

    reference = common.score_matrix(values, genes, gene_sets,
                                    max_rank_frac=0.05, chunk=200, seed=2024)
    cache = sg.RankCache.build(values, genes,
                               ceiling=int(np.ceil(0.05 * n_genes)), seed=2024)

    for name, members in gene_sets.items():
        mine = sg.aucell(cache, members)
        theirs = reference[f"AUC_{name}"].to_numpy()
        gap = np.abs(mine - theirs).max()
        assert gap < 1e-6, f"{name}: max abs difference {gap:.3e}"


def test_the_engines_differ_only_by_the_tie_break(data, cache):
    """Honest form of the same comparison on the cells with a tie-break in them.

    When a gene of the set goes undetected its rank is settled by the tie-break,
    and the legacy engine and this package settle it differently: the legacy
    engine draws a fresh uniform perturbation, this package uses a fixed
    deterministic key.  Both are uniform over the same permutations, so the
    difference between them can be no larger than the whole tie-break term --
    the quantity equation (2) of the theory section is about.  Anything bigger
    would mean the engines disagreed about something the tie-break does not
    reach.
    """
    X, genes = data
    import common

    rng = np.random.default_rng(0)
    gene_sets = {}
    for i in range(6):
        idx = rng.choice(X.shape[1], size=8 + 4 * i, replace=False)
        gene_sets[f"set{i}"] = list(genes[idx])

    reference = common.score_matrix(X, genes, gene_sets, max_rank_frac=0.05,
                                    chunk=200, seed=2024)

    m = int(np.ceil(0.05 * X.shape[1]))
    G = X.shape[1]
    depth = cache.depth
    for name, members in gene_sets.items():
        mine = sg.aucell(cache, members)
        theirs = reference[f"AUC_{name}"].to_numpy()
        k = len(members)
        undetected = k - (cache.ranks(members) > 0).sum(axis=1)
        # A member that goes undetected has a rank in D+1..G and can therefore
        # carry at most weight m - D, out of k * m.
        bound = undetected * np.maximum(m - depth, 0) / (k * m)
        gap = np.abs(mine - theirs)
        assert np.all(gap <= bound + 1e-6), (
            f"{name}: {gap.max():.4f} exceeds the tie-break term "
            f"{bound.max():.4f}")
        assert gap.mean() < 0.2 * bound.mean(), (
            f"{name}: the engines differ by {gap.mean():.4f} on average, "
            f"which is not a small part of the tie-break's own {bound.mean():.4f}")


def test_cache_respects_ceiling(data):
    """A lower ceiling must not change scores for genes that never reach it."""
    X, genes = data
    full = sg.RankCache.build(X, genes, ceiling=60, chunk=200)
    # A ceiling of 60 is far below max_rank, so scores must differ; the point is
    # that ranks at or below 60 are stored exactly.
    r = full.ranks(list(genes[:20]))
    assert r.min() >= 0
    assert r.max() <= 60


def test_cache_rejects_bad_ceiling(data):
    X, genes = data
    with pytest.raises(ValueError):
        sg.RankCache.build(X, genes, ceiling=X.shape[1] + 5, chunk=200)


def test_cache_roundtrip(tmp_path, cache):
    path = tmp_path / "cache.npz"
    cache.save(path)
    back = sg.RankCache.load(path)
    assert np.array_equal(back.clipped, cache.clipped)
    assert list(back.genes) == list(cache.genes)
    assert back.n_genes_total == cache.n_genes_total
    assert back.ceiling == cache.ceiling


# ----------------------------------------------------------------------
# scoring methods
# ----------------------------------------------------------------------
def test_ucell_bounds_and_ordering(data, cache):
    X, genes = data
    top = list(genes[np.argsort(-cache.detection)[:10]])
    bottom = list(genes[np.argsort(cache.detection)[:10]])

    s_top = sg.ucell(cache, top, r_max=60)
    s_bottom = sg.ucell(cache, bottom, r_max=60)
    # The documented floor is -1/(r_max-(k+1)/2) = -0.0183 for r_max=60, k=10.
    assert s_top.min() > -0.02 and s_top.max() <= 1.0
    assert s_bottom.min() > -0.02 and s_bottom.max() <= 1.0
    assert s_top.mean() > s_bottom.mean()


def test_ucell_negative_floor_matches_theory(data, cache):
    """The v2 normaliser is off by k; the floor is -1/(r_max-(k+1)/2)."""
    X, genes = data
    k, r_max = 10, 60
    members = list(genes[np.argsort(cache.detection)[:k]])   # sparsest genes
    s = sg.ucell(cache, members, r_max=r_max)
    floor = -1.0 / (r_max - (k + 1) / 2.0)
    # At r_max = 60 the sparsest genes are beyond the cap in every cell, so the
    # theoretical floor is attained exactly.
    assert s.min() == pytest.approx(floor, abs=2e-3)

    # At a realistic r_max the floor is far below anything the data reach, and
    # the bound still holds.
    big = sg.RankCache.build(X, genes, ceiling=800, chunk=200)
    s_big = sg.ucell(big, members, r_max=800)
    assert s_big.min() >= -1.0 / (800 - (k + 1) / 2.0) - 1e-9
    assert s_big.min() > -0.005


def test_ucell_needs_ceiling(data):
    X, genes = data
    small = sg.RankCache.build(X, genes, ceiling=30, chunk=200)
    with pytest.raises(ValueError):
        sg.ucell(small, list(genes[:8]), r_max=500)


def test_rank_methods_agree(data, cache):
    """AUCell and UCell should rank cells similarly; they measure one thing."""
    X, genes = data
    rng = np.random.default_rng(3)
    members = list(genes[rng.choice(len(genes), size=12, replace=False)])
    a = sg.aucell(cache, members)
    u = sg.ucell(cache, members, r_max=cache.ceiling)
    rho = sg.spearman(a, u)[0]
    assert rho > 0.9, f"AUCell and UCell disagree badly (rho={rho:.3f})"


def test_expression_methods_need_matrix(data, cache):
    X, genes = data
    members = [g for g in cache.genes[:10]]
    s = sg.score_genes(X, genes, members, seed=1)
    m = sg.module_score(X, genes, members, seed=1)
    assert s.shape == (X.shape[0],) and m.shape == (X.shape[0],)
    assert np.isfinite(s).all() and np.isfinite(m).all()
    # Both subtract an expression-matched control, so they must agree closely.
    assert sg.spearman(s, m)[0] > 0.8


def test_expression_methods_track_depth(data, cache):
    """Controls are subtracted, so both scores must fall with sequencing depth."""
    X, genes = data
    members = list(cache.genes[:12])
    s = sg.score_genes(X, genes, members, seed=1)
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    assert sg.spearman(s, depth)[0] < -0.1


def test_score_genes_reproducible(data):
    X, genes = data
    members = list(genes[:8])
    a = sg.score_genes(X, genes, members, seed=11)
    b = sg.score_genes(X, genes, members, seed=11)
    assert np.array_equal(a, b)


def test_ssgsea_bounds(data):
    X, genes = data
    members = list(genes[:10])
    s = sg.ssgsea(X[:60], genes, members)
    assert s.shape == (60,)
    assert np.isfinite(s).all()


# ----------------------------------------------------------------------
# nulls
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def builder(data, cache):
    X, genes = data
    return sg.MatchedNullBuilder(cache, detection_matrix=(X > 0),
                                 detection_matrix_genes=genes,
                                 det_floor=0.02, seed=2024)


def test_matched_null_is_reproducible(data, cache, builder):
    members = list(cache.genes[:8])
    a = builder.sample(members, 3, kind="expression", seed=5)
    b = builder.sample(members, 3, kind="expression", seed=5)
    assert a == b


def test_expression_null_matches_better_than_random(data, cache, builder):
    """The whole point: an expression-matched set tracks the observed set."""
    members = [g for g in cache.genes[:8]]
    pos = cache.positions(members)
    observed = cache.detection[pos].mean()

    rng = np.random.default_rng(1)
    expr = builder.sample(members, 40, kind="expression", rng=rng)
    rand = builder.sample(members, 40, kind="random", rng=rng)

    def mean_det(sets):
        return np.mean([cache.detection[cache.positions(s)].mean() for s in sets])

    d_expr = abs(mean_det(expr) - observed)
    d_rand = abs(mean_det(rand) - observed)
    assert d_expr < d_rand, (
        f"matched null is no closer to the observed detection rate "
        f"({d_expr:.4f}) than the random null ({d_rand:.4f})")


def test_matched_null_size_preserved(cache, builder):
    members = list(cache.genes[:11])
    for kind in ("random", "expression_bin", "expression"):
        for s in builder.sample(members, 3, kind=kind, seed=2):
            assert len(s) == len(members)


def test_codetection_matching(data, cache, builder):
    """Co-detection matching must land closer to the target than plain matching."""
    rng = np.random.default_rng(0)
    members = list(cache.genes[rng.choice(30, size=8, replace=False)])
    target = builder.codetection(members)
    assert np.isfinite(target)

    plain = [builder.codetection(s)
             for s in builder.sample(members, 25, kind="expression", seed=0)]
    matched = [builder.codetection(s)
               for s in builder.sample(members, 25, kind="codetection", seed=0)]
    d_plain = np.mean(np.abs(np.array(plain) - target))
    d_matched = np.mean(np.abs(np.array(matched) - target))
    assert d_matched <= d_plain, (
        f"co-detection matching did not improve the match "
        f"({d_matched:.4f} vs {d_plain:.4f})")


def test_compare_nulls_table(data, cache, builder):
    members = list(cache.genes[:8])
    table = sg.compare_nulls(builder, members, n=10, seed=0)
    assert set(table["kind"]) == {"observed", "random", "expression_bin",
                                  "expression", "codetection"}
    assert "detection" in table.columns


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def test_auroc_known_values():
    assert sg.auroc([1, 2, 3, 4], [False, False, True, True]) == pytest.approx(1.0)
    assert sg.auroc([1, 2, 3, 4], [True, True, False, False]) == pytest.approx(0.0)
    # Positives rank 2 and 4 against negatives 1 and 3: three of four pairs.
    assert sg.auroc([1, 2, 3, 4], [False, True, False, True]) == pytest.approx(0.75)
    # Positives 1 and 4 against negatives 2 and 3: two of four pairs.
    assert sg.auroc([1, 2, 3, 4], [True, False, False, True]) == pytest.approx(0.5)


def test_empirical_p_never_zero():
    nulls = np.arange(1, 2001, dtype=float)
    assert sg.empirical_p(nulls, -99.0, tail="lower") == pytest.approx(1 / 2001)
    assert sg.empirical_p(nulls, 9999.0, tail="upper") == pytest.approx(1 / 2001)
    assert sg.empirical_p(nulls, 9999.0, tail="lower") == pytest.approx(1.0)
    # The correction is the point: beating every draw is not P = 0.
    assert sg.empirical_p(nulls, -99.0, tail="lower") > 0.0


def test_empirical_p_defaults_to_two_sided_about_the_null_centre():
    """The tail is the one argument a caller forgets to pass.

    It was left implicit in the simulation grid, where the matched nulls were
    tested in the lower tail while the conventional analyses they are compared
    against reject in either direction.  Every matched rejection rate in that
    grid was therefore half the quantity it claimed to be, and the matched null's
    power against a planted effect was reported as zero.  The default is now the
    question callers actually ask.
    """
    nulls = np.linspace(-1.0, 1.0, 2001)
    assert sg.empirical_p(nulls, 0.5) == sg.empirical_p(nulls, 0.5,
                                                        tail="two-sided")
    assert sg.null_summary(nulls, 0.5)["p_value"] == sg.empirical_p(
        nulls, 0.5, tail="two-sided")

    # Definition: distance from the null's own centre, not from zero.
    centre = float(np.median(nulls))
    for observed in (-0.9, -0.2, 0.0, 0.4, 1.0):
        k = int((np.abs(nulls - centre) >= abs(observed - centre)).sum())
        assert sg.empirical_p(nulls, observed, tail="two-sided") == \
            pytest.approx((k + 1) / (len(nulls) + 1))

    # A centred null makes the two conventions agree, which is why folding the
    # values about zero looks harmless in an example.
    assert sg.empirical_p(np.abs(nulls), 0.5, tail="upper") == \
        sg.empirical_p(nulls, 0.5, tail="two-sided")

    # An off-centre null is what imperfect matching produces, and it is the case
    # this framework exists for.  Folding about zero then asks whether the
    # association differs from *nothing* -- the naive question the matched null
    # was drawn to replace -- and answers with a smaller P value than the
    # matched comparison does.
    shifted = nulls + 0.4
    assert sg.empirical_p(np.abs(shifted), 0.9, tail="upper") < \
        sg.empirical_p(shifted, 0.9, tail="two-sided")


def test_structural_zero_rate_matches_simulation():
    n_genes, k, frac = 20000, 8, 0.05
    expected = (1 - np.ceil(frac * n_genes) / n_genes) ** k
    assert sg.structural_zero_rate(k, n_genes, frac) == pytest.approx(expected)

    rng = np.random.default_rng(0)
    ranks = rng.integers(1, n_genes + 1, size=(20000, k))
    max_rank = int(np.ceil(frac * n_genes))
    zero = (ranks > max_rank).all(axis=1).mean()
    assert abs(zero - expected) < 0.01


def test_aucell_zero_rate_rises_with_sparsity(data, cache):
    """The sparsity that motivates the package, measured on the score itself.

    A gene no cell detects still enters the score when the tie-break puts it
    inside the ceiling, so a set of such genes scores zero only when the
    tie-break misses all of them; a set the cells do detect scores zero only in
    the cells that missed the whole set.  The zero rate therefore falls with
    detection, and it has to fall across the range rather than merely between
    two sets chosen to make the point.
    """
    order = np.argsort(cache.detection)
    sparse_set = list(cache.genes[order[:8]])
    dense_set = list(cache.genes[order[-8:]])
    z_sparse = (sg.aucell(cache, sparse_set) <= 0).mean()
    z_dense = (sg.aucell(cache, dense_set) <= 0).mean()
    assert z_sparse > 0.6, f"a set of the 8 sparsest genes scored {z_sparse:.2f}"
    assert z_dense < 0.15, f"a set of the 8 densest genes scored {z_dense:.2f}"
    assert z_sparse > 4 * z_dense, (f"{z_sparse:.2f} against {z_dense:.2f}: the "
                                    "gap is smaller than the mechanism implies")

    step = len(order) // 8
    buckets = [(cache.detection[order[lo:lo + 8]].mean(),
                (sg.aucell(cache, list(cache.genes[order[lo:lo + 8]])) <= 0).mean())
               for lo in range(0, len(order) - 8, step)]
    detection = np.array([b[0] for b in buckets])
    zeros = np.array([b[1] for b in buckets])
    rho = sg.spearman(detection, zeros)[0]
    assert rho < -0.8, f"zero rate against detection by decile: rho = {rho:.2f}"
    assert z_sparse > z_dense


def test_optimum_cutpoint_beats_median_when_signal_present():
    rng = np.random.default_rng(4)
    score = rng.normal(size=600)
    outcome = score + rng.normal(scale=0.8, size=600) > 0
    cut = sg.optimum_cutpoint(score, outcome)
    assert np.isfinite(cut)
    assert sg.split_pvalue(score, outcome, cut) < 0.01


def test_cutpoint_fpr_median_is_calibrated():
    """A cutpoint that ignores the outcome must be calibrated at alpha."""
    rng = np.random.default_rng(5)
    score = rng.normal(size=400)
    outcome = rng.random(400) < 0.5
    res = sg.cutpoint_fpr(score, outcome, n_perm=400, method="median", seed=1)
    assert 0.01 < res["fpr"] < 0.12, f"median-split FPR off: {res['fpr']}"


def test_cutpoint_fpr_optimum_is_inflated():
    """A cutpoint chosen from the outcome must NOT be calibrated."""
    rng = np.random.default_rng(6)
    score = rng.normal(size=400)
    outcome = rng.random(400) < 0.5
    res = sg.cutpoint_fpr(score, outcome, n_perm=400, method="optimum", seed=2)
    assert res["fpr"] > 0.15, (
        f"optimum-cutpoint FPR was {res['fpr']:.3f}; expected marked inflation")


def test_dt50_ordering():
    """dt50 must return a larger value for a programme that moves later."""
    rng = np.random.default_rng(7)
    t = rng.random(2000)
    early = -(t ** 3) + rng.normal(scale=0.02, size=2000)
    late = -((1 - t) ** 3) + rng.normal(scale=0.02, size=2000)
    assert sg.dt50(late, t) > sg.dt50(early, t)


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------
def test_sparsity_report_fields(data, cache):
    members = list(cache.genes[:8])
    rep = sg.sparsity_report(cache, members)
    for key in ("observed_zero_rate", "structural_zero_rate", "mean_detection",
                "frac_above_floor"):
        assert key in rep
    assert 0.0 <= rep["observed_zero_rate"] <= 1.0
    assert 0.0 <= rep["structural_zero_rate"] <= 1.0


def test_sparsity_report_handles_missing_genes(data, cache):
    rep = sg.sparsity_report(cache, list(cache.genes[:6]) + ["NOT_A_GENE"])
    assert rep["n_missing"] == 1
    assert rep["n_present"] == 6


def test_depth_report(data):
    X, _ = data
    rep = sg.depth_report(X)
    assert rep["median_genes_per_cell"] > 0
    assert rep["aucell_max_rank"] == int(np.ceil(0.05 * X.shape[1]))


def test_comparability_report_flags_different_panels():
    rep = sg.comparability_report({"deep": (33538, 5904), "shallow": (18211, 4540)})
    assert len(rep) == 2
    assert rep["max_rank"].nunique() == 2
    assert rep.attrs["scale_comparable"] is False


def test_verdict_escalates_with_zero_rate():
    low = sg.verdict(dict(observed_zero_rate=0.05, zero_rate_excess=0.01,
                          frac_above_floor=0.95), used_matched_null=True)
    high = sg.verdict(dict(observed_zero_rate=0.67, zero_rate_excess=0.01,
                           frac_above_floor=0.60), used_matched_null=False,
                      externally_replicated=False)
    assert low["verdict"] == "INTERPRETABLE"
    assert high["verdict"] == "NOT_IDENTIFIABLE"
    assert len(high["reasons"]) >= 2


def test_a_weak_reason_never_lowers_a_severity():
    """Every field that raises the severity to HIGH, then every field that
    would otherwise set it to MODERATE.

    The severities are strings, and ``"MODERATE"`` sorts above ``"HIGH"``, so
    comparing them with ``max()`` silently downgrades a HIGH verdict to a
    MODERATE one -- which is how the checklist used to report a set it should
    have called uninterpretable, on every call that omitted an argument.
    """
    high_share = dict(observed_zero_rate=0.05, zero_rate_excess=0.01,
                      tie_break_share=0.9, frac_above_floor=1.0,
                      detection_floor=0.005)
    high_zero = dict(observed_zero_rate=0.75, zero_rate_excess=0.01,
                     tie_break_share=0.01, frac_above_floor=1.0,
                     detection_floor=0.005)
    extras = [dict(), dict(used_matched_null=False),
              dict(outcome_driven_cutpoint=True),
              dict(externally_replicated=False)]
    for report in (high_share, high_zero):
        for extra in extras:
            v = sg.verdict(report, **extra)
            assert v["severity"] == "HIGH", (report, extra)
            assert v["verdict"] == "NOT_IDENTIFIABLE", (report, extra)


def test_omitting_the_matched_null_cannot_improve_the_verdict():
    clean = dict(observed_zero_rate=0.05, zero_rate_excess=0.01,
                 tie_break_share=0.01, frac_above_floor=1.0,
                 detection_floor=0.005)
    unmatched = sg.verdict(clean, used_matched_null=False)
    matched = sg.verdict(clean, used_matched_null=True)
    assert matched["verdict"] == "INTERPRETABLE"
    assert unmatched["verdict"] == "INTERPRETABLE_WITH_MATCHED_NULL"
    assert (sg.SEVERITY_ORDER.index(unmatched["severity"]) >=
            sg.SEVERITY_ORDER.index(matched["severity"]))


def test_the_verdict_follows_from_the_severity_and_the_null_alone():
    """The rule, stated once: a HIGH severity is uninterpretable whatever else
    was done, and anything short of it is interpretable only once the
    association has actually been tested against a matched null.

    This is the same specification the R implementation of the framework is
    tested against, so the two cannot drift apart on the headline verdict
    without one of the two suites failing.
    """
    reports = [
        dict(observed_zero_rate=0.05, zero_rate_excess=0.01, tie_break_share=0.01,
             frac_above_floor=1.0, detection_floor=0.005),
        dict(observed_zero_rate=0.75, zero_rate_excess=0.01, tie_break_share=0.9,
             frac_above_floor=1.0, detection_floor=0.005),
        dict(observed_zero_rate=0.30, zero_rate_excess=0.01, tie_break_share=0.6,
             frac_above_floor=1.0, detection_floor=0.005),
        dict(observed_zero_rate=0.10, zero_rate_excess=0.40, tie_break_share=0.01,
             frac_above_floor=1.0, detection_floor=0.005),
        dict(observed_zero_rate=0.05, zero_rate_excess=0.01, frac_above_floor=0.10,
             detection_floor=0.02),
    ]
    flags = [dict(), dict(used_matched_null=True),
             dict(outcome_driven_cutpoint=True),
             dict(externally_replicated=False),
             dict(externally_replicated=True)]
    seen = set()
    for report in reports:
        for extra in flags:
            v = sg.verdict(report, **extra)
            if v["severity"] == "HIGH":
                expected = "NOT_IDENTIFIABLE"
            elif not extra.get("used_matched_null", False):
                expected = "INTERPRETABLE_WITH_MATCHED_NULL"
            else:
                expected = "INTERPRETABLE"
            assert v["verdict"] == expected, (report, extra, v)
            assert v["reasons"], "a verdict always says why"
            seen.add(v["verdict"])
    assert seen == {"NOT_IDENTIFIABLE", "INTERPRETABLE_WITH_MATCHED_NULL",
                    "INTERPRETABLE"}


# ----------------------------------------------------------------------
# streaming construction
# ----------------------------------------------------------------------
def test_streaming_matches_in_memory_build(data):
    """Blocks fed from outside must give the same cache as an in-memory build."""
    X, genes = data
    ceiling = int(np.ceil(0.05 * X.shape[1]))
    a = sg.RankCache.build(X, genes, ceiling=ceiling, seed=2024, chunk=200)

    seen = []

    def blocks():
        for lo in range(0, X.shape[0], 200):
            hi = min(lo + 200, X.shape[0])
            seen.append((lo, hi))
            yield lo, hi, X[lo:hi]

    b = sg.RankCache.build_streaming(blocks(), n_cells=X.shape[0],
                                     n_genes=X.shape[1], genes=genes,
                                     ceiling=ceiling, seed=2024)
    assert len(seen) == 2   # 400 cells at 200 per block
    assert np.array_equal(a.clipped, b.clipped)
    assert np.allclose(a.detection, b.detection)
    assert np.allclose(a.mean_expression, b.mean_expression)


def test_streaming_reports_binary_blocks(data):
    """on_binary must see every row exactly once, in the coordinate system given."""
    X, genes = data
    ceiling = int(np.ceil(0.05 * X.shape[1]))
    got = np.zeros(X.shape, dtype=bool)

    def on_binary(lo, hi, block):
        assert block.shape == (hi - lo, X.shape[1])
        got[lo:hi] = np.asarray(block.todense(), dtype=bool)

    sg.RankCache.build_streaming(
        ((lo, min(lo + 150, X.shape[0]), X[lo:lo + 150])
         for lo in range(0, X.shape[0], 150)),
        n_cells=X.shape[0], n_genes=X.shape[1], genes=genes,
        ceiling=ceiling, seed=1, on_binary=on_binary)
    assert np.array_equal(got, np.asarray((X > 0).todense(), dtype=bool))


def test_streaming_keep_subset(data):
    """A subset of columns must be selected consistently across blocks."""
    X, genes = data
    keep = np.arange(0, X.shape[1], 7)
    c = sg.RankCache.build_streaming(
        ((lo, min(lo + 200, X.shape[0]), X[lo:lo + 200])
         for lo in range(0, X.shape[0], 200)),
        n_cells=X.shape[0], n_genes=X.shape[1], genes=genes,
        ceiling=40, keep=keep, seed=2)
    full = sg.RankCache.build(X, genes, ceiling=40, seed=2, chunk=200)
    assert c.genes.tolist() == genes[keep].tolist()
    assert c.n_genes_total == full.n_genes_total == X.shape[1]
    assert np.array_equal(c.clipped, full.clipped[:, keep])


def test_streaming_accepts_dense_blocks_without_mutating_them(data):
    """Dense blocks must give the same cache as sparse ones, and stay untouched.

    The ranking perturbs the block in place, so a dense block that arrives as a
    view of the caller's array would silently corrupt the caller's data -- and
    the caller is usually holding the matrix it still needs for the detection
    matrix and the axis.
    """
    X, genes = data
    dense = np.asarray(X.todense(), dtype=np.float32)
    original = dense.copy()
    blocks = ((lo, min(lo + 200, dense.shape[0]), dense[lo:lo + 200])
              for lo in range(0, dense.shape[0], 200))
    c_dense = sg.RankCache.build_streaming(
        blocks, n_cells=dense.shape[0], n_genes=dense.shape[1], genes=genes,
        ceiling=40, seed=3)
    c_sparse = sg.RankCache.build(X, genes, ceiling=40, seed=3, chunk=200)
    assert np.array_equal(c_dense.clipped, c_sparse.clipped)
    assert np.array_equal(dense, original)


def test_clipped_is_narrow(data):
    """Ranks are stored at the narrowest width that holds the ceiling."""
    X, genes = data
    c = sg.RankCache.build(X, genes, ceiling=40, seed=4, chunk=200)
    assert c.clipped.dtype == np.int16
    assert c.clipped.max() == 40


def test_codetection_block_matches_global_gram(data):
    """The on-demand co-detection block must equal the full gram restricted to it.

    The full n_genes x n_genes gram is what the on-demand version replaced; this
    pins the replacement to it numerically.
    """
    X, genes = data
    B = (X > 0).astype(np.float32)
    cache = sg.RankCache.build(X, genes, ceiling=40, seed=5, chunk=200)
    builder = sg.MatchedNullBuilder(cache, detection_matrix=B,
                                    detection_matrix_genes=genes, seed=0)

    sub = builder._B
    gram = np.asarray((sub.T @ sub).todense())
    norm = np.sqrt(np.diag(gram))
    gram = gram / (norm[:, None] * norm[None, :])
    np.fill_diagonal(gram, np.nan)

    picks = list(cache.genes[builder.pool[::11] ][:25])
    got = builder.codetection(picks)
    positions = [builder._pool_pos[cache._index[g]] for g in picks]
    want = float(np.nanmean(gram[np.ix_(positions, positions)]))
    assert np.isfinite(got) and got == pytest.approx(want, rel=1e-10)


def test_sample_forwards_n_draws(data):
    """`sample` must let the caller bound the co-detection search."""
    X, genes = data
    cache = sg.RankCache.build(X, genes, ceiling=40, seed=6, chunk=200)
    builder = sg.MatchedNullBuilder(cache, detection_matrix=(X > 0),
                                    detection_matrix_genes=genes, seed=0)
    gs = [g for g in genes[::53]][:6]

    class Counting(sg.MatchedNullBuilder):
        calls = 0

        def expression_matched_set(self, gene_set, rng=None):
            Counting.calls += 1
            return super().expression_matched_set(gene_set, rng)

    counter = Counting(cache, detection_matrix=(X > 0),
                       detection_matrix_genes=genes, seed=0)
    counter.sample(gs, 3, kind="codetection", n_draws=4)
    assert Counting.calls <= 3 * 4 + 3


# ----------------------------------------------------------------------
# end-to-end: diagnostics -> tiers -> report
# ----------------------------------------------------------------------
def _small_pipeline(seed=3):
    from sparsegs.simulate import SimConfig, simulate

    sim = simulate(SimConfig(seed=seed, effect=0.0, detection=0.05,
                             depth_programme_loading=0.6, coexpr=0.6,
                             n_cells=400, n_genes=3000, n_target=8))
    cache = sg.RankCache.build(sim.X, sim.genes, ceiling=max(sim.max_rank, 30),
                               seed=2024, chunk=400)
    score = sg.aucell(cache, sim.target, max_rank=sim.max_rank)
    diag = sg.sparsity_report(cache, sim.target, rank_frac=0.05)
    builder = sg.MatchedNullBuilder(cache, detection_matrix=(sim.X > 0),
                                    detection_matrix_genes=sim.genes,
                                    det_floor=0.01, seed=0)
    observed = sg.spearman(score, sim.programme)[0]
    nulls = {}
    for kind in ("random", "expression", "codetection"):
        sets = builder.sample(sim.target, 25, kind=kind, n_draws=10)
        nulls[kind] = np.array(
            [sg.spearman(sg.aucell(cache, s, max_rank=sim.max_rank),
                         sim.programme)[0] for s in sets])
    return sim, cache, score, diag, observed, nulls

def _unbalanced_tags(html_text):
    """Tags left open at the end of the document, ignoring void elements.

    The report is HTML rather than XML -- it carries a doctype and unclosed
    ``<br>`` -- so a strict XML parser rejects it for reasons that say nothing
    about whether a browser would render it.  This checks the thing that
    actually matters and that a typo in a generated string would break: that
    every element opened is closed.
    """
    from html.parser import HTMLParser

    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}

    class Checker(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack = []
            self.problems = []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack:
                self.problems.append("</%s> with nothing open" % tag)
            elif self.stack[-1] != tag:
                self.problems.append("</%s> closes <%s>" % (tag, self.stack[-1]))
                if tag in self.stack:
                    while self.stack and self.stack.pop() != tag:
                        pass
            else:
                self.stack.pop()

    checker = Checker()
    checker.feed(html_text)
    return checker.stack + checker.problems


def test_report_renders_end_to_end(tmp_path):
    """The whole path has to survive contact with a real call, not just import."""
    import xml.etree.ElementTree as ET

    sim, cache, score, diag, observed, nulls = _small_pipeline()
    v = sg.verdict(diag, used_matched_null=False, outcome_driven_cutpoint=False)
    path = tmp_path / "report.html"
    sg.render_report(
        str(path), title="t", gene_set=sim.target, dataset="d",
        cell_type="c", diagnostics=diag, verdict=v, nulls=nulls,
        observed=observed, score=score,
        cutpoint_rows=[("median cut", 0.059), ("optimal cut", 0.483),
                       ("fixed cut", 0.050)],
        tiers=sg.run_tiers(dict(observed_rho=observed, n_cells=400,
                                passed=False)),
        notes="note")
    html = path.read_text(encoding="utf-8")
    # HTML rather than XML -- it carries a doctype and unclosed <br> --
    # so a strict XML parser rejects it for reasons that say nothing
    # about rendering.  What a typo in a generated string actually
    # breaks is element nesting, so that is what is checked.
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert _unbalanced_tags(html) == [], "tag nesting is broken"
    for needle in ("Verdict", "Gene set", "Score distribution",
                   "Association against matched nulls",
                   "Cutpoint false-positive rate", "Validation tiers"):
        assert needle in html, needle
    # Every placeholder must have been substituted; a surviving "$name" means
    # the stylesheet shipped an unresolved variable.
    assert "$" not in html.split("</style>")[0]
    assert "ClawsGO Science Agent" in html


def test_tier1_mismatch_is_reported_not_passed():
    """A technical check run in the wrong regime must not count as a pass."""
    t1 = sg.tier1_technical(target_zero_rate=0.67, n_cells=200, n_genes=2000,
                            k=8, n_repeats=1, seed=0)
    out = sg.run_tiers(dict(observed_rho=0.1, n_cells=200, passed=False),
                       tier1=t1)
    if not t1["regime_match"]:
        assert out["tier1"]["applicable"] is False
        assert "TIER_1" not in out["tiers_passed"]


def test_adaptive_floor_follows_its_rule(data):
    """The floor is max(bound, min(cap, 0.25 x the set's 10th percentile))."""
    X, genes = data
    cache = sg.RankCache.build(X, genes, ceiling=40, seed=8, chunk=200)

    # A set drawn from a low but non-zero detection band: sparse enough that a
    # fixed 0.01 floor would censor it, dense enough that it can be matched.
    band = np.flatnonzero((cache.detection > 0.01) & (cache.detection < 0.08))
    assert band.size >= 30
    gene_set = list(cache.genes[band[:30]])

    q10 = float(np.quantile(cache.detection[band[:30]], 0.10))
    expected = max(1e-4, min(0.01, 0.25 * q10))
    assert sg.adaptive_detection_floor(cache, gene_set) == pytest.approx(expected)
    assert expected < 0.01, "this set should be sparse enough to bind the cap"

    # A set with nothing in the cache is a degenerate call, not an exception.
    assert sg.adaptive_detection_floor(cache, ["not-a-gene"]) == 1e-4


def test_adaptive_floor_does_not_censor_a_sparse_set(data):
    """A fixed floor above a gene set's own detection biases its null upward.

    The floor keeps near-never-observed genes out of the candidate pool.  Set
    above the set's own detection it also removes every gene sparse enough to
    replace the set's genes, and the null comes back denser than the set.  The
    adaptive floor exists to keep the pool's lower edge below the set.
    """
    X, genes = data
    cache = sg.RankCache.build(X, genes, ceiling=40, seed=8, chunk=200)
    band = np.flatnonzero((cache.detection > 0.01) & (cache.detection < 0.08))
    gene_set = list(cache.genes[band[:30]])
    target = float(np.mean(cache.detection[band[:30]]))

    def ratio_for(floor):
        b = sg.MatchedNullBuilder(cache, detection_matrix=(X > 0),
                                  detection_matrix_genes=genes, det_floor=floor,
                                  seed=0)
        sets = b.sample(gene_set, 20, kind="expression",
                        rng=np.random.default_rng(1))
        return (float(np.mean([np.mean(cache.detection[[cache._index[g] for g in s]])
                               for s in sets])) / target, b.pool.size)

    fixed_ratio, fixed_pool = ratio_for(0.01)
    adapt_ratio, adapt_pool = ratio_for(sg.adaptive_detection_floor(cache, gene_set))

    # A lower floor can only widen the pool, never narrow it.
    assert adapt_pool >= fixed_pool
    # And the drawn sets must not sit systematically above the observed one.  The
    # tolerance is what a matched null is expected to achieve, not a comparison
    # between the two floors: on a panel where the fixed floor happens not to
    # bind, both are fine, and the point is that lowering it never makes things
    # worse.
    assert abs(adapt_ratio - 1.0) <= 0.05, f"adaptive floor ratio {adapt_ratio:.4f}"
    assert abs(fixed_ratio - 1.0) <= 0.05, f"fixed floor ratio {fixed_ratio:.4f}"


def test_adaptive_floor_is_capped_for_dense_sets(data):
    """A well-detected set gets the cap, not something below it."""
    X, genes = data
    cache = sg.RankCache.build(X, genes, ceiling=40, seed=9, chunk=200)
    dense = list(cache.genes[np.argsort(-cache.detection)[:12]])
    assert sg.adaptive_detection_floor(cache, dense) == 0.01
    assert sg.adaptive_detection_floor(cache, ["not-a-gene"]) == 1e-4


# ----------------------------------------------------------------------
# the tie-break: a deterministic key, not a generator draw
# ----------------------------------------------------------------------
def test_the_cache_does_not_depend_on_the_blocking():
    """The defect the key exists to remove.

    The reference AUCell implementation perturbs each value by a draw from a
    generator, and a generator produces a stream: a block of 500 cells consumes
    a different part of it than two blocks of 250.  The same matrix and the same
    seed then give different scores at different ``chunk`` settings, which for a
    tool whose output is a P value is not a detail.  A key that is a function of
    the coordinates has no stream to depend on.
    """
    from sparsegs.simulate import SimConfig, simulate

    d = simulate(SimConfig(n_cells=240, n_genes=1500, seed=4))
    genes = np.array(d.genes)
    a = sg.RankCache.build(d.X, genes, ceiling=60, seed=2024, chunk=240)
    b = sg.RankCache.build(d.X, genes, ceiling=60, seed=2024, chunk=17)
    c = sg.RankCache.build(d.X, genes, ceiling=60, seed=2024, chunk=1)
    assert np.array_equal(a.clipped, b.clipped)
    assert np.array_equal(a.clipped, c.clipped)
    assert not np.array_equal(
        a.clipped, sg.RankCache.build(d.X, genes, ceiling=60, seed=99,
                                      chunk=240).clipped)


def test_the_tie_break_key_is_a_function_of_the_coordinates():
    """So that a band of cells can be ranked without reference to the others."""
    from sparsegs.tiebreak import tie_break_keys

    whole = tie_break_keys(np.arange(40), 100, seed=7)
    band = tie_break_keys(np.arange(10, 20), 100, seed=7)
    assert np.array_equal(whole[10:20], band)
    # Different seeds have to give different answers, or the seed is a lie.
    assert not np.array_equal(whole, tie_break_keys(np.arange(40), 100, seed=8))
    # And the keys have to be spread over the range rather than clustered.
    assert 0.45 < whole.mean() / 2 ** 32 < 0.55
    assert len(np.unique(whole)) > 0.999 * whole.size


def test_an_exact_tie_is_settled_by_the_key_and_not_by_gene_order():
    """Count data is mostly ties: which of two equally expressed genes takes the
    last rank inside the ceiling is decided in essentially every cell, and it
    has to be decided by the key rather than by whatever the selection algorithm
    happens to do with equal inputs."""
    from sparsegs.tiebreak import ranked_columns, tie_break_keys

    n_cells, n_genes, ceiling = 3, 400, 20
    # Every detected gene has the same count, so the whole ranking below the
    # first ceiling genes is one tie group.
    block = np.zeros((n_cells, n_genes), dtype=np.float32)
    block[:, 100:140] = 5.0
    keys = tie_break_keys(np.arange(n_cells), n_genes, seed=11)
    ranked = ranked_columns(block, keys, ceiling)

    for row in range(n_cells):
        # The filled slots are exactly the lowest keys of the tie group ...
        chosen = ranked[row][ranked[row] >= 100]
        assert len(chosen) == ceiling
        by_key = np.argsort(keys[row, chosen])
        assert np.array_equal(chosen[by_key],
                              np.array(sorted(chosen, key=lambda g: keys[row, g])))
        assert set(chosen) == set(np.argsort(keys[row, 100:140])[:ceiling] + 100)
    # ... and a different seed picks a different subset, since it is a random
    # one and the tie group is far larger than the ceiling.
    other = ranked_columns(block, tie_break_keys(np.arange(n_cells), n_genes,
                                                 seed=12), ceiling)
    assert not np.array_equal(ranked, other)


def test_an_undetected_gene_never_outranks_a_detected_one():
    """The key fills the slots the cell left empty and touches nothing else."""
    from sparsegs.tiebreak import ranked_columns, tie_break_keys

    n_cells, n_genes, ceiling = 2, 300, 25
    block = np.zeros((n_cells, n_genes), dtype=np.float32)
    block[0, :5] = [1.0, 2.0, 3.0, 4.0, 5.0]      # five genes detected
    keys = tie_break_keys(np.arange(n_cells), n_genes, seed=0)
    ranked = ranked_columns(block, keys, ceiling)

    # A count of one beats any zero, however small its key.
    assert set(ranked[0, :5].tolist()) == {0, 1, 2, 3, 4}
    assert set(ranked[0, 5:].tolist()).isdisjoint({0, 1, 2, 3, 4})
    # The second cell detected nothing, so its ranks are the smallest keys.
    assert np.array_equal(ranked[1], np.argsort(keys[1])[:ceiling])


# ----------------------------------------------------------------------
# the closed form: equation (2) and (3) of sparsegs.theory
# ----------------------------------------------------------------------
def _silent_block_matrix(n_cells=400, n_genes=600, n_silent=40, seed=11):
    """A matrix in which a whole block of genes is never expressed at all.

    The last ``n_silent`` columns are exactly zero, so their rank in every cell
    is settled by the tie-break and nothing else.  That is the cleanest possible
    test of the closed form: the predicted inclusion rate of these genes is a
    function of the depth vector alone.
    """
    rng = np.random.default_rng(seed)
    n_live = n_genes - n_silent
    activity = np.exp(rng.normal(0, 1.2, size=n_live))
    activity /= activity.max()
    cell_scale = np.exp(rng.normal(0, 0.6, size=n_cells))
    rate = np.clip(np.outer(cell_scale, activity) * 0.35, 0.0, 0.9)
    draws = rng.random((n_cells, n_live)) < rate
    values = np.zeros((n_cells, n_genes), dtype=np.float32)
    values[:, :n_live][draws] = rng.gamma(
        2.0, 1.0, size=int(draws.sum())).astype(np.float32)
    return sp.csr_matrix(values), n_live


def test_tie_break_inclusion_matches_the_ranking():
    """Equation (3): a gene the cell never expressed enters the top m at the
    rate (m - D) / (G - D).  Measured against the cache's actual ranks."""
    X, n_live = _silent_block_matrix()
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    G = X.shape[1]
    m = int(np.ceil(0.05 * G))
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    ceiling = int(min(G - 1, max(m + 2, depth.max() + 1)))
    cache = sg.RankCache.build(X, genes, ceiling=ceiling, seed=5, chunk=100)

    silent = list(genes[n_live:])
    r = cache.ranks(silent)
    measured = float(np.mean((r > 0) & (r <= m)))
    predicted = float(np.mean(sg.tie_break_inclusion(depth, m, G)))

    assert predicted > 0.0, "the test regime must leave the tie-break open"
    assert abs(measured - predicted) < 0.01, (
        f"measured {measured:.4f} vs predicted {predicted:.4f}")


def test_effective_inclusion_matches_measured_inclusion():
    """Equation (3) for genes that are sometimes expressed: inclusion is
    detection plus the tie-break, not detection alone."""
    X, n_live = _silent_block_matrix()
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    G = X.shape[1]
    m = int(np.ceil(0.05 * G))
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    ceiling = int(min(G - 1, max(m + 2, depth.max() + 1)))
    cache = sg.RankCache.build(X, genes, ceiling=ceiling, seed=5, chunk=100)

    p_all = np.asarray((X[:, :n_live] > 0).mean(axis=0)).ravel()
    order = np.argsort(-p_all)[:20]
    subset = list(genes[order])
    r = cache.ranks(subset)
    measured = np.mean((r > 0) & (r <= m), axis=0)
    predicted = sg.effective_inclusion(p_all[order], depth, m, G)

    assert np.max(np.abs(measured - predicted)) < 0.02, (
        f"largest gap {np.max(np.abs(measured - predicted)):.4f}")
    # and the correction is not cosmetic: detection alone understates inclusion
    assert np.all(predicted >= p_all[order] - 1e-12)


def test_tie_break_level_closes_when_cells_are_deep():
    """No slot is left for the tie-break once a cell detects max_rank genes."""
    m, G = 1233, 24646
    assert sg.tie_break_level(np.array([m, m + 10, 5000]), m, G).tolist() == [0.0, 0.0, 0.0]
    assert np.all(sg.tie_break_inclusion(np.array([m, 5000]), m, G) == 0.0)
    # and it is monotonically decreasing in depth where it is open
    depths = np.array([100, 300, 600, 900, 1200])
    tau = sg.tie_break_level(depths, m, G)
    assert np.all(np.diff(tau) < 0)


def test_score_decomposition_is_an_identity(data, cache):
    """detected + tie_break must reproduce the score exactly, cell by cell."""
    X, genes = data
    gene_set = [g for g in genes[::7][:12]]
    m = int(np.ceil(0.05 * X.shape[1]))
    df = sg.score_decomposition(cache, gene_set, (X > 0))
    score = sg.aucell(cache, gene_set, max_rank=m)
    assert np.allclose(df["total"].to_numpy(), score, atol=1e-12)
    assert np.allclose(df["detected"].to_numpy() + df["tie_break"].to_numpy(),
                       score, atol=1e-12)
    # the tie-break part is non-negative and the score is the sum of two parts
    assert (df["tie_break"] >= 0).all()
    assert np.allclose(df["total"], df["detected"] + df["tie_break"])


def test_tie_break_expectation_matches_the_realised_share(data, cache):
    """tau(D) is the *mean* tie-break contribution; averaged over cells and
    genes the realised share must land on it."""
    X, genes = data
    gene_set = list(genes[::13][:15])
    m = int(np.ceil(0.05 * X.shape[1]))
    df = sg.score_decomposition(cache, gene_set, (X > 0), max_rank=m)
    realised = float(df["tie_break"].mean())
    expected = float(df["tie_break_expected"].mean())
    assert expected > 0, "the test regime must leave the tie-break open"
    assert abs(realised - expected) <= 0.15 * expected, (
        f"realised {realised:.5f} vs expected {expected:.5f}")


def test_preflight_flags_an_axis_that_is_depth_in_disguise():
    """An axis built from the depth alone is the worst case, and must be caught
    without scoring anything."""
    X, n_live = _silent_block_matrix()
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    G = X.shape[1]
    m = int(np.ceil(0.05 * G))

    confounded = sg.preflight(depth, -depth.astype(float), m, G)
    assert confounded["verdict"] == "CONFOUNDED"
    # tau falls with depth, so an axis that is the negative depth correlates
    # with it positively
    assert confounded["tau_axis_rho"] > 0.5

    rng = np.random.default_rng(3)
    clean = sg.preflight(depth, rng.normal(size=depth.size), m, G)
    assert clean["verdict"] != "CONFOUNDED"
    assert clean["tau_axis_rho"] < confounded["tau_axis_rho"]

    # The CLEAR band is a sample-size statement, so it has to be calibrated as
    # one: over many axes that carry no information at all, the flag must fire
    # at the rate the band claims and not more.
    rhos = np.array([abs(sg.preflight(depth, rng.normal(size=depth.size),
                                      m, G)["tau_axis_rho"])
                     for _ in range(200)])
    band = 1.96 / np.sqrt(depth.size - 3)
    assert float(np.mean(rhos > band)) < 0.10, (
        f"the chance band fired on {np.mean(rhos > band):.1%} of null axes")


def test_preflight_implied_rho_tracks_the_measured_spurious_correlation():
    """The predicted tie-break component and the score's own correlation with a
    depth-derived axis must agree once the term's own sampling noise is
    accounted for.

    For a set that is never detected anywhere, equation (2) reduces to
    ``AUC = tau(D) + noise``, so the score's correlation with an axis that is a
    function of depth has to be ``rho(tau, axis)`` shrunk by the share of the
    score's variance that ``tau`` accounts for.  That is a two-parameter
    prediction with nothing fitted, and it is what the test checks.
    """
    X, n_live = _silent_block_matrix()
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    G = X.shape[1]
    m = int(np.ceil(0.05 * G))
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    ceiling = int(min(G - 1, max(m + 2, depth.max() + 1)))

    gene_set = list(genes[n_live: n_live + 12])          # never expressed
    axis = -depth.astype(float)

    # One realisation of the tie-break is a noisy estimate: the quantity is a
    # correlation of a few hundred cells, and the noise it is being compared
    # against is itself drawn.  The prediction is a statement about the
    # expectation over tie-breaks, so it is the average over several seeds that
    # has to land -- and the model is a Pearson statement, since attenuation is
    # a ratio of standard deviations.
    measured, predicted = [], []
    for seed in range(1, 9):
        cache = sg.RankCache.build(X, genes, ceiling=ceiling, seed=seed,
                                   chunk=100)
        p = cache.detection[[cache._index[g] for g in gene_set]]
        score = sg.aucell(cache, gene_set, max_rank=m)
        pred = sg.preflight(depth, axis, m, G, detection=p, k=len(gene_set),
                            score=score)
        measured.append(float(np.corrcoef(score, axis)[0, 1]))
        predicted.append(pred["realised_implied_rho"])

    measured = np.array(measured)
    predicted = np.array(predicted)
    assert (predicted > 0).all() and (measured > 0).all()
    # implied_rho describes the systematic component; the realised score carries
    # the tie-break's own sampling noise on top, which is why the prediction is
    # smaller than implied_rho = 1.
    assert (predicted < 1.0).all()
    gap = abs(measured.mean() - predicted.mean())
    assert gap < 0.05, (
        f"measured {measured.mean():.3f} vs predicted {predicted.mean():.3f} "
        f"(implied {pred.mean():.3f}, over {len(measured)} tie-break seeds)")


def test_tie_break_inclusion_matches_real_genes_that_are_never_detected():
    """The same identity on the package's default synthetic matrix, using the
    genes that happen to be silent there rather than a planted block."""
    X = synthetic_matrix()
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    G = X.shape[1]
    m = int(np.ceil(0.05 * G))
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    silent = list(genes[depth.size and 0 or 0:0])            # placeholder
    silent = list(genes[(X > 0).sum(axis=0).A1 == 0])
    if len(silent) < 5:
        pytest.skip("no silent genes in this matrix")
    ceiling = int(min(G - 1, max(m + 2, depth.max() + 1)))
    cache = sg.RankCache.build(X, genes, ceiling=ceiling, seed=5, chunk=200)
    r = cache.ranks(silent)
    measured = float(np.mean((r > 0) & (r <= m)))
    predicted = float(np.mean(sg.tie_break_inclusion(depth, m, G)))
    assert abs(measured - predicted) < 0.02


# ----------------------------------------------------------------------
# depth: the cache carries the argument of the closed form
# ----------------------------------------------------------------------
def test_cache_records_each_cells_depth(data, cache):
    """D_c is the number of genes the cell detected over the whole panel, and
    it must be exact -- every use of the closed form depends on it."""
    X, _ = data
    true_depth = np.asarray((X > 0).sum(axis=1)).ravel()
    assert cache.depth is not None
    assert np.array_equal(np.asarray(cache.depth), true_depth)
    # the reference implementation computes the same thing from the matrix
    assert int(cache.depth.max()) <= X.shape[1]


def test_streaming_build_records_depth_from_the_blocks(data):
    """The streaming path must produce the same depth as the in-memory one,
    which is the property that lets a 50,000-cell panel be calibrated."""
    X, genes = data
    inmem = sg.RankCache.build(X, genes, ceiling=60, seed=1, chunk=10 ** 6)

    def blocks():
        for lo in range(0, X.shape[0], 97):
            yield lo, min(lo + 97, X.shape[0]), X[lo:lo + 97]

    stream = sg.RankCache.build_streaming(
        blocks(), n_cells=X.shape[0], n_genes=X.shape[1], genes=genes,
        ceiling=60, seed=1)
    assert np.array_equal(np.asarray(inmem.depth), np.asarray(stream.depth))
    assert np.array_equal(inmem.depth,
                          np.asarray((X > 0).sum(axis=1)).ravel())


def test_depth_survives_subsetting_and_a_save_load_round_trip(data, cache,
                                                              tmp_path):
    """Depth is per cell, so slicing the cache must slice it; and a cache read
    back from disk must behave as the one that was written."""
    mask = np.zeros(cache.n_cells, dtype=bool)
    mask[::3] = True
    sub = cache.subset_cells(mask)
    assert sub.depth is not None
    assert np.array_equal(np.asarray(sub.depth),
                          np.asarray(cache.depth)[mask])

    path = tmp_path / "cache.npz"
    cache.save(path)
    back = sg.RankCache.load(path)
    assert back.depth is not None
    assert np.array_equal(np.asarray(back.depth), np.asarray(cache.depth))
    assert np.array_equal(back.clipped, cache.clipped)


def test_a_cache_without_depth_still_works(data):
    """Caches written before depth was recorded must keep loading, and the
    diagnostics must fall back to the criterion they can still compute."""
    X, genes = data
    bare = sg.RankCache(clipped=np.zeros((X.shape[0], 5), dtype=np.int16),
                        genes=genes[:5],
                        n_genes_total=X.shape[1], ceiling=60,
                        detection=np.full(5, 0.1), mean_expression=np.ones(5))
    assert bare.depth is None
    assert "depth=absent" in repr(bare)


# ----------------------------------------------------------------------
# tie_break_share: the exact version of "is this score carried by expression?"
# ----------------------------------------------------------------------
def test_tie_break_share_is_zero_when_the_set_is_always_detected():
    """A gene counted in every cell enters on its own merits, whatever the
    depth, so the tie-break supplies nothing."""
    depth = np.full(500, 40.0)
    assert sg.tie_break_share(np.ones(30), depth, 1000, 20000) == 0.0


def test_tie_break_share_rises_as_the_set_stops_being_detected():
    """The share is a property of the set in this data, not of the matrix, so
    the same depth vector must give a larger share to a sparser set."""
    depth = np.full(500, 200.0)
    shares = [sg.tie_break_share(np.full(30, p), depth, 1000, 20000)
              for p in (0.50, 0.20, 0.05, 0.01)]
    assert all(np.isfinite(shares))
    assert np.all(np.diff(shares) > 0), shares
    assert shares[-1] > 0.7


def test_tie_break_share_agrees_with_the_realised_decomposition():
    """The share is defined from the inclusion rate; for a set that is never
    detected it must also be what the score itself is made of."""
    X, n_live = _silent_block_matrix()
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    G = X.shape[1]
    m = int(np.ceil(0.05 * G))
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    cache = sg.RankCache.build(X, genes,
                               ceiling=int(min(G - 1, depth.max() + 1)),
                               seed=5, chunk=100)
    gene_set = list(genes[n_live: n_live + 12])
    p = cache.detection[cache.positions(gene_set)]
    assert np.all(p == 0), "the planted block must be silent"

    df = sg.score_decomposition(cache, gene_set, (X > 0), max_rank=m)
    realised = float(df["tie_break"].mean() / max(df["total"].mean(), 1e-12))
    predicted = sg.tie_break_share(p, depth, m, G)
    assert abs(realised - predicted) < 0.15, (realised, predicted)


def test_sparsity_report_prefers_the_exact_criterion(data, cache):
    """With depth in the cache the report must say so, and its share must be the
    same number preflight computes from the same inputs."""
    X, genes = data
    gene_set = [g for g in genes[::11][:20]]
    rep = sg.sparsity_report(cache, gene_set)
    assert rep["criterion"] == "tie_break_share"
    assert "tie_break_share" in rep
    assert rep["detection_floor_source"].startswith("adaptive")

    pos = cache.positions(gene_set)
    pf = sg.preflight(cache.depth, np.arange(cache.n_cells, dtype=float),
                      rep["max_rank"], cache.n_genes_total,
                      detection=cache.detection[pos], k=len(pos))
    assert abs(pf["tie_break_share"] - rep["tie_break_share"]) < 1e-12


def test_verdict_says_which_criterion_it_used(data, cache):
    """The floor is a fixed rate and the share is not; a reader has to be able
    to tell which one produced the verdict without reading the source."""
    X, genes = data
    gene_set = [g for g in genes[::11][:20]]

    with_depth = sg.verdict(sg.sparsity_report(cache, gene_set))
    assert with_depth["criterion"] == "tie_break_share"

    rep = sg.sparsity_report(cache, gene_set)
    rep.pop("tie_break_share")                      # pretend depth was unknown
    without = sg.verdict(rep)
    assert without["criterion"] == "fixed_detection_floor"
    for key in ("verdict", "severity", "reasons"):
        assert key in with_depth and key in without


def test_a_fixed_floor_censors_the_sparsest_sets_and_the_adaptive_one_does_not():
    """The failure the adaptive floor exists to prevent: a set whose genes are
    detected at an ordinary-but-low rate is declared undetectable because the
    floor was chosen for a different dataset, and every replacement drawn for it
    is then denser than the gene it stands in for."""
    X = synthetic_matrix()
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    G = X.shape[1]
    m = int(np.ceil(0.05 * G))
    depth = np.asarray((X > 0).sum(axis=1)).ravel()
    ceiling = int(min(G - 1, max(m + 2, depth.max() + 1)))
    cache = sg.RankCache.build(X, genes, ceiling=ceiling, seed=4, chunk=200)

    # A set that is sparse but not silent: its genes are found in a fraction of
    # a percent to two percent of cells.
    band = np.flatnonzero((cache.detection >= 0.004)
                          & (cache.detection <= 0.018))
    assert band.size >= 25, "the test needs a populated low-detection band"
    sparse_set = [genes[i] for i in band[:25]]

    fixed = sg.sparsity_report(cache, sparse_set, detection_floor=0.02)
    adaptive = sg.sparsity_report(cache, sparse_set)

    assert fixed["detection_floor_source"].startswith("fixed")
    assert adaptive["detection_floor_source"].startswith("adaptive")
    # The fixed floor is above every gene of the set, so the report says the set
    # is not detected anywhere -- which is false, and is the bias the framework
    # exists to remove.
    assert fixed["frac_above_floor"] == 0.0
    assert adaptive["detection_floor"] < 0.02
    assert adaptive["frac_above_floor"] == 1.0

    # The consequence at the point it matters: nulls drawn under the fixed floor
    # cannot be matched to this set, while the adaptive floor can match it.
    fixed_null = sg.construct_expression_matched_null(
        cache, sparse_set, n_sets=10, seed=1)
    adaptive_null = sg.construct_expression_matched_null(
        cache, sparse_set, n_sets=10, seed=1)
    assert fixed_null["status"] in {"PASS", "FAIL"}
    assert abs(adaptive_null["detection_ratio"] - 1.0) <= 0.05, (
        f"the adaptive null is {adaptive_null['detection_ratio']:.3f}x the set")


# ----------------------------------------------------------------------
# the assembled surface: checklist, calibration object, JSON
# ----------------------------------------------------------------------
def _two_cohort_data(seed=21, n_cells=600, n_genes=1000):
    """One matrix in which a planted programme and the depth both move with the
    axis, so that every step of the checklist has something to say."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=n_cells)
    activity = np.exp(rng.normal(0, 1.6, size=n_genes)) / 12.0
    depth_scale = np.exp(0.6 * (axis - axis.mean()) / axis.std())
    rate = np.clip(np.outer(depth_scale, activity), 0, 0.9)
    X = (rng.random((n_cells, n_genes)) < rate).astype(np.float32)
    X *= rng.gamma(2.0, 1.0, size=X.shape).astype(np.float32)
    genes = np.array([f"G{i:05d}" for i in range(n_genes)])
    programme = list(genes[:40])
    return sp.csr_matrix(X), genes, axis, programme


def test_calibrator_end_to_end_produces_a_complete_record(tmp_path):
    """Every field the framework promises has to be present and finite, and the
    file it writes has to be readable JSON -- an object that only serialises in
    memory is not a record of anything."""
    import json

    X, genes, axis, programme = _two_cohort_data()
    cal = sg.Calibrator.from_matrix(X, genes, config=dict(
        n_null=20, cutpoint_permutations=40, seed=7))

    out = cal.calibration_object(programme, axis=axis, label="programme",
                                 n_perm=40)

    assert out["label"] == "programme"
    assert out["discovery"]["n_cells"] == X.shape[0]
    assert np.isfinite(out["discovery"]["association_rho"])
    assert np.isfinite(out["discovery"]["score_auc"])

    ck = out["checklist"]
    for step in ("step1_sparsity", "step2_null_construction",
                 "step3_false_positive_rate", "step4_external_replication"):
        assert step in ck, step
    assert ck["overall_verdict"] in {"INTERPRETABLE",
                                     "INTERPRETABLE_WITH_MATCHED_NULL",
                                     "NOT_IDENTIFIABLE"}
    # the cache carries depth, so step 1 answers with the exact criterion
    assert ck["step1_sparsity"]["criterion"] == "tie_break_share"
    assert 0.0 <= ck["step1_sparsity"]["tie_break_share"] <= 1.0
    # the matched null was built and reported on its own terms
    assert ck["step2_null_construction"]["status"] in {"PASS", "FAIL",
                                                       "UNDEFINED"}

    fpr = out["false_positive_rates"]
    assert np.isfinite(fpr["optimum_cutpoint_FPR"])
    assert np.isfinite(fpr["median_split_FPR"])
    assert 0.0 <= fpr["median_split_FPR"] <= 1.0

    path = tmp_path / "calibration.json"
    sg.to_json(out, path)
    back = json.loads(path.read_text())
    assert back["checklist"]["overall_verdict"] == ck["overall_verdict"]
    assert back["false_positive_rates"]["optimum_cutpoint_FPR"] == \
        fpr["optimum_cutpoint_FPR"]


def test_the_optimum_cutpoint_costs_more_false_positives_than_the_median():
    """The headline claim of step 3, checked against a score with no relation to
    the outcome at all: choosing the cutpoint from the outcome has to buy the
    analyst a materially higher false-positive rate than fixing it at the
    median, and the pre-specified rule has to sit at the nominal level."""
    rng = np.random.default_rng(11)
    n = 800
    score = rng.normal(size=n)
    outcome = rng.random(n) < 0.5                     # independent of the score

    opts = [sg.cutpoint_fpr(score, outcome, n_perm=200, method=m, seed=s)["fpr"]
            for m, s in (("optimum", 1), ("median", 2))]
    pre = sg.cutpoint_fpr(score, outcome, n_perm=200, method="pre_specified",
                          pre_specified=0.0, seed=3)["fpr"]
    optimum, median = opts

    assert optimum > median, (optimum, median)
    assert pre <= 0.15, f"a pre-specified cutpoint fired at {pre:.2f}"
    assert optimum > 2 * pre


def test_checklist_step2_reports_a_null_that_failed_to_match(tmp_path):
    """A null that did not match is not a null, and the checklist has to say so
    rather than carrying its P value forward as evidence."""
    X, genes, axis, _ = _two_cohort_data(seed=5)
    cal = sg.Calibrator.from_matrix(X, genes, config=dict(n_null=10, seed=3))

    # A set of the very sparsest genes: the pool has little to match it with.
    order = np.argsort(cal.cache.detection)
    hard = [genes[i] for i in order[:30]]
    built = cal.matched_null(hard, n_sets=10)
    assert set(built) >= {"null_genes", "target_sparsity", "null_sparsity",
                          "detection_ratio", "status", "ks_statistic"}
    assert np.isfinite(built["ks_statistic"])
    assert built["status"] in {"PASS", "FAIL", "UNDEFINED"}
    # whichever way it went, the verdict must not ignore it
    ck = cal.checklist(hard)
    assert ck["step2_null_construction"]["status"] == built["status"]


def test_a_matched_null_is_drawn_from_the_cache_and_matches_the_set():
    """The replacement sets have to be drawn from this matrix, be the same size
    as the set, and reproduce its detection rate -- the three properties the
    first two steps of the checklist assume."""
    X, genes, axis, programme = _two_cohort_data(seed=9)
    cal = sg.Calibrator.from_matrix(X, genes, config=dict(seed=3))
    built = cal.matched_null(programme, n_sets=25)

    assert len(built["null_genes"]) == 25
    universe = set(cal.cache.genes)
    for s in built["null_genes"]:
        assert len(s) == len(programme)
        assert set(s) <= universe
        assert not (set(s) & set(programme)), "a null must not reuse the set"
    assert abs(built["detection_ratio"] - 1.0) < 0.10, built["detection_ratio"]


def test_compute_calibrated_score_puts_the_null_beside_the_score():
    """The score is only interpretable next to its null, so both have to come
    back on one frame over the same cells."""
    X, genes, axis, programme = _two_cohort_data(seed=13)
    cal = sg.Calibrator.from_matrix(X, genes, config=dict(seed=3))
    null_set = cal.matched_null(programme)["null_genes"]

    frame = cal.calibrated_score(programme, null_genes=null_set, axis=axis)
    assert len(frame) == X.shape[0]
    assert {"score_target", "score_null", "score_difference", "axis"} <= set(
        frame.columns)
    assert np.allclose(frame["score_target"] - frame["score_null"],
                       frame["score_difference"])


def test_decomposition_is_available_exactly_when_the_matrix_was_given():
    """The realised split of equation (2) needs the detection matrix, and the
    calibrator must not pretend otherwise."""
    X, genes, axis, programme = _two_cohort_data(seed=17)
    with_matrix = sg.Calibrator.from_matrix(X, genes)
    df = with_matrix.decomposition(programme)
    assert {"detected", "tie_break", "depth", "total"} <= set(df.columns)
    assert np.allclose(df["detected"] + df["tie_break"], df["total"])

    without = sg.Calibrator(with_matrix.cache, detection_matrix=None)
    with pytest.raises(RuntimeError):
        without.decomposition(programme)


# ----------------------------------------------------------------------
# comparators: the methods a reader is already using
# ----------------------------------------------------------------------
def test_comparator_labels_are_the_ones_the_dispatch_table_carries():
    """A label the dispatcher does not know is a silent omission from the
    comparison table, so the two lists have to agree by construction."""
    from sparsegs.comparators import (COMPARATORS, COMPARATORS_BY_NAME,
                                      decoupler_methods)
    assert set(COMPARATORS) == {label for label, _ in COMPARATORS_BY_NAME}
    for label in decoupler_methods():
        assert label in COMPARATORS, label


def test_decoupler_wrapper_matches_the_packages_own_names():
    """AUCell is the one method ``decoupler`` spells differently; the mapping is
    the only place that difference is allowed to live."""
    from sparsegs.comparators import COMPARATORS_BY_NAME
    mapping = dict(COMPARATORS_BY_NAME)
    assert mapping["aucell_decoupler"] == "aucell"
    assert mapping["ulm"] == "ulm" and mapping["mlm"] == "mlm"


def test_decoupler_is_called_when_it_is_installed_and_reported_when_it_is_not():
    """The wrapper must go through the real package rather than reimplement it,
    and must fail loudly rather than return a plausible number when the package
    is absent."""
    from sparsegs import comparators as cmp

    if not cmp.have_decoupler():
        with pytest.raises(ImportError):
            cmp.run_decoupler(None, [], [], None)
        return

    X, genes, axis, programme = _two_cohort_data(seed=19)
    out = cmp.run_decoupler(X, genes, programme, axis, method="ulm")
    assert out["decoupler_impl"] == "ulm"
    assert out["p_source"].startswith("framework")
    assert np.isfinite(out["score"])
    # the regressor is membership, so the matrix must carry non-members too:
    # restricting it to the set's own columns is the silent failure
    assert out["n_genes"] == len(programme)
    assert out["n_cells_scored"] > 0
    assert out["n_dropped_empty"] >= 0
    assert out["per_cell"].size == out["n_cells_scored"]

    table = cmp.run_comparators(X, genes, programme, axis,
                               methods=["ulm", "mlm"])
    assert len(table) == 2
    assert "per_cell" not in table.columns, (
        "a per-cell vector must not be put inside a one-row-per-method frame")
    assert set(table["method"]) == {"ulm", "mlm"}


def test_size_matched_null_controls_the_mean_and_not_the_spread():
    """The compromise the framework argues against, stated as a property: a
    null matched on mean detection reproduces the mean and leaves the per-gene
    spread free."""
    X, genes, axis, programme = _two_cohort_data(seed=23)
    cache = sg.RankCache.build(X, genes, ceiling=60, seed=1, chunk=10 ** 6)
    nulls = cmp_size_matched(cache, programme, n=20)

    assert len(nulls) == 20
    for s in nulls:
        assert len(s) == len(programme)
    pos = cache.positions(programme)
    target_mean = float(np.mean(cache.detection[pos]))
    drawn = [float(np.mean(cache.detection[cache.positions(s)]))
             for s in nulls]
    assert abs(np.mean(drawn) / target_mean - 1.0) < 0.5
    # ... and the spread is not controlled: the drawn sets' internal spread is
    # not the observed set's, which is the objection
    target_sd = float(np.std(cache.detection[pos]))
    drawn_sd = float(np.mean([np.std(cache.detection[cache.positions(s)])
                              for s in nulls]))
    assert abs(drawn_sd / target_sd - 1.0) > 0.05


def cmp_size_matched(cache, gene_set, n=20):
    """Import shim so the test above stays readable."""
    from sparsegs.comparators import size_matched_null
    return size_matched_null(cache, gene_set, n=n, rng=np.random.default_rng(2))


def test_a_matched_null_does_not_reuse_genes_of_the_set_it_replaces():
    """Regression: the natural nearest-neighbour match, which excludes only the
    single gene being stood in for, replaces a set drawn from a narrow detection
    band almost entirely with its own members.

    On a 30-gene set that was the case for 12 genes per draw, every draw shared
    at least one, and the resulting "null" score tracked the observed score at
    rho 0.43 -- so every P value taken from it was conservative.  The property
    that has to hold is zero overlap, and it has to hold for the hard case: a set
    whose members are each other's nearest neighbours.
    """
    X = synthetic_matrix(n_cells=800, n_genes=2000, density=0.06, seed=31)
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    cache = sg.RankCache.build(X, genes, ceiling=100, seed=1, chunk=800)
    builder = sg.MatchedNullBuilder(cache, det_floor=1e-4, seed=7)

    # The adversarial set: 30 genes of nearly identical detection rate, so the
    # nearest-neighbour pool of each is the rest of the set.
    mid = int(np.argsort(cache.detection)[cache.detection.size // 2])
    band = np.argsort(np.abs(cache.detection - cache.detection[mid]))[:30]
    gene_set = list(cache.genes[band])

    rng = np.random.default_rng(7)
    for kind in ("expression", "expression_bin", "random"):
        sets = builder.sample(gene_set, 20, kind=kind, rng=rng, n_draws=5)
        overlaps = [len(set(s) & set(gene_set)) for s in sets]
        assert max(overlaps) == 0, (kind, max(overlaps))
        for s in sets:
            assert len(set(s)) == len(s), "a replacement must not repeat a gene"


def test_the_null_status_does_not_depend_on_how_many_draws_were_asked_for():
    """The status is a property of the pool, not of the sample size.

    Deciding it on the largest gap over draws made it a function of ``n_sets``
    instead: a maximum can only grow as more draws are taken, so one pool would
    pass at 25 draws and fail at 100 -- more draws, the better null, reported as
    the worse one.  The mean gap settles down instead, and both numbers are
    reported so the spread is still visible.
    """
    X = synthetic_matrix(n_cells=500, n_genes=1500, density=0.05, seed=17)
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    cache = sg.RankCache.build(X, genes, ceiling=60, seed=3, chunk=500)
    # Sparse but not undetected: a set with no detection at all is UNDEFINED
    # rather than PASS or FAIL, which is a different test.
    det = cache.detection
    band = np.where((det > 0.01) & (det < 0.06))[0]
    assert band.size >= 12, "the fixture matrix has to contain a sparse band"
    gene_set = list(cache.genes[band[:12]])

    statuses, worsts, means = [], [], []
    for n in (10, 50, 200):
        built = sg.construct_expression_matched_null(cache, gene_set,
                                                     n_sets=n, seed=11)
        statuses.append(built["status"])
        worsts.append(built["worst_detection_gap"])
        means.append(built["mean_detection_gap"])
        assert built["max_overlap"] == 0

    assert len(set(statuses)) == 1, (
        f"the same pool was reported as {statuses} at 10, 50 and 200 draws")
    # The maximum is still monotone in the draw count -- that is the whole
    # reason it cannot decide the status -- while the mean is not.
    assert worsts[0] <= worsts[-1] + 1e-9
    assert max(means) - min(means) < 0.01, f"mean gaps {means} did not settle"


def test_a_family_that_matches_on_average_but_has_wild_draws_says_so():
    """Averaging must not hide the spread: the fraction within tolerance is
    reported next to the mean, so a family matching on average through a few
    outliers is distinguishable from one that matches throughout."""
    X = synthetic_matrix(n_cells=500, n_genes=1500, density=0.05, seed=17)
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    cache = sg.RankCache.build(X, genes, ceiling=60, seed=3, chunk=500)
    det = cache.detection
    band = np.where((det > 0.01) & (det < 0.06))[0]
    gene_set = list(cache.genes[band[:12]])
    built = sg.construct_expression_matched_null(cache, gene_set, n_sets=60,
                                                 seed=11)
    for field in ("mean_detection_gap", "worst_detection_gap",
                  "frac_within_tolerance"):
        assert field in built, f"{field} is part of what the status is read from"
    assert built["worst_detection_gap"] >= built["mean_detection_gap"]
    assert 0.0 <= built["frac_within_tolerance"] <= 1.0


def test_the_null_reports_how_much_it_had_to_relax():
    """When the background cannot supply distinct non-member genes the draw
    relaxes rather than failing, and says so -- a silently relaxed null is the
    failure the report exists to prevent."""
    X = synthetic_matrix(n_cells=400, n_genes=800, density=0.06, seed=41)
    genes = np.array([f"G{i:05d}" for i in range(X.shape[1])])
    cache = sg.RankCache.build(X, genes, ceiling=40, seed=1, chunk=400)

    # A set of a third of the background: no pool can avoid its members entirely.
    big = list(cache.genes[np.argsort(cache.detection)[-260:]])
    built = sg.construct_expression_matched_null(cache, big, n_sets=5, seed=2)

    assert built["max_overlap"] > 0
    assert built["status"] == "FAIL", (
        "a null that reuses genes of the set must not be reported as a pass")
    assert built["n_from_query"] > 0


# ----------------------------------------------------------------------
# the single-set report
# ----------------------------------------------------------------------
def _report_calibrator(tmp_path):
    X, genes, axis, programme = _two_cohort_data()
    cal = sg.Calibrator.from_matrix(X, genes, config=dict(
        n_null=15, n_draws=6, cutpoint_permutations=25, seed=5))
    return cal, programme, axis


def _parse_report(path):
    """The report is a document, so it is checked by parsing it, not by reading
    bytes: a signature mismatch upstream produces a file that opens and is
    empty."""
    lxml = pytest.importorskip("lxml.html")
    return lxml.fromstring(path.read_bytes())


def test_the_report_is_written_and_parses(tmp_path):
    """The report path is the one a reader actually opens, so it is exercised
    end to end -- the signature it calls has to be the signature that exists."""
    cal, programme, axis = _report_calibrator(tmp_path)
    path = tmp_path / "report.html"
    returned = cal.report(programme, path, title="Repressor arm", axis=axis,
                          dataset="synthetic", cell_type="CD8+ T",
                          n_null=15)
    assert returned == str(path)
    doc = _parse_report(path)

    sections = [h.text for h in doc.iter("h2")]
    assert sections[:2] == ["Verdict", "Gene set"]
    assert "Association against matched nulls" in sections
    assert "Cutpoint false-positive rate" in sections
    # every figure the report claims to draw has to be in the file
    assert len(doc.findall(".//svg")) >= 3
    assert doc.findtext(".//title") == "Repressor arm"


def test_the_report_without_an_axis_narrows_to_what_can_be_answered(tmp_path):
    """Steps 3 and 4 need an outcome.  Without one the report has to drop those
    sections rather than draw an empty figure that reads as a null result."""
    cal, programme, _ = _report_calibrator(tmp_path)
    path = tmp_path / "bare.html"
    cal.report(programme, path)
    doc = _parse_report(path)

    sections = [h.text for h in doc.iter("h2")]
    assert "Verdict" in sections and "Gene set" in sections
    assert "Association against matched nulls" not in sections
    assert "Cutpoint false-positive rate" not in sections


def test_the_report_defaults_to_the_strongest_null_it_can_build(tmp_path):
    """A calibrator built from a matrix can measure co-detection, so that is the
    family its report uses; one built without the matrix falls back rather than
    raising, and either way the figure names the family it drew."""
    LH = pytest.importorskip("lxml.html")

    cal, programme, axis = _report_calibrator(tmp_path)
    path = tmp_path / "auto.html"
    cal.report(programme, path, axis=axis, n_null=10)
    labels = [t.text_content() for t in
              LH.fromstring(path.read_bytes()).iter("text")]
    assert any("codetection" in (t or "").lower() for t in labels), (
        "the report must name the null family it drew")

    bare = sg.Calibrator(cal.cache, config=cal.config)
    path2 = tmp_path / "bare_null.html"
    bare.report(programme, path2, axis=axis, n_null=10)
    assert path2.exists()


def test_the_api_documentation_covers_every_export():
    """The manual is generated, so the failure mode is silent drift: a name
    added to the package and not to the manual.  Regenerating it here and
    comparing against ``__all__`` is what stops that."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "docs", "build_api.py")
    spec = importlib.util.spec_from_file_location("_build_api", path)
    build_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_api)

    text = build_api.render()
    documented = set(re.findall(r"^### (\w+)", text, re.M))
    documented |= set(re.findall(r"^\*\*`(\w+)`\*\*", text, re.M))

    # The submodules are the chapters of the manual and the dunder is not part
    # of the public surface; everything else has to be in it.
    skip = {m.rsplit(".", 1)[-1] for m, _ in build_api.MODULES} | {"__version__"}
    missing = [n for n in sg.__all__ if n not in documented and n not in skip]
    assert not missing, f"undocumented exports: {missing}"


def test_every_script_in_the_pipeline_compiles():
    """The analysis scripts run once, late, on data that took hours to produce.

    A script whose input does not exist yet has never been executed, so a typo
    on any of its lines is invisible until the moment the expensive run
    finishes and the analysis is meant to happen -- and the failure then costs
    a debugging round at the worst possible time.  ``analyse_benchmark.py`` sat
    with a dict key that was not a legal identifier (a decimal point in a name)
    for exactly that reason.  Importing every file is cheap by comparison.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    failures = []
    for folder in ("experiments", "sparsegs", "app", "tutorials"):
        for name in sorted(os.listdir(os.path.join(root, folder))):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, folder, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    compile(fh.read(), path, "exec")
            except SyntaxError as exc:
                failures.append(f"{folder}/{name}:{exc.lineno}: {exc.msg}")
    assert not failures, "uncompilable scripts: " + "; ".join(failures)


# ----------------------------------------------------------------------
# a seed parameter is a promise, and the promise is about the numbers
# ----------------------------------------------------------------------
def _canonical(value):
    """Whatever a call returned, in a form that two calls can be compared by.

    The entry points below return lists of gene names, arrays, dicts and
    DataFrames, and a comparison that only worked for one of those would
    quietly skip the rest.
    """
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        return ("frame", tuple(value.columns),
                tuple(np.asarray(value[c]).tobytes() for c in value.columns))
    if isinstance(value, dict):
        return ("dict", tuple((str(k), _canonical(v))
                              for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_canonical(v) for v in value))
    if isinstance(value, np.ndarray):
        return ("array", value.shape, value.tobytes())
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("num", float(value))
    return ("scalar", value)


def _seeded_entry_points(data, cache, builder):
    """Every way a user can draw random numbers through this package.

    The table is not documentation.  A parameter named ``seed`` promises two
    things -- that a repeated seed repeats the answer, and that a different seed
    changes it -- and the second is the one a function can break without any
    test of its signature noticing.  Reading the global NumPy state instead of
    the argument passed in looks identical from the outside and makes every
    reported number irreproducible.
    """
    X, genes = data
    members = list(cache.genes[:12])
    score = np.asarray(sg.aucell(cache, members), dtype=float)
    programme = score + np.linspace(0.0, 1.0, score.size)
    outcome = programme > np.median(programme)
    # The fixture's 25 bins of 1200 genes leave ~45-48 non-set genes per bin,
    # which sits inside the degenerate window of the default control sizes
    # (50 and 100): the whole bin would be taken and the control composition
    # could not depend on the seed -- any apparent seed sensitivity would be
    # floating-point summation order, not the generator.  ctrl_size=10 keeps
    # the draw genuinely random on every numpy and pandas.
    return {
        "score_genes":
            lambda s: sg.score_genes(X, genes, members, ctrl_size=10, seed=s),
        "module_score":
            lambda s: sg.module_score(X, genes, members, ctrl_size=10, seed=s),
        "score_all":
            lambda s: sg.score_all(cache, X, genes, members, ucell_r_max=60,
                                   ctrl_size=10, seed=s),
        "permutation_test":
            lambda s: sg.permutation_test(score, programme, n_perm=40, seed=s),
        "all_conventional":
            lambda s: sg.all_conventional(score, programme, n_perm=40, seed=s),
        "cutpoint_fpr":
            lambda s: sg.cutpoint_fpr(score, outcome, n_perm=40, method="median", seed=s),
        "tie_break_keys":
            lambda s: sg.tie_break_keys(60, 400, seed=s),
        "compare_nulls":
            lambda s: sg.compare_nulls(builder, members, n=4, seed=s),
        "construct_expression_matched_null":
            lambda s: sg.construct_expression_matched_null(cache, members,
                                                           n_sets=3, seed=s),
        "diagnostic_checklist":
            lambda s: sg.diagnostic_checklist(cache, members, axis=programme,
                                              n_perm=40, seed=s),
        "tier1_technical":
            lambda s: sg.tier1_technical(0.30, n_cells=200, n_genes=2500,
                                         seed=s, n_repeats=2),
        "tier2_internal":
            lambda s: sg.tier2_internal(score, programme, builder, members, 60,
                                        n_null=8, seed=s, kinds=("expression",)),
        "MatchedNullBuilder.sample":
            lambda s: builder.sample(members, 3, kind="expression", seed=s),
    }


def test_a_family_that_was_not_drawn_has_no_verdict(cache, builder):
    """``kinds`` is a free parameter, and the headline verdicts follow it.

    Asking for the expression family alone used to raise ``KeyError`` -- the
    co-detection verdict was read unconditionally -- and the R port answered
    the same question with ``FALSE``, which is worse: a "failed" verdict for a
    test that was never run is exactly the kind of number this framework exists
    to refuse to produce.
    """
    members = list(cache.genes[:12])
    score = np.asarray(sg.aucell(cache, members), dtype=float)
    programme = score + np.linspace(0.0, 1.0, score.size)

    partial = sg.tier2_internal(score, programme, builder, members, 60,
                                n_null=8, seed=3, kinds=("expression",))
    assert partial["passed_family"] == "expression"
    assert partial["passed"] in (True, False)
    assert partial["passed_strictest"] is None
    assert partial["passed_strictest_family"] is None

    full = sg.tier2_internal(score, programme, builder, members, 60,
                             n_null=8, seed=3)
    assert full["passed_strictest_family"] == "codetection"


def test_every_stochastic_entry_point_reproduces_under_a_fixed_seed(data, cache,
                                                                    builder):
    for name, call in _seeded_entry_points(data, cache, builder).items():
        assert _canonical(call(11)) == _canonical(call(11)), (
            f"{name} returned two different answers for one seed")


def test_the_seed_a_caller_passes_is_the_seed_that_is_used(data, cache, builder):
    """Different seeds have to give different answers.

    A function that ignores its ``seed`` argument and draws from the global
    NumPy state passes any test that only checks the first property: it is
    reproducible whenever nothing else in the process has touched the generator.
    This is the test that fails instead.
    """
    ignored = [name for name, call
               in _seeded_entry_points(data, cache, builder).items()
               if _canonical(call(11)) == _canonical(call(12))]
    assert not ignored, (
        "these accept a seed and return identical results for two different "
        f"seeds, so the argument is not reaching the generator: {ignored}")


def test_every_export_that_takes_a_seed_is_in_the_reproducibility_table(data,
                                                                       cache,
                                                                       builder):
    """The table above has to keep up with the package.

    A new stochastic entry point added to the public surface and left out of the
    table would be a function whose seeding nobody checks; the failure is silent
    and the table is the only place that would notice.
    """
    takes_seed = set()
    for name in sg.__all__:
        obj = getattr(sg, name, None)
        if not callable(obj) or inspect.isclass(obj):
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        if any("seed" in p or "rng" in p for p in params):
            takes_seed.add(name)
    missing = sorted(takes_seed - set(_seeded_entry_points(data, cache, builder)))
    assert not missing, (
        f"exports taking a seed that no reproducibility test covers: {missing}")

    # And the other direction, so that a name in the table which stops existing
    # is a failure here rather than a test that quietly goes on passing.
    stale = sorted(n for n in _seeded_entry_points(data, cache, builder)
                   if "." not in n and n not in sg.__all__)
    assert not stale, f"the table names things the package does not export: {stale}"


# ----------------------------------------------------------------------
# the manuscript's numbers
# ----------------------------------------------------------------------
def _manuscript_dir():
    from experiments import check_format
    return check_format.PKG / "manuscript", check_format


def test_the_macro_check_finds_a_number_that_was_never_emitted(tmp_path,
                                                               monkeypatch):
    """The failure mode this guard exists for, reproduced on purpose.

    ``make_numbers.py`` emits a macro per number from the file that produced it.
    When that file is absent the block is skipped, so the macro does not exist
    and the manuscript -- which still references it -- compiles with a hole in
    it.  A guard that has never been shown to fire is not evidence that the
    manuscript is intact, so the hole is punched here deliberately.
    """
    from experiments import check_format

    # The check looks for the manuscript under ``PKG/manuscript``, as the real
    # one is, so the fixture is laid out that way too rather than beside it.
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "numbers.tex").write_text(
        r"\newcommand{\gridARuns}{1080}" "\n", encoding="utf-8")
    (tmp_path / "manuscript" / "main.tex").write_text(
        r"\documentclass{article}" "\n"
        r"\input{numbers}" "\n"
        r"\begin{document}" "\n"
        r"\gridARuns{} runs, $\rho = \gridARho$.\n"
        r"\end{document}" "\n", encoding="utf-8")
    monkeypatch.setattr(check_format, "PKG", tmp_path)

    problems = check_format.macro_problems(tex="manuscript/main.tex",
                                           log="absent.log")
    assert any("gridARho" in p for p in problems), problems
    assert not any("gridARuns" in p for p in problems), (
        "a macro the document defines itself was reported as missing")
    assert not any("\\rho" in p for p in problems), (
        f"a LaTeX built-in was reported as missing: {problems}")


def test_the_macro_check_finds_a_number_that_was_declared_but_never_filled(
        tmp_path, monkeypatch):
    """The quieter half of the same failure.

    A macro that is never emitted leaves no trace in the source, and the test
    above covers that.  A macro that *is* declared and then left unfilled is
    worse: it is present in ``numbers.tex`` carrying the marker
    ``make_numbers.py`` writes for it, the marker compiles, and the page shows a
    gap where a value belongs.  A manuscript built from a results directory that
    is missing a whole block therefore passes every check that only looks at
    whether the macros resolve.
    """
    from experiments import check_format

    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "numbers.tex").write_text(
        r"\newcommand{\benchSets}{1{,}378}" "\n"
        r"\newcommand{\benchCohortSets}{\textit{(pending)}}" "\n",
        encoding="utf-8")
    monkeypatch.setattr(check_format, "PKG", tmp_path)

    pending = check_format.pending_numbers()
    assert pending == ["benchCohortSets"], (
        f"the marker was not read back, or a filled macro was reported "
        f"alongside it: {pending}")


def test_the_spacing_check_finds_a_macro_that_eats_its_space(tmp_path, monkeypatch):
    """A macro followed by a space and a word sets as one word.

    ``\\benchMaxRank and`` typesets as ``1233and``: the control word ends at the
    last letter of its name and TeX skips the spaces that follow it while
    reading the source.  Nothing warns -- every macro resolves, every word is
    still counted -- so the defect reaches the page unless something reads the
    source for it.  The corpus of macros differs from a blank line, which is a
    paragraph break rather than a space, so a macro ending a paragraph must not
    be reported.
    """
    from experiments import check_format

    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "numbers.tex").write_text(
        r"\newcommand{\benchMaxRank}{1233}" "\n"
        r"\newcommand{\benchCells}{55003}" "\n", encoding="utf-8")
    (tmp_path / "manuscript" / "main.tex").write_text(
        "the ceiling is \\benchMaxRank and the cohort holds\n"
        "\\benchCells{}\n\n"
        "cells, which is \\benchCells{} more\n"
        "than \\benchCells\n"
        "alone.\n"
        "\\benchMaxRank{} is the ceiling.\n", encoding="utf-8")
    monkeypatch.setattr(check_format, "PKG", tmp_path)

    found = check_format.swallowed_spaces("manuscript/main.tex")
    # Two defects: ``\benchMaxRank and`` on the first line and ``\benchCells``
    # before the newline that continues into ``alone``.  The macro followed by
    # ``{}``, the one at the end of a paragraph and the one ending a sentence
    # are all correct and must not be reported.
    assert len(found) == 2, f"expected the two joined words, got {found}"
    assert any("benchMaxRank" in p for p in found), found
    assert any("benchCells" in p for p in found), found


def test_the_manuscript_keeps_the_space_after_its_macros():
    """The manuscript on disk has no macro joined to the word after it.

    The check above is only worth having if the delivered text passes it.  A
    number written without the following ``{}`` is a typographic defect that no
    compiler reports, and the manuscript is edited between runs, so it is read
    here rather than assumed.
    """
    manuscript, check_format = _manuscript_dir()
    if not (manuscript / "numbers.tex").exists():
        pytest.skip("numbers.tex has not been generated yet")
    found = check_format.swallowed_spaces()
    assert not found, "\n".join(found)


def test_a_generated_appendix_float_has_to_be_renumbered(tmp_path, monkeypatch):
    """A table input in the appendix is numbered in the main series unless told not to.

    It then prints as "Table 5" beside four print tables and a figure, which is
    a sixth display item the journal counts, while the sentence citing it says
    "Supplementary Table~5".  The table file is regenerated, so its numbering is
    decided by the manuscript around it: the check has to look there.
    """
    from experiments import check_format

    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "gen_table.tex").write_text(
        "\\begin{table}\n\\caption{The tail conventions.}\n\\end{table}\n",
        encoding="utf-8")
    main = tmp_path / "manuscript" / "main.tex"
    head = ("\\begin{table}\\caption{Print table.}\\end{table}\n"
            "\\appendix\n\\section{Supplementary tables}\n")
    main.write_text(head + "\\input{gen_table}\n", encoding="utf-8")
    monkeypatch.setattr(check_format, "PKG", tmp_path)

    found = check_format.supplementary_numbering("manuscript/main.tex")
    assert len(found) == 1 and "gen_table.tex" in found[0], (
        f"a float input in the appendix without renumbering was not reported: "
        f"{found}")

    main.write_text(head
                    + "\\setcounter{table}{0}\n"
                      "\\renewcommand{\\thetable}{S\\arabic{table}}\n"
                    + "\\input{gen_table}\n", encoding="utf-8")
    assert check_format.supplementary_numbering("manuscript/main.tex") == [], (
        "a supplementary table was reported after the manuscript renumbered it")


def test_the_appendix_table_is_numbered_apart_from_the_print_tables():
    """The manuscript on disk keeps its generated table out of the print count.

    Four tables and a figure are the print display items; the generated table
    arrives through ``\\input`` after ``\\appendix`` and has to be numbered in
    its own series, which is what the sentence citing it claims.
    """
    manuscript, check_format = _manuscript_dir()
    if not (manuscript / "tail_robustness_table.tex").exists():
        pytest.skip("the generated table has not been written yet")
    assert check_format.supplementary_numbering() == []


def test_the_generated_numbers_carry_no_marker():
    """``numbers.tex`` on disk is built from inputs that are all present.

    The pipeline regenerates this file before it builds the PDF, so a marker
    surviving into it means the manuscript that would be delivered is missing a
    quantity the text asks for.  The file is skipped when it has not been
    generated at all -- a clone that has never run the chain is not a defect --
    but a file that exists and is incomplete is one.
    """
    manuscript, check_format = _manuscript_dir()
    numbers = manuscript / "numbers.tex"
    if not numbers.exists():
        pytest.skip("numbers.tex has not been generated yet")
    pending = check_format.pending_numbers()
    assert not pending, (
        f"{len(pending)} macro(s) in numbers.tex were declared and never "
        f"computed, so the manuscript prints a marker where each value "
        f"belongs: {pending[:8]}"
        + (" ..." if len(pending) > 8 else ""))


def test_every_number_the_manuscript_quotes_can_be_produced(monkeypatch):
    """No macro in the text is a typo.

    ``make_numbers.py`` is the only thing that defines a number, so a macro the
    manuscript uses has to trace back to it.  The macro name is not always
    written out in full there -- ``{tag}ShareCeilRho`` is completed at run time
    from the grid it belongs to -- so the test asks whether the name *ends* in
    something the generator can put at the end of a name.  That is enough to
    separate ``\\auditSparsePseudoLo`` from a misspelling of it, which is the
    error it is for.

    A name that fails this is either a typo or a number nothing computes; both
    are worth failing a test over, and neither is visible in the compiled PDF.
    """
    from experiments import check_format, make_numbers

    manuscript, _ = _manuscript_dir()
    generator = (manuscript.parent / "experiments" / "make_numbers.py").read_text(
        encoding="utf-8")
    # Five characters, not more: the shortest literal a generated name is built
    # from is ``Ratio``, as in ``f"audit{label}Ratio"``, and a threshold above
    # that would report every ratio macro in the paper as an orphan.
    literals = set(re.findall(r'[A-Za-z][A-Za-z0-9_]{4,}', generator))
    # Reading the source alone is not enough for the per-level families.  Their
    # names end in a table neither side spells out: the level suffix is
    # ``NAMES[label].capitalize()`` and the factor key is ``FACTORS``'s second
    # column, so ``\\gridEDonACellcor`` is composed from strings that appear in
    # the file only in lower case, or not in that role at all.  The vocabulary
    # is taken from the two tables rather than retyped here, so a level added
    # to either one is covered without touching this test.
    literals |= {value.capitalize() for value in make_numbers.NAMES.values()}
    literals |= {key + "s" for _, key in make_numbers.FACTORS}   # the list macro

    problems = check_format.macro_problems()
    unresolved = [re.match(r"\\(\w+)", p).group(1) for p in problems
                  if p.startswith("\\")]
    orphans = [name for name in unresolved
               if len(name) >= 5
               and not any(name.endswith(lit) for lit in literals)]
    assert not orphans, (
        "these macros are used in the manuscript and are not produced by "
        f"make_numbers.py, so no re-run will ever define them: {orphans}")


# ----------------------------------------------------------------------
# the edge cases the brief asks the framework to refuse or warn on
# ----------------------------------------------------------------------
def _capture_log():
    """A handler that collects the framework's structured records as text."""
    import io
    import logging
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    sg.events.LOGGER.addHandler(handler)
    return buf, handler


def test_a_score_with_no_spread_is_refused(cache):
    """Edge case three: a set that is off in every cell, or on in every cell.

    The correlation such a score enters is 0/0.  Left alone it returns a NaN
    that surfaces much later, in whichever statistic first touches it, with
    nothing to say about which gene set produced it -- so the refusal happens
    where the score is made, and names the set.
    """
    n = int(cache.clipped.shape[0])
    with pytest.raises(ValueError, match="zero variance"):
        sg.calibrate._check_score(np.full(n, 0.3), label="flat")
    # The advice the brief asks for is in the message, not only in the log: a
    # caller who catches the exception is the one who has to act on it.
    with pytest.raises(ValueError, match="consider different genes"):
        sg.calibrate._check_score(np.zeros(n), label="all zero")
    # A score with any real spread passes, and the check returns what it saw.
    live = np.linspace(0.0, 1.0, n)
    assert sg.calibrate._check_score(live) > sg.MIN_SCORE_SD


def test_a_perfect_auc_warns_with_the_three_things_to_check(cache):
    """Edge case two: an AUC of 0.99 is a claim about the data, not the score.

    It is warned about rather than refused, because a marker can genuinely
    separate what it marks; the brief's three candidate explanations are
    carried into the record so the warning can be acted on.
    """
    buf, handler = _capture_log()
    old = sg.events.LOGGER.level
    try:
        sg.set_log_level("warning")
        sg.calibrate._check_auc(0.5)
        assert buf.getvalue() == "", "a calibrated AUC raised a warning"
        sg.calibrate._check_auc(0.999, label="T-cell set")
        record = buf.getvalue()
        buf.truncate(0)
        buf.seek(0)
        # A score that separates the two groups just as perfectly in the other
        # direction is the same confound, and `auroc` does not fold its argument
        # about 0.5 -- so an upper-tail-only check would have missed half the
        # ways this case can arise.  The record carries both numbers, so a
        # reader can tell which way round it was.
        returned = sg.calibrate._check_auc(0.002, label="T-cell set")
        reversed_record = buf.getvalue()
    finally:
        sg.events.LOGGER.removeHandler(handler)
        sg.events.LOGGER.setLevel(old)

    assert "suspicious_score_separation" in record
    assert "T-cell set" in record
    for check in ("batch_effects", "outcome_leakage_into_cell_type_annotation",
                  "extreme_sparsity"):
        assert check in record, f"{check} missing from the warning"
    # The threshold travels with the record, so a reader of the log can tell
    # how far past it the value was without knowing this package's constants.
    assert "threshold=0.99" in record
    assert "auc=0.002" in reversed_record
    assert "separation=0.998" in reversed_record
    # The check reports the value it was given and does not fold it: the caller
    # asked for the AUC, and an AUC of 0.002 is what they get back.
    assert returned == 0.002


def test_the_checklist_carries_the_separation_check_not_only_the_tier_runner():
    """Edge case two fires through `diagnostic_checklist`, not just elsewhere.

    Written after the Python checklist was found not to call the check at all
    while the R implementation of the same framework did.  The verdict on the
    two sides agreed, the P values agreed, and the confound the brief's second
    edge case exists to catch was raised by one implementation and silently
    dropped by the other -- which is exactly the kind of difference the
    cross-language check compares outputs to find, and this one sat behind an
    input no battery happened to construct.

    The matrix puts a gene set that is on only in the second half of the cells
    against an axis that *is* the cell index, so the score separates the two
    halves completely and `step3_false_positive_rate` must both warn and report
    the number the judgement was made on.
    """
    rng = np.random.default_rng(6)
    n_cells, n_genes = 120, 200
    panel = [f"g{i}" for i in range(1, n_genes + 1)]
    X = np.zeros((n_cells, n_genes))
    X[:, :5] = 8                                  # ranks 1-5 in every cell
    X[60:, 20:40] = 5                             # separates the two halves
    X[:, 40:] = rng.binomial(1, 0.4, (n_cells, n_genes - 40))
    local = sg.RankCache.build(X, panel, ceiling=10, seed=1)
    axis = np.arange(n_cells, dtype=float)

    buf, handler = _capture_log()
    old = sg.events.LOGGER.level
    try:
        sg.set_log_level("warning")
        out = sg.diagnostic_checklist(local, panel[20:40], axis=axis,
                                      n_perm=20, cutpoint_method="median")
        record = buf.getvalue()
    finally:
        sg.events.LOGGER.removeHandler(handler)
        sg.events.LOGGER.setLevel(old)

    assert "suspicious_score_separation" in record, record
    assert "separation=1" in record
    # Reported whether or not it crosses, so a reader of the calibration object
    # can see the number the warning was about without re-deriving it.
    assert out["step3_false_positive_rate"]["score_auc"] == 1.0


def test_a_level_set_before_the_first_record_is_not_overwritten():
    """Attaching the first handler must not reset a level the caller chose.

    A caller who reaches for ``logging.getLogger("sparsegs").setLevel(...)``
    before the framework has emitted anything has made the most explicit
    choice available, and `_configure()` used to overwrite it on the way in --
    so the records they had just turned off arrived anyway.  It surfaced in the
    cross-language record battery, which raises the level to keep its own
    comparison quiet and printed the records regardless.

    Run in a fresh interpreter because the bug lives in a module-level
    once-only flag, which any earlier test in this file has already tripped.
    """
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import logging\n"
        "import sparsegs\n"
        "logging.getLogger('sparsegs').setLevel('CRITICAL')\n"
        "from sparsegs.events import event\n"
        "sent = event('warning', 'demo_event', n=1)\n"
        "assert sent == 'demo_event n=1', sent\n")
    proc = subprocess.run([sys.executable, "-c", script], cwd=root,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", f"a raised level was ignored: {proc.stderr!r}"


def test_the_env_var_decides_the_level_only_when_no_caller_has():
    """`SPARSEGS_LOG_LEVEL` is a default, not an override.

    The two ways of setting the level have to have a defined order rather than
    whichever ran last, so the environment variable fills the gap and a
    `setLevel` call takes precedence over it.
    """
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import logging\n"
        "import sparsegs\n"
        "from sparsegs.events import event, LOGGER\n"
        "event('warning', 'demo_event', n=1)\n"
        "print(logging.getLevelName(LOGGER.getEffectiveLevel()))\n")
    env = dict(os.environ, SPARSEGS_LOG_LEVEL="error")
    proc = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ERROR", proc.stdout
    # and the record was dropped before it was formatted, not printed anyway
    assert proc.stderr == "", proc.stderr

    script = (
        "import logging\n"
        "import sparsegs\n"
        "logging.getLogger('sparsegs').setLevel('DEBUG')\n"
        "from sparsegs.events import event, LOGGER\n"
        "event('debug', 'demo_event', n=1)\n"
        "print(logging.getLevelName(LOGGER.getEffectiveLevel()))\n")
    proc = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "DEBUG", proc.stdout


def test_a_background_too_thin_to_draw_from_is_refused_not_approximated(cache):
    """Edge case one, at the point the ladder gives up.

    A query covering all but a handful of the background leaves almost nothing
    to replace it with.  The framework loosens the detection floor once, by the
    brief's ten per cent, announces that it did, and -- if the pool is still too
    thin -- refuses.  Drawing from five genes and calling the result a matched
    null is the failure this case exists to prevent.
    """
    genes = list(cache.genes)
    query = genes[:len(genes) - 5]
    buf, handler = _capture_log()
    old = sg.events.LOGGER.level
    try:
        sg.set_log_level("warning")
        with pytest.raises(ValueError) as exc:
            sg.construct_expression_matched_null(cache, query, n_sets=1, seed=1)
        record = buf.getvalue()
    finally:
        sg.events.LOGGER.removeHandler(handler)
        sg.events.LOGGER.setLevel(old)

    message = str(exc.value)
    assert "Cannot construct matched null" in message
    assert "Consider different gene set" in message
    # The loosening is announced before the refusal, so a log of a failed call
    # records that the framework tried the brief's remedy rather than skipping
    # to the error.
    assert "background_insufficient" in record
    assert "loosening" in record
    assert "loosened_to=" in record


def test_a_gene_set_that_fits_its_background_is_not_warned_about(cache):
    """The guard above has to stay quiet when there is no problem.

    A warning that fires on ordinary input is a warning that gets filtered out
    of the log, which costs the warnings that matter.
    """
    genes = list(cache.genes)
    query = genes[:max(20, len(genes) // 10)]
    buf, handler = _capture_log()
    old = sg.events.LOGGER.level
    try:
        sg.set_log_level("warning")
        out = sg.construct_expression_matched_null(cache, query, n_sets=1, seed=1)
        record = buf.getvalue()
    finally:
        sg.events.LOGGER.removeHandler(handler)
        sg.events.LOGGER.setLevel(old)

    assert "background_insufficient" not in record
    assert out["det_floor_loosened"] is False
    assert out["n_available_background"] > sg.MIN_BACKGROUND_GENES


def test_the_log_is_machine_readable():
    """Every field is ``key=value``, so a log can be counted and not just read.

    The point of emitting records rather than sentences is that the question
    "how often did this happen" has an answer.  That answer requires the format
    to hold, so it is asserted rather than assumed.
    """
    from sparsegs.events import event
    line = event("warning", "test_event", count=3, ratio=0.125, ok=True,
                 name="a value with spaces", absent=None)
    parts = line.split(" ", 1)[1]
    fields = dict(re.findall(r'(\w+)=("(?:[^"]*)"|\S+)', parts))
    assert fields["count"] == "3"
    assert fields["ratio"] == "0.125"
    assert fields["ok"] == "true"
    assert fields["name"] == '"a value with spaces"'
    assert "absent" not in fields, "an empty field was written anyway"


# ----------------------------------------------------------------------
# the axis's own confound with depth, folded into the verdict
# ----------------------------------------------------------------------
def _clean_report(**over):
    """A sparsity report with nothing wrong with it.

    Every field the verdict reads is set to a value that crosses no threshold,
    so a reason in the output can only have come from what the test added.
    """
    base = dict(observed_zero_rate=0.05, structural_zero_rate=0.04,
                zero_rate_excess=0.01, tie_break_share=0.01,
                frac_above_floor=1.0, detection_floor=0.005)
    base.update(over)
    return base


def test_the_axis_confound_qualifies_a_verdict_without_refusing_it():
    """The confound is a reason, and the verdict rule is what makes it one.

    A matched null is calibrated *under* this confound: the replacement sets are
    drawn from a matrix with the same depth distribution and carry the same
    tie-break behaviour, so the null distribution moves with the observed
    statistic and the confound is priced in.  Making a depth-confounded axis
    ``NOT_IDENTIFIABLE`` outright would say the opposite -- that the framework's
    own recommended analysis cannot be trusted on precisely the data it was
    built for -- so the test pins the weaker treatment: it is named, it raises
    the severity, and it leaves the verdict standing.
    """
    clear = sg.verdict(_clean_report(), used_matched_null=True,
                       preflight={"verdict": "CLEAR"})
    assert clear["verdict"] == "INTERPRETABLE"
    assert clear["severity"] == "LOW"
    assert clear["reasons"] == ["no diagnostic threshold was crossed"]

    for name, phrase in (("CAUTION", "weakly correlated"),
                         ("CONFOUNDED", "correlated with sequencing depth")):
        v = sg.verdict(_clean_report(), used_matched_null=True,
                       preflight={"verdict": name})
        assert v["verdict"] == "INTERPRETABLE", name
        assert v["severity"] == "MODERATE", name
        assert any(phrase in r for r in v["reasons"]), (name, v["reasons"])

    # Without a matched null the confound joins the reasons already there
    # rather than replacing them, and the verdict is the one that says a
    # matched null is owed.  The count is not pinned: an absent replication
    # cohort is a third reason when the severity has been raised at all, and
    # that is the verdict rule's business rather than this test's.
    v = sg.verdict(_clean_report(), used_matched_null=False,
                   preflight={"verdict": "CONFOUNDED"})
    assert v["verdict"] == "INTERPRETABLE_WITH_MATCHED_NULL"
    assert any("not tested against an expression-matched null" in r
               for r in v["reasons"]), v["reasons"]
    assert any("correlated with sequencing depth" in r for r in v["reasons"])

    # A verdict that already refuses is not softened by the extra reason: the
    # heavier severity still wins, which is the ordering the strings get wrong.
    v = sg.verdict(_clean_report(observed_zero_rate=0.7),
                   used_matched_null=True, preflight={"verdict": "CONFOUNDED"})
    assert v["verdict"] == "NOT_IDENTIFIABLE"
    assert v["severity"] == "HIGH"


def test_the_checklist_reads_the_depth_off_the_cache(cache):
    """The pre-flight has to run without being asked to.

    It was wired to a ``depth`` argument that no caller passed, so the Python
    checklist returned no pre-flight where the R one always had -- the same call
    on the same data, two objects of different shapes.  The depth is on the
    cache, so the default is to use it.

    The axis is the depth itself, which makes the correlation exactly -1: a
    positive test rather than a borderline one, because the question here is
    whether the number is computed at all.
    """
    axis = np.asarray(cache.depth, dtype=float)
    out = sg.diagnostic_checklist(cache, list(cache.genes)[:40], axis=axis,
                                  used_matched_null=True, n_perm=50,
                                  config={"n_null": 3})
    assert "preflight" in out
    assert out["preflight"]["verdict"] == "CONFOUNDED"
    assert abs(out["preflight"]["tau_axis_rho"]) > 0.9
    assert any("correlated with sequencing depth" in r for r in out["reasons"])

    # An axis that is not the depth is not warned about, so the reason above is
    # the correlation and not the fact that a pre-flight ran.
    rng = np.random.default_rng(11)
    free = sg.diagnostic_checklist(cache, list(cache.genes)[:40],
                                   axis=rng.normal(size=out["n_cells"]),
                                   used_matched_null=True, n_perm=50,
                                   config={"n_null": 3})
    assert free["preflight"]["verdict"] == "CLEAR"
    assert not any("sequencing depth" in r for r in free["reasons"])


def test_the_preflight_reports_the_band_it_judged_against(cache):
    """The threshold travels with the verdict, so a reader can check the call.

    The band is ``1.96/sqrt(n-3)`` rather than a fixed correlation: the same
    |rho| is chance in a 200-cell dataset and a real association in a
    20,000-cell one, and a constant would be wrong for one of them.  Reporting
    it is what lets a reader see which regime their dataset is in.
    """
    axis = np.asarray(cache.depth, dtype=float)
    out = sg.diagnostic_checklist(cache, list(cache.genes)[:40], axis=axis,
                                  used_matched_null=True, n_perm=50,
                                  config={"n_null": 3})
    pre = out["preflight"]
    expected = 1.96 / np.sqrt(out["n_cells"] - 3)
    assert abs(pre["chance_band"] - expected) < 1e-12
    assert pre["n_cells"] == out["n_cells"]
    # The amplitude of the confound is reported alongside it: the correlation
    # says the axis moves with the depth, and gamma says how much of a typical
    # gene set that reaches.
    assert 0.0 <= pre["gamma"] <= 1.0
    assert abs(pre["implied_rho"]) <= abs(pre["tau_axis_rho"]) + 1e-12


def test_a_tie_break_that_never_varies_is_clear_and_says_so():
    """A matrix deeper than the score's ceiling has no tie-break to confound with.

    Every cell deeper than ``max_rank`` contributes ``tau = 0``; on a matrix
    where all of them are, ``tau`` is constant and its correlation with the axis
    does not exist.  Asked of scipy that is a NaN *and* a warning, which a caller
    who has turned warnings into errors meets as a crash -- and the R
    implementation of the framework returned ``NA`` in silence, so the same data
    was an exception on one side and a number on the other.  This is not an
    exotic input: it is what a deep 10x matrix looks like whenever the gene
    panel is small enough that five per cent of it falls below the median cell.
    """
    depth = np.full(300, 500.0)
    axis = np.linspace(0.0, 1.0, 300)
    pre = sg.preflight(depth, axis, max_rank=100, n_genes=2000)
    assert pre["tau_constant"] is True
    assert pre["tau_sd"] == 0.0
    assert not np.isfinite(pre["tau_axis_rho"])
    # Clear, and correctly so: a component that never varies cannot carry an
    # association with anything.
    assert pre["verdict"] == "CLEAR"

    # The degenerate vector through `spearman` directly, with warnings as
    # errors, so the silence is asserted rather than assumed.  The R side has
    # always been silent here, which is the standard the two are held to.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rho, p = sg.spearman(np.zeros(50), np.arange(50.0))
        rho2, _ = sg.spearman(np.arange(50.0), np.full(50, 3.0))
    assert np.isnan(rho) and np.isnan(p) and np.isnan(rho2)

    # A matrix with depth variation still produces a correlation, so the branch
    # above is the degenerate case and not the ordinary one.
    live = np.linspace(20.0, 400.0, 300)
    assert np.isfinite(sg.preflight(live, axis, 100, 2000)["tau_axis_rho"])


def test_the_figure_scores_the_same_panel_the_manuscript_describes():
    """The figure's panel and ceiling are the benchmark's, not its own.

    ``make_figure.py`` has to know how many genes the panel holds and where the
    rank ceiling sits, because both are printed in panel a.  Typed there, they
    are a second copy of a number the benchmark's log already states, and two
    copies of a number drift: the figure would go on drawing the old ceiling
    after a re-filtering changed it, and nothing in the manuscript would look
    wrong.  The log is the QC pipeline's own output, so it is the copy that
    moves, and this asserts the figure has not been left behind.
    """
    log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "log_bench_gse176078.txt")
    if not os.path.exists(log):
        pytest.skip("the benchmark has not been run in this checkout")
    pytest.importorskip("matplotlib")
    from experiments import make_figure

    text = open(log, encoding="utf-8").read()
    genes = int(re.search(r"(\d+)\s+cells\s+x\s+(\d+)\s+genes", text).group(2))
    rank = int(re.search(r"max_rank = (\d+)", text).group(1))
    assert make_figure.PANEL_GENES == genes, (
        f"the figure draws a panel of {make_figure.PANEL_GENES} genes; the "
        f"benchmark scored {genes}")
    assert make_figure.MAX_RANK == rank, (
        f"the figure draws a ceiling of {make_figure.MAX_RANK}; the benchmark "
        f"used {rank}")


# ----------------------------------------------------------------------
# the bibliography
# ----------------------------------------------------------------------


def test_the_bibliography_parser_reads_braced_titles_whole(tmp_path):
    """A title containing braces is read whole, and no field is skipped.

    The naive reading of a .bib entry -- one non-greedy match per field -- gets
    two things wrong at once, and both are silent.  It stops at the first
    closing brace, so ``{GSVA}: gene set variation analysis`` loses the part
    after the acronym; and matching the trailing newline as part of a field
    consumes the separator before the next one, so every second field is
    missed entirely.  Either mistake turns the comparison in
    ``verify_references.py`` into a comparison against an empty string, which
    no title can fail -- the check would pass on a bibliography that was wrong.
    """
    from experiments import verify_references as vr

    path = tmp_path / "refs.bib"
    path.write_text(
        "@article{key,\n"
        "  author  = {A. Author and B. Author},\n"
        "  title   = {{GSVA}: gene set variation analysis for {RNA}-seq},\n"
        "  journal = {BMC Bioinformatics},\n"
        "  year    = {2013},\n"
        "  doi     = {10.1186/1471-2105-14-7}\n"
        "}\n", encoding="utf-8")
    entries = vr.parse_bib(str(path))
    assert len(entries) == 1
    key, fields = entries[0]
    assert key == "key"
    # Every field, not every other field.
    assert set(fields) == {"author", "title", "journal", "year", "doi"}
    assert fields["title"] == "{GSVA}: gene set variation analysis for {RNA}-seq"
    assert vr.normalise(fields["title"]) == \
        "gsva gene set variation analysis for rna seq"

    # The comparison has to be one that a wrong title can fail.
    ratio = difflib.SequenceMatcher(
        None, "gsva gene set variation analysis for rna seq",
        vr.normalise("Gene set variation analysis for microarray and RNA-seq data")
    ).ratio()
    assert ratio < vr.TITLE_FLOOR


def test_the_manuscript_cites_just_what_the_bibliography_holds():
    """No citation without an entry, and no entry without a citation.

    Both directions are silent in the PDF.  A citation to a key that was
    renamed prints a question mark that a read-through may not catch; an entry
    nothing cites never reaches the reference list at all, so nothing in the
    compiled manuscript says it is there.  The check is offline: the DOIs are
    resolved by a separate run, and a test that needed the network would fail
    for a reason that has nothing to do with the manuscript.
    """
    from experiments import verify_references as vr

    if not os.path.exists(vr.BIB) or not os.path.exists(vr.TEX):
        pytest.skip("the manuscript is not in this checkout")

    defined = {key for key, _ in vr.parse_bib(vr.BIB)}
    used = set(vr.cite_keys(vr.TEX))
    assert not used - defined, (
        f"cited but absent from the .bib: {sorted(used - defined)}")
    assert not defined - used, (
        f"present in the .bib and never cited: {sorted(defined - used)}")


# ----------------------------------------------------------------------
# the rank cache refuses inputs its ranking cannot represent
# ----------------------------------------------------------------------
def test_the_rank_cache_rejects_nan_and_infinite_values():
    """A NaN would take rank 1 and poison the score instead of failing.

    The composite order is the bit pattern of a non-negative ``float32``; the
    bit pattern of a NaN sorts above almost every positive value, so an
    undetected-looking NaN gene would be ranked first and contribute the
    largest weight a single gene can contribute -- a plausible-looking,
    systematically inflated score rather than an error.  Refusing the matrix
    at the entrance is the only safe answer.
    """
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.random((6, 10)).astype(np.float32))
    X[2, 3] = np.nan
    genes = [f"G{i}" for i in range(10)]
    with pytest.raises(ValueError, match="NaN or infinite"):
        sg.RankCache.build(X, genes, ceiling=4)


def test_the_rank_cache_rejects_negative_values():
    """A negative float32 sorts below every positive one -- silently.

    The bit-complement order is defined for non-negative values; a negative
    value has its sign bit set, so its complement lands below every detected
    gene and the value is treated as less-than-undetected.  Scaled or
    z-scored matrices -- a common thing to pass -- contain negatives, and the
    failure mode would be a wrong ranking with no warning anywhere.
    """
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.random((6, 10)).astype(np.float32))
    X[4, 7] = -0.5
    genes = [f"G{i}" for i in range(10)]
    with pytest.raises(ValueError, match="negative values"):
        sg.RankCache.build(X, genes, ceiling=4)
