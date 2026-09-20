"""Turn the simulation grid into the framework's operating characteristics.

Four questions, in the order a reader will ask them:

1. **Is the conventional null valid?**  Under ``effect = 0`` every rejection is a
   false positive with respect to the claim being made.  Grid A varies the
   confounding between depth and the programme, the sparsity and the background
   co-expression; grid B varies the regime (``max_rank`` over median detected);
   grid D the set size.  All three hold donor structure out, so that what they
   measure is the tie-break channel and not the donor channel.

2. **Is the matched null valid where the conventional one is not?**  Same grids,
   three matched families.  The claim is that their false-positive rate stays at
   the nominal level as the conventional rate climbs.

3. **What does validity cost in power?**  Grid C plants a real effect.  A null
   that never rejects is trivially calibrated, so the power grid is what makes
   the calibration claim meaningful.

4. **How much of an apparent association is the donors?**  Grid E varies only the
   number of donors, with no confounder and no effect.  It is kept separate
   because it is a different failure with a different remedy: the tie-break
   channel is a property of the score and the depth, while donor structure is a
   property of the design, and a cell-level test of a donor-level claim is
   anticonservative however well the score behaves.  Pooling the two would let
   each be reported as the other.

Accuracy note
-------------
With 60 replicates a rate near 0.05 carries a standard error of about 0.028, so
a single cell of the grid cannot resolve 0.05 from 0.10.  The marginal summaries
pool over the factors that should not matter for the quantity being estimated,
and every rate is reported with a Wilson interval.  Where a headline number
needs to be tighter than the grid can make it, it is re-estimated with targeted
replicates rather than read off a single cell.

Usage
-----
    python experiments/analyse_grid.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

RESULTS = os.path.join(PKG, "results")
ALPHA = 0.05

#: The analyses under test, in the order they are reported.  The first four are
#: conventional; the last three are the matched families this package supplies.
CONVENTIONAL = [("naive", "naive_p"), ("permutation", "perm_p"),
                ("cutpoint_median", "cut_median_p"),
                ("cutpoint_optimal", "cut_opt_p")]
MATCHED = [("random", "random_p"), ("expression", "expression_p"),
           ("codetection", "codetection_p")]
ALL_TESTS = CONVENTIONAL + MATCHED


def wilson(k, n, z=1.96):
    """Wilson score interval; behaves at the boundaries where Wald does not."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def expected_runs(name):
    """How many runs grid ``name`` is defined to produce, or None if unknown.

    The grid is written to with ``open(path, "a")`` as the runs come in, so a
    sweep that is still running leaves a file that is a perfectly valid prefix
    of the finished one.  Every summary computed from it is then a real number,
    correctly calculated, describing two-thirds of an experiment -- and nothing
    in the arithmetic can notice.  The count the grid defines is the only thing
    that can, so it is read from the grid definition rather than restated here.
    """
    try:
        from run_grid import GRIDS
    except Exception:
        return None
    entry = GRIDS.get(name)
    if entry is None:
        return None
    fn, reps = entry
    return len(list(fn())) * int(reps)


def load(name, allow_partial=False):
    path = os.path.join(RESULTS, f"grid_{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run run_grid.py first")
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # The grid may still be appending to this file; a torn last line
                # is expected, not a corruption.
                continue
    df = pd.DataFrame(rows)
    expected = None if allow_partial else expected_runs(name)
    if expected is not None and len(df) < expected:
        raise SystemExit(
            f"[{name}] {path} holds {len(df)} runs of the {expected} the grid "
            f"defines.  The sweep is still running, so every rate below would be "
            f"computed on part of the experiment and would look entirely normal. "
            f"Wait for run_grid.py to finish, or pass --allow-partial to look at "
            f"the partial data on purpose.")
    err = df["error"] if "error" in df else None
    if err is not None and err.notna().any():
        print(f"[{name}] {int(err.notna().sum())} runs carry an error and are "
              f"excluded")
        df = df[df["error"].isna()]
    return df


def reject_table(df, tests=ALL_TESTS):
    """Rejection rate per test, with a Wilson interval, over whatever is passed."""
    out = {}
    n = len(df)
    for label, col in tests:
        if col not in df:
            continue
        k = int((df[col] < ALPHA).sum())
        lo, hi = wilson(k, n)
        out[label] = dict(k=k, n=n, rate=k / n if n else np.nan,
                          ci_low=lo, ci_high=hi)
    return out


#: Every matched test in each of the three tails.  The framework reports the
#: two-sided value; the directional ones come from the same null draws, so the
#: reader of a table built from these pairs can see exactly which question each
#: number answers, and that no conclusion in the paper turns on the choice.
def tail_tests():
    return [(f"{kind} ({tail})", f"{kind}_p{sfx}")
            for kind in ("random", "expression", "codetection")
            for tail, sfx in (("two-sided", ""), ("lower", "_lower"),
                              ("upper", "_upper"))]


def tail_robustness(df):
    """Rejection rate of every matched family, read in all three tails.

    On a grid with no planted effect these are false-positive rates; on grid C
    they are power.  Both are needed: the tail convention was wrong once, and
    the symptom in the power grid -- a matched null that could never reject --
    is not one a false-positive-rate table would have shown.
    """
    table = reject_table(df, tail_tests())
    return pd.DataFrame([
        dict(test=label, k=v["k"], n=v["n"], rate=v["rate"],
             ci_low=v["ci_low"], ci_high=v["ci_high"])
        for label, v in table.items()])


def grouping(df, factor, max_levels=12):
    """The values to group a grid factor by, and a printable label for them.

    Several grid factors are continuous -- detection rate, co-expression,
    median genes per cell.  Grouping those by their raw value gives one group
    per run and a marginals table whose rows have n = 1, which reads like a
    result and is not one.  A continuous factor is therefore binned into
    quantiles; a factor with few enough distinct values is used as it is.
    """
    col = df[factor]
    if not pd.api.types.is_numeric_dtype(col) or col.nunique() <= max_levels:
        return col, col
    binned = pd.qcut(col, max_levels, duplicates="drop")
    return binned, binned


def top_stratum(df, factor, quantile=0.75):
    """The runs at the top of a continuous factor, as a quantile band.

    ``df[factor] == df[factor].max()`` selects the single largest value, which
    on a continuous factor is a handful of runs and on a grid with three seeds
    is three.  The claim "the effect is largest where the confounder is
    strongest" is a claim about a region, so the region is a band.
    """
    col = df[factor]
    if not pd.api.types.is_numeric_dtype(col) or col.nunique() <= 4:
        return df[col == col.max()], col.max()
    cut = float(col.quantile(quantile))
    return df[col >= cut], cut


def marginal(df, factor, tests=ALL_TESTS):
    """Rejection rate by one grid factor, pooling the others."""
    rows = []
    levels, _ = grouping(df, factor)
    for level, sub in df.groupby(levels, observed=True):
        row = {factor: level, "n": len(sub)}
        for label, col in tests:
            if col in sub:
                row[label] = float((sub[col] < ALPHA).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(factor)


def interaction(df, row_factor, col_factor, test="naive", tests=ALL_TESTS):
    """Rejection rate of one test over a two-way grid.

    The marginals are not enough here.  Two factors that push the rejection rate
    in opposite directions average into a flat marginal, and the mechanism this
    project is about is exactly of that kind: the tie-break's share of the score
    rises as the set gets sparser, while the score's variance falls, so the
    false-positive rate is not monotone in either and only the joint table shows
    where it peaks.
    """
    col = dict(tests).get(test, test)
    if col not in df:
        return None
    rows, _ = grouping(df, row_factor)
    cols, _ = grouping(df, col_factor)
    # The rate is the share of runs whose P value falls below alpha, not the mean
    # P value: those are different numbers and only the first is a rejection rate.
    work = df.assign(_reject=(df[col] < ALPHA).astype(float))
    piv = work.groupby([rows, cols], observed=True)["_reject"].agg(
        ["mean", "size"]).reset_index()
    piv.columns = [row_factor, col_factor, f"{test}_rate", "n"]
    return piv


def format_interaction(piv, test="naive"):
    """A two-way table as a printable grid, blanks where a cell is empty."""
    rate = piv.pivot(index=piv.columns[0], columns=piv.columns[1],
                     values=f"{test}_rate")
    count = piv.pivot(index=piv.columns[0], columns=piv.columns[1],
                      values="n")
    lines = [f"{test} rejection rate (n in brackets)"]
    header = "  " + " " * 26 + "".join(f"{str(c):>16s}" for c in rate.columns)
    lines.append(header)
    for idx, vals in rate.iterrows():
        cells = []
        for c in rate.columns:
            v, k = vals[c], count.loc[idx, c]
            cells.append(f"{'--':>16s}" if not np.isfinite(v)
                         else f"{v:>10.3f} ({int(k):>3d})")
        lines.append(f"  {str(idx):26s}" + "".join(cells))
    return "\n".join(lines)


def channel_summary(df, tests=ALL_TESTS):
    """Rejection rates split by whether the tie-break channel is open at all.

    ``depth_ratio = max_rank / median detected`` decides whether a cell can fill
    its top block with genes it actually detected.  Below one, the tie-break
    never reaches the ceiling and there is no channel; the diagnostics report a
    share of zero, correctly, and the naive test has nothing to be fooled by.
    Pooling the two regimes averages a real effect with its absence, so the grid
    is split before it is summarised -- and the split is the framework's own
    claim about when the problem exists.
    """
    if "depth_ratio" not in df:
        return None
    open_ = df[df["depth_ratio"] > 1.0]
    shut = df[df["depth_ratio"] <= 1.0]
    return dict(
        threshold=1.0,
        open=dict(n=int(len(open_)), **reject_table(open_, tests)),
        closed=dict(n=int(len(shut)), **reject_table(shut, tests)))


def share_response(df, test="naive", n_bins=6):
    """Rejection rate against the diagnostic, in quantile bins of its own value.

    The framework's claim is not "sparse data inflates P values" but "this
    quantity predicts by how much".  That is a claim about a response curve, and
    this is the curve: bin the runs by the tie-break share the diagnostic
    reported, and count rejections in each bin.
    """
    col = dict(ALL_TESTS).get(test, test)
    if "tie_break_share" not in df or col not in df:
        return None
    work = df[df["tie_break_share"].notna()].copy()
    if work["tie_break_share"].nunique() < 2:
        return None
    work["_bin"] = pd.qcut(work["tie_break_share"], n_bins, duplicates="drop")
    rows = []
    for level, sub in work.groupby("_bin", observed=True):
        k = int((sub[col] < ALPHA).sum())
        lo, hi = wilson(k, len(sub))
        rows.append(dict(tie_break_share_low=float(level.left),
                         tie_break_share_high=float(level.right),
                         n=len(sub), rate=k / len(sub),
                         ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def implied_rho(df):
    """The spurious association a score derives from depth, as a product.

    Two correlations, neither of which uses the outcome: how strongly the score
    tracks sequencing depth, and how strongly the axis under test tracks it.
    Where a score has no depth dependence, or the axis has none, the product is
    zero and no amount of sparsity can manufacture an association between them.

    This is the classical attenuation-of-confound identity.  It is the quantity
    ``preflight`` estimates structurally from the detection probabilities; here
    it is measured from the score itself, which is cheaper and makes no model
    assumption about the score.
    """
    need = ("score_depth_rho", "programme_depth_rho")
    if any(c not in df for c in need):
        return None
    return df["score_depth_rho"] * df["programme_depth_rho"]


def implied_response(df, test="naive", n_bins=6):
    """Rejection rate against ``implied_rho``, in quantile bins of its magnitude.

    The sign of the product is not what the test responds to -- the test is
    two-sided, so an axis that anti-correlates with depth fails it just as an
    axis that correlates does -- so the bins are taken on the magnitude.
    """
    col = dict(ALL_TESTS).get(test, test)
    imp = implied_rho(df)
    if imp is None or col not in df:
        return None
    work = df.assign(_implied=imp).dropna(subset=["_implied"])
    if work["_implied"].abs().nunique() < 2:
        return None
    work["_bin"] = pd.qcut(work["_implied"].abs(), n_bins, duplicates="drop")
    rows = []
    for level, sub in work.groupby("_bin", observed=True):
        k = int((sub[col] < ALPHA).sum())
        lo, hi = wilson(k, len(sub))
        rows.append(dict(implied_low=float(level.left),
                         implied_high=float(level.right),
                         implied_median=float(sub["_implied"].abs().median()),
                         n=len(sub), rate=k / len(sub),
                         ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def implied_calibration(df, column="observed_rho"):
    """How well the pre-flight product recovers the association that appears.

    The response curve above shows the product tracks the *failure rate*; this
    shows it tracks the *association itself*, which is the stronger statement
    and the one the framework's users act on.  A slope near one means the
    product can be read as the spurious association directly.
    """
    imp = implied_rho(df)
    if imp is None or column not in df:
        return None
    work = df.assign(_implied=imp).dropna(subset=["_implied", column])
    if len(work) < 3 or work["_implied"].std() == 0:
        return None
    x, y = work["_implied"].to_numpy(), work[column].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    return dict(n=len(work),
                pearson=float(np.corrcoef(x, y)[0, 1]),
                spearman=float(stats.spearmanr(x, y)[0]),
                slope=float(slope), intercept=float(intercept),
                mean_implied=float(x.mean()), mean_observed=float(y.mean()),
                realised=float(np.mean(y - x)))


def trend_test(df, test="naive", column="implied", n_bins=6):
    """Cochran-Armitage trend in rejection rate across ordered bins of ``column``.

    A response curve read off a table can look monotone by eye while resting on
    a hundred runs per bin; this is the test for "the rate rises with the
    diagnostic" against "it does not".  It is the right test here because the
    bins are ordered and the contrast is one-sided -- the framework predicts the
    rate rises with the pre-flight quantity and is refuted if it does not.

    ``column`` is ``"implied"`` for the pre-flight product or ``"tie_break_share"``
    for the diagnostic the first version of this study proposed.
    """
    if column == "implied":
        values = implied_rho(df)
    elif column in df:
        values = df[column]
    else:
        return None
    if values is None:
        return None
    col = dict(ALL_TESTS).get(test, test)
    if col not in df:
        return None
    work = df.assign(_v=values).dropna(subset=["_v", col])
    if len(work) < 6 or work["_v"].abs().nunique() < 2:
        return None
    work["_bin"] = pd.qcut(work["_v"].abs(), n_bins, duplicates="drop")
    n = work.groupby("_bin", observed=True)[col].size().to_numpy(float)
    x = work.assign(_r=work[col] < ALPHA).groupby("_bin", observed=True)["_r"].sum().to_numpy(float)
    t = np.arange(1, len(n) + 1, dtype=float)
    N, X = n.sum(), x.sum()
    p = X / N
    num = float(np.sum(t * (x - n * p)))
    var = p * (1 - p) * (float(np.sum(n * t**2)) - float(np.sum(n * t)) ** 2 / N)
    if var <= 0:
        return None
    z = num / np.sqrt(var)
    return dict(z=float(z), p=float(2 * stats.norm.sf(abs(z))), n=int(N),
                n_bins=int(len(n)),
                rates=[float(a / b) for a, b in zip(x, n)])


def logistic_irls(X, y, max_iter=200, tol=1e-11):
    """Logistic regression by iteratively reweighted least squares.

    Written out rather than imported because the analysis has to run in a frozen
    environment and a two-predictor fit is ten lines.  Returns the coefficients,
    their standard errors and the Wald z statistics.
    """
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        root = np.sqrt(w)
        step = np.linalg.lstsq(X * root[:, None], (eta + (y - p) / w) * root,
                               rcond=None)[0]
        if np.max(np.abs(step - beta)) < tol:
            beta = step
            break
        beta = step
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-10, None)
    cov = np.linalg.pinv((X * w[:, None]).T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, beta / np.where(se > 0, se, np.nan)


def joint_logistic(df, test="naive"):
    """Does the refuted diagnostic add anything once the product is in the model?

    Binning is a coarse way to compare two candidate pre-flight quantities; the
    direct one is to put both in the same model of whether the run rejected, each
    standardised so the coefficients are per standard deviation and comparable.
    A quantity that earns its place has a positive coefficient of its own, and
    keeps it when the other is present.
    """
    col = dict(ALL_TESTS).get(test, test)
    if col not in df:
        return None
    imp = implied_rho(df)
    if imp is None:
        return None
    work = pd.DataFrame({"imp": imp.abs(), "share": df["tie_break_share"],
                         "reject": (df[col] < ALPHA).astype(float)}).dropna()
    if len(work) < 20 or work["reject"].nunique() < 2:
        return None
    z = lambda v: (v - v.mean()) / (v.std() or 1.0)
    rows = []
    models = [("product alone", {"product": z(work["imp"])}),
              ("share alone", {"share": z(work["share"])}),
              ("product and share", {"product": z(work["imp"]),
                                     "share": z(work["share"])})]
    for label, terms in models:
        X = pd.DataFrame(terms)
        beta, se, zs = logistic_irls(X, work["reject"])
        for name, b, s, zz in zip(["intercept"] + list(X.columns), beta, se, zs):
            p = float(2 * stats.norm.sf(abs(zz))) if np.isfinite(zz) else np.nan
            rows.append(dict(test=test, model=label, term=name, n=int(len(work)),
                             coef=float(b), se=float(s), z=float(zz), p=p))
    return pd.DataFrame(rows)


def share_ceiling_correlation(df):
    """How nearly the tie-break share is a relabelling of the ceiling ratio.

    Where this is high, the share's marginal association with a rejection is the
    ceiling ratio's association wearing the share's name -- which is why the
    sign of that association is not a property of the share, and why the share
    cannot be a pre-flight quantity: a quantity whose meaning changes with what
    else is being varied cannot be read off one analysis in isolation.
    """
    if not {"tie_break_share", "depth_ratio"} <= set(df.columns):
        return None
    r = stats.spearmanr(df["tie_break_share"], df["depth_ratio"])
    return dict(rho=float(r.statistic), p=float(r.pvalue), n=int(len(df)))


def format_rate(entry):
    return (f"{entry['rate']:.3f} "
            f"[{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]")


def main(allow_partial=False):
    report = {}
    frames = {}
    for name in ("A_calibration", "B_regime", "C_power", "D_size",
                 "E_donor"):
        path = os.path.join(RESULTS, f"grid_{name}.jsonl")
        if not os.path.exists(path):
            print(f"\n[{name}] not present, skipped")
            continue
        df = load(name, allow_partial=allow_partial)
        frames[name] = df
        print(f"\n{'=' * 72}\n{name}: {len(df)} runs\n{'=' * 72}")

        # a single run per configuration is not enough to say anything, so the
        # first thing printed is always the pooled rate.
        overall = reject_table(df)
        print("\npooled rejection rate at alpha = 0.05")
        for label, _ in ALL_TESTS:
            if label in overall:
                print(f"  {label:18s} {format_rate(overall[label])}")

        is_null = (df["effect"] == 0).all() if "effect" in df else True
        if not is_null:
            print("\n  (effect > 0: these rates are power, not false-positive rate)")

        report[name] = dict(n_runs=int(len(df)), overall=overall,
                            null_grid=bool(is_null))

        # The tail is a convention, and it was set wrongly here once: the
        # matched nulls were tested in the lower tail while the conventional
        # analyses they are compared against reject in either direction.  The
        # same null draws are therefore read all three ways and written out, so
        # that the rates the paper quotes and the rates a reader would get from
        # the directional values are on the record together.
        tails = tail_robustness(df)
        if len(tails):
            kind = "power" if not is_null else "false-positive rate"
            print(f"\nmatched tests in all three tails ({kind})")
            print(tails.to_string(index=False,
                                  float_format=lambda v: f"{v:.3f}"))
            tails.to_csv(os.path.join(
                RESULTS, f"grid_{name}_tail_robustness.csv"), index=False)
            report[name]["tail_robustness"] = tails.to_dict(orient="list")

        # per-factor marginals
        for factor in ("depth_programme_loading", "detection", "coexpr",
                       "median_detected", "effect", "n_target",
                       "n_samples", "depth_ratio", "tie_break_share"):
            if factor not in df or df[factor].nunique() < 2:
                continue
            tab = marginal(df, factor)
            print(f"\nby {factor}")
            print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            tab.to_csv(os.path.join(
                RESULTS, f"grid_{name}_by_{factor}.csv"), index=False)
            report[name][f"by_{factor}"] = tab.to_dict(orient="list")

        # The two factors that move the rejection rate in opposite directions,
        # so that the marginal does not hide where the rate actually peaks.
        for row_f, col_f in (("depth_programme_loading", "detection"),
                             ("depth_programme_loading", "coexpr"),
                             ("depth_programme_loading", "n_samples")):
            if row_f not in df or col_f not in df:
                continue
            if df[row_f].nunique() < 2 or df[col_f].nunique() < 2:
                continue
            for test in ("naive", "cutpoint_optimal", "codetection"):
                piv = interaction(df, row_f, col_f, test=test)
                if piv is None:
                    continue
                print(f"\n{row_f} x {col_f}")
                print(format_interaction(piv, test=test))
                piv.to_csv(os.path.join(
                    RESULTS, f"grid_{name}_{row_f}_x_{col_f}_{test}.csv"),
                    index=False)
                report[name][f"{row_f}_x_{col_f}_{test}"] = piv.to_dict(
                    orient="list")

        # Whether the channel is open at all, and whether the diagnostic that
        # claims to measure it actually tracks the failure.
        cs = channel_summary(df)
        if cs is not None:
            print("\nby whether the tie-break channel is open "
                  "(depth_ratio > 1)")
            for regime in ("open", "closed"):
                sub = cs[regime]
                print(f"  {regime:7s} n = {sub['n']:5d}  "
                      + "  ".join(f"{t} {sub[t]['rate']:.3f}"
                                  for t, _ in ALL_TESTS if t in sub))
            report[name]["channel"] = cs

        sr = share_response(df)
        if sr is not None:
            print("\nrejection against the reported tie-break share")
            print(sr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            sr.to_csv(os.path.join(RESULTS, f"grid_{name}_share_response.csv"),
                      index=False)
            report[name]["share_response"] = sr.to_dict(orient="list")

        # The response curve the framework actually rests on, with the
        # refuted one beside it so that a reader can see both were measured.
        rows = []
        for label, _ in ALL_TESTS:
            ir = implied_response(df, test=label)
            if ir is None:
                continue
            ir.insert(0, "test", label)
            rows.append(ir)
            tr = trend_test(df, test=label, column="implied")
            ts = trend_test(df, test=label, column="tie_break_share")
            if tr is not None:
                print(f"\n{label}: rejection against the pre-flight product "
                      f"(trend z = {tr['z']:+.2f}, P = {tr['p']:.2e})")
                print(ir.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
                report[name][f"implied_response_{label}"] = tr
            if ts is not None:
                mark = "rises" if ts["z"] > 0 else "FALLS"
                print(f"  for comparison, the same test against the tie-break "
                      f"share: z = {ts['z']:+.2f}, P = {ts['p']:.2e} ({mark})")
                report[name][f"share_trend_{label}"] = ts
        if rows:
            pd.concat(rows).to_csv(
                os.path.join(RESULTS, f"grid_{name}_implied_response.csv"),
                index=False)

        # The same comparison as a single model, so that "the share adds
        # nothing" rests on a coefficient rather than on a grid of rates.  Runs
        # with a planted effect are excluded: there a rejection is a detection.
        jl = joint_logistic(df[df["effect"] == 0] if "effect" in df else df)
        if jl is not None:
            print(f"\nrejection regressed on both pre-flight quantities "
                  f"(standardised)")
            print(jl.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
            jl.to_csv(os.path.join(RESULTS, f"grid_{name}_joint_logistic.csv"),
                      index=False)
            for term in ("product", "share"):
                hit = jl[(jl["model"] == "product and share")
                         & (jl["term"] == term)]
                if len(hit):
                    report[name][f"joint_logistic_{term}"] = {
                        "coef": float(hit["coef"].iloc[0]),
                        "z": float(hit["z"].iloc[0]),
                        "p": float(hit["p"].iloc[0])}

        sc = share_ceiling_correlation(df)
        if sc is not None:
            print(f"\ntie-break share against the ceiling ratio: "
                  f"rho = {sc['rho']:+.3f} (P = {sc['p']:.2e}, n = {sc['n']})")
            report[name]["share_ceiling"] = sc

    # ---- the summary the paper leads with ------------------------------
    if "A_calibration" in frames:
        a = frames["A_calibration"]
        print(f"\n{'=' * 72}\ncalibration summary\n{'=' * 72}")
        print("False-positive rate under effect = 0, pooled over the whole grid.\n"
              "A valid test at alpha = 0.05 sits at 0.05.\n")
        print(f"  {'test':18s} {'FPR':>7s}  {'95% CI':>16s}  {'vs 0.05':>9s}")
        for label, _ in ALL_TESTS:
            if label not in report["A_calibration"]["overall"]:
                continue
            e = report["A_calibration"]["overall"][label]
            verdict = ("calibrated" if e["ci_low"] <= ALPHA <= e["ci_high"]
                       else ("anti-conservative" if e["ci_low"] > ALPHA
                             else "conservative"))
            print(f"  {label:18s} {e['rate']:7.4f}  "
                  f"[{e['ci_low']:.4f}, {e['ci_high']:.4f}]  {verdict:>9s}")

        # the same, at the highest confounding, where the claim bites hardest.
        hi, cut = top_stratum(a, "depth_programme_loading")
        if len(hi) >= 20:
            print(f"\nAt the highest confounding "
                  f"(depth_programme_loading >= {cut}):")
            for label, _ in ALL_TESTS:
                col = dict(ALL_TESTS)[label]
                if col in hi:
                    k = int((hi[col] < ALPHA).sum())
                    lo, hh = wilson(k, len(hi))
                    print(f"  {label:18s} {k / len(hi):7.4f}  "
                          f"[{lo:.4f}, {hh:.4f}]  (n = {len(hi)})")
            report["A_calibration"]["high_confounding"] = reject_table(hi)
            report["A_calibration"]["high_confounding_cut"] = float(cut)

    # ---- the two candidate diagnostics, head to head -------------------
    if "A_calibration" in frames:
        # Pooled over both calibration grids, so that the verdict on the
        # quantity the study first proposed rests on every run where the
        # confounder was varied rather than on one sweep of it.  Only the
        # effect-free runs enter: under an effect the response is power, and a
        # rejection is no longer a false one.
        pool = pd.concat([frames[n][frames[n]["effect"] == 0]
                          for n in ("A_calibration", "B_regime")
                          if n in frames and len(frames[n])],
                         ignore_index=True)

        # The supplementary table on the tail convention: the two calibration
        # grids pooled, so that the table and the sentence that points at it
        # quote the same runs.  Grid C is pooled separately under its own
        # heading, because there the same table is power.
        if len(pool) and {"expression_p_lower", "expression_p_upper"} <= set(pool):
            pt = tail_robustness(pool)
            pt.to_csv(os.path.join(RESULTS,
                                   "grid_A_B_tail_robustness.csv"), index=False)
            print(f"\nmatched tests in all three tails, pooled over the "
                  f"calibration grids (n = {len(pool)})")
            print(pt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            report["tail_robustness_pooled"] = pt.to_dict(orient="list")
            if "C_power" in frames:
                pc = frames["C_power"]
                if {"expression_p_lower", "expression_p_upper"} <= set(pc):
                    ct = tail_robustness(pc)
                    ct.to_csv(os.path.join(RESULTS,
                                           "grid_C_tail_robustness.csv"),
                              index=False)
                    print("\nthe same three readings on the power grid")
                    print(ct.to_string(index=False,
                                       float_format=lambda v: f"{v:.3f}"))
                    report["tail_robustness_power"] = ct.to_dict(orient="list")

        jl = joint_logistic(pool) if len(pool) else None
        if jl is not None:
            print(f"\n{'=' * 72}\nboth candidate diagnostics in one model, "
                  f"pooled over the calibration grids (n = {len(pool)})\n"
                  f"{'=' * 72}")
            print("Standardised predictors of a rejection under effect = 0.  A "
                  "quantity\nthat forecasts the failure has a positive "
                  "coefficient of its own.\n")
            print(jl.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
            jl.to_csv(os.path.join(RESULTS, "grid_A_B_joint_logistic.csv"),
                      index=False)
            joint = jl[jl["model"] == "product and share"].set_index("term")
            if {"product", "share"} <= set(joint.index):
                report["joint_logistic_pooled"] = dict(
                    n=int(len(pool)),
                    **{f"{term}_{stat}": float(joint.loc[term, stat])
                       for term in ("product", "share")
                       for stat in ("coef", "z", "p")})
                # The marginal fits are quoted too: the share's coefficient
                # changes sign between them and the joint one, which is the
                # clearest evidence that its marginal association belongs to
                # whatever else is varying.
                marg = jl[jl["model"] != "product and share"].set_index("term")
                for term in ("product", "share"):
                    if term in marg.index:
                        report["joint_logistic_pooled"][f"marginal_{term}_coef"] = \
                            float(marg.loc[term, "coef"])
                        report["joint_logistic_pooled"][f"marginal_{term}_p"] = \
                            float(marg.loc[term, "p"])

    # ---- what does validity cost in power? -----------------------------
    if "C_power" in frames:
        c = frames["C_power"]
        print(f"\n{'=' * 72}\npower\n{'=' * 72}")
        tab = marginal(c, "effect")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        report["power"] = tab.to_dict(orient="list")

        # Power in each tail, per planted effect.  Pooling the effect sizes
        # hides the question, because a one-sided convention does not lose power
        # uniformly: it loses it where the planted association points away from
        # the tested tail, which is exactly the case a two-sided test covers and
        # a lower-tail test cannot.
        rows = []
        for effect, sub in c.groupby("effect"):
            t = tail_robustness(sub)
            if len(t):
                t.insert(0, "effect", effect)
                rows.append(t)
        if rows:
            pt = pd.concat(rows, ignore_index=True)
            print("\nmatched tests in all three tails, by planted effect")
            print(pt.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            pt.to_csv(os.path.join(
                RESULTS, "grid_C_tail_robustness_by_effect.csv"), index=False)
            report["tail_robustness_power_by_effect"] = pt.to_dict(orient="list")
        # The price of a valid null has to be read where the conventional test is
        # honest.  Under confounding the conventional test's extra rejections are
        # partly its own false positives, so the two-way table is what separates
        # power lost to the correction from power the correction never had.
        for label in ("naive", "expression", "codetection"):
            piv = interaction(c, "effect", "depth_programme_loading", test=label)
            if piv is None:
                continue
            print("\n" + format_interaction(piv, test=label))
            piv.to_csv(os.path.join(
                RESULTS, f"grid_C_power_effect_x_loading_{label}.csv"),
                index=False)
            report.setdefault("power_by_loading", {})[label] = \
                piv.to_dict(orient="list")
        # a matched null that never rejects has no power either; the ratio of
        # the matched power to the conventional power is the price of validity.
        for label in ("expression", "codetection"):
            if (f"{label}_p" in c and "naive_p" in c):
                pw_m = float((c[f"{label}_p"] < ALPHA).mean())
                pw_c = float((c["naive_p"] < ALPHA).mean())
                report.setdefault("power_cost", {})[label] = dict(
                    matched=pw_m, conventional=pw_c,
                    ratio=pw_m / pw_c if pw_c else np.nan)
                # the same ratio with the confounder switched off, where both
                # tests are measuring detection and nothing else.
                clean = c[c["depth_programme_loading"] == 0]
                if len(clean):
                    m0 = float((clean[f"{label}_p"] < ALPHA).mean())
                    c0 = float((clean["naive_p"] < ALPHA).mean())
                    report["power_cost"][label].update(
                        unconfounded_matched=m0, unconfounded_conventional=c0,
                        unconfounded_ratio=m0 / c0 if c0 else np.nan)

    # ---- did the nulls actually match? ---------------------------------
    for name in ("A_calibration", "C_power"):
        if name not in frames:
            continue
        df = frames[name]
        print(f"\n{'=' * 72}\nnull composition achieved, {name}\n{'=' * 72}")
        for label in ("random", "expression", "codetection"):
            d, e, g = (f"{label}_null_detection_ratio",
                       f"{label}_null_expression_ratio",
                       f"{label}_null_codetection_gap")
            if d in df:
                print(f"  {label:14s} detection ratio "
                      f"{df[d].mean():.3f} +- {df[d].std():.3f}   "
                      f"expression ratio {df[e].mean():.3f}   "
                      f"co-detection gap {df[g].mean():+.4f}")
                report.setdefault("null_composition", {}).setdefault(name, {})[
                    label] = dict(detection_ratio=float(df[d].mean()),
                                  expression_ratio=float(df[e].mean()),
                                  codetection_gap=float(df[g].mean()))

    out = os.path.join(RESULTS, "grid_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {out}")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-partial", action="store_true",
                    help="analyse a sweep that is still running, on purpose")
    main(allow_partial=ap.parse_args().allow_partial)
