#!/usr/bin/env python
"""Does the checklist flag the gene sets whose analysis would mislead?

The brief states the success criterion as a rate -- the diagnostic checklist has
to flag 95%+ of the problematic gene sets in synthetic tests -- and most of the
work here went into deciding what "problematic" means, because three readings of
it are available and two of them measure nothing.

*Per-dataset rejection.*  Under ``effect = 0`` every rejection is a false
positive, so one can label each dataset by whether its conventional test
rejected and ask whether the checklist flagged it.  Run that way the sensitivity
lands near chance, structurally: a rejection is a coin flip whose probability is
the regime, while the checklist's flag is a property *of* the regime -- it reads
the matrix and the gene set and never sees the outcome.  A later stage will be
far from the best stage at both.

*Conventional anticonservatism, read as a refusal.*  The tempting repair is to
take the regimes where the conventional test over-rejects as the problematic
class and ask whether the checklist refuses there.  In the regimes where the
conventional test fails hardest -- a well-detected gene set whose axis tracks
sequencing depth -- the checklist correctly does *not* refuse: it reports that
the axis is confounded, which is a statement that the matched null is required
rather than that the question is unanswerable, and the matched null is in fact
calibrated there.  Scoring those as misses would mark the framework down for
being right.

So the sweep reports three numbers, and the point of reporting three is that
they answer different questions.

**sensitivity** -- over the regimes where the conventional test's false-positive
rate is significantly above the nominal level, the share of datasets the
checklist does not give a clean pass to, *called as the analyst who ran that
test would call it* (``used_matched_null=False``).  This is the brief's
criterion, and it is 100% by construction rather than by measurement: the
verdict rule never returns a plain ``INTERPRETABLE`` for an association that was
not tested against a matched null.  It is reported as a consistency check on
that rule, not as a result.

**clean-pass relief** -- over the regimes where the conventional test is *well*
calibrated, the share of datasets the checklist nonetheless gives a clean pass
to once the matched null has been used.  This is the measured operating
characteristic, and it is the complement of the cost the severity raises.

**refusal recall** -- over the regimes where the expression-matched null is
*itself* miscalibrated (its own false-positive rate significantly above the
nominal level, so no analysis of this data answers the question), the share of
datasets the checklist refuses outright with ``NOT_IDENTIFIABLE``.  This is the
class the framework reserves its refusal for, and the one the earlier readings
confused with "the conventional test failed".

A power arm runs beside all three: each regime is simulated a second time with
an effect planted, and the matched null's rejection rate there is its power.  It
is a companion rather than a reference class -- in this grid the matched null is
never powerless -- and it is what lets the manuscript say the calibrated test is
not merely conservative.

    python experiments/checklist_acceptance.py --workers 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_GENES = 8000                  # max_rank = ceil(0.05 * 8000) = 400 throughout
RANK_FRAC = 0.05
ALPHA = 0.05
#: The effect planted in the power arm.  The same value grid C uses, so a regime
#: that looks underpowered here and well-powered there is a disagreement between
#: two scripts rather than a property of the simulation.
PLANTED_EFFECT = 0.6
#: One-sided level for calling a regime's error rate above the nominal one, and
#: for calling a regime well calibrated.  Both are one-sided tests of the same
#: count against the same null, so a regime can be anticonservative, calibrated,
#: or neither -- and "neither" means the replicates do not separate it from
#: alpha in either direction, which is the honest answer at these sample sizes
#: rather than a coin flip assigned to one class.
REGIME_LEVEL = 0.05


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion.

    Reported beside the rates.  At the sample sizes here the normal
    approximation can put a bound above 1 or below 0, and a rate with an
    impossible interval invites the reader to distrust the rest of the row.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _matched_p(sg, np, cache, target, z, max_rank, n_null, seed):
    """The expression-matched P value for one dataset."""
    built = sg.construct_expression_matched_null(cache, target, n_sets=n_null,
                                                 seed=seed)
    null_rho = np.array([sg.spearman(sg.aucell(cache, s, max_rank=max_rank), z)[0]
                         for s in built["null_genes"]])
    obs_rho = sg.spearman(sg.aucell(cache, target, max_rank=max_rank), z)[0]
    # A replacement set can be sparse enough that its score never varies, in
    # which case its correlation is undefined and carries no information about
    # the null.  Dropped rather than counted as zero: a zero would enter the
    # null distribution as a real observation.
    usable = null_rho[np.isfinite(null_rho)]
    if not usable.size:
        return float("nan")
    return float(sg.empirical_p(usable, obs_rho, tail="two-sided"))


def one_run(task):
    """One regime at two effect sizes, plus the checklist's verdict on it."""
    import numpy as np

    import sparsegs as sg
    from sparsegs.simulate import SimConfig, simulate

    params, rep, n_null, n_perm = task
    seed = 5000 + 17 * rep
    t0 = time.time()
    try:
        def build(effect):
            cfg = SimConfig(seed=seed, rank_frac=RANK_FRAC, n_genes=N_GENES,
                            effect=effect, **{k: v for k, v in params.items()
                                              if k != "effect"})
            sim = simulate(cfg)
            cache = sg.RankCache.build(sim.X, sim.genes,
                                       ceiling=max(sim.max_rank, 30),
                                       seed=2024, chunk=500)
            return sim, cache

        # --- the null arm: what each analysis concludes with nothing there ---
        sim, cache = build(0.0)
        score = sg.aucell(cache, sim.target, max_rank=sim.max_rank)
        naive = sg.naive_test(score, sim.programme)
        matched_p_null = _matched_p(sg, np, cache, sim.target, sim.programme,
                                    sim.max_rank, n_null, seed + 1)

        # --- the power arm: can the recommended analysis find a real effect? ---
        sim_p, cache_p = build(PLANTED_EFFECT)
        matched_p_power = _matched_p(sg, np, cache_p, sim_p.target, sim_p.programme,
                                     sim_p.max_rank, n_null, seed + 2)

        # --- the checklist, called the way a user of the framework calls it ---
        # One call is enough for both readings of the verdict.  The severity does
        # not depend on `used_matched_null` -- that flag only chooses between the
        # two verdicts a severity allows -- so the conventional analyst's reading
        # is derived from the same object rather than bought with a second call.
        refused = None
        try:
            check = sg.diagnostic_checklist(
                cache, sim.target, axis=sim.programme, used_matched_null=True,
                cutpoint_method="median", n_perm=n_perm,
                config={"n_null": n_null})
            severity = check["severity"]
            zero_rate = float(check["step1_sparsity"]["percent_zero"] / 100.0)
            preflight_verdict = (check.get("preflight") or {}).get("verdict")
            tie_break_share = check["step1_sparsity"].get("tie_break_share")
        except ValueError as exc:
            refused = str(exc)
            severity = "HIGH"
            zero_rate = tie_break_share = preflight_verdict = None

        return dict(
            params=params, rep=rep, seed=seed, planted_effect=PLANTED_EFFECT,
            naive_p=float(naive["p_value"]), naive_rho=float(naive["rho"]),
            matched_p_null=matched_p_null,
            matched_p_power=matched_p_power,
            conv_reject=bool(naive["reject"]),
            matched_reject=bool(np.isfinite(matched_p_null)
                                and matched_p_null < ALPHA),
            power_reject=bool(np.isfinite(matched_p_power)
                              and matched_p_power < ALPHA),
            severity=severity, refused=refused,
            # The two readers of the checklist are different people and only
            # one of them gets a clean pass.  `INTERPRETABLE` needs the
            # severity below HIGH *and* a matched null behind the association,
            # so the analyst who ran the conventional test needs severity LOW,
            # while the analyst who ran the recommended analysis needs only
            # that the checklist did not refuse.  Recording both is what lets
            # the sweep score a miss the way the brief means it -- the
            # conventional analyst was told the analysis was fine -- without
            # crediting the framework for a clean pass it never gave.
            clean_pass_calibrated=bool(severity != "HIGH"),
            clean_pass_conventional=bool(severity == "LOW"),
            observed_zero_rate=zero_rate,
            tie_break_share=(None if tie_break_share is None
                             else float(tie_break_share)),
            preflight_verdict=preflight_verdict,
            depth_ratio=float(sim.max_rank / max(np.median(sim.detected_per_cell), 1)),
            elapsed=time.time() - t0)
    except Exception as exc:                          # keep the sweep running
        return dict(params=params, rep=rep, seed=seed, error=str(exc),
                    traceback=traceback.format_exc()[-800:])


# ----------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------
def configurations():
    """Sparsity x confounder x regime, under a true null, one donor.

    The detection range runs to 1.0, where the gene set is expressed in every
    cell and the score is as far from the sparse regime as it gets.  Those
    regimes are not padding: a checklist that flags everything scores 100% on
    the miss rate and is useless, and the only way to measure the other side of
    that trade is to include gene sets with nothing wrong with them.  An
    earlier version of this grid stopped at 0.15 detection, where the framework
    is *right* to decline a clean pass on nearly every dataset, and it could
    not have detected a checklist that never passed anything.
    """
    for det, conf, med, co in itertools.product(
            (0.01, 0.05, 0.15, 0.5, 1.0),   # sparsity, out to no sparsity at all
            (0.0, 0.3, 0.6),           # the confounder; 0.0 is the control
            (200, 900),                # regime: below and above max_rank = 400
            (0.0, 0.6)):               # co-detection structure
        yield dict(detection=det, depth_programme_loading=conf,
                   median_detected=med, coexpr=co, effect=0.0, n_samples=1)


def key_of(params):
    return json.dumps(params, sort_keys=True)


def classify(k, n, level=REGIME_LEVEL):
    """Is a regime's error rate above alpha, below it, or indistinguishable?

    Three classes rather than two, because at ten replicates a rate of one in
    ten is neither: it is above the nominal level and not evidence of
    inflation, and assigning it to either side would move a headline number by
    more than the effect being measured.  The two classes the summary uses are
    the ones the data actually separates -- significantly above alpha, and at
    or below it on the point estimate.
    """
    from scipy import stats as st
    if n == 0:
        return "unknown"
    if st.binomtest(k, n, ALPHA, alternative="greater").pvalue < level:
        return "inflated"
    if k / n <= ALPHA:
        return "not_inflated"
    return "indistinguishable"


def group_by_config(records, power_split):
    """Per-regime rates, which are what the criteria are defined on."""
    import numpy as np

    ok = [r for r in records if "error" not in r]
    by = {}
    for r in ok:
        by.setdefault(key_of(r["params"]), []).append(r)
    out = []
    for key, rs in sorted(by.items()):
        n = len(rs)
        conv_k = sum(1 for r in rs if r["conv_reject"])
        matched_k = sum(1 for r in rs if r["matched_reject"])
        power = float(np.mean([r["power_reject"] for r in rs]))
        out.append(dict(
            params=rs[0]["params"], n=n,
            power=power,
            conv_k=conv_k, conv_error_rate=conv_k / n,
            matched_k=matched_k, matched_error_rate=matched_k / n,
            conv_class=classify(conv_k, n),
            matched_class=classify(matched_k, n),
            clean_pass_calibrated=float(np.mean(
                [r["clean_pass_calibrated"] for r in rs])),
            clean_pass_conventional=float(np.mean(
                [r["clean_pass_conventional"] for r in rs])),
            high_severity=float(np.mean([r["severity"] == "HIGH" for r in rs])),
            zero_rate=float(np.nanmean([r["observed_zero_rate"] for r in rs
                                        if r["observed_zero_rate"] is not None]))
            if any(r["observed_zero_rate"] is not None for r in rs) else None,
            underpowered=bool(power < power_split),
            n_refused=sum(1 for r in rs if r["refused"] is not None)))
    return out


def _pool(configs, field):
    """A rate over datasets, weighted by each regime's dataset count.

    The unit of the claim is the dataset, so a rate pooled over regimes counts
    datasets.  Averaging the per-regime proportions instead would give a regime
    with one dataset the same weight as one with fifty, which is the wrong
    question and, at unbalanced classes, a visibly different number.
    """
    n = sum(c["n"] for c in configs)
    if not n:
        return float("nan"), 0, 0
    k = sum(round(c[field] * c["n"]) for c in configs)
    return k / n, k, n


def summarise(runs, configs):
    """The three criteria and their denominators."""
    bad_conv = [c for c in configs if c["conv_class"] == "inflated"]
    ok_conv = [c for c in configs if c["conv_class"] == "not_inflated"]
    bad_match = [c for c in configs if c["matched_class"] == "inflated"]
    ok_match = [c for c in configs if c["matched_class"] == "not_inflated"]

    sens, sens_k, sens_n = _pool(bad_conv, "clean_pass_conventional")
    relief, relief_k, relief_n = _pool(ok_conv, "clean_pass_calibrated")
    recall, recall_k, recall_n = _pool(bad_match, "high_severity")
    cost, cost_k, cost_n = _pool(ok_match, "high_severity")

    def wmean(cs, field):
        n = sum(c["n"] for c in cs)
        return sum(c[field] * c["n"] for c in cs) / n if n else float("nan")

    # Does the severity ordering track the thing it is a proxy for?  The two
    # rates above compare extremes; this uses every regime, including the
    # ambiguous ones, and asks whether the checklist's own criterion -- the
    # share of datasets it refuses -- rises with the conventional test's actual
    # error rate.  A threshold can pass both extremes and still be flat in the
    # middle, which is where most real gene sets live.
    import numpy as np
    from scipy import stats as st
    sev = np.array([c["high_severity"] for c in configs], dtype=float)
    conv = np.array([c["conv_error_rate"] for c in configs], dtype=float)
    zero = np.array([c["zero_rate"] if c["zero_rate"] is not None else np.nan
                     for c in configs], dtype=float)
    keep = np.isfinite(sev) & np.isfinite(conv)
    rho_sev = float(st.spearmanr(sev[keep], conv[keep])[0]) if keep.sum() > 2 else float("nan")
    keep_z = keep & np.isfinite(zero)
    rho_zero = (float(st.spearmanr(zero[keep_z], conv[keep_z])[0])
                if keep_z.sum() > 2 else float("nan"))

    return dict(
        n_runs=len(runs), n_failed=sum(1 for r in runs if "error" in r),
        n_configs=len(configs), reps=configs[0]["n"] if configs else 0,
        # the brief's criterion, as a miss rate: datasets in an
        # anticonservative regime that were still given a clean pass by the
        # analyst who ran the conventional test
        conventional_clean_pass=sens,
        conventional_clean_pass_ci=list(wilson(sens_k, sens_n)),
        n_anticonservative=len(bad_conv), n_anticonservative_datasets=sens_n,
        # the measured operating characteristic
        clean_pass_relief=relief,
        clean_pass_relief_ci=list(wilson(relief_k, relief_n)),
        n_well_calibrated=len(ok_conv), n_well_calibrated_datasets=relief_n,
        # the class the refusal is reserved for
        refusal_recall=recall,
        refusal_recall_ci=list(wilson(recall_k, recall_n)),
        n_unrepairable=len(bad_match), n_unrepairable_datasets=recall_n,
        refusal_cost=cost,
        refusal_cost_ci=list(wilson(cost_k, cost_n)),
        n_repairable=len(ok_match), n_repairable_datasets=cost_n,
        rho_severity_vs_conv_error=rho_sev,
        rho_zero_rate_vs_conv_error=rho_zero,
        # The boundary of the framework's own remedy.  The expression-matched
        # null is calibrated against the tie-break -- that is the framework's
        # central result -- and this is the number that says whether it survives
        # an axis that is itself a depth proxy.  It is broken out by the
        # pre-flight's own verdict so the diagnostic can be checked against the
        # thing it claims to predict, rather than asserted.
        by_preflight=[dict(
            preflight=name,
            n=len(rs), n_regimes=len({key_of(r["params"]) for r in rs}),
            conv_error_rate=float(np.mean([r["conv_reject"] for r in rs])),
            matched_error_rate=float(np.mean([r["matched_reject"] for r in rs])),
            power=float(np.mean([r["power_reject"] for r in rs])),
            high_severity=float(np.mean([r["severity"] == "HIGH" for r in rs])))
            for name, rs in sorted(
                ((k, [r for r in runs
                      if "error" not in r and r.get("preflight_verdict") == k])
                 for k in ("CLEAR", "CAUTION", "CONFOUNDED")),
                key=lambda kv: kv[0]) if rs],
        n_underpowered=sum(1 for c in configs if c["underpowered"]),
        power_under=wmean([c for c in configs if c["underpowered"]], "power"),
        power_over=wmean([c for c in configs if not c["underpowered"]], "power"),
        power_overall=wmean(configs, "power"),
        conv_error_overall=wmean(configs, "conv_error_rate"),
        matched_error_overall=wmean(configs, "matched_error_rate"),
        clean_pass_overall=wmean(configs, "clean_pass_calibrated"),
        n_refusals=sum(1 for r in runs if r.get("refused")),
        alpha=ALPHA, planted_effect=PLANTED_EFFECT, regime_level=REGIME_LEVEL)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=10,
                    help="datasets per regime; 10 puts the binomial bar for a "
                         "conventional error rate above alpha at 3 of 10")
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--power-split", type=float, default=0.5,
                    help="below this the power arm is reported separately")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = ap.parse_args(argv)

    configs = list(configurations())
    tasks = [(p, r, args.n_null, args.n_perm)
             for p in configs for r in range(args.reps)]
    print(f"{len(configs)} regimes x {args.reps} datasets x 2 effect sizes = "
          f"{len(tasks)} runs", flush=True)

    t0 = time.time()
    records = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(one_run, tasks,
                                                        chunksize=1), 1):
                records.append(rec)
                if i % 20 == 0 or i == len(tasks):
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"  {i}/{len(tasks)}  {rate:.2f} runs/s  "
                          f"ETA {(len(tasks) - i) / max(rate, 1e-9) / 60:.1f} min",
                          flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            records.append(one_run(task))
            if i % 20 == 0 or i == len(tasks):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(tasks)}  {rate:.2f} runs/s  "
                      f"ETA {(len(tasks) - i) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)

    grouped = group_by_config(records, args.power_split)
    summary = summarise(records, grouped)
    summary["elapsed"] = time.time() - t0
    summary["n_null"] = args.n_null
    summary["n_perm"] = args.n_perm
    summary["power_split"] = args.power_split

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "checklist_acceptance.json")
    with open(path, "w") as fh:
        json.dump(dict(summary=summary, by_config=grouped, runs=records), fh,
                  indent=2)

    def pct(x):
        return "  n/a" if x != x else f"{100 * x:5.1f}%"

    def band(ci):
        return f"[{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]"

    s = summary
    print(f"\n{s['n_runs']} datasets over {s['n_configs']} regimes "
          f"({s['n_failed']} failed), nominal alpha {ALPHA:g}")
    print(f"conventional false-positive rate {pct(s['conv_error_overall'])}; "
          f"matched-null {pct(s['matched_error_overall'])}; "
          f"clean passes under the matched null {pct(s['clean_pass_overall'])}")

    print(f"\nregimes where the conventional test is inflated            "
          f"{s['n_anticonservative']:>3}  ({s['n_anticonservative_datasets']} datasets)")
    print(f"  given a clean pass by the conventional analyst  "
          f"{pct(s['conventional_clean_pass'])} {band(s['conventional_clean_pass_ci'])}"
          f"   <- the criterion, 0% is the target")
    print(f"regimes where its error rate is at or below alpha        "
          f"{s['n_well_calibrated']:>3}  ({s['n_well_calibrated_datasets']} datasets)")
    print(f"  given a clean pass once a matched null was used "
          f"{pct(s['clean_pass_relief'])} {band(s['clean_pass_relief_ci'])}"
          f"   <- the measured operating characteristic")
    print(f"  severity tracks the error rate: rho = "
          f"{s['rho_severity_vs_conv_error']:+.3f}, zero rate "
          f"{s['rho_zero_rate_vs_conv_error']:+.3f}")

    print(f"\nregimes where the matched null is itself inflated             "
          f"{s['n_unrepairable']:>3}  ({s['n_unrepairable_datasets']} datasets)")
    print(f"  refused outright (NOT_IDENTIFIABLE)             "
          f"{pct(s['refusal_recall'])} {band(s['refusal_recall_ci'])}"
          f"   <- the refusal's recall")
    print(f"regimes where its error rate is at or below alpha        "
          f"{s['n_repairable']:>3}  ({s['n_repairable_datasets']} datasets)")
    print(f"  refused outright anyway                         "
          f"{pct(s['refusal_cost'])} {band(s['refusal_cost_ci'])}"
          f"   <- the refusal's cost")

    print(f"\npower of the matched null: mean {pct(s['power_overall'])}, "
          f"{s['n_underpowered']} regimes below {args.power_split:g}")
    print(f"refusals raised: {s['n_refusals']}")

    # The framework's own remedy, held up against the diagnostic that claims to
    # predict where it stops working.
    print(f"\nwhere the matched null is calibrated, by the pre-flight's verdict:")
    print(f"  {'verdict':<12}{'n':>5}{'regimes':>9}{'conv':>9}"
          f"{'matched':>10}{'power':>8}{'refused':>9}")
    for row in s["by_preflight"]:
        print(f"  {row['preflight']:<12}{row['n']:>5}{row['n_regimes']:>9}"
              f"{pct(row['conv_error_rate']):>9}{pct(row['matched_error_rate']):>10}"
              f"{pct(row['power']):>8}{pct(row['high_severity']):>9}")
    print(f"  nominal alpha is {ALPHA:g}; the column to read is 'matched'")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
