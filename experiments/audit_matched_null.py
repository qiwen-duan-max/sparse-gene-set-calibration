#!/usr/bin/env python
"""Is the set under test exchangeable with the sets the null draws?

Motivation
----------
The calibration grid measures the matched nulls' false-positive rate under
``effect = 0`` and finds them not exactly nominal: the expression-matched family
sits above 0.05, and the excess does not grow with the confounder.  That is a
different failure from the conventional tests' -- theirs grows from 0.06 to 0.69
as depth is made to track the programme -- and it has
two candidate explanations which the grid cannot separate:

1. **The null is under-dispersed.**  The draws are more alike than independent
   sets of the same composition would be, so the observed rho lands in their
   tail more often than it should.  This would be a defect of the matching.

2. **The set under test is not exchangeable with the draws.**  A simulated
   target set is built with its own activity spread and, when ``coexpr > 0``,
   its own module factor; a matched draw reproduces each gene's detection rate
   and nothing else.  The target is then over-dispersed relative to the draws,
   and again the observed rho lands in the tail.  This is a property of the
   simulator's construction of the target set.

The audit separates them by adding a third kind of set to the same test:
**pseudo-targets** drawn from the null's own generator.  A pseudo-target is by
construction exchangeable with the draws, so if the test is valid for them and
anti-conservative for the simulator's targets, the residual belongs to (2).  If
it is anti-conservative for both, it belongs to (1).

The measured quantity is the dispersion of the observed rho across replicate
datasets, against the null's internal dispersion.  A valid test has the two
equal at the same configuration.

    python experiments/audit_matched_null.py --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

RESULTS = PKG / "results"
ALPHA = 0.05

#: The configurations to audit.  The first is the grid cell where the
#: expression-matched family is furthest from nominal; the second adds the
#: co-expression that per-gene matching cannot reproduce; the third is the
#: control, where no confounder exists and every valid test should sit at 0.05.
CONFIGS = {
    "confounded_sparse": dict(depth_programme_loading=0.6, detection=0.01,
                              coexpr=0.0),
    "confounded_coexpressed": dict(depth_programme_loading=0.6, detection=0.05,
                                   coexpr=0.6),
    "control_unconfounded": dict(depth_programme_loading=0.0, detection=0.05,
                                 coexpr=0.0),
}

BASE = dict(n_cells=800, n_genes=12000, n_target=8, median_detected=430,
            rank_frac=0.05)
#: ``n_genes = 12,000`` and ``median_detected = 430`` put ``max_rank = 600`` and
#: the ceiling-to-depth ratio at 1.40, which is the ratio measured in the CD8+ T
#: cell compartment of the real dataset.  The first version of this audit used
#: 6,000 genes with the same depth, giving a ratio of 0.70 -- the tie-break never
#: reached the ceiling, and the "confounded" configurations had no channel
#: through which to be confounded.  It reported, correctly and uselessly, that
#: the confounder did nothing.


def score_rho(cache, genes, z, max_rank):
    from sparsegs import aucell, spearman
    s = aucell(cache, genes, max_rank=max_rank)
    return spearman(s, z)[0]


def one_dataset(name, params, seed, n_null=60, n_pseudo=8):
    """One dataset, one real target, and ``n_pseudo`` exchangeable pseudo-targets."""
    from sparsegs import (RankCache, MatchedNullBuilder, empirical_p,
                          sparsity_report)
    from sparsegs.nulls import adaptive_detection_floor
    from sparsegs.simulate import SimConfig, simulate

    cfg = SimConfig(seed=seed, **{**BASE, **params})
    sim = simulate(cfg)
    cache = RankCache.build(sim.X, sim.genes, ceiling=max(sim.max_rank, 30),
                            seed=2024, chunk=500)
    builder = MatchedNullBuilder(
        cache, detection_matrix=(sim.X > 0), detection_matrix_genes=sim.genes,
        det_floor=adaptive_detection_floor(cache, sim.target), seed=seed)
    rng = np.random.default_rng(seed + 1)
    z, m = sim.programme, sim.max_rank

    def test(genes):
        """Observed rho, matched P, and the null's own dispersion."""
        draws = builder.sample(genes, n_null, kind="expression", rng=rng)
        obs = score_rho(cache, genes, z, m)
        nr = np.array([score_rho(cache, g, z, m) for g in draws])
        finite = nr[np.isfinite(nr)]
        # Through the package rather than written out here.  The same three
        # lines were written out in four places in this project and one of them
        # was read in the wrong tail; a P value is not the place for a fourth
        # opinion about what the test is.
        return dict(rho=obs,
                    p=empirical_p(finite, obs, tail="two-sided"),
                    p_lower=empirical_p(finite, obs, tail="lower"),
                    p_upper=empirical_p(finite, obs, tail="upper"),
                    null_sd=float(np.std(finite, ddof=1)),
                    null_median=float(np.median(finite)),
                    null_undefined=int(np.sum(~np.isfinite(nr))))

    diag = sparsity_report(cache, sim.target, rank_frac=cfg.rank_frac)
    # How much more co-detected the target set is than the sets drawn to replace
    # it.  This is the property per-gene matching cannot reproduce, and it is
    # what hypothesis (2) turns on, so it is carried through the audit.
    trial = builder.sample(sim.target, 8, kind="expression",
                           rng=np.random.default_rng(7))
    cod_gap = float(np.mean([builder.codetection(s) for s in trial])
                    - builder.codetection(sim.target))
    out = dict(audit=name, seed=seed, kind="real_target",
               tie_break_share=float(diag["tie_break_share"]),
               codetection_gap=cod_gap,
               # The Monte-Carlo draw count belongs in the record: the p value is
               # (k+1)/(n+1), so a reader cannot say what resolution the audit
               # had without it, and it cannot be recovered after the fact.
               n_null=int(n_null),
               # Recorded so that a configuration which never opens the channel
               # cannot pass for a confounded one again.
               depth_ratio=float(m / max(np.median(sim.detected_per_cell), 1)),
               **BASE, **params)
    out.update(test(sim.target))

    rows = [out]
    for k in range(n_pseudo):
        pseudo = builder.sample(sim.target, 1, kind="expression", rng=rng)[0]
        rec = dict(audit=name, seed=seed, kind="pseudo_target", pseudo=k,
                   tie_break_share=float(diag["tie_break_share"]),
                   n_null=int(n_null),
                   **BASE, **params)
        rec.update(test(pseudo))
        rows.append(rec)
    return rows


def _worker(task):
    name, params, seed = task
    try:
        return one_dataset(name, params, seed)
    except Exception as exc:                                   # keep going
        return [dict(audit=name, seed=seed, error=f"{type(exc).__name__}: {exc}")]


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    if "error" in df and df["error"].notna().any():
        bad = int(df["error"].notna().sum())
        print(f"  {bad} datasets failed and are excluded")
        df = df[df["error"].isna()]
    lines = []
    for name, sub in df.groupby("audit"):
        for kind, s in sub.groupby("kind"):
            k = int((s["p"] < ALPHA).sum())
            # The directional rates travel with the two-sided one for the same
            # reason they do in the grid: the audit's headline is the dispersion
            # ratio, which does not depend on the tail, but its false-positive
            # rate does, and a reader comparing the two tables should not have
            # to work out which convention each was computed under.
            k_lo = int((s["p_lower"] < ALPHA).sum()) if "p_lower" in s else None
            k_hi = int((s["p_upper"] < ALPHA).sum()) if "p_upper" in s else None
            lines.append(dict(
                audit=name, kind=kind, n=len(s), fpr=k / len(s),
                fpr_lower=k_lo / len(s) if k_lo is not None else np.nan,
                fpr_upper=k_hi / len(s) if k_hi is not None else np.nan,
                depth_ratio=float(s["depth_ratio"].mean()),
                tie_break=float(s["tie_break_share"].mean()),
                rho_mean=float(s["rho"].mean()),
                rho_sd=float(s["rho"].std(ddof=1)),
                null_sd=float(s["null_sd"].mean()),
                dispersion_ratio=float(s["rho"].std(ddof=1) / s["null_sd"].mean()),
                median_p=float(s["p"].median())))
    return pd.DataFrame(lines)


def main(workers=2, reps=24, n_null=60, n_pseudo=8):
    import multiprocessing as mp
    tasks = [(name, params, 3000 + r * 17)
             for name, params in CONFIGS.items() for r in range(reps)]
    print(f"{len(tasks)} datasets x ({1 + n_pseudo} sets x {n_null + 1} draws)")

    t0 = time.time()
    rows = []
    with mp.Pool(workers) as pool:
        for i, recs in enumerate(pool.imap_unordered(_worker, tasks), 1):
            rows.extend(recs)
            if i % 4 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {i / (time.time() - t0):.3f} sets/s",
                      flush=True)

    out = RESULTS / "audit_matched_null.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    tab = summarise(rows)
    print("\n" + "=" * 78)
    print("FPR of the expression-matched test, and the dispersion that explains it")
    print("=" * 78)
    print("dispersion_ratio = sd of the observed rho across datasets / the null's "
          "own sd.\nA valid test has this at 1.0 and the FPR at 0.05.\n")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tab.to_csv(RESULTS / "audit_matched_null.csv", index=False)
    print(f"\nwrote {out}")
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--n-null", type=int, default=60)
    ap.add_argument("--n-pseudo", type=int, default=8)
    main(**vars(ap.parse_args()))
