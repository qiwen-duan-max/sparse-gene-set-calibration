"""Measure the residual of the score decomposition, cell by cell.

Equation (3) of the manuscript writes an AUCell score as the detected part,
plus the closed form's expectation for the tie-break, plus a residual.  The
split into the detected part and the *realised* tie-break is exact -- a unit
test asserts it cell by cell -- and the closed form is the expectation of the
tie-break under the uniform-order model.  What this script measures is the
residual between the two: how far the realised tie-break sits from its closed
form in a real matrix, cell by cell, across the calibration grid's own
parameter space.  The tie-break is a deterministic hash, so the residual is a
reproducible property of the data, not sampling noise; the numbers it produces
are the ones the manuscript quotes.

    python experiments/measure_residual.py

Writes ``results/decomposition_residual.json``: one record per simulated
dataset, plus an ``overall`` summary.  Rerun by the pipeline before the
manuscript numbers are generated.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

from sparsegs import RankCache, aucell                      # noqa: E402
from sparsegs.simulate import SimConfig, simulate           # noqa: E402
from sparsegs.theory import score_decomposition             # noqa: E402

#: The calibration grid's own factorial: confounder x sparsity x co-detection,
#: under a true null, the same configurations whose operating characteristics
#: the manuscript reports.  Five replicates each keeps the full sweep under a
#: minute while covering every configuration the grid defines.
REPS = 5
RANK_FRAC = 0.05
SEED0 = 1000


def configs():
    for conf, det, co in itertools.product(
            (0.0, 0.3, 0.6), (0.01, 0.05, 0.15), (0.0, 0.6)):
        for rep in range(REPS):
            yield dict(depth_programme_loading=conf, detection=det,
                       coexpr=co, effect=0.0, n_samples=1), rep


def one(params, rep):
    seed = SEED0 + rep * 17
    cfg = SimConfig(seed=seed, rank_frac=RANK_FRAC, **params)
    sim = simulate(cfg)
    cache = RankCache.build(sim.X, sim.genes, ceiling=max(sim.max_rank, 30),
                            seed=2024, chunk=500)
    score = aucell(cache, sim.target, max_rank=sim.max_rank)
    dec = score_decomposition(cache, sim.target, (sim.X > 0),
                              max_rank=sim.max_rank, depth=sim.detected_per_cell)
    # The exact half of the claim: the realised split reproduces the score.
    gap = float(np.max(np.abs(dec["total"].to_numpy() - np.asarray(score))))
    eps = (dec["tie_break"] - dec["tie_break_expected"]).to_numpy()
    depth = dec["depth"].to_numpy()
    return dict(
        params=params, rep=rep, seed=seed, n_cells=int(eps.size),
        max_rank=int(sim.max_rank),
        exact_gap=gap,
        eps_mean=float(eps.mean()), eps_sd=float(eps.std(ddof=1)),
        eps_max_abs=float(np.max(np.abs(eps))),
        eps_q999_abs=float(np.quantile(np.abs(eps), 0.999)),
        eps_depth_corr=float(np.corrcoef(eps, depth)[0, 1]),
        tb_expected_sd=float(dec["tie_break_expected"].std(ddof=1)),
    )


def main():
    t0 = time.time()
    records = [dict(**one(params, rep), grid="residual_sweep")
               for params, rep in configs()]
    all_eps_sd = np.array([r["eps_sd"] for r in records])
    all_max = np.array([r["eps_max_abs"] for r in records])
    all_gap = np.array([r["exact_gap"] for r in records])
    n_cells = sum(r["n_cells"] for r in records)
    worst = max(records, key=lambda r: r["eps_max_abs"])
    out = dict(
        n_datasets=len(records), n_cells=n_cells,
        exact_gap_max=float(all_gap.max()),
        eps_sd_median=float(np.median(all_eps_sd)),
        eps_sd_max=float(all_eps_sd.max()),
        eps_max_abs_over_datasets=float(all_max.max()),
        worst=dict(params=worst["params"], rep=worst["rep"],
                   eps_max_abs=worst["eps_max_abs"],
                   eps_sd=worst["eps_sd"], max_rank=worst["max_rank"]),
        elapsed=time.time() - t0,
    )
    path = os.path.join(HERE, "..", "results", "decomposition_residual.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(overall=out, records=records), fh, indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
