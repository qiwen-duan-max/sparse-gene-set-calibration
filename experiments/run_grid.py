"""The simulation grid behind the framework's operating characteristics.

Four grids, each answering one question:

``A_calibration``
    Under a true null, how often does each of the four conventional analyses
    declare an association, and how often do the matched nulls?  Swept over the
    depth-programme confounder, the detection rate of the gene set, and whether
    the set has co-detection structure.
``B_regime``
    The same, swept over the ratio of ``max_rank`` to the median detected count.
    Above one, the top block is filled partly by tie-broken zeros; this grid
    measures what that does to inference.
``C_power``
    When there *is* an effect, which analyses find it and which miss it?  A
    calibration fix that destroys power is not a fix.
``D_size``
    Does the verdict depend on how many genes are in the set?

Every run writes one JSON object.  Runs are keyed by (grid, configuration,
replicate) so the driver resumes rather than repeating work.

Usage
-----
    python experiments/run_grid.py --grid A_calibration --workers 3
    python experiments/run_grid.py --all --workers 3
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

N_GENES = 8000          # max_rank = ceil(0.05 * 8000) = 400 throughout
RANK_FRAC = 0.05
ALPHA = 0.05
#: Candidate draws per returned co-detection null.  Matching on a single scalar
#: summary of the joint distribution costs this many extra sets per set
#: returned, so it sets the runtime of every grid.  It is not assumed that the
#: match succeeds: ``evaluate`` reports the co-detection each family actually
#: achieved, and the analysis checks it against the observed set's own value.
CODETECTION_DRAWS = 50


def _jsonable(value):
    """Coerce numpy scalars so a record can be serialised without surprises."""
    import numpy as np
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


# ----------------------------------------------------------------------
# grid definitions
# ----------------------------------------------------------------------
def grid_A_calibration():
    """Confounder x sparsity x co-detection, under a true null.

    Donor structure is switched off (``n_samples = 1``), and it has to be.  The
    programme and the depth both carry a donor-level component by default, and a
    cell-level test of a donor-level claim is anticonservative through
    pseudo-replication alone: at four donors with no confounder and no biological
    effect, the naive test already rejects 8% of the time at a target detection
    of 0.01 and 32% at 0.15.  Leaving that in grid A would attribute to the
    tie-break an inflation the tie-break did not cause, and the attribution is
    the whole claim.  Donor structure is measured in grid E instead, where it is
    the only thing varying.
    """
    for conf, det, co in itertools.product(
            (0.0, 0.3, 0.6), (0.01, 0.05, 0.15), (0.0, 0.6)):
        yield dict(depth_programme_loading=conf, detection=det, coexpr=co,
                   effect=0.0, n_samples=1)


def grid_B_regime():
    """max_rank / median-detected, the axis that opens the tie-break channel."""
    # n_genes = 8000 -> max_rank = 400; these give ratios 2.80 down to 0.47.
    for med, conf in itertools.product((143, 215, 286, 430, 572, 860),
                                       (0.0, 0.6)):
        yield dict(median_detected=med, depth_programme_loading=conf,
                   detection=0.05, coexpr=0.0, effect=0.0, n_samples=1)


def grid_C_power():
    """Non-zero effect: which analyses still detect it."""
    for eff, det, conf in itertools.product((0.25, 0.5), (0.01, 0.05, 0.15),
                                            (0.0, 0.6)):
        yield dict(effect=eff, detection=det, depth_programme_loading=conf,
                   coexpr=0.0, n_samples=1)


def grid_D_size():
    """Gene-set size, including sizes at which a set is barely scorable."""
    for k, det in itertools.product((4, 8, 20, 50), (0.01, 0.05)):
        yield dict(n_target=k, detection=det, depth_programme_loading=0.6,
                   coexpr=0.0, effect=0.0, n_samples=1)


def grid_E_donor():
    """Donor-level structure alone, as the separate failure mode it is.

    Every cell from one donor shares a depth offset and a programme offset, so a
    cell-level test of a donor-level claim treats 1,200 cells as 1,200
    independent observations of four.  No confounder, no biological effect, and
    no sparsity sweep: this grid exists to put a number on how much of an
    apparent association that structure manufactures by itself, and how it
    scales with the number of donors.
    """
    for n_smp, conf in itertools.product((1, 4, 16), (0.0, 0.6)):
        yield dict(n_samples=n_smp, depth_programme_loading=conf,
                   detection=0.05, coexpr=0.0, effect=0.0)


GRIDS = {
    "A_calibration": (grid_A_calibration, 60),
    "B_regime": (grid_B_regime, 50),
    "C_power": (grid_C_power, 40),
    "D_size": (grid_D_size, 50),
    "E_donor": (grid_E_donor, 40),
}


def config_key(grid, params, rep):
    return f"{grid}|{json.dumps(params, sort_keys=True)}|{rep}"


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------
def run_one(task):
    """Simulate one dataset and subject it to every analysis.  Top-level for pickling."""
    grid, params, rep = task
    from sparsegs.simulate import SimConfig, simulate, evaluate

    t0 = time.time()
    seed = 1000 + rep * 17
    cfg = SimConfig(seed=seed, rank_frac=RANK_FRAC, **params)
    try:
        sim = simulate(cfg)
        out = evaluate(sim, n_null=100, n_perm=1000, seed=seed,
                       n_draws=CODETECTION_DRAWS)
    except Exception as exc:                     # keep the grid running
        return dict(grid=grid, rep=rep, seed=seed, params=params, error=str(exc),
                    traceback=traceback.format_exc()[-800:])
    out.update(grid=grid, rep=rep, params=params, elapsed=time.time() - t0,
               n_null=100, n_perm=1000, n_draws=CODETECTION_DRAWS)
    return _jsonable(out)


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                continue                          # retry failed runs
            done.add(config_key(rec["grid"], rec["params"], rec["rep"]))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", action="append", choices=sorted(GRIDS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = ap.parse_args(argv)

    names = sorted(GRIDS) if args.all else (args.grid or [])
    if not names:
        ap.error("choose --grid or --all")

    os.makedirs(args.outdir, exist_ok=True)
    for name in names:
        gen, n_rep = GRIDS[name]
        path = os.path.join(args.outdir, f"grid_{name}.jsonl")
        done = load_done(path)
        configs = list(gen())
        tasks = [(name, p, r) for p in configs for r in range(n_rep)
                 if config_key(name, p, r) not in done]
        total = len(configs) * n_rep
        print(f"[{name}] {total} runs, {total - len(tasks)} already done, "
              f"{len(tasks)} to go", flush=True)
        if not tasks:
            continue

        import multiprocessing as mp
        t0 = time.time()
        n_done = 0
        with open(path, "a") as fh, mp.Pool(args.workers) as pool:
            for rec in pool.imap_unordered(run_one, tasks, chunksize=1):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done == len(tasks):
                    rate = n_done / max(time.time() - t0, 1e-9)
                    eta = (len(tasks) - n_done) / max(rate, 1e-9)
                    print(f"  {n_done}/{len(tasks)}  {rate:.2f} runs/s  "
                          f"ETA {eta/60:.1f} min", flush=True)
        print(f"[{name}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
