#!/usr/bin/env bash
#
# Every step from a finished sweep to a checked manuscript, in order.
#
# The steps below have a dependency order that is not obvious from their names,
# and getting it wrong does not always announce itself.  ``make_numbers.py``
# reads the per-compartment tie-break share that ``compartment_tie_break.py``
# writes, so running it first leaves those macros unfilled and the manuscript
# compiles with a gap that looks like a quantity nobody computed.  ``latexmk``
# has to come after ``make_numbers.py`` because the manuscript quotes macros
# that file defines, not constants.  And the format check has to come last,
# because it reads the build log to decide whether every reference resolved.
#
# Each step is checked before the next begins, and the first failure stops the
# chain: a pipeline that carries on after a failed step produces a complete-
# looking set of outputs built from a mixture of old and new inputs, which is
# the failure this file exists to prevent.
#
#     bash experiments/run_pipeline.sh              # full chain
#     bash experiments/run_pipeline.sh --no-latex   # stop before the build
#
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(dirname "$HERE")"
cd "$PKG" || exit 1
LOG="${PKG}/results/log_pipeline.txt"
: > "$LOG"

step() {
    local name="$1"; shift
    printf '\n=== %s  %s ===\n' "$(date '+%H:%M:%S')" "$name" | tee -a "$LOG"
    if "$@" >>"$LOG" 2>&1; then
        printf '    ok\n' | tee -a "$LOG"
        return 0
    fi
    printf '    FAILED: %s\n' "$*" | tee -a "$LOG"
    printf '\nlast 25 lines of %s:\n' "$LOG"
    tail -25 "$LOG"
    exit 1
}

need() {
    for f in "$@"; do
        if [ ! -s "$f" ]; then
            printf 'missing input: %s\n' "$f" >&2
            printf 'the chain cannot start; run the sweep or the benchmark first\n' >&2
            exit 1
        fi
    done
}

# The same as ``step``, except that a checker which needs the network is allowed
# to report that it could not reach it (exit 2) without stopping the chain.  A
# bibliography that does not match the registry (exit 1) still does stop it: the
# distinction is between a check that did not run and a check that ran and
# failed, and only the second is evidence about the manuscript.
step_soft() {
    local name="$1"; shift
    printf '\n=== %s  %s ===\n' "$(date '+%H:%M:%S')" "$name" | tee -a "$LOG"
    "$@" >>"$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        printf '    ok\n' | tee -a "$LOG"
    elif [ $rc -eq 2 ]; then
        printf '    skipped: no registry reachable, references unchecked\n' \
            | tee -a "$LOG"
    else
        printf '    FAILED: %s\n' "$*" | tee -a "$LOG"
        printf '\nlast 25 lines of %s:\n' "$LOG"
        tail -25 "$LOG"
        exit 1
    fi
}

# The sweeps and the benchmark take hours and are run separately; this chain
# assumes they are finished and refuses to guess.
need results/grid_A_calibration.jsonl results/grid_B_regime.jsonl
need results/grid_C_power.jsonl results/grid_D_size.jsonl
need results/grid_E_donor.jsonl
need results/benchmark_gse176078_all.csv results/benchmark_gse161529_all.csv

# The realised tie-break share comes first, and this is the one ordering in the
# chain that looks wrong until it is read: ``analyse_benchmark.py`` *merges* that
# file into the benchmark table when it is present and prints a note when it is
# not, so running the two the other way round does not fail.  It quietly drops
# level 2 to the detection profile alone and writes a summary that is a mixture
# of the two runs -- the failure the header of this file is about, and the only
# one here that a passing chain would not announce.
step "realised tie-break share per compartment" \
     python experiments/compartment_tie_break.py --dataset gse176078
step "the same for the second cohort" \
     python experiments/compartment_tie_break.py --dataset gse161529
step "operating characteristics from the sweeps" \
     python experiments/analyse_grid.py
step "real-data benchmark summary" \
     python experiments/analyse_benchmark.py
# The audit is rerun here rather than reused: it applies the matched-null test,
# and the test's tail convention changed after the sweep it was first run on.
step "audit of the matched null's residual" \
     python experiments/audit_matched_null.py
# The decomposition residual is remeasured rather than reused, for the same
# reason the audit is: the manuscript quotes its magnitude as a measured
# number, and a stale copy would keep quoting after the scoring changed.
step "residual of the score decomposition" \
     python experiments/measure_residual.py
step "manuscript numbers" \
     python experiments/make_numbers.py
step "companion figure" \
     python experiments/make_figure.py
# The bibliography is the one input nothing else regenerates, so it is checked
# against the registry rather than against a previous run.  It sits after the
# figure because it is the only step that needs the network, and a machine
# without one should still get that far.
step_soft "references against Crossref" \
     python experiments/verify_references.py

if [ "${1:-}" != "--no-latex" ]; then
    # ``-cd`` builds from the manuscript's own directory, and it is required
    # rather than cosmetic: the manuscript inputs ``numbers.tex`` by a relative
    # path, and ``\input`` is resolved against the working directory, so a
    # build launched from the package root stops at "File `numbers.tex' not
    # found" and produces no PDF.  The format check below reads the log the
    # build writes beside the source, which is where ``-cd`` puts it.
    step "manuscript build" \
         latexmk -cd -pdf -interaction=nonstopmode manuscript/main.tex
    # Last, because it reads the log the build just wrote: an unresolved
    # control sequence is only visible there.
    step "format and macro check" \
         python experiments/check_format.py
fi

printf '\n%s  pipeline complete\n' "$(date '+%H:%M:%S')" | tee -a "$LOG"
