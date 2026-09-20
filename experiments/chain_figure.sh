#!/bin/bash
# Wait for the grid, then render Figure 1.
#
# Two inputs to the figure are still missing: the per-compartment tie-break
# share (panel b's y axis), which needs the rank cache rather than the
# benchmark's summary columns, and the figure itself.  Both are queued behind
# the grid so they do not compete with it for the four cores the machine has.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")" || exit 1

while pgrep -f "experiments/[r]un_grid.py" >/dev/null; do sleep 20; done
echo "=== grid finished at $(date +%H:%M:%S) ==="

echo "=== per-compartment tie-break share ==="
python experiments/compartment_tie_break.py --dataset gse176078 2>&1 | tail -25

echo "=== figure ==="
python experiments/make_figure.py 2>&1 | tail -25

echo "=== rendered files ==="
ls -la figures/ 2>&1

echo "=== done at $(date +%H:%M:%S) ==="
