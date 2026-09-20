#!/bin/bash
# Wait for the grid to finish, then run the steps that read its output.
#
# The audit re-derives the matched null's residual with the corrected tail
# direction, which the grid rerun is also applying; running it before the grid
# finishes would have it contend for cores that the grid is already using, and
# running it before the code change would have it measure the old rule.  So it
# is queued behind the grid rather than beside it.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")" || exit 1

while pgrep -f "experiments/[r]un_grid.py" >/dev/null; do sleep 20; done
echo "=== grid finished at $(date +%H:%M:%S) ==="

echo "=== audit ==="
python experiments/audit_matched_null.py --workers 3 2>&1 | tail -20

echo "=== numbers ==="
python experiments/make_numbers.py 2>&1 | tail -20

echo "=== compile ==="
cd manuscript
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "undefined control sequences in log: $(grep -c '^! ' main.log)"
grep "Output written" main.log
echo "=== chain done at $(date +%H:%M:%S) ==="
