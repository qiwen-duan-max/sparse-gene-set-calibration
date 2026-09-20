#!/usr/bin/env bash
# Build the two distributable artefacts from the current source tree.
#
#   dist/sparse_gene_set_calibration-<version>-py3-none-any.whl
#   dist/sparse_gene_set_calibration-<version>.tar.gz
#   dist/sparseGenSetCal_<version>.tar.gz
#
# Everything downstream of the packages -- the manuscript's numbers, the
# accompanying figure, the tutorials -- is produced by run_pipeline.sh.  This
# script is the other direction: it takes the packages as they now stand and
# writes what a reader would install.  It is deliberately not part of that
# pipeline, because a manuscript revision does not change a wheel.
#
# The R package is built with R CMD build, which runs the vignette; that needs
# knitr and rmarkdown, and a missing one is reported here rather than producing
# a tarball whose vignette is a stub.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Python: building wheel and sdist"
python -m build --outdir dist >/dev/null

echo "==> Python: checking the wheel imports and answers"
python - <<'PY'
import glob
import subprocess
import sys
import tempfile
import zipfile

wheel = sorted(glob.glob("dist/sparse_gene_set_calibration-*.whl"))[-1]
with zipfile.ZipFile(wheel) as z:
    names = z.namelist()
missing = [n for n in ("sparsegs/__init__.py", "sparsegs/calibrate.py",
                       "sparsegs/events.py") if n not in names]
if missing:
    sys.exit(f"wheel is missing {missing}")

# Imported from a clean copy rather than from the source tree, so the check is
# about what the wheel contains and not about what happens to be beside it.
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--target", tmp, wheel], check=True)
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import sparsegs as sg; "
         "print(sg.__version__, len(sg.__all__), "
         "sorted(n for n in ('set_log_level', 'MIN_BACKGROUND_GENES', "
         "'MIN_SCORE_SD', 'SUSPICIOUS_AUC') if n in sg.__all__))" % tmp],
        capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(out.stderr[-2000:])
    print("    " + out.stdout.strip())
PY

echo "==> R: checking the package loads, then building the source tarball"
Rscript -e 'suppressMessages(pkgload::load_all("r-package/sparseGenSetCal", quiet = TRUE));
            missing <- setdiff(c("set_log_level", "MIN_BACKGROUND_GENES",
                                 "MIN_SCORE_SD", "SUSPICIOUS_AUC"), ls("package:sparseGenSetCal"));
            if (length(missing)) stop("not exported: ", paste(missing, collapse = ", "));
            cat("    exports:", length(ls("package:sparseGenSetCal")), "\n")'

( cd dist && R CMD build ../r-package/sparseGenSetCal >/dev/null )

echo "==> dist/"
ls -1sh dist/
