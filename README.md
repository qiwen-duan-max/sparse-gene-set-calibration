# sparsegs — calibration for sparse gene-set scores

A gene-set score computed on sparse single-cell data is a number with a null
attached. The number gets reported; the null gets assumed. This package supplies
the null the data can support, and a diagnostic that says — before any outcome is
looked at — whether the score is measuring the gene set or the sequencing depth.

Written by **ClawsGO Science Agent**. Every statistic in this README is computed
from the matrices and simulations shipped with the project; none is reproduced
from a published source.

---

## The problem, stated exactly

Every rank-based score (AUCell, UCell and relatives) asks where a gene sits
among *all* genes in a cell. A cell that detects fewer genes than the score's
rank ceiling has no counts to place in the remaining positions, and the
tie-break fills them with genes the cell never expressed.

Let the panel have $$G$$ genes, let cell $$c$$ detect $$D_c$$ of them, and let
$$m = \lceil fG \rceil$$ be the rank ceiling ($$f = 0.05$$ for AUCell). A gene
the cell never expressed lands in one of the remaining $$m - D_c$$ slots with
probability $$(m-D_c)/(G-D_c)$$, and its expected share of an AUCell score is

$$\tau(D) = \frac{(m-D)(m-D+1)}{2\,m\,(G-D)}. \tag{1}$$

Writing $$d_c(S)$$ for the number of genes of set $$S$$ detected in cell $$c$$,
the score splits exactly:

$$\mathrm{AUC}_c = \mathrm{Det}_c + \frac{k-d_c(S)}{k}\,\tau(D_c) + \text{noise}. \tag{2}$$

The first term is the gene set. The middle term is the same for every set with
the same detection profile, and it is a *decreasing function of depth*. Any axis
that correlates with sequencing depth therefore acquires a correlation with the
score that owes nothing to biology — and unlike most confounds, this one is
computable in closed form from the depth vector alone, before any score exists.

The quantity an expression-matched null has to reproduce follows from the same
algebra. A gene detected in a fraction $$p$$ of cells enters the top $$m$$
either because it was counted or, failing that, by tie-break:

$$P(\text{in top } m) = p + (1-p)\,\overline{\frac{m-D}{G-D}}. \tag{3}$$

Matching on $$p$$ alone leaves the second term free; because it varies with the
cell's depth, the mismatch is not a constant that cancels — it is a term that
tracks the confounder.

### How open the channel is

At the 5% AUCell default on a 24,646-gene panel, $$m = 1233$$. A CD8⁺ T cell in
GSE176078 detects a median of 882 genes, so the ceiling sits 40% above the
median cell's entire expressed set. Most of the "ranking" is a coin toss among
genes the cell does not have.

---

## What is in the package

| Module | What it is for |
|---|---|
| `theory` | The closed form: `tie_break_level` (τ(D), the expected tie-break share), `effective_inclusion` (the tie-break's inclusion probability), `tie_break_share`, `score_decomposition` (the exact realised split, with its expectation), `preflight`, `confound_band`. |
| `rankcache` | Ranks computed once and clipped at a ceiling, with each cell's depth recorded. `build_streaming` never materialises the matrix; `subset_cells` slices a cache to one cell type. |
| `tiebreak` | How a tie is settled — deterministically and identically in both languages, instead of by a draw from a generator. |
| `scores` | `aucell`, `ucell`, `score_genes`, `module_score`, `ssgsea`, `score_all` over the cache. |
| `nulls` | Three nested null families — uniform, expression-matched, co-detection-matched — each reporting the composition it actually achieved. Plus `adaptive_detection_floor`. |
| `diagnostics` | `sparsity_report`, `depth_report`, `comparability_report`, `verdict`. |
| `calibrate` | The assembled surface: `Calibrator`, the four-step `diagnostic_checklist`, `construct_expression_matched_null`, `compute_calibrated_score`, `validate_tier_framework`, `to_json`. |
| `convenience` | `select_null_method` — the decision table as a pure function — and `predict_fpr` — the fitted rejection probability, with its provenance attached. Run before any scoring, from quantities the caller already has. |
| `analyses` | The four conventional tests the framework is calibrated against: naive correlation, label permutation, median cutpoint, outcome-chosen cutpoint. |
| `comparators` | Head-to-head against `decoupler`'s published methods, called through the real package rather than reimplemented, plus the two null strategies a careful analyst uses unaided. |
| `stats` | Spearman, AUC, Monte-Carlo P values, `cutpoint_fpr`. |
| `tiers` | Technical / internal / external validation, with the tiers regime-matched. |
| `simulate` | Ground-truth simulation with an explicit depth–programme confounder, used by the grid in `experiments/`. |
| `report` | A self-contained HTML calibration report for one gene set. |
| `events` | The structured records: one `event field=value` line per diagnostic, so a run can be counted as well as read. |

The reference for every name above is [`docs/API.md`](docs/API.md), which is
generated from the package docstrings by `python docs/build_api.py`. A test
regenerates it and checks that every export appears in it, so the manual cannot
drift away from the code.

---

## Quick start

```python
import sparsegs as sg

# 1. Ranks, once, clipped.  Keep only the columns a gene set can query.
cache = sg.RankCache.build(X, genes, ceiling=1700,
                           genes_of_interest=candidate_genes)
# cache.depth is D_c — the D of τ(D) and of the channel probability π(D).

# 2. The pre-flight check.  No gene set scored, no outcome looked at.
depth = cache.depth
pf = sg.preflight(depth, exhaustion_axis, max_rank=1233, n_genes=24646,
                  detection=cache.detection[cache.positions(gene_set)], k=30)
pf["tau_axis_rho"], pf["implied_rho"], pf["verdict"]     # 'CONFOUNDED'

# 3. A null matched on what the gene set looks like, not just how big it is.
null = sg.construct_expression_matched_null(cache, gene_set, n_sets=200)
null["status"], null["max_overlap"]                      # 'PASS', 0

# 4. Test against it.
score = sg.aucell(cache, gene_set)
observed = sg.spearman(score, axis)[0]
null_rho = [sg.spearman(sg.aucell(cache, s), axis)[0] for s in null["null_genes"]]
sg.null_summary(null_rho, observed, tail="lower")["p_value"]
```

Or the whole thing in one call:

```python
cal = sg.Calibrator.from_matrix(X, genes, rank_frac=0.05, seed=42)
out = cal.calibration_object(CLOCK_REPRESSORS, axis=exhaustion,
                             cutpoint_pre_specified=0.0)
sg.to_json(out, "calibration.json")

out["checklist"]["overall_verdict"]        # 'NOT_IDENTIFIABLE'
out["checklist"]["step1_sparsity"]["tie_break_share"]   # 0.635
out["false_positive_rates"]                # {'optimum_cutpoint_FPR': 0.44, ...}
```

---

## The four-step checklist

The order is the argument. Each step is cheaper and more decisive than the one
after it, and step 1 can invalidate the rest before anything is tested.

**Step 1 — sparsity.** `tie_break_share` is the exact share of the set's chance
of entering the score that comes from tie-breaking rather than from expression,
from the inclusion probability. It is a property of the gene set *in this
data*, not of a
threshold anyone chose. Above 0.75 the score is largely a depth proxy and the
verdict is `NOT_IDENTIFIABLE`.

Before this quantity existed the check was a fixed detection floor, and the
package keeps that as an explicitly labelled fallback for caches built without
depth. It is weaker, and the difference is not academic: a floor of 0.02 applied
to a set whose genes are detected in 0.4–1.8% of cells reports that the set is
detected nowhere. Every replacement drawn for it is then denser than the gene it
stands in for.

**Step 2 — null construction.** A replacement set matched on detection,
expression and co-detection, reported with the composition it *achieved*. A null
that failed to match is not a null, and its P value is not evidence.

One failure mode is worth stating separately because it is invisible in the
output unless it is measured. The natural nearest-neighbour match — for each
gene of the set, find the genes closest to it in detection and mean expression —
excludes only the single gene being stood in for. The set's *other* members
remain eligible. On a 30-gene set drawn from a narrow detection band, which is
what a real pathway looks like, the draws that resulted contained a median of
**12 of the set's own 30 genes**, every draw shared at least one, and the
resulting "null" score correlated with the observed score at $$\rho = 0.43$$.
Every P value taken from such a null is conservative. `construct_expression_matched_null`
therefore returns `max_overlap` and `n_from_query` as measured values, and
`status` is `PASS` only when the overlap is zero and the detection rate matched.

**Step 3 — cutpoint calibration.** If the analysis dichotomises the score,
measure the false-positive rate of the rule rather than assuming the nominal
level. `cutpoint_fpr` does this by permutation. The gap between an
outcome-driven cutpoint and a pre-specified one is routinely a factor of five or
more.

**Step 4 — external replication.** Sign and significance in an independent
cohort. Because `max_rank` is a fraction of the gene panel, scores from two
cohorts are not comparable as *values*; only direction and rank association are
compared.

Running the step shows what it can carry. Between GSE176078 and GSE161529 the
channel reproduces — median tie-break inclusion 0.006 against 0.017, ceiling
1,233 against 1,677 — while the association does not: of the 1,077 gene sets
scored in both, 57% keep the sign of their association with the compartment's
dominant axis and the median set moves by 0.06. Restricted to the 70 sets that
clear $$|\rho| = 0.2$$ in the first cohort, 83% keep the sign, but the
association itself falls from a median of 0.25 to 0.04. A replication of this
design can confirm a direction; it cannot confirm an effect size, and the
checklist asks it for no more than that.

---

## Reading a diagnostic

`sparsity_report` returns two zero rates, and the gap between them is the signal:

- `structural_zero_rate` — the fraction of cells that *must* score zero because
  the gene set's genes cannot all be ranked inside `max_rank`. Computed from set
  size and panel size alone.
- `observed_zero_rate` — the fraction that *does* score zero.

A large positive gap means the set is sparser than its size implies: its genes
are rarely detected, so more of the score comes from tie-broken empty ranks than
the set size predicts. That is the regime where a conventional null stops being
valid, and it is visible before the score is compared to anything.

`verdict` reports which criterion produced it, in `criterion` — `tie_break_share`
when depth is known and `fixed_detection_floor` when it is not — so a reader can
tell how much weight the verdict carries without reading the source.

---

## The three null families

They are nested, and the nesting is the point. Each refines the previous one:

| Family | Matches | Typical mismatch it leaves |
|---|---|---|
| `random` | nothing beyond set size | detection rate, expression, co-detection |
| `expression_bin` | the set's *mean* detection rate | per-gene detection, expression, co-detection |
| `expression` | per-gene detection rate and mean expression, within decile strata, refined by relative distance | co-detection structure |
| `codetection` | the above, plus mean pairwise co-detection | residual correlation structure |

All four exclude the set's own genes. The package reports the composition each
family achieved rather than asserting it matched, so a family that failed to
match is visible in `compare_nulls` output.

---

## When it refuses, and when it only warns

Three inputs break the framework rather than stretching it, and the two
responses are different on purpose.

**A gene set with no spread.** If the set's genes are detected in no cell, or in
every cell, the score is the same number for every cell and its correlation with
any axis is `0/0`. The call raises `ValueError` ("*Gene set score has zero
variance...*") at the point the score is made, rather than returning a `NaN`
that surfaces much later with nothing left to say which set produced it. The
check is against `MIN_SCORE_SD`, not against zero: the score is a sum of
doubles, and bit-identical sums are not guaranteed.

**A null that cannot be drawn.** Every draw excludes the query set, so what can
stand in for a 400-gene set is the background *minus those 400 genes*. When
fewer than `MIN_BACKGROUND_GENES` candidates remain, the detection floor is
lowered by `background_loosen_factor` (10%) and the widening is announced in the
log; if that still leaves too few, the call raises "*Cannot construct matched
null. Consider different gene set.*" A replacement drawn from five genes is not
a null, and the P value it produced would be read as a test.

**A score that separates its groups almost perfectly.** An AUC at or above
`SUSPICIOUS_AUC` is a statement about the data, not about the score, and the
record carries the three explanations worth checking (`batch_effects`,
`outcome_leakage_into_cell_type_annotation`, `extreme_sparsity`). This one warns
rather than refusing, because a marker that marks exactly what it is supposed to
mark also lands here. The check is on `max(auc, 1 - auc)`: `auroc` does not fold
its argument about 0.5, and a score that lands every high-axis cell *below*
every low-axis cell separates them just as perfectly.

---

## The log

Every diagnostic is a *record* rather than a sentence — an event name and a set
of named fields, written `event field=value`, with values containing a space
quoted and empty fields dropped:

```
suspicious_score_separation context="cutpoint calibration" auc=1 separation=1 threshold=0.99 check=batch_effects,outcome_leakage_into_cell_type_annotation,extreme_sparsity
```

That makes a run countable and not only readable: *how many gene sets had to be
drawn from a widened pool* is a question with an answer. Python writes the
records through `logging` on the `sparsegs` logger — `set_log_level("error")`
moves the floor for the framework's records and nothing else — and R writes the
same text through `message()`/`warning()` with conditions of class
`sparsegs_event`. `experiments/cross_language_check.py` asserts that the two
implementations emit the same record, character for character, which is a
stronger claim than comparing them by eye.

---

## Cost

The rank cache is why the calibration is affordable: scoring is
$$O(n_\text{cells} \times k)$$ per gene set after a one-time
$$O(n_\text{cells} \times n_\text{genes})$$ pass, instead of a full re-ranking
per gene set per null draw.

Memory shapes the API. A cache over all 55,003 cells of GSE176078 is 2.7 GB at
`int16`; over one cell type's 4,000 cells it is 134 MB and the ranks are
identical, because a rank is computed within one cell and does not depend on
which other cells are present. `build_streaming` and `subset_cells` exist for
that reason, and `RankCache` records `D_c` for the same reason — the closed form
needs it, and recomputing it means holding a matrix the caller may no longer
have.

---

## Tests

```bash
python -m pytest tests/ -q
```

127 tests. Three of them are load-bearing and the rest depend on them:

- `test_cache_reproduces_reference_aucell` — the cache must reproduce the
  scoring engine this project already used, exactly. Everything downstream
  inherits its correctness.
- `test_score_decomposition_is_an_identity` — `detected + tie_break` must equal
  the score cell by cell: the exact half of the decomposition, verified rather
  than asserted.
- `test_preflight_implied_rho_tracks_the_measured_spurious_correlation` — for a
  set that is never detected anywhere, the score's correlation with a
  depth-derived axis must equal the predicted value once the tie-break's own
  cell-to-cell fluctuation is accounted for. Two parameters, nothing fitted.

`test_preflight_flags_an_axis_that_is_depth_in_disguise` also checks the
chance band empirically: over 200 null axes the flag must fire at the rate the
band claims and not more, which is what stops the CLEAR threshold from being a
number someone liked.

---

## From a finished sweep to a checked manuscript

The manuscript quotes no number that is not computed here, and the chain that
computes them refuses to run out of order:

```bash
bash experiments/run_pipeline.sh          # add --no-latex to stop before the build
```

It recomputes the per-compartment tie-break shares, the grid summaries and the
benchmark summary, regenerates `manuscript/numbers.tex`, redraws the figure,
builds the PDF and then checks the build log for an undefined macro or an
unresolved reference. The order is not incidental: `analyse_benchmark.py`
*merges* the per-compartment file when it is present and merely notes it when it
is not, so running that pair the wrong way round produces a complete-looking
summary of a mixture of two runs rather than an error. Each step is checked and
the first failure stops the chain.

Four failures a compiled PDF shows but does not announce are caught at the end.
A macro the manuscript uses and nothing defines compiles into a missing word;
a macro `make_numbers.py` *declared* and then left unfilled compiles into a
visible marker; and a number macro followed by a space and a word loses the
space, because TeX ends a control word at its last letter and skips what follows
— `\benchMaxRank and` sets as `1233and`, cleanly and without a warning. So
`check_format.py` reads the generated `numbers.tex` back for the first two and
the manuscript source for the third, separately from the word counts it takes.
The fourth is the appendix table, which arrives through an `\input` of a
generated file: numbered in the main series it prints as "Table 5" beside four
print tables and a figure, which is a display item the journal counts and the
sentence citing it calls "Supplementary Table 5" — so the check requires the
manuscript to move that table into its own series before it inputs it. A results
directory missing a whole block therefore fails the chain instead of producing a
paper that reads as though it had never quoted those quantities.

The bibliography is checked separately, because it is the one input nothing
regenerates:

```bash
python experiments/verify_references.py   # --offline to check only the keys
```

It resolves every DOI at Crossref and compares title, journal and year against
the entry, and matches the manuscript's `\cite` keys against the file in both
directions — a citation to a renamed key and a key nothing cites are both
invisible in the PDF. The year is compared against the print date where the
registry has one: an article published online in December and printed in March
is cited by the print year, which is not the date Crossref reports first.

### What feeds what

Every artefact the manuscript shows traces to exactly one generating script:

| Artefact | Generated by | Input |
|---|---|---|
| Grid sweeps, `results/grid_*.jsonl` | `experiments/run_grid.py` | recorded seeds, five parameter grids |
| Real-data benchmark, `benchmark_*_all.csv` | `experiments/real_benchmark.py` | GSE176078, GSE161529 |
| Per-compartment tie-break shares | `experiments/compartment_tie_break.py` | each cohort |
| Operating characteristics (grid) | `experiments/analyse_grid.py` | `results/grid_*.jsonl` |
| Benchmark summary | `experiments/analyse_benchmark.py` | benchmark CSVs |
| Matched-null audit | `experiments/audit_matched_null.py` | grid sweep draws |
| Decomposition residual | `experiments/measure_residual.py` | grid-A configurations, 5 reps |
| Every macro in `manuscript/numbers.tex`, supplementary Tables S1–S6 | `experiments/make_numbers.py` | the summaries above |
| Companion figure | `experiments/make_figure.py` | the same summaries |
| Reference check | `experiments/verify_references.py` | `manuscript/refs.bib` vs Crossref |
| PDF build + measured-limit check | `experiments/run_pipeline.sh` | everything above, in order |

One input the table does not cover: the test suite's reference engine.
`tests/test_core.py` imports `common.score_matrix` -- the legacy
random-tie-break implementation the package is validated against -- which
lives at `reference/common.py`, extracted verbatim from the project's early
scoring scripts.  And on a fresh checkout of this repository the few tests
that read the simulation grids' records skip themselves, because those
records travel as the release attachment `raw_data_results.zip` rather than
in the repository; in the authors' working tree they run in full.

The two raw layers are the only steps with real wall-clock cost; everything
after them reruns in minutes. The environment files pin what the runs used:
`environment.yml` (Python 3.12, numpy 2.4.6, scipy 1.18.1, pandas 2.3.3 —
`pyproject.toml` holds the looser bounds) and `renv.lock` (R 4.3.3; generated
from `installed.packages()` on the host that produced the results — renv itself
was not used there, so the lockfile carries no Hash fields, which renv restores
from Package/Version without). `.github/workflows/ci.yml` runs the test suites
and the cross-language check on push; it does not run the pipeline, whose
benchmark stages need the two public cohorts.

The generated records and summaries themselves — the five grids' 2,800
JSON-line records, the benchmark CSVs and the audit outputs under `results/`
(131 files, ~52 MB) — are distributed with the release as the archive
`raw_data_results.zip`, attached to the v0.1.0 release and checksummed by
`results_md5.txt` in the repository root; `make_verification_table.py`
recomputes the headline rates from them and must report a worst difference
of 0.000.

---

## Attribution

Code written by **ClawsGO Science Agent**; the author reviewed and takes
responsibility for it. Sources of the methods compared against are cited in the
module docstrings, and every statistic reported here is computed from the data
in this repository.
