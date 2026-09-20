#!/usr/bin/env python
"""Every reference, checked against the registry that issued its DOI.

A bibliography is the one part of a manuscript that nothing else in the
pipeline touches.  The numbers are regenerated from the results, the figure is
redrawn from the results, and both would break loudly if the inputs changed --
but a reference is typed once and then only read, and a wrong volume, a wrong
year or a plausible-looking title attached to a DOI that resolves to a
different paper is invisible in the compiled PDF.  So each entry is sent to
Crossref and compared field by field.

The year needs one rule, because a DOI carries several dates and the one a
reference should show is not the first of them: an article published online in
December 2016 and printed in the March 2017 issue is cited as 2017, and
Crossref reports ``issued`` as the online date.  The comparison therefore reads
``published-print`` when the registry has it and falls back to ``issued`` for
records that predate the distinction -- which is what a reader copying the
citation from the publisher's own page would get.

Two checks run here.  The DOIs are resolved and compared, and the manuscript's
``\\cite`` keys are matched against the file both ways, because a citation to a
key that was renamed and a key nothing cites are each silent in the PDF: one
prints a question mark that survives a read-through, the other prints nothing at
all and leaves a reference list entry as a loose end.

    python experiments/verify_references.py            # check and report
    python experiments/verify_references.py --offline  # keys only, no network
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RESULTS = os.path.join(PKG, "results")

BIB = os.path.join(PKG, "manuscript", "references.bib")
TEX = os.path.join(PKG, "manuscript", "main.tex")

#: Crossref asks that automated callers identify themselves, and treats a
#: contact address as the difference between a client and a scraper.
UA = "sparsegs-reference-check/0.1 (mailto:noreply@example.org)"

#: How close two titles have to be to count as the same work.  Subtitle
#: punctuation and capitalisation differ between the registry and any BibTeX
#: file, so the comparison is made case- and punctuation-blind and the
#: threshold is set to catch a *different paper*, which is the failure that
#: matters, rather than to police editorial changes, which is not.
TITLE_FLOOR = 0.85
JOURNAL_FLOOR = 0.80


def normalise(text):
    """A string reduced to the words in it, for comparison across styles."""
    text = re.sub(r"<[^>]+>", "", text or "")
    for entity, char in (("&amp;", "&"), ("&#x2019;", "'"), ("&#39;", "'"),
                         ("&nbsp;", " ")):
        text = text.replace(entity, char)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_bib(path):
    """The entries of a .bib file, as (key, {field: value}).

    The fields are read one at a time rather than in a single sweep, and the
    value is taken by matching braces rather than by finding the next closing
    one.  Both choices are forced by the format: a single ``findall`` over the
    whole entry consumes the newline that separates one field from the next and
    so misses every second field, and a value may contain braces of its own --
    ``{GSVA}: gene set variation analysis`` -- which a non-greedy match stops
    at, quietly truncating the title that is about to be compared.
    """
    text = open(path, encoding="utf-8").read()
    chunks = [c.strip() for c in re.split(r"\n(?=@)", text)
              if c.lstrip().startswith("@")]
    entries = []
    for chunk in chunks:
        key = re.match(r"@\w+\{([^,]+),", chunk).group(1)
        fields = {}
        for match in re.finditer(r"(?m)^\s*(\w+)\s*=\s*", chunk):
            name = match.group(1).lower()
            rest = chunk[match.end():]
            if rest.startswith("{"):
                depth, value = 0, None
                for i, char in enumerate(rest):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            value = rest[1:i]
                            break
                if value is None:
                    raise ValueError(f"{key}.{name} has an unclosed brace")
            else:
                value = rest.split(",")[0]      # a bare value, e.g. a number
            fields[name] = re.sub(r"\s+", " ", value).strip()
        entries.append((key, fields))
    return entries


def cite_keys(path):
    """The citation keys the manuscript uses, with the lines they are on."""
    text = open(path, encoding="utf-8").read()
    keys = {}
    for match in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", text):
        line = text.count("\n", 0, match.start()) + 1
        for key in match.group(1).split(","):
            keys.setdefault(key.strip(), line)
    return keys


def fetch(doi, timeout=30):
    """The Crossref record for a DOI."""
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["message"]


def citation_year(message):
    """The year a reference to this record should carry."""
    for field in ("published-print", "issued", "published"):
        parts = (message.get(field) or {}).get("date-parts") or [[]]
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0]), field
    return "", "none"


def check_entry(key, fields, delay=0.4):
    """Resolve one entry and compare it with the registry's record."""
    row = {"key": key, "doi": fields.get("doi", ""), "status": "", "note": ""}
    doi = fields.get("doi", "")
    if not doi:
        row["status"] = "no-doi"
        row["note"] = "no identifier to check against; verify by hand"
        return row
    try:
        message = fetch(doi)
    except urllib.error.HTTPError as exc:
        row["status"] = "unresolved"
        row["note"] = f"Crossref returned {exc.code} for this DOI"
        return row
    except Exception as exc:                        # offline, DNS, timeout
        row["status"] = "error"
        row["note"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        if delay:
            time.sleep(delay)

    problems = []
    title = normalise((message.get("title") or [""])[0])
    mine = normalise(fields.get("title", ""))
    ratio = difflib.SequenceMatcher(None, title, mine).ratio()
    row["title_match"] = f"{ratio:.2f}"
    if ratio < TITLE_FLOOR:
        problems.append(f"title differs ({ratio:.2f}): the DOI resolves to "
                        f"{(message.get('title') or [''])[0]!r}")

    theirs, source = citation_year(message)
    row["crossref_year"] = theirs
    row["year_source"] = source
    if fields.get("year") and theirs and fields["year"] != theirs:
        problems.append(f"year {fields['year']} in the .bib, {theirs} at "
                        f"Crossref ({source})")

    journal = normalise((message.get("container-title") or [""])[0])
    mine_journal = normalise(fields.get("journal") or fields.get("booktitle", ""))
    if journal and mine_journal:
        jratio = difflib.SequenceMatcher(None, journal, mine_journal).ratio()
        row["journal_match"] = f"{jratio:.2f}"
        if jratio < JOURNAL_FLOOR:
            problems.append(f"journal differs: {(message.get('container-title') or [''])[0]!r}")

    row["status"] = "ok" if not problems else "check"
    row["note"] = "; ".join(problems)
    return row


def main(offline=False, outdir=RESULTS):
    entries = parse_bib(BIB)
    used = cite_keys(TEX)
    defined = {key for key, _ in entries}

    missing = sorted(set(used) - defined)
    uncited = sorted(defined - set(used))

    rows = []
    counts = {}
    for key, fields in entries:
        if offline:
            row = {"key": key, "doi": fields.get("doi", ""),
                   "status": "skipped", "note": "offline"}
        else:
            row = check_entry(key, fields)
            print(f"  {key:16s} {row['status']:11s} {row['note']}")
        rows.append(row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "reference_check.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {k for row in rows for k in row}), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    print(f"\n{len(entries)} entries, {len(used)} distinct keys cited")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")
    if missing:
        print(f"\ncited but not in the .bib: {missing}")
    if uncited:
        print(f"in the .bib but never cited: {uncited}")
    print(f"wrote {path}")

    # The exit code is what a pipeline step reads.  An unreachable registry is
    # not a bibliography defect, so it stops the run without claiming the
    # references are wrong; a mismatch is a defect and does.
    if missing or uncited:
        return 1
    if counts.get("unresolved") or counts.get("check"):
        return 1
    if counts.get("error"):
        return 2
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="skip the registry and check only the citation keys")
    ap.add_argument("--outdir", default=RESULTS)
    sys.exit(main(**vars(ap.parse_args())))
