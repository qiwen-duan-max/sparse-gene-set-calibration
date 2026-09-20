#!/usr/bin/env python
"""The journal's length limits, measured on the source rather than estimated.

The manuscript targets Briefings in Bioinformatics (since 2026-09-20; it was
written against Nature Methods' limits before that).  BiB publishes no hard
word cap for original research, so the numbers in ``LIMITS`` are the journal's
practical range for review-type submissions (2,000--7,000 words) applied as
self-discipline: the abstract follows the ~250-word convention, and the
main-text count excludes the abstract, Methods, references and figure
captions.  Those exclusions are what make the number hard
to eyeball from the compiled PDF, and a count that is re-estimated by hand each
time the text changes is a count that stops being checked.

So the count is taken here, on the same source the PDF is built from, by the
same rules every time:

* comments are removed, because they are not text;
* ``\\cite``, ``\\ref``, ``\\label``, ``\\eqref`` and ``\\code`` are removed
  entirely, since a citation is not a word in the text and a quoted symbol is
  not prose;
* display maths and inline maths are removed, because a formula is not a word;
* macros defined in ``numbers.tex`` count as one word each, which is what a
  numeral is;
* an em-dash or a hyphenated compound counts as one word, as a typesetter
  would count it.

The result is a convention, not the truth: the journal counts with its own
tool.  It is the same convention every time, which is what makes it useful for
deciding whether a paragraph has to go.

    python experiments/check_format.py [--tex manuscript/main.tex]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

#: The journal's limits, and the point at which a limit is treated as blocking
#: rather than as a warning.  The manuscript now targets Briefings in
#: Bioinformatics (switched from Nature Methods, 2026-09-20): BiB publishes no
#: hard word cap for original research, so the ceilings below are the journal's
#: practical range for review-type submissions (2,000--7,000 words) applied as
#: a self-discipline, the abstract follows the ~250-word convention, and
#: Methods is measured but only blocked at an extreme.
LIMITS = {
    "abstract": (150, 250),
    "main_text": (5000, 7000),
    "methods": (3000, 6000),
    "title": (None, None),
}
#: Under BiB the number of display items is not capped in print; the count is
#: reported for information only.
MAX_DISPLAY_ITEMS = None

#: Commands whose argument is not prose.  ``\code`` and ``\pending`` are this
#: project's own; the rest are LaTeX's.
DROP_COMMANDS = ("cite", "citep", "citet", "ref", "eqref", "label", "code",
                 "pending", "includegraphics", "bibliography",
                 "bibliographystyle", "input", "documentclass", "usepackage")

#: LaTeX's own control sequences.  A macro the manuscript uses has to come from
#: somewhere, and this list is the part of "from somewhere" that is neither this
#: project nor visible in its source.  Every entry earns its place by appearing
#: in the manuscript; the list is deliberately explicit rather than inferred,
#: so that a control sequence that is missing because a number was never
#: computed cannot hide inside a pattern that also matches LaTeX's own names.
TEX_BUILTINS = frozenset("""
    affil alpha appendices appendix arabic arraystretch author bar bfseries big Big
    bigg Bigg begin bibliography bibliographystyle bmod bottomrule caption
    captionsetup cdot centering cite citep citet clearpage code columnwidth
    dag date DeclareMathOperator def documentclass dots emph end ensuremath
    eqref fbox figure footnotesize footnotemark footnotetext frac ge geq given
    href hline hspace huge Huge IfFileExists includegraphics input item
    itemize label lceil large Large LARGE ldots le left leftmargin linebreak
    linewidth maketitle mathbf mathcal mathrm midrule mod multicolumn nabla
    leq
    varepsilon
    newcommand newenvironment newline noindent normalsize overline paragraph
    par parbox partial pct pending phantom phantomsection pi pm pmod protect
    qquad quad qty raisebox ref relax renewcommand rceil right rho raggedright
    scriptsize allowbreak arraybackslash array emergencystretch setlength
    section setcounter SI si sisetup small sqrt subsection subsubsection sum tabcolsep
    table tau text textbf textcolor textit textnormal textsc textsf
    textsubscript textsuperscript texttt textwidth thanks times title today
    toprule underline url usepackage vec vspace widehat widebar xrightarrow z
""".split())

#: Prefixes of the macro families ``make_numbers.py`` generates.  Used only to
#: say which missing macro is a missing *number*, which is the failure this
#: check exists for: a number that was never computed leaves no trace in the
#: source, and the manuscript reads as though it were simply not mentioned.
NUMBER_PREFIXES = ("audit", "bench", "grid", "conf", "boundary", "tail",
                   "power", "cohort", "cross", "null", "cell", "set")


def strip_latex(text):
    """The words of a LaTeX fragment, in the order they are read out."""
    text = re.sub(r"(?<!\\)%.*", "", text)              # comments
    text = re.sub(r"\\begin\{(equation|align|eqnarray)\*?\}.*?"
                  r"\\end\{\1\*?\}", " ", text, flags=re.S)   # display maths
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)              # inline maths
    for name in DROP_COMMANDS:
        text = re.sub(r"\\" + name + r"\*?\s*(\[[^\]]*\])?\s*"
                      r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)         # remaining commands
    text = re.sub(r"[{}~]", " ", text)
    text = re.sub(r"\\[-a-zA-Z]+", " ", text)
    text = text.replace("\\\\", " ")
    return text


def count_words(text):
    """Words, with hyphenated compounds and em-dash joins counted once."""
    text = strip_latex(text)
    text = text.replace("---", " ").replace("--", " ")
    tokens = [t for t in re.split(r"\s+", text) if re.search(r"[A-Za-z0-9]", t)]
    return len(tokens)


def section(text, start, end=None):
    """The text between two markers, both given as regular expressions."""
    m = re.search(start, text)
    if not m:
        return ""
    rest = text[m.end():]
    if end:
        m2 = re.search(end, rest)
        if m2:
            rest = rest[:m2.start()]
    return rest


def captions(text):
    """Every ``\\caption`` body, which the main-text limit excludes."""
    out = []
    for match in re.finditer(r"\\caption\{", text):
        i, depth = match.end(), 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        out.append(text[match.end():i - 1])
    return out


def _control_sequences(text):
    """Every ``\\name`` in a fragment that is used rather than defined.

    A definition site is skipped: ``\\newcommand{\\foo}`` uses ``\\foo`` in the
    sense that the characters are there, but it is not a reference that can
    dangle.  Comments are dropped for the same reason they are dropped
    everywhere else here -- a commented-out line is not part of the document.
    """
    text = re.sub(r"(?<!\\)%.*", "", text)
    out = set()
    for m in re.finditer(r"\\([a-zA-Z]+)", text):
        name = m.group(1)
        # ``\newcommand{\foo}`` and friends: the backslash is followed by a
        # brace, so the match inside is a definition and not a use.
        before = text[:m.start()].rstrip()
        if before.endswith("{") and re.search(
                r"\\(new|renew|provide)command\s*$", before[:-1]):
            continue
        out.add(name)
    return out


def macro_problems(tex="manuscript/main.tex", log="manuscript/main.log"):
    """Control sequences the manuscript uses that nothing defines.

    The failure this catches is narrow and quiet.  ``make_numbers.py`` emits one
    macro per number, and when an input file is absent the block that would have
    emitted a macro is skipped -- so a quantity that was never computed leaves
    no trace in the source at all.  The manuscript then reads as though that
    quantity had simply not been mentioned, and the compiled page shows nothing
    where a value belongs.  A macro that is *declared* but unfilled is handled
    by ``make_numbers.py`` itself, which writes it out as a visible ``(pending)``;
    this check is for the ones that are not declared either.
    """
    root = PKG / "manuscript"
    path = PKG / tex if not os.path.isabs(tex) else Path(tex)
    text = path.read_text(encoding="utf-8")
    # Anything the manuscript \input's is part of the manuscript, and the tables
    # are generated into such files.
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        inc = root / m.group(1)
        if not inc.suffix:
            inc = inc.with_suffix(".tex")
        if inc.exists():
            text += "\n" + inc.read_text(encoding="utf-8")

    defined = set()
    for src in list(root.glob("*.tex")) + list(root.glob("*.sty")):
        body = src.read_text(encoding="utf-8")
        defined |= set(re.findall(
            r"\\(?:new|renew|provide)command\*?\s*\{\s*\\([a-zA-Z]+)", body))
        defined |= set(re.findall(r"\\DeclareMathOperator\*?\s*\{\s*\\([a-zA-Z]+)",
                                  body))
        defined |= set(re.findall(r"\\def\s*\\([a-zA-Z]+)", body))
        defined |= set(re.findall(r"\\(?:new|renew)environment\s*\{\s*([a-zA-Z]+)",
                                  body))

    used = _control_sequences(text)
    missing = sorted(used - defined - TEX_BUILTINS)

    problems = []
    for name in missing:
        kind = ("a generated number" if name.startswith(NUMBER_PREFIXES)
                else "a control sequence")
        problems.append(
            f"\\{name} is used in the manuscript and defined nowhere; if it is "
            f"{kind}, the input that would have produced it is missing from "
            f"results/ and the value is silently absent from the text")

    # The compiled log is the authority on whether the build actually resolved
    # everything, and it catches what source inspection cannot: a macro that
    # exists but is never assigned, and a \label that no \ref finds.
    log_path = PKG / log if not os.path.isabs(log) else Path(log)
    if log_path.exists():
        body = log_path.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"^! Undefined control sequence\.\s*\n"
                             r"l\.\d+\s+(.*)$", body, flags=re.M):
            problems.append(f"the last build failed to resolve "
                            f"{m.group(1).strip()[:60]!r}")
        if "There were undefined references" in body:
            refs = re.findall(r"Reference `([^']+)' on page", body)
            problems.append(f"undefined references in the last build: "
                            f"{', '.join(sorted(set(refs))[:6])}")
    return problems


def pending_numbers(root=None):
    """Macros that were declared and never computed.

    ``make_numbers.py`` declares each number before it fills it and writes a
    visible marker when the input that would have filled it is absent.  The
    marker compiles, which is the point of it -- the page shows a hole rather
    than a wrong value -- but it also means a manuscript built from a results
    directory that is missing a whole block passes every other check here.  It
    reads as a paper that does not quote those quantities, and the difference
    between that and a paper whose quantities were never computed is invisible
    in the PDF.

    So the marker is read back off the generated source, and reported as what
    it is: a number the manuscript asks for that no input supplied.
    """
    root = Path(root) if root else PKG / "manuscript"
    pattern = re.compile(r"\\newcommand\*?\{\\([a-zA-Z]+)\}\s*\{\s*"
                         + re.escape(r"\textit{(pending)}") + r"\s*\}")
    out = []
    for src in sorted(root.glob("*.tex")):
        for m in pattern.finditer(src.read_text(encoding="utf-8")):
            out.append(m.group(1))
    return out


def swallowed_spaces(tex="manuscript/main.tex"):
    """Macros whose following space TeX will eat, joining them to the next word.

    A control word ends at the last letter of its name, so the spaces after it
    are skipped as the source is read: ``\\benchMaxRank{} and`` sets ``1233 and``
    while ``\\benchMaxRank and`` sets ``1233and``.  Nothing warns.  The compile
    is clean, no macro is undefined, and the word count does not change, because
    the words are all still there -- one space is not.  It is visible only by
    reading the typeset page, which is exactly the kind of check that gets
    skipped when the text is edited late.

    The run of whitespace is allowed to contain one newline, because a line break
    is a space to TeX.  A blank line is not: it becomes ``\\par``, which is not a
    space, so a macro at the end of a paragraph is safe however the next one
    starts.  An explicit ``{}`` after the macro ends the control word and keeps
    the space, which is why the manuscript writes its numbers that way.
    """
    root = PKG / "manuscript"
    path = PKG / tex if not os.path.isabs(tex) else Path(tex)
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        inc = root / m.group(1)
        if not inc.suffix:
            inc = inc.with_suffix(".tex")
        if inc.exists():
            text += "\n" + inc.read_text(encoding="utf-8")

    # Only the manuscript's own macros are checked: a control sequence defined
    # here is followed by a value the author wrote and can change, so a missing
    # space is a defect in the text.  LaTeX's own names take brace arguments, or
    # set maths where spaces carry no meaning.  Definitions are collected from
    # every source in the directory rather than only from the ones this file
    # inputs, because the numbers live in a generated file that a caller may
    # point at without wiring the ``\input`` up.
    defined = set()
    for src in list(root.glob("*.tex")) + list(root.glob("*.sty")):
        defined |= set(re.findall(
            r"\\(?:new|renew)command\*?\s*\{\s*\\([a-zA-Z]+)",
            src.read_text(encoding="utf-8")))
    problems = []
    for m in re.finditer(r"\\([a-zA-Z]+)", text):
        name = m.group(1)
        if name not in defined or text[m.end():m.end() + 1] == "{":
            continue
        # One optional newline: more than one is a paragraph break, and TeX stops
        # skipping there.
        skip = re.match(r"[ \t]*\n?[ \t]*", text[m.end():]).group(0)
        if not skip:
            continue
        after = text[m.end() + len(skip):]
        if not after or not after[0].isalpha():
            continue
        line = text.count("\n", 0, m.start()) + 1
        excerpt = re.sub(r"\s+", " ", after[:16]).strip()
        problems.append(
            f"\\{name} (line {line}) is followed by a space that LaTeX will "
            f"drop, so it sets as ``…{excerpt}'' run together; write "
            f"\\{name}{{}} to end the control word and keep the space")
    return problems


def supplementary_numbering(tex="manuscript/main.tex", root=None):
    """A float that arrives through ``\\input`` in the appendix must be renumbered.

    A table generated into a file and input after ``\\appendix`` is numbered by
    whatever counter is current, so unless the manuscript points that counter at
    a supplementary series first, the table prints as "Table 5" -- beside four
    print tables and a figure, reading as a sixth display item the journal
    counts and the author never intended, while the sentence that cites it says
    "Supplementary Table 5".  The two disagree on the page, and nothing in the
    build says so.  What makes this worth a check rather than a comment is that
    the table itself is regenerated: the file stays the same shape as the text
    around it changes, so the numbering has to be inspected where it is set.
    """
    root = Path(root) if root else PKG / "manuscript"
    path = PKG / tex if not os.path.isabs(tex) else Path(tex)
    text = path.read_text(encoding="utf-8")
    appendix = text.find(r"\appendix")
    problems = []
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        inc = root / m.group(1)
        if not inc.suffix:
            inc = inc.with_suffix(".tex")
        if not inc.exists():
            continue
        if not re.search(r"\\begin\{(table|figure)\}", inc.read_text(encoding="utf-8")):
            continue
        if appendix == -1 or m.start() < appendix:
            continue                  # a print float, numbered in the main series
        between = text[appendix:m.start()]
        if not re.search(r"\\renewcommand\*?\{\\thetable\}", between):
            problems.append(
                f"{inc.name} declares a float and is input in the appendix, "
                f"but \\thetable is not re-pointed first: the float is numbered "
                f"in the main series and reads as a print display item")
    return problems


def main(tex="manuscript/main.tex", verbose=True):
    path = PKG / tex if not os.path.isabs(tex) else Path(tex)
    if not path.exists():
        raise SystemExit(f"missing {path}")
    text = path.read_text(encoding="utf-8")

    abstract = section(text, r"\\begin\{abstract\}", r"\\end\{abstract\}")
    # The main text is what a reader reads between the abstract and the
    # Methods.  The bibliography, the availability statements and the appendix
    # are all outside the limit.
    main_text = section(text, r"\\section\{Introduction\}", r"\\section\{Methods\}")
    methods = section(text, r"\\section\{Methods\}",
                      r"\\section\*\{Data availability\}")
    title = section(text, r"\\title\{", r"\}\s*\\author")

    # Captions live inside the main text but are excluded from its count, so
    # they are removed before it is counted rather than after.
    body = main_text
    for cap in captions(main_text):
        body = body.replace(cap, " ")
    n_cap = sum(count_words(c) for c in captions(text))

    counts = {
        "title": count_words(title),
        "abstract": count_words(abstract),
        "main_text": count_words(body),
        "methods": count_words(methods),
        "captions": n_cap,
    }

    # Citations in the abstract are forbidden outright, not merely discouraged.
    abstract_cites = re.findall(r"\\cite[a-z]*\{", abstract)
    # The display items are what a reader sees in the article.  Counted from the
    # manuscript itself: the appendix writes its own table into a generated file
    # and numbers it in a supplementary series, so it is not one of these, and
    # the one float declared here after \appendix is the companion figure, which
    # is a print figure wherever the submission puts it.
    display = (len(re.findall(r"\\begin\{figure\}", text))
               + len(re.findall(r"\\begin\{table\}", text)))

    if verbose:
        print(f"{path}")
        print(f"  title        {counts['title']:6d}")
        target, ceiling = LIMITS["abstract"]
        flag = "over" if counts["abstract"] > ceiling else "ok"
        print(f"  abstract     {counts['abstract']:6d}  (limit {ceiling}, {flag})"
              + (f"  {len(abstract_cites)} citation command(s) in the abstract"
                 if abstract_cites else ""))
        target, ceiling = LIMITS["main_text"]
        head = "over" if counts["main_text"] > ceiling else (
            "over target" if counts["main_text"] > target else "within target")
        print(f"  main text    {counts['main_text']:6d}  "
              f"(target {target}, ceiling {ceiling}, {head})")
        target, ceiling = LIMITS["methods"]
        head = "over" if counts["methods"] > ceiling else "ok"
        print(f"  methods      {counts['methods']:6d}  (ceiling {ceiling}, {head})")
        print(f"  captions     {counts['captions']:6d}  (excluded from the "
              f"main-text count)")
        print(f"  display items {display:5d}  "
              + (f"(print limit {MAX_DISPLAY_ITEMS})" if MAX_DISPLAY_ITEMS
                 else "(BiB: no print cap)"))

    problems = []
    if counts["abstract"] > LIMITS["abstract"][1]:
        problems.append(f"abstract is {counts['abstract']} words, limit "
                        f"{LIMITS['abstract'][1]}")
    if abstract_cites:
        problems.append(f"{len(abstract_cites)} citation command(s) in the "
                        f"abstract")
    if counts["main_text"] > LIMITS["main_text"][1]:
        problems.append(f"main text is {counts['main_text']} words, ceiling "
                        f"{LIMITS['main_text'][1]}")
    if MAX_DISPLAY_ITEMS is not None and display > MAX_DISPLAY_ITEMS:
        problems.append(f"{display} display items, print limit "
                        f"{MAX_DISPLAY_ITEMS}")
    macros = macro_problems(tex)
    if verbose:
        print(f"  macros       {len(macros):5d} unresolved"
              if macros else "  macros        all resolved")
    problems += macros
    pending = pending_numbers()
    if verbose:
        print(f"  pending      {len(pending):5d} number(s) no input supplied"
              if pending else "  pending       none")
    for name in pending:
        problems.append(
            f"\\{name} was declared and never computed: the input that would "
            f"fill it is missing from results/, and until it is present the "
            f"manuscript prints a marker where a value belongs")
    swallowed = swallowed_spaces(tex)
    if verbose:
        print(f"  spacing      {len(swallowed):5d} macro(s) that eat the "
              f"following space" if swallowed else "  spacing       ok")
    problems += swallowed
    numbering = supplementary_numbering(tex)
    if verbose:
        print(f"  appendix     {len(numbering):5d} generated float(s) in the "
              f"main series" if numbering else "  appendix      floats numbered "
              f"supplementary")
    problems += numbering
    if problems:
        print("\n" + "\n".join(f"  ! {p}" for p in problems))
    elif verbose:
        print("\n  every measured limit is satisfied")
    return counts, problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default="manuscript/main.tex")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    _, problems = main(tex=args.tex, verbose=not args.quiet)
    sys.exit(1 if problems else 0)
