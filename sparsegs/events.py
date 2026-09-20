"""Structured logging, one event per line.

Every diagnostic the framework emits is a *record*: an event name and a set of
named fields.  That is what makes a log worth keeping.  The alternative --- a
formatted sentence --- carries the same information and loses it again the
moment anyone wants to count how often a gene set had to be drawn from a
loosened pool, which is exactly the question the framework's edge cases exist
to answer.

Why this module rather than a logging library
---------------------------------------------
The brief this package was built to asks for ``loguru`` in Python and
``futile.logger`` in R.  Neither is a dependency here, and the reason is that a
package whose whole claim is "these numbers are reproducible" should be
installable without a logging framework that the caller did not choose.  What
the brief asks for is *structured* logging, and the standard library provides
it: the records below are emitted through :mod:`logging`, so they integrate with
whatever handler the calling application already has, and a caller who wants
``loguru`` gets it by adding a sink to the ``sparsegs`` logger --

    from loguru import logger
    import logging
    logger.add(lambda m: None)
    logging.getLogger("sparsegs").addHandler(MyLoguruBridge())

-- without this package having to decide for them.  The R port makes the same
choice for the same reason and reports through ``message()``/``warning()`` with
the same ``key=value`` body, so a log from either language parses the same way.

The format
----------
``event field=value field=value``, with values that contain spaces quoted.  A
line therefore reads as a sentence and parses as a record.
"""

from __future__ import annotations

import logging
import os

__all__ = ["set_log_level"]

#: The one logger this package writes to.  Named rather than the root logger so
#: that a caller can silence the framework without silencing their application,
#: and so that ``logging.getLogger("sparsegs").setLevel(...)`` is all it takes
#: to turn the volume up.
LOGGER = logging.getLogger("sparsegs")

#: Level names accepted by :func:`set_log_level`, mapping the words a caller would
#: use onto the standard library's constants.
_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO,
           "warning": logging.WARNING, "error": logging.ERROR,
           "critical": logging.CRITICAL}

#: A handler is attached only once, and only when the caller has not configured
#: logging themselves.  A library that installs a handler unconditionally takes
#: over the application's logging configuration, which is a side effect nobody
#: asked for; a library with no handler at all is silent, which hides the very
#: warnings the edge cases exist to raise.  So: attach a plain handler only if
#: the logger has none, and leave it alone if it has.
_configured = False


def _configure():
    global _configured
    if _configured:
        return
    _configured = True
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
        # ``lastResort`` would otherwise print anything at WARNING or above
        # through a handler of its own, so the check above has to be paired with
        # a level rather than relying on the absence of handlers alone.
        LOGGER.propagate = False
    # The level is set only when the caller has not set one.  ``Logger.level``
    # is ``NOTSET`` until someone sets it, and a caller who reached for
    # ``logging.getLogger("sparsegs").setLevel(...)`` before the framework's
    # first record made the most explicit choice available -- overwriting it
    # here would print the warnings they had just turned off, and the effect
    # reads as the framework ignoring its own switch.  Found by a check that
    # raised the level before emitting anything and watched the records arrive
    # anyway.
    if LOGGER.level == logging.NOTSET:
        LOGGER.setLevel(_LEVELS[os.environ.get("SPARSEGS_LOG_LEVEL", "warning")
                               .lower()] if os.environ.get(
                                   "SPARSEGS_LOG_LEVEL", "").lower() in _LEVELS
                        else logging.WARNING)


def set_log_level(level="warning"):
    """Turn the framework's own logging up or down.

    Parameters
    ----------
    level : str
        One of ``debug``, ``info``, ``warning``, ``error``, ``critical``.

    Returns
    -------
    str
        The level that was set, so a caller can echo it back into a record.
    """
    _configure()
    key = str(level).lower()
    if key not in _LEVELS:
        raise ValueError(f"unknown log level {level!r}; "
                         f"expected one of {sorted(_LEVELS)}")
    LOGGER.setLevel(_LEVELS[key])
    return key


def _codify(value):
    """One field value, in a form that survives a round trip through text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # A float in a log is a measurement, and its precision is part of it;
        # ``6`` for ``6.02`` would be a different number.
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set, frozenset)):
        # A field that holds several values is written as a comma-joined list
        # rather than as its ``repr``.  The repr would put the value in Python's
        # syntax -- quotes around each element, brackets around the whole --
        # which is not what a reader of the log wants and not what the R port
        # can produce: the two implementations write the same record or the log
        # is not comparable across them.
        value = sorted(value) if isinstance(value, (set, frozenset)) else value
        return _codify(",".join(str(v) for v in value))
    text = str(value)
    if text == "" or any(c.isspace() or c == "=" for c in text):
        return '"' + text.replace('"', "'") + '"'
    return text


def event(level, event_name, **fields):
    """Emit one structured record.

    Parameters
    ----------
    level : str
        Standard library level name, lower case.
    event_name : str
        The event, as ``snake_case``.  Events are looked for by name, so the
        name is part of the interface and does not change once used.  The
        parameter is spelled ``event_name`` rather than ``name`` so that
        ``name`` stays available as a *field*: "which gene set was this" is the
        first thing anyone asks of a record, and a signature that reserved the
        obvious word for it would be reserving the wrong one.
    **fields
        The record's named values.

    Examples
    --------
    >>> from sparsegs.events import event
    >>> event("warning", "pool_loosened", gene="CD8A", stratum=3, kept=2)
    'pool_loosened gene=CD8A stratum=3 kept=2'
    """
    _configure()
    # An empty field is left out rather than written as ``context=""``: a record
    # whose fields are sometimes empty trains the reader to skip them, and the
    # one that matters is then skipped with the rest.
    body = " ".join(f"{k}={_codify(v)}" for k, v in fields.items()
                    if v is not None and v != "")
    record = f"{event_name} {body}".rstrip()
    LOGGER.log(_LEVELS.get(str(level).lower(), logging.WARNING), record)
    return record
