#' Structured logging, one event per line
#'
#' Every diagnostic the framework emits is a *record*: an event name and a set
#' of named fields.  That is what makes a log worth keeping.  The alternative --
#' a formatted sentence -- carries the same information and loses it again the
#' moment anyone wants to count how often a gene set had to be drawn from a
#' loosened pool, which is exactly the question the framework's edge cases exist
#' to answer.
#'
#' @section Why this rather than a logging package:
#' The brief this package was built to asks for `futile.logger`.  It is not a
#' dependency here, for the same reason the Python port declines `loguru`: a
#' package whose whole claim is "these numbers are reproducible" should be
#' installable without a logging framework that the caller did not choose.  What
#' the brief asks for is *structured* logging, and base R provides the channel:
#' records go out through the condition system, so they arrive wherever the
#' caller already sends messages and warnings, and a caller who wants
#' `futile.logger` gets it by handing the records to it --
#'
#' ```
#' futile.logger::flog.info(record)
#' ```
#'
#' -- without this package having to decide for them.  The record format is
#' identical to the Python port's, so a log from either language parses the same
#' way.
#'
#' @section The levels:
#' `debug`, `info`, `warning`, `error`, `critical`, in increasing order.
#' Records below the current level are dropped before they are formatted, so
#' turning the volume down costs nothing.  A record at `warning` is signalled
#' with [warning()] and so is catchable with [tryCatch()] -- specifically, by the
#' class `sparsegs_event`, which lets a caller handle the framework's records
#' without touching their application's own warnings.  Every other level goes to
#' [message()]: a record at `error` or `critical` is emitted on the way to the
#' `stop()` that raises the failure, and signalling both would deliver one
#' failure as two conditions.  That split is the R-idiomatic one and is the only
#' respect in which this module differs from the Python port, which writes every
#' level to one logger.
#'
#' @name events
#' @keywords internal
NULL

#: The levels, in increasing order, with the number each is ranked at.  Named
#: rather than positional because the rank is the only thing the filter reads.
.sparsegs_levels <- c(debug = 10L, info = 20L, warning = 30L, error = 40L,
                      critical = 50L)

#: The level the framework starts at.  Warnings are the point of the edge cases
#: and are on by default; the two levels below would narrate every draw.
.sparsegs_state <- new.env(parent = emptyenv())
.sparsegs_state$level <- "warning"

#: A level name, validated and turned into its rank.  An unknown name is
#: refused rather than silently mapped onto the default, since a caller who
#: asked for "warn" and got "warning" cannot tell the difference between a
#: level that was accepted and one that was guessed at.
.sparsegs_level_rank <- function(level) {
    key <- tolower(as.character(level)[1])
    if (length(key) != 1L || is.na(key) || !key %in% names(.sparsegs_levels)) {
        stop(sprintf("unknown log level '%s'; expected one of %s",
                     as.character(level)[1],
                     paste(names(.sparsegs_levels), collapse = ", ")),
             call. = FALSE)
    }
    .sparsegs_levels[[key]]
}

#' Turn the framework's own logging up or down
#'
#' @param level One of `"debug"`, `"info"`, `"warning"`, `"error"`,
#'   `"critical"`.
#'
#' @return The level that was set, invisibly, so a caller can echo it back into
#'   a record.
#'
#' @export
set_log_level <- function(level = "warning") {
    rank <- .sparsegs_level_rank(level)
    key <- tolower(as.character(level)[1])
    .sparsegs_state$level <- key
    invisible(key)
}

#: One field value, in a form that survives a round trip through text.  Matches
#: the Python port's `_codify` character for character: booleans lower case,
#: numbers at six significant figures, anything holding a space or an `=`
#: quoted.  The two implementations write the same log, so a record can be
#: compared across them rather than merely understood.
.sparsegs_codify <- function(value) {
    if (is.null(value) || length(value) == 0L) {
        return(NULL)
    }
    if (is.logical(value) && length(value) == 1L) {
        return(if (is.na(value)) "NA" else if (value) "true" else "false")
    }
    if (is.numeric(value) && length(value) == 1L) {
        # NA and the infinities are not measurements and have no six-figure
        # form; `sprintf` would render NA as "NA" anyway but Inf as "inf",
        # which is neither R's spelling nor Python's.
        if (!is.finite(value)) {
            return(as.character(value))
        }
        return(sprintf("%.6g", value))
    }
    text <- paste(as.character(value), collapse = ",")
    if (!nzchar(text) || grepl("[[:space:]=]", text)) {
        return(paste0('"', gsub('"', "'", text), '"'))
    }
    text
}

#: Build the record.  Fields with no value are left out rather than written as
#: `context=""`: a record whose fields are sometimes empty trains the reader to
#: skip them, and the one that matters is then skipped with the rest.
.sparsegs_record <- function(event_name, fields) {
    if (length(fields) == 0L) {
        return(as.character(event_name)[1])
    }
    # A field with nothing in it is dropped before it is formatted, matching the
    # Python port's `v is not None and v != ""`.  `context = ""` is the default
    # at several call sites, and writing it out would put `context=""` on every
    # record that has no context -- noise that trains the reader to skip the
    # field, including the times it carries something.
    keep <- vapply(fields, function(v) {
        !is.null(v) && !(is.character(v) && length(v) == 1L && !nzchar(v))
    }, logical(1))
    fields <- fields[keep]
    if (length(fields) == 0L) {
        return(as.character(event_name)[1])
    }
    body <- vapply(seq_along(fields), function(i) {
        coded <- .sparsegs_codify(fields[[i]])
        if (is.null(coded)) NA_character_ else
            paste0(names(fields)[i], "=", coded)
    }, character(1))
    body <- body[!is.na(body)]
    paste(c(as.character(event_name)[1], body), collapse = " ")
}

#' Emit one structured record
#'
#' @param level Standard level name, lower case.
#' @param event_name The event, as `snake_case`.  Events are looked for by
#'   name, so the name is part of the interface and does not change once used.
#'   The parameter is spelled `event_name` rather than `name` so that `name`
#'   stays available as a *field*: "which gene set was this" is the first thing
#'   anyone asks of a record, and a signature that reserved the obvious word for
#'   it would be reserving the wrong one.
#' @param ... The record's named values.
#'
#' @return The record, invisibly, so a caller can assert on its text.
#'
#' @name event
#' @keywords internal
event <- function(level, event_name, ...) {
    rank <- .sparsegs_level_rank(level)
    if (rank < .sparsegs_level_rank(.sparsegs_state$level)) {
        return(invisible(NULL))
    }
    record <- .sparsegs_record(event_name, list(...))
    if (rank == .sparsegs_levels[["warning"]]) {
        # `call. = FALSE`: the record is the message and a call stack would only
        # add the framework's own internals to it.  The condition carries the
        # record as its message, so `tryCatch(warning = function(w) ...)` reads
        # the same text that was logged.
        warning(structure(class = c("sparsegs_event", "warning", "condition"),
                          list(message = record, call = NULL)))
    } else {
        # debug/info, and error/critical.  The higher two go here rather than to
        # `warning()` because every call site emits them immediately before the
        # `stop()` that raises the failure itself: signalled as a warning too, a
        # single failure would arrive as two conditions, and the one that
        # precedes the error reads as a warning the code went on to survive.
        message(record)
    }
    invisible(record)
}
