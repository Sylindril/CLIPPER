"""errors.py - error/warning messages.

Errors and warnings print to stderr; fatal() and an exceeded error limit
raise ``SystemExit(1)``.
"""

import sys

ERROR_LIMIT = 20
_total_errors = 0


def _fmt(fmt, args):
    return fmt % args if args else fmt


def fatal(fmt, *args):
    print("fatal: " + _fmt(fmt, args), file=sys.stderr)
    raise SystemExit(1)


def error(fmt, *args):
    global _total_errors
    print("error: " + _fmt(fmt, args), file=sys.stderr)
    _total_errors += 1
    if _total_errors > ERROR_LIMIT:
        print("error limit exceeded, goodbye", file=sys.stderr)
        raise SystemExit(1)


def warning(fmt, *args):
    print("warning: " + _fmt(fmt, args), file=sys.stderr)


def lex_error(fmt, *args):
    # There is no lexer here; behave like error() without file:line context.
    error(fmt, *args)


# Exit status of a run stopped by an LLM account error.  The suite runner
# (scripts/run_learning_curve.py) keys on it to stop launching cells.
ACCOUNT_ERROR_EXIT = 3


class LLMAccountError(RuntimeError):
    """The LLM provider refused service for account reasons -- no credit,
    exhausted quota, bad or unauthorized key.  Never retried and never turned
    into a no-op rule: a run that carried on would finish with a plausible-
    looking hypothesis learned from nothing, which is how two claude-sonnet-5
    suites silently produced unusable rows when the balance ran out."""
