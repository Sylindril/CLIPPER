"""trace.py - trace levels.

Guard verbose output with ``if trace.tracing(SUMM): print(...)``.
"""

NONE = 0
SUMM = 1
LONG = 2
DBUG = 3
DBG1 = 4

_level = NONE


def set_trace_level(n):
    global _level
    _level = n


def trace_level():
    return _level


def tracing(lev):
    """True when messages at level ``lev`` should be shown (lev <= Trace_level)."""
    return lev <= _level
