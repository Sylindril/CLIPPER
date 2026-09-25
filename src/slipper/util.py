"""util.py - small timing and CLI helpers."""

import time as _time

_clock_start = 0.0


def start_clock():
    global _clock_start
    _clock_start = _time.process_time()


def elapsed_time():
    """User-ish CPU time since the last start_clock()."""
    return _time.process_time() - _clock_start


def add_ext(stem, ext):
    return stem + ext
