"""rng.py - random number source.

Uses a clean seedable PRNG so runs are deterministic and reproducible given
a fixed seed.

A separate numpy Generator drives weighted_sample (data.py); seed() reseeds
both so --seed makes the whole run reproducible.
"""

import random as _random

import numpy as _np

DEFAULT_SEED = 1
_seed = DEFAULT_SEED
_rng = _random.Random(DEFAULT_SEED)
_np_rng = _np.random.default_rng(DEFAULT_SEED)


def seed(s=DEFAULT_SEED):
    global _seed
    _seed = s
    _rng.seed(s)
    global _np_rng
    _np_rng = _np.random.default_rng(s)


def below(m):
    """A uniform integer draw in ``range(m)`` for m > 0, else 0."""
    if m <= 0:
        return 0
    return _rng.randrange(m)


def np_rng():
    """Return the current numpy Generator (re-bound by seed())."""
    return _np_rng


def derived_np_rng(stream, index=0):
    """Independent reproducible stream; does not advance either learner RNG."""
    return _np.random.default_rng(_np.random.SeedSequence([_seed, stream, index]))
