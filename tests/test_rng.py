"""Reseeding rng.seed() must reset both the Python and numpy generators."""
from slipper import rng


def test_below_is_deterministic_after_seed():
    rng.seed(123)
    first = [rng.below(100) for _ in range(10)]
    rng.seed(123)
    second = [rng.below(100) for _ in range(10)]
    assert first == second


def test_below_changes_with_different_seed():
    rng.seed(1)
    a = [rng.below(1_000_000) for _ in range(10)]
    rng.seed(2)
    b = [rng.below(1_000_000) for _ in range(10)]
    assert a != b


def test_np_rng_reseeds_with_seed():
    rng.seed(7)
    a = rng.np_rng().random(8).tolist()
    rng.seed(7)
    b = rng.np_rng().random(8).tolist()
    assert a == b


def test_np_rng_different_seeds_diverge():
    rng.seed(7)
    a = rng.np_rng().random(8).tolist()
    rng.seed(8)
    b = rng.np_rng().random(8).tolist()
    assert a != b
