"""data.weighted_sample: renormalization, freshness, determinism, bias."""
import math

import pytest

from slipper import data, rng
from slipper.data import Example


def _mk(n, weight_fn=lambda i: 1.0):
    """Make n Examples with .index=i, .wt=weight_fn(i), trivial inst/lab."""
    return [Example(inst=("inst", i), lab=("lab", i), wt=weight_fn(i), index=i)
            for i in range(n)]


def _normalize(items):
    z = sum(ex.wt for ex in items)
    for ex in items:
        ex.wt /= z
    return items


def test_full_copy_when_k_is_zero():
    rng.seed(1)
    src = _normalize(_mk(20))
    out = data.weighted_sample(src, 0)
    assert len(out) == len(src)
    # fresh wrappers, not the originals
    assert all(o is not s for o, s in zip(out, src))


def test_full_copy_when_k_exceeds_n():
    rng.seed(1)
    src = _normalize(_mk(20))
    out = data.weighted_sample(src, 999)
    assert len(out) == len(src)


def test_sample_weights_sum_to_one():
    rng.seed(1)
    src = _normalize(_mk(100, weight_fn=lambda i: i + 1.0))
    out = data.weighted_sample(src, 25)
    assert len(out) == 25
    assert math.isclose(sum(ex.wt for ex in out), 1.0, rel_tol=1e-9)


def test_full_dataset_weights_sum_to_one_after_renorm():
    rng.seed(1)
    src = _mk(10, weight_fn=lambda i: 2.0)        # unnormalized
    out = data.weighted_sample(src, 0)
    assert math.isclose(sum(ex.wt for ex in out), 1.0, rel_tol=1e-9)


def test_returned_examples_share_index_and_payload():
    rng.seed(1)
    src = _normalize(_mk(50))
    out = data.weighted_sample(src, 10)
    by_index = {ex.index: ex for ex in src}
    for o in out:
        assert o.index in by_index
        assert o.inst is by_index[o.index].inst
        assert o.lab is by_index[o.index].lab


def test_mutating_sample_wt_does_not_touch_parent():
    rng.seed(1)
    src = _normalize(_mk(20))
    orig_wts = [ex.wt for ex in src]
    out = data.weighted_sample(src, 5)
    for o in out:
        o.wt = 999.0
    assert [ex.wt for ex in src] == orig_wts


def test_deterministic_under_same_seed():
    src = _normalize(_mk(200, weight_fn=lambda i: i + 1.0))
    rng.seed(42)
    a = [ex.index for ex in data.weighted_sample(src, 30)]
    rng.seed(42)
    b = [ex.index for ex in data.weighted_sample(src, 30)]
    assert a == b


def test_different_seeds_pick_different_indices():
    src = _normalize(_mk(200, weight_fn=lambda i: i + 1.0))
    rng.seed(1)
    a = set(ex.index for ex in data.weighted_sample(src, 30))
    rng.seed(2)
    b = set(ex.index for ex in data.weighted_sample(src, 30))
    assert a != b


def test_higher_weight_examples_dominate_in_expectation():
    """Index 99 has 100x the weight of index 0 — should land in the sample
    far more often across many draws."""
    src = _normalize(_mk(100, weight_fn=lambda i: i + 1.0))
    rng.seed(2024)
    hits = [0] * 100
    trials = 200
    for _ in range(trials):
        for ex in data.weighted_sample(src, 10):
            hits[ex.index] += 1
    # the top-10-weight bucket should massively outdraw the bottom-10-weight bucket
    top = sum(hits[-10:])
    bot = sum(hits[:10])
    assert top > 5 * bot, f"top={top} bot={bot} (top:bot ratio too small)"


def test_empty_dataset_returns_empty():
    rng.seed(1)
    out = data.weighted_sample([], 5)
    assert out == []
