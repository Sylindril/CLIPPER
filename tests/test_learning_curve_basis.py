"""Regressions for comparing partial result grids on a common dataset basis."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_learning_curve import curve_basis, macro


def test_missing_hard_dataset_cannot_make_a_worse_learner_look_better():
    stats = {"a": {100: {"easy": (10, 0, 10), "hard": (90, 0, 10)}},
             "b": {100: {"easy": (20, 0, 10)}}}
    basis = curve_basis(["a", "b"], stats, [100])
    means = [macro({d: v for d, v in stats[l][100].items()
                    if d in basis[100]})[0] for l in ("a", "b")]
    assert means == [10, 20]  # The old per-learner means were 50 and 20.


def test_fixed_basis_intersects_every_size_and_learner():
    st = (10, 0, 10)
    stats = {"a": {100: {"x": st, "y": st}, 200: {"x": st, "y": st}},
             "b": {100: {"x": st}, 200: {"x": st, "y": st}}}
    assert curve_basis(["a", "b"], stats, [100, 200], fixed=True) == {
        100: {"x"}, 200: {"x"}}
    assert curve_basis(["a", "b"], stats, [100, 200]) == {
        100: {"x"}, 200: {"x", "y"}}


def test_absent_training_size_adds_no_curve_or_dataset():
    st = (10, 0, 10)
    stats = {"a": {100: {"x": st}, 200: {"x": st}},
             "b": {100: {"x": st}}}
    assert curve_basis(["a", "b"], stats, [100, 200], fixed=True) == {
        100: {"x"}, 200: {"x"}}
