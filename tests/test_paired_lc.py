"""scripts/paired_lc.py: CSV loading, the paired test, and p formatting."""
import importlib.util
import os
import sys

import numpy as np
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
sys.path.insert(0, SCRIPTS)
_spec = importlib.util.spec_from_file_location(
    "paired_lc", os.path.join(SCRIPTS, "paired_lc.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


CSV = """label,dataset,size,seed,bestj,err
a,titanic,100,1,5,10.0
a,titanic,100,2,5,12.0
a,titanic,200,1,7,9.0
b,titanic,100,1,5,11.0
b,titanic,100,2,5,14.0
b,move1,100,1,9,30.0
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text(CSV)
    return str(p)


def test_load_keys_on_dataset_size_seed_and_filters_by_label(csv_path):
    a = P.load(csv_path, "a")
    assert a == {("titanic", 100, 1): 10.0,
                 ("titanic", 100, 2): 12.0,
                 ("titanic", 200, 1): 9.0}


def test_load_unknown_label_is_empty(csv_path):
    assert P.load(csv_path, "nope") == {}


def test_common_cells_exclude_unpaired_rows(csv_path):
    common = set(P.load(csv_path, "a")) & set(P.load(csv_path, "b"))
    # a has titanic@200 that b lacks; b has move1 that a lacks.
    assert common == {("titanic", 100, 1), ("titanic", 100, 2)}


def test_paired_identical_arrays_are_not_significant():
    x = np.array([1.0, 2.0, 3.0])
    assert P.paired(x, x) == (1.0, "n.s.")


def test_paired_detects_a_consistent_shift():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + np.array([3.0, 3.2, 2.9, 3.1, 3.0])   # same sign every pair, small spread
    p, sig = P.paired(a, b)
    assert p < 0.001 and sig == "***"


def test_paired_needs_at_least_two_pairs():
    p, sig = P.paired(np.array([1.0]), np.array([2.0]))
    assert np.isnan(p) and sig == "?"


def test_paired_mismatched_lengths_are_not_compared():
    p, sig = P.paired(np.array([1.0, 2.0]), np.array([1.0]))
    assert np.isnan(p) and sig == "?"


@pytest.mark.parametrize("p,want", [(0.0005, "<.001"), (0.0431, "0.043"),
                                    (0.5, "0.500"), (float("nan"), "  —  ")])
def test_fmt_p(p, want):
    assert P.fmt_p(p) == want
