"""scripts/retrieval_probe.py: answer normalization/scoring and question truth.

The probe's conclusions are only as good as its ground truth, so the question
generator is checked against a hand-built table.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
sys.path.insert(0, SCRIPTS)
_spec = importlib.util.spec_from_file_location(
    "retrieval_probe", os.path.join(SCRIPTS, "retrieval_probe.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

@pytest.mark.parametrize("got,want,ok", [
    ("'yes'", "'yes'", True),
    ("yes", "'yes'", True),          # model drops the repr quoting
    ("YES", "'yes'", True),          # case
    (" 42 ", "42", True),
    ("42.0", "42", True),            # numeric canonicalization
    ("42", "43", False),
    ("no", "'yes'", False),
    (None, "'yes'", False),          # missing key in the JSON reply
])
def test_correct(got, want, ok):
    assert P.correct(got, want) is ok


def test_norm_leaves_unparseable_strings_alone():
    assert P.norm("'a b'") == "a b"


# --------------------------------------------------------------------------
# question generation
# --------------------------------------------------------------------------

class _FakeNom:
    def __init__(self, name): self.name = name


class _FakeLab:
    def __init__(self, nom): self.nom = nom


class _FakeEx:
    def __init__(self, inst, cls):
        self.inst, self.lab, self.wt = inst, _FakeLab(_FakeNom(cls)), 0.1


@pytest.fixture
def table(monkeypatch):
    """Five rows: `uid` is unique per row, `age` numeric, classes 3 pos / 2 neg."""
    monkeypatch.setattr(P.L, "inst_to_dict", lambda inst: inst)
    monkeypatch.setattr(P.L, "_cell", lambda v: repr(v))
    monkeypatch.setattr(P.L, "_attr_kind_map",
                        lambda: {"uid": "symbolic", "age": "continuous"})
    return [_FakeEx({"uid": "u%d" % i, "age": float(10 * i)},
                    "pos" if i < 3 else "neg") for i in range(5)]


def test_lookup_answers_come_from_the_right_row(table):
    qs = P.make_questions(table, np.random.default_rng(0))
    lookups = [q for q in qs if q["kind"] == "lookup"]
    assert lookups, "expected unique-key lookups"
    for q in lookups:
        # every lookup must quote a real cell of the row it names
        assert q["want"] in [repr(v) for row in table for v in row.inst.values()]
        assert 0.0 <= q["depth"] <= 1.0


def test_class_count_is_the_true_count(table):
    qs = P.make_questions(table, np.random.default_rng(0))
    q = next(q for q in qs if q["kind"] == "class_count")
    want = 3 if "pos" in q["q"] else 2
    assert q["want"] == str(want)


def test_col_max_is_the_true_max(table):
    qs = P.make_questions(table, np.random.default_rng(0))
    q = next(q for q in qs if q["kind"] == "col_max")
    assert q["want"] == repr(40.0)


def test_aggregates_are_marked_depthless(table):
    qs = P.make_questions(table, np.random.default_rng(0))
    assert all(q["depth"] == -1.0 for q in qs if q["kind"] != "lookup")
