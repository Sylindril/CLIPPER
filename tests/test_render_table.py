"""Table-format sample rendering, incl. --table-minibatch-size subtables."""
import pytest

from slipper import config
from slipper import llm_rule_builder as L
from slipper.data import Example


class _FakeNom:
    def __init__(self, name): self.name = name


class _FakeLab:
    def __init__(self, nom): self.nom = nom


POS = _FakeNom("pos")


@pytest.fixture
def sample(monkeypatch):
    """Five examples over two attributes, with the schema lookups stubbed out."""
    monkeypatch.setattr(L, "inst_to_dict", lambda inst: inst)
    monkeypatch.setattr(L, "_attr_kind_map", lambda: {"a": "continuous", "b": "symbolic"})
    monkeypatch.setattr(config, "Table_minibatch_size", 0)
    return [Example(inst={"a": i, "b": "v%d" % i}, lab=_FakeLab(POS), wt=0.2)
            for i in range(5)]


def _headers(text):
    return [ln for ln in text.splitlines() if ln.startswith("| class |")]


def _datarows(text):
    return [ln for ln in text.splitlines() if ln.startswith("| pos |")]


def test_one_table_by_default(sample):
    out = L._render_examples_table(sample)
    assert len(_headers(out)) == 1
    assert len(_datarows(out)) == 5
    assert "\n\n" not in out


def test_minibatch_repeats_the_header_and_keeps_row_order(sample, monkeypatch):
    monkeypatch.setattr(config, "Table_minibatch_size", 2)
    out = L._render_examples_table(sample)
    # 5 rows at 2 per subtable -> 3 subtables (2, 2, 1), each with its own header.
    assert len(out.split("\n\n")) == 3
    assert len(_headers(out)) == 3
    rows = _datarows(out)
    assert len(rows) == 5
    assert [r.split(" | ")[2] for r in rows] == ["0", "1", "2", "3", "4"]


def test_minibatch_larger_than_sample_is_identical_to_off(sample, monkeypatch):
    off = L._render_examples_table(sample)            # fixture leaves it at 0
    monkeypatch.setattr(config, "Table_minibatch_size", 99)
    assert L._render_examples_table(sample) == off


def test_kind_row_repeats_with_each_subtable(sample, monkeypatch):
    monkeypatch.setattr(config, "Table_minibatch_size", 2)
    out = L._render_examples_table(sample)
    assert out.count("| — | — | continuous | symbolic |") == 3


def test_empty_sample_renders_just_a_header(sample, monkeypatch):
    monkeypatch.setattr(config, "Table_minibatch_size", 2)
    out = L._render_examples_table([])
    assert _datarows(out) == []
    assert len(_headers(out)) == 1
