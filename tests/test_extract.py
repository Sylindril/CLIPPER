"""extract: RegexRule semantics, (p, n) scoring, the alpha vote over values,
the loader's "task": "extract" branch, and the --rule-pool CLI path."""
import json
import os
import re
import subprocess
import sys

import pytest

from slipper import boosting, config, dataset, extract
from slipper.concept import Concept

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _reset_config_globals():
    """Extraction mode lives in config globals; don't leak it between tests."""
    saved = {k: getattr(config, k) for k in (
        "Task_mode", "Rule_pool", "Epsilon", "N_boost", "Slipper_folds",
        "Rule_llm_model", "Sample_size",
    )}
    yield
    for k, v in saved.items():
        setattr(config, k, v)
    dataset.reset_all()


# ---------------------------------------------------------------------------
# a tiny synthetic extraction dataset: "<n>-year-old" notes with two traps --
# one note the pattern misses (abstain) and one where it captures the wrong
# number (fires, wrong value).
# ---------------------------------------------------------------------------

AGES = [87, 34, 61, 22, 45, 70, 19, 53]

ROWS = ([{"values": ["a %d-year-old patient was admitted" % a, "calc=1", "age"],
          "label": "[%d, 'years']" % a, "weight": 1.0} for a in AGES]
        + [{"values": ["a 6-month-old infant was admitted", "calc=1", "age"],
            "label": "[6, 'months']", "weight": 1.0},
           {"values": ["the mother of a 3-year-old is 40 years old", "calc=1", "age"],
            "label": "[40, 'years']", "weight": 1.0}])

SPEC = {"task": "extract",
        "attributes": [{"name": "NOTE", "kind": "symbol"},
                       {"name": "CALC_ID", "kind": "symbol"},
                       {"name": "ENTITY", "kind": "symbol"}],
        "train": ROWS, "test": ROWS}

YEARS = {"pattern": r"(\d+)-year-old", "flags": "", "template": [1, "years"]}
MONTHS = {"pattern": r"(\d+)-month-old", "flags": "", "template": [1, "months"]}


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


@pytest.fixture
def ages(tmp_path):
    """The synthetic dataset, loaded (so names_mod is set up for inst_dict)."""
    train, test = dataset.load_dataset(_write(tmp_path, "ages.json", SPEC))
    config.Epsilon = 0.5 / len(train)
    return train, test


# ---------------------------------------------------------------------------
# RegexRule
# ---------------------------------------------------------------------------

def test_regex_template_builds_a_list_of_groups_and_literals():
    r = extract.RegexRule(r"(\d+)-year-old", template=[1, "years"])
    assert r.extract({"NOTE": "a 87-year-old man"}) == ["87", "years"]


def test_regex_single_item_template_yields_a_bare_value():
    assert extract.RegexRule(r"\bfemale\b", flags="i",
                             template=["Female"]).extract({"NOTE": "A Female pt"}) == "Female"
    assert extract.RegexRule(r"(\d+) kg",
                             template=[1]).extract({"NOTE": "is 72 kg today"}) == "72"


def test_regex_no_template_means_group_1():
    r = extract.RegexRule(r"bmi of ([\d.]+)", flags="i")
    assert r.extract({"NOTE": "BMI of 24.3"}) == "24.3"


def test_regex_abstains_when_nothing_matches():
    r = extract.RegexRule(r"(\d+)-year-old", template=[1, "years"])
    assert r.extract({"NOTE": "a 6-month-old infant"}) is None


def test_regex_spec_round_trips_and_is_the_dedup_key():
    r = extract.RegexRule(**YEARS)
    assert r.spec() == YEARS
    assert r.source == extract.RegexRule(**YEARS).source
    assert r.source != extract.RegexRule(**MONTHS).source
    assert r.shyp_line().startswith("rerule ")


def test_unknown_regex_flag_is_fatal():
    with pytest.raises(SystemExit):
        extract.RegexRule(r"x", flags="z")


# ---------------------------------------------------------------------------
# ExtractRule (the sandboxed, value-returning sibling of PyRule)
# ---------------------------------------------------------------------------

def test_extract_rule_runs_in_the_pyrule_sandbox():
    pytest.importorskip("smolagents")
    r = extract.sandbox_rule(
        "def extract(x):\n"
        "    import re\n"
        "    m = re.search(r'(\\d+)-year-old', x['NOTE'])\n"
        "    return [m.group(1), 'years'] if m else None\n")
    assert r.conseq is None                      # no consequent in extraction
    assert r.extract({"NOTE": "a 87-year-old man"}) == ["87", "years"]
    assert r.extract({"NOTE": "a 6-month-old"}) is None   # abstain, not False
    assert r.shyp_line().startswith("exrule ")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def test_extract_evaluate_splits_right_wrong_and_abstain(ages):
    train, _ = ages
    p, n = extract.extract_evaluate(extract.RegexRule(**YEARS), train)
    # 8 rows right, the "mother of a 3-year-old" row wrong, the month row abstains
    assert (p, n) == (8.0, 1.0)
    assert p + n == len(train) - 1


def test_extract_evaluate_honours_weights(ages):
    train, _ = ages
    for ex in train:
        ex.wt = 0.5
    p, n = extract.extract_evaluate(extract.RegexRule(**YEARS), train)
    assert (p, n) == (4.0, 0.5)


def test_rule_y_is_plus_one_minus_one_zero(ages):
    train, _ = ages
    r = extract.RegexRule(**YEARS)
    assert extract.rule_y(r, train[0]) == 1        # 87-year-old
    assert extract.rule_y(r, train[-2]) == 0       # 6-month-old: abstain
    assert extract.rule_y(r, train[-1]) == -1      # captures 3, gold is 40


def test_a_crashing_rule_abstains_and_is_tallied(ages):
    train, _ = ages

    class Boom:
        conseq = None
        nposx = nnegx = 0.0

        def extract(self, x):
            raise KeyError("nope")

    boosting._round_exceptions.clear()
    assert extract.extract_evaluate(Boom(), train) == (0.0, 0.0)
    assert boosting._round_exceptions[("KeyError", "'nope'")] == len(train)


# ---------------------------------------------------------------------------
# the alpha vote
# ---------------------------------------------------------------------------

def _concept(*rules):
    c = Concept()
    c.rules = list(rules)
    return c


def test_classify_sums_alpha_per_canonical_value(ages):
    config.Epsilon = 0.1
    weak_f = extract.RegexRule(r"\bfemale\b", flags="i", template=["Female"],
                               nposx=2.0, nnegx=1.0)
    weak_f2 = extract.RegexRule(r"\bwoman\b", flags="i", template=["Female"],
                                nposx=2.0, nnegx=1.0)
    strong_m = extract.RegexRule(r"\bmale\b", flags="i", template=["Male"],
                                 nposx=3.0, nnegx=1.0)
    x = {"NOTE": "a female woman, not male"}
    # one rule for Male beats either Female rule alone, but not their sum
    assert extract.extract_classify(_concept(weak_f, strong_m), x)[0] == "Male"
    v, conf = extract.extract_classify(_concept(weak_f, weak_f2, strong_m), x)
    assert v == "Female"
    assert conf == pytest.approx(weak_f.alpha + weak_f2.alpha)


def test_classify_abstains_when_no_rule_fires():
    assert extract.extract_classify(
        _concept(extract.RegexRule(**YEARS)), {"NOTE": "no ages here"}) == (None, 0.0)


def test_classify_of_an_empty_ruleset_abstains():
    assert extract.extract_classify(_concept(), {"NOTE": "a 5-year-old"}) == (None, 0.0)


def test_error_rate_counts_abstain_as_wrong(ages):
    train, _ = ages
    config.Epsilon = 0.1
    c = _concept(extract.RegexRule(nposx=8.0, nnegx=1.0, **YEARS))
    # 8/10 exact; the month row abstains and the decoy row is wrong
    assert boosting.soft_error_rate(c, train) == pytest.approx(0.2)
    assert extract.extract_error_rate(c, train) == pytest.approx(0.2)
    assert extract.extract_counts(c, train) == (8, 1, 10)


def test_error_rate_is_weighted(ages):
    train, _ = ages
    config.Epsilon = 0.1
    train[-1].wt = 3.0                       # the wrong row counts triple
    c = _concept(extract.RegexRule(nposx=8.0, nnegx=1.0, **YEARS))
    assert extract.extract_error_rate(c, train) == pytest.approx(4.0 / 12.0)


def test_best_pool_rule_picks_the_higher_value_rule(ages, tmp_path):
    train, _ = ages
    config.Rule_pool = _write(tmp_path, "pool.json", [MONTHS, YEARS])
    r, p, n = extract.best_pool_rule(train)
    assert r.spec() == YEARS and (p, n) == (8.0, 1.0)
    assert r.nposx == 8.0 and r.nnegx == 1.0


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

def test_loader_reads_the_extract_format(ages):
    train, test = ages
    assert config.Task_mode == "extract"
    assert len(train) == len(test) == 10
    # the gold repr is kept verbatim; normalize parses it
    assert train[0].lab.nom.name == "[87, 'years']"
    assert extract.gold(train[0]) == [87, "years"]
    # no class list: one abstain slot, so the stratified shuffles still work
    from slipper import names as names_mod
    assert [c.nom.name for c in names_mod.Classes] == [dataset.ABSTAIN]
    assert extract.inst_dict(train[0].inst)["ENTITY"] == "age"


CLASSIFY_SPEC = {"classes": ["yes", "no"],
                 "attributes": [{"name": "A", "kind": "symbolic"}],
                 "train": [{"values": ["x"], "label": "yes", "weight": 1.0},
                           {"values": ["y"], "label": "no", "weight": 1.0}]}


def test_reset_all_clears_extraction_state(ages, tmp_path):
    """Task_mode and the pool cache are registered in dataset.reset_all, so a
    later classification load doesn't inherit extraction mode (CV folds would
    otherwise leak it)."""
    config.Rule_pool = _write(tmp_path, "pool.json", [YEARS])
    extract.load_pool(config.Rule_pool)
    assert extract._pool_cache and config.Task_mode == "extract"
    train, test = dataset.load_dataset(_write(tmp_path, "cls.json", CLASSIFY_SPEC))
    assert config.Task_mode == "classify"
    assert not extract._pool_cache
    assert train[0].lab.nom.name == "yes" and test is None


# ---------------------------------------------------------------------------
# end to end through the CLI
# ---------------------------------------------------------------------------

def _run(*args, cwd):
    return subprocess.run([sys.executable, "-m", "slipper", *args],
                          capture_output=True, text=True, cwd=cwd, timeout=180)


def test_cli_rule_pool_end_to_end(tmp_path):
    stem = _write(tmp_path, "ages.json", SPEC)
    pool = _write(tmp_path, "pool.json", [YEARS])
    proc = _run("--rule-pool", pool, "-i0", "-n3", "-p", stem, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    # The seeded pool fires on 9 of 10 rows and is right on 8 of them, on both
    # splits (train == test here), so both error rates are exactly 20%.
    for which, pad in (("Train", "  "), ("Test", "   ")):
        m = re.search(r"^%s error rate:%s([\d.]+)%%" % (which, pad), out, re.MULTILINE)
        assert m, out
        assert float(m.group(1)) == 20.00
    assert "Train exact match: 8 of 10" in out
    assert "Test abstain:      10.00% (1 of 10)" in out
    # one pool entry, so every round proposes the same rule -> compressed to one
    assert "Hypothesis size:   1 rules" in out

    with open(str(tmp_path / "ages.shyp")) as fp:
        assert fp.readline().startswith("rerule ")
    preds = [l.rstrip("\n").split("\t")
             for l in open(str(tmp_path / "ages.pred"))]
    assert len(preds) == 10 and {len(r) for r in preds} == {4}
    assert preds[0][0] == "['87', 'years']" and preds[0][3] == "[87, 'years']"
    assert preds[-2][0] == "None"                     # abstained on the month row
    assert not os.path.exists(str(tmp_path / "ages.pyhyp.py"))


def test_cli_extraction_requires_a_rule_pool(tmp_path):
    stem = _write(tmp_path, "ages.json", SPEC)
    proc = _run("-i0", "-n1", stem, cwd=REPO_ROOT)
    assert proc.returncode != 0
    assert "needs --rule-pool" in proc.stdout + proc.stderr


def test_cli_extraction_rejects_pool_and_llm_together(tmp_path):
    """Both flags fill the rule-builder slot, so passing both is an error --
    and it is caught before any model is contacted."""
    stem = _write(tmp_path, "ages.json", SPEC)
    pool = _write(tmp_path, "pool.json", [YEARS])
    proc = _run("--rule-pool", pool, "-R", "fake/model", "-i0", "-n1", stem,
                cwd=REPO_ROOT)
    assert proc.returncode != 0
    assert "pass one, not both" in proc.stdout + proc.stderr


def test_cli_stops_when_no_rule_beats_abstaining(tmp_path):
    """A pool whose only rule is wrong everywhere it fires never beats the
    do-nothing baseline (_value(0, 0) == 1), so no rule is added at all."""
    stem = _write(tmp_path, "ages.json", SPEC)
    pool = _write(tmp_path, "pool.json",
                  [{"pattern": r"(\d+)-year-old", "template": [1, "furlongs"]}])
    proc = _run("--rule-pool", pool, "-i0", "-n3", "-v1", stem, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no candidate rule beats abstaining" in proc.stdout
    assert "Hypothesis size:   0 rules" in proc.stdout
    assert "Test abstain:      100.00% (10 of 10)" in proc.stdout
