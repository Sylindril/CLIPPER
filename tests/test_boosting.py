"""boosting: PyRule exception tally + the sampling-collapse regression."""
import math
import os

import pytest

from slipper import boosting, config, data, dataset, grammar, refine, rng
from slipper import names as names_mod
from slipper.concept import Concept, PyRule


# ---------------------------------------------------------------------------
# _rule_covers / _round_exceptions
# ---------------------------------------------------------------------------

EMPTY_INST = []   # inst_to_dict iterates inst directly; empty list -> {}


def test_round_exceptions_tallies_pyrule_failures():
    boosting._round_exceptions.clear()

    def bad_fn(_x):
        raise KeyError("missing_attr")

    r = PyRule(conseq=None, source="bad", fn=bad_fn, nposx=0.0, nnegx=0.0)
    assert boosting._rule_covers(r, EMPTY_INST) is False
    assert boosting._rule_covers(r, EMPTY_INST) is False
    assert boosting._round_exceptions[("KeyError", "'missing_attr'")] == 2


def test_round_exceptions_keys_by_type_and_message():
    boosting._round_exceptions.clear()

    def double_fault(_x):
        double_fault.n = getattr(double_fault, "n", 0) + 1
        if double_fault.n % 2:
            raise KeyError("a")
        raise TypeError("b")

    r = PyRule(conseq=None, source="dual", fn=double_fault,
               nposx=0.0, nnegx=0.0)
    for _ in range(4):
        boosting._rule_covers(r, EMPTY_INST)
    assert boosting._round_exceptions[("KeyError", "'a'")] == 2
    assert boosting._round_exceptions[("TypeError", "b")] == 2


# ---------------------------------------------------------------------------
# Sampling-collapse regression (Option B keeps M/N < 1/3 useful).
# ---------------------------------------------------------------------------

VOTE1_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "pdata", "vote1.json",
)


@pytest.fixture(autouse=True)
def _reset_config_globals():
    """Each test starts from a clean(ish) config so test ordering doesn't
    leak Sample_size / Rule_llm_model / etc. between tests."""
    saved = {k: getattr(config, k) for k in (
        "Sample_size", "Rule_llm_model", "Slipper_folds", "N_boost",
        "Allow_negative_rules", "Weighting", "Simplify_function",
        "Cover_objective", "Cover_sizing", "Cover_patience",
    )}
    yield
    for k, v in saved.items():
        setattr(config, k, v)


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_small_sample_does_not_collapse_hypothesis():
    """Before Option B, weighted_partition's hardcoded 1/3 threshold meant
    any sample whose raw weight summed to <= 1/3 (i.e. M/N <= 1/3 at the
    uniform initial round) produced an empty grow_data, so the rule was
    always universal and the default-update always won.  Vote1 has 300
    train examples; M=80 -> 80/300 ~= 0.267, comfortably in the old
    collapse zone."""
    config.Rule_llm_model = ""
    config.Sample_size = 80
    config.Slipper_folds = 0
    config.N_boost = 5
    rng.seed(1)

    train, test = dataset.load_dataset(VOTE1_JSON)
    assert grammar.ld_grammar(train)
    hyp = boosting.soft_model(train, test)

    assert len(hyp.rules) > 0, (
        "M=80 / N=300 should still grow at least one rule "
        "after Option B's per-sample renormalization")
    # And the rule should be doing real work, not just matching the default.
    err = boosting.soft_error_rate(hyp, test)
    assert err < 0.3, f"expected reasonable test err on vote1, got {err:.3f}"


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_no_sampling_reproducible_with_seed():
    """-M0 (no sampling) on vote1 with --seed 1 should be deterministic."""
    def _run():
        config.Rule_llm_model = ""
        config.Sample_size = 0
        config.Slipper_folds = 0
        config.N_boost = 5
        rng.seed(1)
        train, test = dataset.load_dataset(VOTE1_JSON)
        assert grammar.ld_grammar(train)
        hyp = boosting.soft_model(train, test)
        return len(hyp.rules), round(boosting.soft_error_rate(hyp, test), 6)

    a = _run()
    b = _run()
    assert a == b


# ---------------------------------------------------------------------------
# --weighting cover (greedy set cover)
# ---------------------------------------------------------------------------

def _cover_hyp(nboost=10, simplify_fn=4, weighting="cover"):
    config.Rule_llm_model = ""
    config.Sample_size = 0
    config.Slipper_folds = 0
    config.N_boost = nboost
    config.Weighting = weighting
    config.Simplify_function = simplify_fn      # cli defaults this under cover
    rng.seed(1)
    train, test = dataset.load_dataset(VOTE1_JSON)
    assert grammar.ld_grammar(train)
    return boosting.soft_model(train, test), train, test


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cover_learns_rules_and_beats_base_rate():
    hyp, _train, test = _cover_hyp()
    assert len(hyp.rules) > 0
    assert boosting.soft_error_rate(hyp, test) < 0.2


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cover_classifies_as_a_disjunction():
    """Cover hypotheses ignore confidences: covered by any rule -> positive,
    otherwise the default class.  (All slipper2 rules share the positive
    consequent, so 'first rule that fires' and 'any rule fires' coincide.)"""
    hyp, _train, test = _cover_hyp()
    for ex in test:
        covered = any(boosting._rule_covers(r, ex.inst) for r in hyp.rules)
        cls, conf = boosting.soft_classify(hyp, ex.inst)
        if covered:
            assert cls is not hyp.def_.nom and conf == 1.0
        else:
            assert cls is hyp.def_.nom and conf == -1.0


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cover_stops_before_the_round_budget():
    """A covering learner is done when nothing is left worth covering, so a
    big -n must not keep appending rules (the universal rule in particular)."""
    small, _, _ = _cover_hyp(nboost=5)
    big, _, _ = _cover_hyp(nboost=100)
    assert len(big.rules) == len(small.rules) or len(big.rules) < 100
    for r in big.rules:
        assert r.antec, "cover accepted the universal (no-condition) rule"


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cover_leaves_boost_mode_untouched():
    """Regression guard: the cover path must not perturb default boosting."""
    def _boost():
        config.Rule_llm_model = ""
        config.Sample_size = 0
        config.Slipper_folds = 0
        config.N_boost = 5
        config.Weighting = "boost"
        rng.seed(1)
        train, test = dataset.load_dataset(VOTE1_JSON)
        assert grammar.ld_grammar(train)
        hyp = boosting.soft_model(train, test)
        return len(hyp.rules), round(boosting.soft_error_rate(hyp, test), 6)

    before = _boost()
    _cover_hyp()
    assert _boost() == before


def test_cover_dist_zeroes_covered_and_stops_when_nothing_left(monkeypatch):
    """recover_dist: covered examples drop to 0, survivors stay uniform, and it
    reports False once no uncovered example carries the target class."""
    pos, neg = object(), object()

    class _Ex:
        def __init__(self, i, lab, inst):
            self.index, self.inst = i, inst
            self.lab = type("L", (), {"nom": lab})()

    ds = [_Ex(0, pos, "a"), _Ex(1, pos, "b"), _Ex(2, neg, "c"), _Ex(3, neg, "d")]
    d = [0.25] * 4
    rule = PyRule(conseq=pos, source="", fn=lambda x: False, nposx=1.0, nnegx=0.0)

    monkeypatch.setattr(boosting, "_rule_covers", lambda r, inst: inst == "a")
    assert boosting.recover_dist(rule, pos, ds, d) is True
    assert d[0] == 0.0
    assert d[1] == d[2] == d[3] == pytest.approx(1 / 3)

    # cover the remaining positive -> nothing left worth covering
    monkeypatch.setattr(boosting, "_rule_covers", lambda r, inst: inst == "b")
    assert boosting.recover_dist(rule, pos, ds, d) is False


# ---------------------------------------------------------------------------
# soft_simplify: --cover-sizing gate (gate-aware, specific-first)
# ---------------------------------------------------------------------------

class _Tok:
    """Stand-in for a GSym: soft_simplify only ever copies form elements."""

    def __init__(self, tag):
        self.tag = tag

    def copy(self):
        return _Tok(self.tag)


def _stub_sizing(monkeypatch, orig, gens):
    """Drive soft_simplify off scripted scores instead of a real grammar.

    ``orig``/``gens`` are (val, p, n) triples for the un-generalized form and
    for each designated generalization, in order.  Returns the starting form,
    whose single token's tag says which candidate soft_simplify settled on.
    """
    scores = {"orig": orig}
    scores.update(("g%d" % j, g) for j, g in enumerate(gens))
    forms = {tag: [_Tok(tag)] for tag in scores}

    monkeypatch.setattr(refine, "compute_designated_generalizations",
                        lambda cl, form, deriv: None)
    monkeypatch.setattr(refine, "n_designated_generalizations", lambda: len(gens))
    monkeypatch.setattr(refine, "generalization", lambda j: forms["g%d" % j])
    monkeypatch.setattr(refine, "derivation", lambda j: [])
    monkeypatch.setattr(boosting, "_simplify_fct",
                        lambda form, *a, **k: scores[form[0].tag])
    return forms["orig"]


def test_cover_sizing_gate_keeps_the_specific_rule_on_a_tie(monkeypatch):
    """Under legacy sizing a tie hands the round to the last and most general
    candidate -- which is how a rule that passed `p > n` gets pruned into one
    that fails it."""
    config.Weighting = "cover"
    config.Cover_sizing = "gate"
    form = _stub_sizing(monkeypatch, (1.0, 0.6, 0.2), [(1.0, 0.9, 0.4)])
    assert boosting.soft_simplify(form, [], None, [], []) == (0.6, 0.2)
    assert form[0].tag == "orig"


def test_cover_sizing_gate_skips_generalizations_that_fail_the_gate(monkeypatch):
    """A strictly better sizing score is not enough: soft_fit rejects (and can
    then halt on) any rule with p <= n, so sizing must not produce one."""
    config.Weighting = "cover"
    config.Cover_sizing = "gate"
    form = _stub_sizing(monkeypatch, (1.0, 0.6, 0.2),
                        [(-5.0, 0.9, 0.9), (-9.0, 0.95, 1.2)])
    assert boosting.soft_simplify(form, [], None, [], []) == (0.6, 0.2)
    assert form[0].tag == "orig"


def test_cover_sizing_gate_still_takes_a_better_passing_generalization(monkeypatch):
    """The gate check must not freeze the rule at its grown form."""
    config.Weighting = "cover"
    config.Cover_sizing = "gate"
    form = _stub_sizing(monkeypatch, (1.0, 0.6, 0.2),
                        [(-5.0, 0.9, 0.9), (-2.0, 0.8, 0.3)])
    assert boosting.soft_simplify(form, [], None, [], []) == (0.8, 0.3)
    assert form[0].tag == "g1"


def test_cover_sizing_legacy_is_the_default_and_unchanged(monkeypatch):
    """The published baseline: at the default sizing, cover mode still lets the
    last tying candidate win and still ignores the gate entirely."""
    config.Weighting = "cover"
    assert config.Cover_sizing == "legacy"
    form = _stub_sizing(monkeypatch, (1.0, 0.6, 0.2),
                        [(1.0, 0.9, 0.4), (1.0, 0.95, 1.2)])
    assert boosting.soft_simplify(form, [], None, [], []) == (0.95, 1.2)
    assert form[0].tag == "g1"


def test_boost_sizing_tie_break_is_unchanged(monkeypatch):
    """Boost mode keeps `<=` whatever Cover_sizing says: last tie wins, gate or
    no gate."""
    config.Weighting = "boost"
    config.Cover_sizing = "gate"
    form = _stub_sizing(monkeypatch, (1.0, 0.6, 0.2),
                        [(1.0, 0.9, 0.4), (1.0, 0.95, 1.2)])
    assert boosting.soft_simplify(form, [], None, [], []) == (0.95, 1.2)
    assert form[0].tag == "g1"


# ---------------------------------------------------------------------------
# soft_fit: skip-and-retry with patience on a failed p > n gate
# ---------------------------------------------------------------------------

# The consequents _scripted_fit hands soft_fit, so tests can assert which one
# an accepted rule ended up carrying.
_POS, _NEG = object(), object()


def _scripted_fit(monkeypatch, script, patience=None, nboost=None, exhausted=False,
                  weighting="cover"):
    """Run soft_fit's cover branch off scripted per-round (p, n) counts.

    ``script`` is one (p, n) pair per round (the last repeats if the budget
    outlasts it); the rule search is stubbed out so the acceptance gate is the
    only live logic.  ``patience`` None leaves Cover_patience at its default.
    ``exhausted`` makes recover_dist report "no uncovered positives left" after
    the first accepted rule.  Returns (hyp, #rounds run).
    """
    pos, neg = _POS, _NEG
    lab = lambda c: type("L", (), {"nom": c})()
    train = [data.Example([], lab(pos), 1.0, 0), data.Example([], lab(neg), 1.0, 1)]
    dist = [0.5, 0.5]

    config.Rule_llm_model = ""
    config.Sample_size = 0
    config.Weighting = weighting
    if patience is not None:
        config.Cover_patience = patience
    config.N_boost = nboost if nboost is not None else len(script)

    rounds = []

    def _count(form, cl, ds):
        rounds.append(script[min(len(rounds), len(script) - 1)])
        return rounds[-1]

    monkeypatch.setattr(refine, "make_the_universe", lambda cl, form: None)
    monkeypatch.setattr(boosting, "build_rule", lambda *a: None)
    monkeypatch.setattr(boosting.counting, "count_examples", _count)
    monkeypatch.setattr(boosting, "recover_dist", lambda *a: not exhausted)

    hyp = Concept()
    boosting.soft_fit(pos, neg, train, dist, None, None, hyp)
    return hyp, len(rounds)


def test_cover_patience_skips_failed_rounds_and_resets_on_accept(monkeypatch):
    """Two failed gates under patience 3 skip their rounds and the third rule is
    still accepted; the accept resets the counter, so the next two failures
    don't stop the run either."""
    fail, ok = (0.1, 0.3), (0.4, 0.1)
    hyp, rounds = _scripted_fit(monkeypatch, [fail, fail, ok, fail, fail, ok],
                                patience=3)
    assert len(hyp.rules) == 2
    assert rounds == 6, "skipped rounds must still consume the budget"


def test_cover_patience_stops_after_consecutive_failures(monkeypatch):
    """Cover_patience consecutive failures end the run, having added nothing."""
    hyp, rounds = _scripted_fit(monkeypatch, [(0.1, 0.3)], patience=3, nboost=10)
    assert hyp.rules == []
    assert rounds == 3


def test_cover_patience_defaults_to_the_legacy_hard_stop(monkeypatch):
    """The fix is opt-in: out of the box the first failure still ends the run."""
    assert config.Cover_patience == 1
    hyp, rounds = _scripted_fit(monkeypatch, [(0.1, 0.3), (0.4, 0.1)], nboost=10)
    assert hyp.rules == []
    assert rounds == 1


def test_cover_stops_immediately_when_no_uncovered_positives_left(monkeypatch):
    """The other cover stop is not a failed gate and patience must not defer it."""
    hyp, rounds = _scripted_fit(monkeypatch, [(0.4, 0.1)], patience=3,
                                nboost=10, exhausted=True)
    assert len(hyp.rules) == 1
    assert rounds == 1


def test_cover_rules_all_carry_the_positive_consequent(monkeypatch):
    """cover is a positive disjunction: an accepted rule is never relabelled."""
    hyp, _ = _scripted_fit(monkeypatch, [(0.4, 0.1)], nboost=1)
    assert [r.conseq for r in hyp.rules] == [_POS]


# ---------------------------------------------------------------------------
# --weighting coverlist (ordered decision list, per-rule consequent)
# ---------------------------------------------------------------------------

def test_coverlist_accepts_a_majority_negative_rule_as_a_negative_rule(monkeypatch):
    """The whole point of the mode: p < n is not a rejected rule, it is a rule
    about the other class.  Its (p, n) stay measured against the positive."""
    hyp, rounds = _scripted_fit(monkeypatch, [(0.1, 0.4)], weighting="coverlist",
                                nboost=1)
    assert len(hyp.rules) == 1
    assert hyp.rules[0].conseq is _NEG
    assert (hyp.rules[0].nposx, hyp.rules[0].nnegx) == (0.1, 0.4)
    assert rounds == 1


def test_coverlist_keeps_the_positive_consequent_on_a_majority_positive_rule(monkeypatch):
    hyp, _ = _scripted_fit(monkeypatch, [(0.4, 0.1)], weighting="coverlist", nboost=1)
    assert hyp.rules[0].conseq is _POS


def test_cover_still_rejects_what_coverlist_accepts(monkeypatch):
    """Regression guard: the same (p, n) that coverlist turns into a negative
    rule must still fail cover's p > n gate and stop the run."""
    hyp, rounds = _scripted_fit(monkeypatch, [(0.1, 0.4)], nboost=10)
    assert hyp.rules == []
    assert rounds == 1


def test_coverlist_tie_is_a_gate_failure_that_respects_patience(monkeypatch):
    """p == n leaves the rule with nothing to conclude, so it is the failure
    case -- and it feeds Cover_patience exactly like cover's p <= n does."""
    hyp, rounds = _scripted_fit(monkeypatch, [(0.25, 0.25)], weighting="coverlist",
                                patience=3, nboost=10)
    assert hyp.rules == []
    assert rounds == 3, "a tie should skip and retry until patience runs out"


def test_coverlist_tie_stops_at_the_default_patience(monkeypatch):
    """At the legacy patience of 1 a tie ends the run on the first occurrence,
    without consuming the round that would have succeeded."""
    assert config.Cover_patience == 1
    hyp, rounds = _scripted_fit(monkeypatch, [(0.25, 0.25), (0.4, 0.1)],
                                weighting="coverlist", nboost=10)
    assert hyp.rules == []
    assert rounds == 1


class _Cls:
    """Stand-in for a class Atom: soft_classify reads only .nom and .index."""

    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.nom = self


def _list_hyp(monkeypatch, *conseqs):
    """A Concept whose rules carry ``conseqs`` in order, with _rule_covers
    stubbed so an instance is just the list of rules that cover it."""
    pos, neg = _Cls(0, "pos"), _Cls(1, "neg")
    monkeypatch.setattr(names_mod, "Classes", [pos, neg])
    monkeypatch.setattr(boosting, "_rule_covers", lambda r, inst: r in inst)
    hyp = Concept()
    hyp.def_ = neg
    hyp.rules = [PyRule(conseq=(pos if c == "pos" else neg), source="", fn=None,
                        nposx=0.4, nnegx=0.1) for c in conseqs]
    return hyp, pos, neg


def test_coverlist_classify_takes_the_first_matching_rules_own_consequent(monkeypatch):
    """A negative rule ahead of a positive one is an exception clause: when both
    fire the list answers negative, and anything uncovered gets the default."""
    config.Weighting = "coverlist"
    hyp, pos, neg = _list_hyp(monkeypatch, "neg", "pos")
    rneg, rpos = hyp.rules
    assert boosting.soft_classify(hyp, [rneg, rpos]) == (neg, -1.0)
    assert boosting.soft_classify(hyp, [rpos]) == (pos, 1.0)
    assert boosting.soft_classify(hyp, []) == (neg, -1.0)


def test_cover_classify_is_unchanged_by_the_per_rule_consequent(monkeypatch):
    """cover's rules are all positive, so first-match-wins still reads as
    'covered by any rule -> positive, otherwise the default' with +-1."""
    config.Weighting = "cover"
    hyp, pos, neg = _list_hyp(monkeypatch, "pos", "pos")
    r1, r2 = hyp.rules
    assert boosting.soft_classify(hyp, [r1, r2]) == (pos, 1.0)
    assert boosting.soft_classify(hyp, [r2]) == (pos, 1.0)
    assert boosting.soft_classify(hyp, []) == (neg, -1.0)


def test_coverlist_sizing_gate_keeps_a_negative_rule_on_its_own_side(monkeypatch):
    """Under coverlist the sizing gate reads on the side the grown rule covers:
    a generalization that flips it to the other class is a different rule, not a
    more general one, so it is skipped however well it scores."""
    config.Weighting = "coverlist"
    config.Cover_sizing = "gate"
    form = _stub_sizing(monkeypatch, (1.0, 0.2, 0.6),
                        [(-5.0, 0.9, 0.3), (-2.0, 0.3, 0.8)])
    assert boosting.soft_simplify(form, [], None, [], []) == (0.3, 0.8)
    assert form[0].tag == "g1"


def test_simplify_fct_sf5_flip_scores_the_other_class(monkeypatch):
    """sf=5's flip is the same FOIL gain with the classes exchanged, base rate
    included; the reported (p, n) still count the positive class."""
    config.Simplify_function = 5
    counts = {"grow": (0.2, 0.9), "prune": (0.1, 0.5)}
    monkeypatch.setattr(boosting.counting, "count_examples",
                        lambda form, cl, ds: counts[ds])
    val, outp, outn = boosting._simplify_fct(None, None, 1.0, "grow", "prune", 0.2)
    fval, foutp, foutn = boosting._simplify_fct(None, None, 1.0, "grow", "prune",
                                                0.2, flip=True)
    assert val == pytest.approx(-0.1 * (math.log2(0.1 / 0.6) - math.log2(0.2)))
    assert fval == pytest.approx(-0.5 * (math.log2(0.5 / 0.6) - math.log2(0.8)))
    assert (outp, outn) == (foutp, foutn) == (pytest.approx(0.3), pytest.approx(1.4))


# ---------------------------------------------------------------------------
# value.foil_gain (the cover-mode growth objective)
# ---------------------------------------------------------------------------

def test_foil_gain_baseline_call_is_zero():
    """soft_refine scores the current rule with oldp=oldn=0; that must be 0 so
    its `best_val <= last_val` test means 'stop when no refinement gains'."""
    from slipper import value
    assert value.foil_gain(None, 0, 0, 0, 0.3, 0.2) == 0.0


def test_foil_gain_prefers_the_purer_rule_at_equal_p_minus_n():
    """The multiplicative tradeoff is the whole point: p-n calls these equal,
    FOIL prefers the small clean rule."""
    from slipper import value
    clean = value.foil_gain(None, 0, 100, 100, 10, 0)
    broad = value.foil_gain(None, 0, 100, 100, 90, 80)
    assert value.rule_accuracy(None, 0, 100, 100, 10, 0) == \
           value.rule_accuracy(None, 0, 100, 100, 90, 80)
    assert clean > broad > 0


def test_foil_gain_is_zero_without_positives():
    from slipper import value
    assert value.foil_gain(None, 0, 100, 100, 0, 50) == 0.0


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cover_objective_default_is_foil_gain():
    from slipper import value
    _cover_hyp()
    assert config.Value_function is value.foil_gain


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cover_objective_linear_installs_cover_value():
    """--cover-objective linear swaps the growth objective for the p - n + 1
    one the acceptance gate measures."""
    from slipper import value
    config.Cover_objective = "linear"
    _cover_hyp()
    assert config.Value_function is value.cover_value


# ---------------------------------------------------------------------------
# the two-sided growth objectives (--weighting coverlist)
# ---------------------------------------------------------------------------

def test_two_sided_objectives_are_symmetric_in_the_two_classes():
    """A coverlist rule may conclude either class, so its growth objective must
    not prefer one direction: exchanging the classes leaves the score alone."""
    from slipper import value
    for p, n in ((0.3, 0.2), (0.05, 0.9), (0.4, 0.4), (0.0, 0.7)):
        assert value.two_sided_cover_value(None, 0, 0, 0, p, n) == \
               pytest.approx(value.two_sided_cover_value(None, 0, 0, 0, n, p))
        assert value.two_sided_foil_gain(None, 0, 0.35, 0.65, p, n) == \
               pytest.approx(value.two_sided_foil_gain(None, 0, 0.65, 0.35, n, p))


def test_two_sided_objectives_dominate_their_one_sided_siblings():
    """Each is the max over the two directions, so it agrees with the one-sided
    form wherever that one already wins and rescues the mirror case."""
    from slipper import value
    # majority-positive: the positive direction wins, both agree
    assert value.two_sided_cover_value(None, 0, 0, 0, 0.6, 0.1) == \
           value.cover_value(None, 0, 0, 0, 0.6, 0.1)
    assert value.two_sided_foil_gain(None, 0, 0.5, 0.5, 0.6, 0.1) == \
           value.foil_gain(None, 0, 0.5, 0.5, 0.6, 0.1)
    # majority-negative: the one-sided forms score it at or below "don't refine"
    assert value.cover_value(None, 0, 0, 0, 0.1, 0.6) < 1.0
    assert value.foil_gain(None, 0, 0.5, 0.5, 0.1, 0.6) < 0.0
    assert value.two_sided_cover_value(None, 0, 0, 0, 0.1, 0.6) == pytest.approx(1.5)
    assert value.two_sided_foil_gain(None, 0, 0.5, 0.5, 0.1, 0.6) > 0.0


def test_two_sided_foil_gain_baseline_call_is_still_zero():
    """soft_refine's oldp=oldn=0 baseline must stay 0 in both directions, or its
    `best_val <= last_val` stop test loses its meaning."""
    from slipper import value
    assert value.two_sided_foil_gain(None, 0, 0, 0, 0.3, 0.2) == 0.0


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
@pytest.mark.parametrize("objective,fname", [("foil", "two_sided_foil_gain"),
                                             ("linear", "two_sided_cover_value")])
def test_coverlist_installs_the_two_sided_objective(objective, fname):
    from slipper import value
    config.Cover_objective = objective
    _cover_hyp(weighting="coverlist")
    assert config.Value_function is getattr(value, fname)
