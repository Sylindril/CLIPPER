"""value.py - scoring a sentential form.

``value()`` is the workhorse called when refining a rule: given a candidate
form (a refinement of ``oldform``), it isolates the newly-added terminal
conditions, resolves any wildcard thresholds/values to their best setting,
and scores the result with the installed ``config.Value_function``.

For slipper2 the value function is always ``newboost_value``
(sqrt(p)-sqrt(n)+1), but the other scoring objectives are available too.
"""

import math

from . import config, counting, data, grammar, optimize, trace
from .symbols import Atom, Operator

_LN2 = 0.69314718055994530942
_NEG_INF = float("-inf")


def _log2(x):
    return math.log(x) / _LN2 if x > 0 else 0.0


def information(p, n):
    if p == 0:
        return 0.0
    return _log2((p + n + 1.0) / (p + 1.0))


def _supported_wildcard(g):
    return grammar.contains_wildcard(g) and g.op != Operator.NULLOP


# ---------------------------------------------------------------------------
# value functions to optimize (signature: form, oldinfo, oldp, oldn, p, n)
# ---------------------------------------------------------------------------

def info_gain(form, oldinfo, oldp, oldn, p, n):
    if oldn > 0:
        return p * (oldinfo - information(p, n))
    return pos_cvg(form, oldinfo, oldp, oldn, p, n)


def pos_cvg(form, oldinfo, oldp, oldn, p, n):
    if form is not None:
        return float(p) / len(form)
    return float(p)


def _H(p, n):
    return (p * information(p, n) + n * information(n, p)) / (p + n)


def entropy_reduction1(form, oldinfo, oldp, oldn, p, n):
    uncovp = oldp - p
    uncovn = oldn - n
    h0 = _H(oldp, oldn)
    hcov = _H(p, n)
    huncov = _H(uncovp, uncovn)
    if hcov < huncov:
        return _NEG_INF
    return h0 - hcov


def newboost_value(form, oldinfo, oldp, oldn, p, n):
    if config.Allow_negative_rules:
        pval = math.sqrt(p) - math.sqrt(n) + 1
        nval = math.sqrt(n) - math.sqrt(p) + 1
        return max(pval, nval)
    return math.sqrt(p) - math.sqrt(n) + 1


def rule_accuracy(form, oldinfo, oldp, oldn, p, n):
    return p - n


def foil_gain(form, oldinfo, oldp, oldn, p, n):
    """FOIL information gain (Quinlan 1990) -- the growth objective under
    ``--weighting cover``.

    Scores refining a rule covering (oldp, oldn) into one covering (p, n) as

        p * (log2(p / (p+n)) - log2(oldp / (oldp+oldn)))

    i.e. bits gained about the target class per surviving positive, times the
    surviving positive weight.  The multiplier is what balances the two
    degenerate extremes: pure precision picks tiny spotless rules, pure coverage
    picks the universal rule.  Unlike ``rule_accuracy`` (p - n) the tradeoff is
    multiplicative, so at equal p - n it prefers the purer, smaller rule -- which
    is what a covering learner wants, since the examples it doesn't cover go back
    into the pool for the next rule.

    ``soft_refine`` scores the *current* rule by calling this with oldp = oldn =
    0; returning 0.0 there makes its ``best_val <= last_val`` stop test read as
    "stop when no refinement has positive gain", the textbook FOIL criterion.
    Weights are fractional here (a boosting/cover distribution rather than
    counts), which only rescales the multiplier.
    """
    if oldp + oldn <= 0:            # soft_refine's baseline call: gain over self
        return 0.0
    if p <= 0:
        return 0.0
    return p * (math.log2(p / (p + n)) - math.log2(oldp / (oldp + oldn)))


def two_sided_foil_gain(form, oldinfo, oldp, oldn, p, n):
    """``foil_gain`` for a learner whose rule may carry either consequent --
    the growth objective under ``--weighting coverlist --cover-objective foil``.

    A decision list can predict the negative class, so a region that is pure in
    the *negative* direction is just as good a rule as a pure positive one.
    Scoring it means measuring the same gain with the classes swapped and taking
    whichever side the refinement is actually buying bits about, exactly as
    ``newboost_value`` does for ``Allow_negative_rules``.  Symmetric by
    construction: swapping (oldp, p) with (oldn, n) leaves the value unchanged.
    """
    return max(foil_gain(form, oldinfo, oldp, oldn, p, n),
               foil_gain(form, oldinfo, oldn, oldp, n, p))


def cover_value(form, oldinfo, oldp, oldn, p, n):
    """Linear greedy-cover objective: the weighted error a rule removes from the
    disjunction, p - n.  Kept as the simpler alternative to ``foil_gain`` (the
    cover default).

    The ``+ 1`` matches newboost_value's offset and is not cosmetic: ``value()``
    returns 0.0 for a refinement that changes no coverage, so a search objective
    must stay above that or the refiner reads a genuinely useful but
    minority-positive rule (p < n, common on imbalanced data before any
    reweighting) as worse than not refining at all, and stops at the universal
    rule.  Weights are normalized, so p - n + 1 lies in [0, 2].
    """
    return p - n + 1


def two_sided_cover_value(form, oldinfo, oldp, oldn, p, n):
    """``cover_value`` for a learner whose rule may carry either consequent --
    the growth objective under ``--weighting coverlist --cover-objective
    linear``, and the one aligned with that mode's ``p != n`` gate.

    ``max(p, n) - min(p, n) + 1`` is |p - n| + 1: the weighted error the rule
    removes once it is free to predict the majority class of what it covers.
    The one-sided form scores a majority-negative region *below* the universal
    rule and so never grows one; this scores it by how decisive it is, whichever
    way it leans.  Weights are normalized, so it lies in [1, 2].
    """
    return max(p, n) - min(p, n) + 1


# ---------------------------------------------------------------------------
# value() - score a refinement
# ---------------------------------------------------------------------------

def value(form, cl, dataset, oldform, old_info, oldp, oldn):
    # removing a nonterminal doesn't change coverage
    if len(form) < len(oldform):
        return 0.0

    # newform = form \ oldform : both share prefix x, oldform=xAz, form=xyz
    i = 0
    while i < len(oldform) and oldform[i] == form[i]:
        i += 1

    newform = []
    span = len(form) - len(oldform) + 1
    for j in range(span):
        gsymj = form[i + j]
        if gsymj.nonterm is None:
            newform.append(gsymj.copy())

    # no new terminal conditions -> no change in coverage
    if len(newform) == 0:
        return 0.0

    contains_wildcard_value = any(_supported_wildcard(g) for g in newform)

    if contains_wildcard_value:
        if trace.tracing(trace.DBG1):
            print("// wildcard values in" + _form_str(form))
        p, n = _set_wildcards(newform, cl, dataset, old_info, oldp, oldn)
        # copy the resolved wildcard values back into form
        k = 0
        for j in range(span):
            gsymj = form[i + j]
            if gsymj.nonterm is None:
                src = newform[k]
                k += 1
                gsymj.value.nom = src.value.nom
                gsymj.value.num = src.value.num
        if trace.tracing(trace.DBUG):
            print("// wildcards set to" + _form_str(form))
    else:
        p, n = counting.count_examples(newform, cl, dataset)

    val = config.Value_function(form, old_info, oldp, oldn, p, n)
    if trace.tracing(trace.DBUG):
        print("// value of" + _form_str(newform) + " = %g (%g/%g)" % (val, p, n))
    return val


def _form_str(form):
    return "".join(" " + grammar.fprint_gsym(g) for g in form)


# ---------------------------------------------------------------------------
# resolving wildcard thresholds / values
# ---------------------------------------------------------------------------

def _set_wildcards(form, cl, dataset, old_info, oldp, oldn):
    """Set each supported wildcard in ``form`` to its best value; return the
    (pos, neg) coverage of the resolved form."""
    if len(form) == 1:
        gsymi = form[0]
        best, pos, neg = _best_wildcard(gsymi, cl, dataset, old_info, oldp, oldn)
        gsymi.value.nom = best.nom
        gsymi.value.num = best.num
        return pos, neg

    # several wildcards (and maybe some known conditions)
    data1 = data.copy_data(dataset)
    tmp_form = [g for g in form if not _supported_wildcard(g)]
    data.remove_uncovered_examples(tmp_form, data1)

    pos = neg = 0.0
    for gsymi in form:
        if _supported_wildcard(gsymi):
            best, pos, neg = _best_wildcard(gsymi, cl, data1, old_info, oldp, oldn)
            gsymi.value.nom = best.nom
            gsymi.value.num = best.num
            data.remove_uncovered_examples([gsymi], data1)
    return pos, neg


def _best_wildcard(gsym, cl, dataset, old_info, oldp, oldn):
    if gsym.op in (Operator.OPLE, Operator.OPGE):
        th, pos, neg = optimize.best_threshold(gsym, cl, dataset, old_info, oldp, oldn)
        return Atom(nom=None, num=th), pos, neg
    if gsym.op in (Operator.OPIN, Operator.OPOUT, Operator.OPEQ, Operator.OPNEQ):
        sym, pos, neg = optimize.best_symbol(gsym, cl, dataset, old_info, oldp, oldn)
        return Atom(nom=sym, num=0.0), pos, neg
    from . import errors
    errors.fatal("unimplemented use of '?' in grammar")
