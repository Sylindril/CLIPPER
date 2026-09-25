"""refine.py - designated refinements / generalizations.

The "designated refinements" of a sentential form x are the forms obtained
by repeatedly applying grammar rules to the nonterminals of x, with no
grammar rule used more than once (and excluding x itself).  Generalizations
undo a single derivation step whose nonterminal can derive the empty string.

A sentential form together with its derivation is represented here by
``_Antec``.  The list of antecedents is rebuilt on each call rather than
reused across calls, for simplicity.

Every GSym is copied by value when it is placed into a new form: this
matters because the caller (value.py) later writes chosen wildcard
thresholds *into* a form's gsyms, and that must not mutate the shared
grammar rules.
"""

from . import grammar
from .grammar import DerivStep, GSym
from .symbols import SymKind, intern


class _Antec:
    __slots__ = ("sform", "deriv")

    def __init__(self):
        self.sform = []
        self.deriv = []


_slots = []   # list[_Antec]: the computed refinements/generalizations


def clear_slots():
    _slots.clear()


def avail_ref():
    a = _Antec()
    _slots.append(a)
    return a


def refinement(index):
    return _slots[index].sform


def generalization(index):
    return _slots[index].sform


def derivation(index):
    return _slots[index].deriv


def n_designated_refinements():
    return len(_slots)


def n_designated_generalizations():
    return len(_slots)


# ---------------------------------------------------------------------------

def make_the_universe(cl, u):
    """Set ``u`` to the start form for class ``cl``: ``[body_<cl>]`` if such a
    nonterminal exists, else ``[body]``."""
    u.clear()
    body_cl = intern("body_" + cl.name)
    if body_cl.kind == SymKind.NONTERM:
        nt = body_cl
    else:
        nt = intern("body")
    u.append(GSym(nonterm=nt))


def _do_deriv_step(oldform, step, newform):
    """Apply grammar rule ``step.rulex`` at position ``step.posn`` of
    ``oldform``, writing the result (as fresh GSym copies) into ``newform``."""
    r = grammar.Grammar[step.rulex]
    i = step.posn
    newform.clear()
    for j in range(i):
        newform.append(oldform[j].copy())
    for g in r.rhs:
        newform.append(g.copy())
    for j in range(i + 1, len(oldform)):
        newform.append(oldform[j].copy())


# ---------------------------------------------------------------------------
# refinements
# ---------------------------------------------------------------------------

def compute_designated_refinements(form, deriv, max_priority):
    used = [False] * len(grammar.Grammar)
    clear_slots()
    root = _Antec()
    root.sform = form
    root.deriv = deriv
    _store_refinements(root, 0, len(form), max_priority, used)


def _store_refinements(antec, lo, hi, max_priority, used):
    for i in range(lo, hi):
        nt = antec.sform[i].nonterm
        if nt is None:
            continue
        for j in range(grammar.Lo_rule[nt.index], grammar.Hi_rule[nt.index]):
            rj = grammar.Grammar[j]
            if used[j] or rj.priority > max_priority:
                continue
            step = DerivStep(i, j)
            newref = avail_ref()
            if antec.deriv:
                newref.deriv = [s.copy() for s in antec.deriv]
            newref.deriv.append(step)
            _do_deriv_step(antec.sform, step, newref.sform)
            used[j] = True
            _store_refinements(newref, i, i + len(rj.rhs), max_priority, used)
            used[j] = False


# ---------------------------------------------------------------------------
# generalizations
# ---------------------------------------------------------------------------

def compute_designated_generalizations(cl, form, deriv):
    clear_slots()
    form1 = []
    tmp = []
    deriv1 = []
    make_the_universe(cl, form1)
    for i in range(len(deriv)):
        step = deriv[i]
        if grammar.Derives_emptystr[form1[step.posn].nonterm.index]:
            new_antec = avail_ref()
            new_antec.sform = [g.copy() for g in form1]
            new_antec.deriv = [s.copy() for s in deriv1]
            lo = step.posn
            hi = step.posn + 1
            lo, hi = _replay_deriv(new_antec.sform, new_antec.deriv, deriv, i, lo, hi)
            _copy_wildcards(new_antec.sform, form)

            # delete the marked subtree conditions (between lo and hi)
            n_remove = hi - lo
            length = len(new_antec.sform)
            for j in range(lo, length - n_remove):
                new_antec.sform[j] = new_antec.sform[j + n_remove]
            del new_antec.sform[length - n_remove:]

            # insert the subtree's root nonterminal in their place
            new_antec.sform.append(GSym(nonterm=intern("????")))
            for j in range(len(new_antec.sform) - 1, lo, -1):
                new_antec.sform[j] = new_antec.sform[j - 1]
            new_antec.sform[step.posn] = form1[step.posn].copy()

        _do_deriv_step(form1, step, tmp)
        form1 = [g.copy() for g in tmp]
        deriv1.append(step.copy())


def _replay_deriv(form, cnd_deriv, deriv, start, lo, hi):
    """Replay derivation steps ``deriv[start:]`` onto ``form`` (mutated),
    appending the corresponding steps to ``cnd_deriv`` and tracking how the
    marked region [lo, hi) shifts.  Returns the updated (lo, hi)."""
    tmp = []
    for i in range(start, len(deriv)):
        step = deriv[i]
        _do_deriv_step(form, step, tmp)
        if step.posn < lo:
            cnd_deriv.append(step.copy())
        elif step.posn >= hi:
            cnd_deriv.append(DerivStep(step.posn - (hi - lo) + 1, step.rulex))
        # else: ignore steps whose position falls inside the marked region
        delta = len(tmp) - len(form)
        if step.posn < hi:
            hi += delta
        if step.posn < lo:
            lo += delta
        form[:] = [g.copy() for g in tmp]
    return lo, hi


def _copy_wildcards(new_form, old_form):
    """Copy chosen threshold/value bindings from ``old_form`` into the
    wildcard positions of ``new_form`` (assumed same length)."""
    assert len(new_form) == len(old_form)
    for i in range(len(old_form)):
        if grammar.contains_wildcard(new_form[i]):
            new_form[i].value.nom = old_form[i].value.nom
            new_form[i].value.num = old_form[i].value.num


# ---------------------------------------------------------------------------
# tracing helpers (print_form / print_deriv)
# ---------------------------------------------------------------------------

def form_str(form):
    return "".join(" " + grammar.fprint_gsym(g) for g in form)


def deriv_str(der):
    return "".join(" %d@%d" % (s.rulex, s.posn) for s in der)
