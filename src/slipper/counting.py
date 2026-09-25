"""counting.py - coverage and weighted counting.

The routines exercised by the slipper2 path: ``form_covers``,
``contains_symbol``, ``count_rule``, ``count_examples`` and
``count_class_freq``, each returning its results as a tuple.

Counts are weighted by ``Example.wt`` (the ``weight()`` helper).
"""

from .names import AttrKind
from .symbols import Operator, SymKind


def contains_symbol(sym, st):
    """True if symbol ``sym`` is in the set ``st`` (a list of Symbol)."""
    for s in st:
        if s is sym:
            return True
    return False


def form_covers(sform, inst):
    """True if the sentential form ``sform`` (list of gsym) covers instance
    ``inst`` (list of Aval).  Nonterminals are ignored (a partially-derived
    form covers iff its terminal conditions all hold)."""
    for cp in sform:
        if cp.nonterm is not None:
            continue
        cp_nom = cp.value.nom
        cp_num = cp.value.num
        ex_val = inst[cp.attr_index]
        if ex_val.kind == AttrKind.MISSING_VALUE:
            # a missing value is covered only by an explicit "= ?" test
            return cp.op == Operator.OPEQ and cp_nom is not None \
                and cp_nom.kind == SymKind.MISSING_MARK
        op = cp.op
        if op == Operator.OPEQ:
            if ex_val.kind == AttrKind.SYMBOL and ex_val.sym is not cp_nom:
                return False
            if ex_val.kind == AttrKind.CONTINUOUS and ex_val.num != cp_num:
                return False
        elif op == Operator.OPNEQ:
            if ex_val.kind == AttrKind.SYMBOL and ex_val.sym is cp_nom:
                return False
            if ex_val.kind == AttrKind.CONTINUOUS and ex_val.num == cp_num:
                return False
        elif op == Operator.OPLE:
            if ex_val.num > cp_num:
                return False
        elif op == Operator.OPGE:
            if ex_val.num < cp_num:
                return False
        elif op == Operator.OPIN:
            if not contains_symbol(cp_nom, ex_val.set):
                return False
        elif op == Operator.OPOUT:
            if contains_symbol(cp_nom, ex_val.set):
                return False
        else:
            raise AssertionError("unknown operator")
    return True


def count_rule(form, cl, data):
    """Return (cov, uncov, fp, fn): weighted coverage, non-coverage,
    false positives and false negatives of ``form`` for class ``cl``."""
    cov = uncov = fp = fn = 0.0
    for exi in data:
        w = exi.wt
        if form_covers(form, exi.inst):
            if exi.lab.nom is not cl:
                fp += w
            cov += w
        else:
            uncov += w
            if exi.lab.nom is cl:
                fn += w
    return cov, uncov, fp, fn


def count_examples(form, cl, data):
    """Return (pos, neg): weighted #correctly and #incorrectly covered."""
    cov, uncov, fp, fn = count_rule(form, cl, data)
    return cov - fp, fp


def count_class_freq(cl, data):
    """Return (p, n): weighted count of members / non-members of class ``cl``."""
    p = n = 0.0
    for exi in data:
        if exi.lab.nom is cl:
            p += exi.wt
        else:
            n += exi.wt
    return p, n
