"""concept.py - rule and concept types.

A Concept is a flat list of Rules plus a default class; this is the shared
core that a multiclass wrapper would combine with codewords.  This module
covers the data types, the size metrics, and the machine-readable (.shyp)
serialization via fshow_*.

The ``soft_*`` printing / classification lives in boosting.py.
"""

import json
from enum import IntEnum

from . import grammar
from .symbols import fmt_g


class Strategy(IntEnum):
    BEST = 0      # use highest-confidence rule on conflicts
    FIRST = 1     # use first matching rule on conflicts


class Rule:
    """A learned rule (rule_t): conseq :- antec, with weighted pos/neg
    coverage and the grammar derivation used to build it."""
    __slots__ = ("conseq", "antec", "nposx", "nnegx", "deriv")

    def __init__(self, conseq=None, antec=None, nposx=0.0, nnegx=0.0, deriv=None):
        self.conseq = conseq          # Symbol
        self.antec = antec if antec is not None else []   # list[GSym]
        self.nposx = nposx
        self.nnegx = nnegx
        self.deriv = deriv if deriv is not None else []   # list[DerivStep]


class PyRule:
    """A rule whose antecedent is a Python predicate, not a grammar form.

    Produced by the LLM rule builder (see llm_rule_builder.py).  ``source`` is
    the LLM-generated text defining ``def covers(x: dict) -> bool``; ``fn`` is
    a Python callable that evaluates that source against an instance dict
    (executed via a smolagents sandbox in the LLM path).  Lives alongside
    Rule in a Concept; boosting and printing branch on isinstance.
    """
    __slots__ = ("conseq", "source", "fn", "nposx", "nnegx")

    def __init__(self, conseq=None, source="", fn=None, nposx=0.0, nnegx=0.0):
        self.conseq = conseq
        self.source = source
        self.fn = fn
        self.nposx = nposx
        self.nnegx = nnegx


class Concept:
    """A ruleset (concept_t): conflict-resolution strategy, default class,
    and the list of rules."""
    __slots__ = ("res", "def_", "nposx", "nnegx", "rules")

    def __init__(self):
        self.res = Strategy.BEST      # newmem() zeroes the struct -> BEST
        self.def_ = None              # Atom (default class)
        self.nposx = 0.0
        self.nnegx = 0.0
        self.rules = []               # list[Rule]


# ---------------------------------------------------------------------------
# size metrics
# ---------------------------------------------------------------------------

def rule_size(r):
    """1 (the consequent) plus the number of terminal conditions.

    A PyRule (or an extraction rule) has no grammar antecedent; we score it as
    a single unit so the summary ``concept_size`` stays meaningful when both
    kinds of rules coexist.
    """
    if not isinstance(r, Rule):
        return 1
    sz = 1
    for g in r.antec:
        if g.nonterm is None:
            sz += 1
    return sz


def form_size(form):
    return sum(1 for g in form if g.nonterm is None)


def concept_size(c):
    return sum(rule_size(r) for r in c.rules)


def free_concept(c):
    c.rules = []


# ---------------------------------------------------------------------------
# machine-readable serialization (.shyp) via fshow_*
# ---------------------------------------------------------------------------

def fshow_rule(fp, r):
    if hasattr(r, "shyp_line"):
        # extraction rules serialize themselves (extract.py): same one-line
        # "sigil stats PAYLOAD <json>" shape as pyrule, but no consequent.
        fp.write(r.shyp_line())
        return
    if isinstance(r, PyRule):
        # .shyp is write-only in this codebase (no reader), so we use a custom
        # one-line form: a leading "pyrule" sigil + JSON-encoded source.
        fp.write("pyrule %s %s %s SOURCE %s .\n"
                 % (r.conseq.name, fmt_g(r.nposx), fmt_g(r.nnegx),
                    json.dumps(r.source)))
        return
    fp.write("%s %s %s IF" % (r.conseq.name, fmt_g(r.nposx), fmt_g(r.nnegx)))
    for g in r.antec:
        if g.nonterm is None:
            fp.write(" " + grammar.fshow_gsym(g))
    fp.write(" .\n")


def fshow_concept(fp, c):
    for r in c.rules:
        fshow_rule(fp, r)
    fp.write("%s %s %s IF .\n.\n" % (c.def_.nom.name, fmt_g(c.nposx), fmt_g(c.nnegx)))
    if c.res != Strategy.FIRST:
        fp.write("use_best_rule\n")
