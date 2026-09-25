"""symbols.py - interned symbols and atoms.

A Symbol is a unique, interned object identified by its name; it carries a
mutable ``kind`` and ``index`` that other passes fill in.  Because symbols
are interned, identity (``is``) is the correct equality test.

An Atom holds either a symbol (``nom``) or, when ``nom is None``, a numeric
value (``num``).
"""

from enum import IntEnum


class SymKind(IntEnum):
    OTHER = 0          # type not yet determined
    MISSING_MARK = 1   # the symbol "?"
    WILDCARD = 2       # the symbol "*"
    ATTRIBUTE = 3      # name of an attribute
    VALUE = 4          # value of a symbolic attribute / element name
    OPERATOR = 5       # name of an operator
    NONTERM = 6        # name of a nonterminal symbol
    CLASS = 7          # name of a class


class Operator(IntEnum):
    NULLOP = 0
    OPEQ = 1
    OPNEQ = 2
    OPLE = 3
    OPGE = 4
    OPIN = 5
    OPOUT = 6


NULLINDEX = -1

# how operators print
OP_NAME = {
    Operator.NULLOP: "(null op)",
    Operator.OPEQ: "=",
    Operator.OPNEQ: "!=",
    Operator.OPLE: "<=",
    Operator.OPGE: ">=",
    Operator.OPIN: " ~ ",
    Operator.OPOUT: " !~ ",
}


class Symbol:
    __slots__ = ("name", "kind", "index")

    def __init__(self, name):
        self.name = name
        self.kind = SymKind.OTHER
        self.index = NULLINDEX

    def __repr__(self):
        return "<symbol %s type=%d indx=%d>" % (self.name, int(self.kind), self.index)


class Atom:
    __slots__ = ("nom", "num")

    def __init__(self, nom=None, num=0.0):
        self.nom = nom
        self.num = num

    def __eq__(self, other):
        # same symbol (by identity) and same number.
        if not isinstance(other, Atom):
            return NotImplemented
        return self.nom is other.nom and self.num == other.num

    def __hash__(self):
        return hash((id(self.nom), self.num))

    def __repr__(self):
        if self.nom is not None:
            return "<atom:'%s'>" % self.nom.name
        return "<atom:%.3f>" % self.num


# ---------------------------------------------------------------------------
# the symbol table
# ---------------------------------------------------------------------------

_table = {}


def intern(name):
    """Map a string to its unique Symbol, creating one if needed."""
    sym = _table.get(name)
    if sym is None:
        sym = Symbol(name)
        _table[name] = sym
    return sym


def reset_symbol_table():
    """Clear the oblist (used between independent runs / tests)."""
    _table.clear()


# ---------------------------------------------------------------------------
# printing helpers
# ---------------------------------------------------------------------------

def fmt_g(x):
    """Format a float roughly like printf's %g."""
    return format(float(x), "g")


def _needs_quote(s):
    if s == "?":
        return False
    if not (s[:1].isalpha() or s[:1] == "_"):
        return True
    for ch in s[1:]:
        if not (ch.isalnum() or ch == "_"):
            return True
    return False


def sym_str(sym):
    if _needs_quote(sym.name):
        return "'%s'" % sym.name
    return sym.name


def atom_str(a):
    if a.nom is not None:
        return sym_str(a.nom)
    return fmt_g(a.num)
