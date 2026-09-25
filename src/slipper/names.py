"""names.py - attribute metadata and class list.

Holds the global ``Names`` (list of AttrDef) and ``Classes`` (list of Atom),
plus the per-field query functions the learner and grammar use.  The
metadata is populated from JSON (see dataset.py) rather than parsed from a
.names file.
"""

from enum import IntEnum

from . import errors
from .symbols import Atom, SymKind, intern, sym_str


class AttrKind(IntEnum):
    MISSING_VALUE = 0
    SYMBOL = 1
    CONTINUOUS = 2
    SET = 3
    IGNORE = 4
    TEXT = 5      # raw free text: LLM path only, no grammar conditions


class AttrDef:
    __slots__ = ("name", "kind", "values", "value_index", "isbag", "suppressed")

    def __init__(self, name):
        self.name = name              # Symbol
        self.kind = AttrKind.MISSING_VALUE
        self.values = []              # list of Symbol (possible values)
        self.value_index = {}         # name -> position in self.values
        self.isbag = False
        self.suppressed = False


# --- module globals ---
Names = []        # list of AttrDef
Classes = []      # list of Atom
_n_values = 0     # next index to assign to a distinct field value (N_values)
_names_defined = False


def reset():
    """Clear all names state (used between independent runs / tests).

    Lists are cleared in place so that any ``names.Classes`` reference held
    elsewhere stays valid.
    """
    global _n_values, _names_defined
    Names.clear()
    Classes.clear()
    _n_values = 0
    _names_defined = False


# ---------------------------------------------------------------------------
# queries (interface to information from the names file)
# ---------------------------------------------------------------------------

def names_defined():
    return _names_defined


def n_symbolic_values():
    return _n_values


def n_fields():
    return len(Names)


def field_name(i):
    return Names[i].name.name


def continuous_field(i):
    return Names[i].kind == AttrKind.CONTINUOUS


def ignored_field(i):
    return Names[i].kind == AttrKind.IGNORE


def suppressed_field(i):
    return Names[i].suppressed


def llm_hidden(i):
    """True if attribute i is kept out of the LLM's view: ignored or
    suppressed, or a text attribute under --hide-text."""
    from . import config
    return (ignored_field(i) or suppressed_field(i)
            or (Names[i].kind == AttrKind.TEXT and config.Hide_text))


def symbolic_field(i):
    return Names[i].kind == AttrKind.SYMBOL


def set_field(i):
    return Names[i].kind == AttrKind.SET


def text_field(i):
    return Names[i].kind == AttrKind.TEXT


def n_different_field_values(i, data):
    """Number of distinct values of field i (symbolic/set: the value list;
    continuous: distinct non-missing numbers actually present in data)."""
    adef = Names[i]
    if adef.kind in (AttrKind.SYMBOL, AttrKind.SET):
        return len(adef.values)
    # continuous: count distinct non-missing numeric values in data
    seen = set()
    for ex in data:
        av = ex.inst[i]
        if av.kind != AttrKind.MISSING_VALUE:
            seen.add(av.num)
    return len(seen)


def attr_value_index(sym, adef):
    return adef.value_index.get(sym.name, -1)


# ---------------------------------------------------------------------------
# value/class bookkeeping
# ---------------------------------------------------------------------------

def update_values(sym, adef):
    """Register symbol ``sym`` as a possible value of attribute ``adef``,
    assigning it a global value index the first time it is seen anywhere."""
    global _n_values
    if sym is None:
        errors.lex_error("symbolic attribute %s has continuous value", adef.name.name)
        return
    if sym.kind == SymKind.OTHER:
        sym.kind = SymKind.VALUE
        sym.index = _n_values
        _n_values += 1
    elif sym.kind != SymKind.VALUE:
        errors.lex_error(
            "symbol '%s' is used as type 'attribute value' and also as another type",
            sym.name,
        )
    if sym.name not in adef.value_index:
        adef.value_index[sym.name] = len(adef.values)
        adef.values.append(sym)


def update_field_values(i, sym):
    update_values(sym, Names[i])


def update_class(atom):
    """Add a class label seen in the data if not already declared."""
    if atom is not None and atom.nom is not None:
        for c in Classes:
            if c.nom is atom.nom:
                return
        Classes.append(Atom(nom=atom.nom))
        atom.nom.kind = SymKind.CLASS
        atom.nom.index = len(Classes) - 1


def mark_names_defined():
    global _names_defined
    _names_defined = True


# ---------------------------------------------------------------------------
# printing (print_names / print_attr_def) - for the -N option
# ---------------------------------------------------------------------------

def print_attr_def(adef):
    out = sym_str(adef.name) + ":\t"
    if adef.kind == AttrKind.CONTINUOUS:
        out += "continuous."
    elif adef.kind == AttrKind.SYMBOL:
        if not adef.values:
            out += "dummy_value1, dummy_value2."
        else:
            out += ", ".join(sym_str(v) for v in adef.values) + "."
    elif adef.kind == AttrKind.SET:
        out += "bag." if adef.isbag else "set."
    elif adef.kind == AttrKind.TEXT:
        out += "text."
    elif adef.kind == AttrKind.IGNORE:
        out += "ignore."
    else:
        out += " ???"
    print(out)


def print_names():
    if not Names:
        print("<attributes undefined>")
        return
    print(",".join(sym_str(c.nom) for c in Classes) + ".")
    for adef in Names:
        print_attr_def(adef)
