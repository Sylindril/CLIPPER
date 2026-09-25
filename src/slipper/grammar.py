"""grammar.py - grammar data types + default-grammar construction.

Covers what slipper2 needs when there is no .gram file: the gsym/grule
types, ``complete_grammar`` (which builds the default
body/conds/cond/cond_<attr> grammar from the attribute metadata),
``index_grammar`` (sort, index, and the derives-empty-string analysis), and
the printing helpers used for the .shyp output.

There is no .gram-file parser or lexer-based grammar reader; only the
auto-generated default grammar is supported.
"""

from . import config, errors
from . import names as names_mod
from .symbols import (NULLINDEX, OP_NAME, Atom, Operator, SymKind, fmt_g,
                      intern, sym_str)


class GSym:
    """A grammar symbol: a nonterminal, or a terminal condition
    ``attr op value``."""
    __slots__ = ("nonterm", "attr_index", "op", "value")

    def __init__(self, nonterm=None, attr_index=NULLINDEX,
                 op=Operator.NULLOP, value=None):
        self.nonterm = nonterm                       # Symbol or None
        self.attr_index = attr_index
        self.op = op
        self.value = value if value is not None else Atom()

    def __eq__(self, other):
        # structural equality: same nonterminal identity, attribute, operator,
        # and value
        if not isinstance(other, GSym):
            return NotImplemented
        return (self.nonterm is other.nonterm
                and self.attr_index == other.attr_index
                and self.op == other.op
                and self.value == other.value)

    def __hash__(self):
        return hash((id(self.nonterm), self.attr_index, int(self.op),
                     id(self.value.nom), self.value.num))

    def copy(self):
        return GSym(self.nonterm, self.attr_index, self.op,
                    Atom(self.value.nom, self.value.num))


class DerivStep:
    """One step in a rule's derivation: grammar rule ``rulex`` was applied
    at position ``posn``."""
    __slots__ = ("posn", "rulex")

    def __init__(self, posn, rulex):
        self.posn = posn
        self.rulex = rulex

    def copy(self):
        return DerivStep(self.posn, self.rulex)


class GRule:
    """A grammar rule: lhs --> rhs, with a priority and the original user
    ordering (used to make sorting deterministic)."""
    __slots__ = ("lhs", "priority", "rhs", "user_order")

    def __init__(self, lhs, rhs, priority=0, user_order=0):
        self.lhs = lhs            # Symbol
        self.priority = priority
        self.rhs = rhs            # list[GSym]
        self.user_order = user_order


# --- module globals ---
Grammar = []                 # list[GRule]
Max_priority = 0
N_nonterms = 0
N_eff_terms = 0
_n_suppressed_terms = 0
Lo_rule = []                 # index in Grammar of first rule for nonterm i
Hi_rule = []                 # one past index of last rule for nonterm i
Nonterm = []                 # Symbol for nonterm i
Derives_emptystr = []        # bool for nonterm i

_Wildcard = None             # Atom whose nom is intern("*")


def reset():
    global Grammar, Max_priority, N_nonterms, N_eff_terms, _n_suppressed_terms
    global Lo_rule, Hi_rule, Nonterm, Derives_emptystr
    Grammar = []
    Max_priority = 0
    N_nonterms = 0
    N_eff_terms = 0
    _n_suppressed_terms = 0
    Lo_rule = []
    Hi_rule = []
    Nonterm = []
    Derives_emptystr = []


def contains_wildcard(gsym):
    return (gsym.nonterm is None
            and gsym.value.nom is not None
            and gsym.value.nom.kind == SymKind.WILDCARD)


# ---------------------------------------------------------------------------
# build the default grammar (no .gram file)
# ---------------------------------------------------------------------------

def _init_operators():
    """Intern the operator and wildcard symbols with their proper kinds."""
    intern("*").kind = SymKind.WILDCARD
    for name, code in (("=", Operator.OPEQ), ("!=", Operator.OPNEQ),
                       ("<=", Operator.OPLE), (">=", Operator.OPGE),
                       ("~", Operator.OPIN), ("!~", Operator.OPOUT)):
        op = intern(name)
        op.kind = SymKind.OPERATOR
        op.index = int(code)


def ld_grammar(data):
    """Build (and index) the default grammar for the given training data.

    Initialise operator symbols, start from an empty grammar, then run
    complete_grammar + index_grammar.
    """
    global Grammar, Max_priority
    _init_operators()
    Grammar = []
    Max_priority = 0
    if not complete_grammar(data):
        errors.error("grammar is incomplete")
        return False
    if not index_grammar(data):
        errors.error("problem in indexing grammar")
        return False
    return True


def _rules_for(sym):
    return sym.kind == SymKind.NONTERM


def _add_rule_nn(lhs, nt1, nt2):
    """Add a rule lhs --> nt1 [nt2] (nonterminals) and mark lhs defined."""
    lhs.kind = SymKind.NONTERM
    rhs = []
    if nt1 is not None:
        rhs.append(GSym(nonterm=nt1))
    if nt2 is not None:
        rhs.append(GSym(nonterm=nt2))
    Grammar.append(GRule(lhs, rhs, priority=0))


def _add_rule_t(lhs, attrx, op, val_atom, r):
    """Add a terminal rule lhs --> (attr op value)."""
    lhs.kind = SymKind.NONTERM
    if val_atom is not None:
        value = Atom(val_atom.nom, val_atom.num)
    else:
        value = Atom(None, r)
    Grammar.append(GRule(lhs, [GSym(None, attrx, op, value)], priority=0))


def complete_grammar(data):
    """Fill in default definitions until every nonterminal is defined."""
    global _Wildcard, _n_suppressed_terms
    body = intern("body")
    conds = intern("body_conds")
    cond = intern("cond")
    _Wildcard = Atom(nom=intern("*"), num=0.0)

    _n_suppressed_terms = 0

    def define_body():
        _add_rule_nn(body, conds, None)
        if not _rules_for(conds):
            return define_conds()
        return True

    def define_conds():
        _add_rule_nn(conds, None, None)        # body_conds --> .
        _add_rule_nn(conds, cond, conds)       # body_conds --> cond, body_conds
        if not _rules_for(cond):
            return define_cond()
        return True

    def define_cond_a(cond_a, index):
        if names_mod.symbolic_field(index):
            _add_rule_t(cond_a, index, Operator.OPEQ, _Wildcard, 0.0)
            if config.Eq_negations and names_mod.n_different_field_values(index, data) > 2:
                _add_rule_t(cond_a, index, Operator.OPNEQ, _Wildcard, 0.0)
        elif names_mod.continuous_field(index):
            _add_rule_t(cond_a, index, Operator.OPLE, _Wildcard, 0.0)
            _add_rule_t(cond_a, index, Operator.OPGE, _Wildcard, 0.0)
        elif names_mod.set_field(index):
            _add_rule_t(cond_a, index, Operator.OPIN, _Wildcard, 0.0)
            if config.Set_negations:
                _add_rule_t(cond_a, index, Operator.OPOUT, _Wildcard, 0.0)
        else:
            errors.fatal("unknown field type")
        return True

    def define_cond():
        global _n_suppressed_terms
        for i in range(names_mod.n_fields()):
            if names_mod.text_field(i):
                # Raw text has no enumerable domain and none of the six
                # operators applies to it, so it gets no cond_<attr>
                # nonterminal at all: the discrete search never sees the
                # column, only the LLM path's covers(x) does.
                continue
            if names_mod.suppressed_field(i):
                if names_mod.continuous_field(i):
                    mul = 2
                elif names_mod.set_field(i) and config.Set_negations:
                    mul = 2
                elif (names_mod.symbolic_field(i) and config.Eq_negations
                      and names_mod.n_different_field_values(i, data) > 2):
                    mul = 2
                else:
                    mul = 1
                _n_suppressed_terms += mul * names_mod.n_different_field_values(i, data)
            elif not (names_mod.ignored_field(i) or names_mod.text_field(i)):
                # text fields are skipped exactly like ignore: no conditions,
                # no N_eff_terms contribution (they exist for the LLM path only)
                cond_a = intern("cond_" + names_mod.field_name(i))
                _add_rule_nn(cond, cond_a, None)
                if not _rules_for(cond_a):
                    define_cond_a(cond_a, i)
        if not _rules_for(cond):
            errors.warning("cannot complete grammar: no usable attributes")
            return False
        return True

    def add_default_defn(s):
        if s is body:
            return define_body()
        if s is conds:
            return define_conds()
        if s is cond:
            return define_cond()
        if s.name.startswith("cond_"):
            a = intern(s.name[5:])
            if a.kind == SymKind.ATTRIBUTE:
                return define_cond_a(s, a.index)
            errors.error("undefined nonterminal %s (%s not an attribute)",
                         s.name, a.name)
            s.kind = SymKind.NONTERM
            return False
        errors.error("undefined nonterminal %s", s.name)
        s.kind = SymKind.NONTERM
        return False

    if not _rules_for(body):
        define_body()

    while True:
        complete = True
        for rule in list(Grammar):
            for gsym in rule.rhs:
                if gsym.nonterm is not None and gsym.nonterm.kind != SymKind.NONTERM:
                    complete = False
                    if not add_default_defn(gsym.nonterm):
                        return False
        if complete:
            break
    return True


def index_grammar(data):
    """Sort grammar by nonterminal, index it, and run the derives-empty
    static analysis."""
    global N_nonterms, N_eff_terms, Lo_rule, Hi_rule, Nonterm, Derives_emptystr
    global Grammar

    for i, rule in enumerate(Grammar):
        rule.user_order = i
    Grammar.sort(key=lambda r: (r.lhs.name, r.user_order))

    # assign an index to each nonterminal (in sorted order)
    N_nonterms = 0
    last_lhs = None
    for rule in Grammar:
        if rule.lhs is not last_lhs:
            rule.lhs.index = N_nonterms
            N_nonterms += 1
            last_lhs = rule.lhs

    # count effective terminal symbols
    N_eff_terms = _n_suppressed_terms
    for rule in Grammar:
        for c in rule.rhs:
            if c.nonterm is None:
                if not contains_wildcard(c):
                    N_eff_terms += 1
                else:
                    N_eff_terms += names_mod.n_different_field_values(c.attr_index, data)

    # first/last rule index for each nonterminal
    Lo_rule = [0] * N_nonterms
    Hi_rule = [0] * N_nonterms
    Nonterm = [None] * N_nonterms
    last_lhs = None
    for i, rule in enumerate(Grammar):
        if rule.lhs is not last_lhs:
            Nonterm[rule.lhs.index] = rule.lhs
            Lo_rule[rule.lhs.index] = i
            if last_lhs is not None:
                Hi_rule[last_lhs.index] = i
            last_lhs = rule.lhs
    if last_lhs is not None:
        Hi_rule[last_lhs.index] = len(Grammar)

    # static analysis: mark nonterminals that can derive the empty string
    Derives_emptystr = [False] * N_nonterms
    changed = True
    while changed:
        changed = False
        for i in range(N_nonterms):
            if Derives_emptystr[i]:
                continue
            for j in range(Lo_rule[i], Hi_rule[i]):
                rhs = Grammar[j].rhs
                if all(c.nonterm is not None and Derives_emptystr[c.nonterm.index]
                       for c in rhs):
                    Derives_emptystr[i] = True
                    changed = True
                    break
    return True


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def fprint_gsym(gsym):
    """Human-readable form: ``field OPNAME value`` (used in rule printing)."""
    if gsym is None:
        return "(null)"
    if gsym.nonterm is not None:
        return sym_str(gsym.nonterm)
    return (names_mod.field_name(gsym.attr_index)
            + OP_NAME[Operator(gsym.op)]
            + (fmt_g(gsym.value.num) if gsym.value.nom is None
               else sym_str(gsym.value.nom)))


def fshow_gsym(gsym):
    """Machine-readable form for .shyp: ``"%s %s %s" % (field, Op_name[op],
    value)`` -- note Op_name keeps its own surrounding spaces (e.g. " ~ "),
    so set tests render as ``WORDS  ~  value``."""
    if gsym is None:
        return "(null)"
    if gsym.nonterm is not None:
        return sym_str(gsym.nonterm)
    val = fmt_g(gsym.value.num) if gsym.value.nom is None else gsym.value.nom.name
    return "%s %s %s" % (names_mod.field_name(gsym.attr_index),
                         OP_NAME[Operator(gsym.op)], val)


def print_grammar():
    for rule in Grammar:
        line = "!" * rule.priority + sym_str(rule.lhs) + " --> "
        line += ",".join(fprint_gsym(g) for g in rule.rhs)
        print(line + ".")
