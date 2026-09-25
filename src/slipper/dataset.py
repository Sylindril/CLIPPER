"""dataset.py - load a dataset from JSON.

There is no separate .names/.data text-format parser; a dataset JSON file
has the form::

    {
      "classes": ["democrat", "republican"],
      "attributes": [{"name": "physician_fee_freeze", "kind": "symbolic"},
                     {"name": "age", "kind": "continuous"},
                     {"name": "WORDS", "kind": "set"}, ...],
      "train": [{"values": [...], "label": "...", "weight": 1.0}, ...],
      "test":  [ ... ]            # optional
    }

A value is a string (symbolic), a number (continuous), a list of strings
(set/bag), a string (text), or null (a missing "?" value).  ``kind`` is one of
"symbolic", "continuous", "set", "bag", "text", "ignore".  An optional "values"
list on a symbolic attribute declares its domain (as a .names file would).

"text" is raw free text -- a whole clinical note, say -- carried verbatim to
the LLM path's ``covers(x)`` and prompt.  Unlike "symbolic" it registers no
domain values, and ``grammar.define_cond_a`` builds no conditions over it, so
the discrete learner ignores the column entirely.

A ``"task": "extract"`` file (see extract.py) has no "classes" key: "label" is
the repr of the gold *value*, kept verbatim on the example and compared only
via normalize.equiv.  The mode is recorded in ``config.Task_mode``.
"""

import json

from . import config, extract, grammar, model
from . import names as names_mod
from .data import Aval, Example
from .names import AttrDef, AttrKind
from .symbols import Atom, SymKind, intern, reset_symbol_table

_KIND = {
    "symbolic": AttrKind.SYMBOL,
    "symbol": AttrKind.SYMBOL,
    "continuous": AttrKind.CONTINUOUS,
    "set": AttrKind.SET,
    "bag": AttrKind.SET,
    "text": AttrKind.TEXT,
    "ignore": AttrKind.IGNORE,
}

# The stand-in class for extraction: there is no class list, but Concept.def_
# needs a default to print and the stratified shuffles index by class.
ABSTAIN = "abstain"


def reset_all():
    """Reset every module that holds global state (for a fresh load)."""
    reset_symbol_table()
    names_mod.reset()
    grammar.reset()
    model.reset()
    extract.reset()
    config.Task_mode = "classify"


def _bagify(syms):
    """Rename the i-th (i>1) occurrence of a term ``t`` to ``t__i``."""
    counts = {}
    out = []
    for s in syms:
        counts[s.name] = counts.get(s.name, 0) + 1
        c = counts[s.name]
        out.append(intern("%s__%d" % (s.name, c)) if c > 1 else s)
    return out


def _setup_names(spec):
    # classes ("task": "extract" files have none -- see ABSTAIN above)
    classes = [ABSTAIN] if config.Task_mode == "extract" else spec["classes"]
    for i, cname in enumerate(classes):
        sym = intern(cname)
        sym.kind = SymKind.CLASS
        sym.index = i
        names_mod.Classes.append(Atom(nom=sym))
    # attributes
    for i, attr in enumerate(spec["attributes"]):
        adef = AttrDef(intern(attr["name"]))
        adef.name.kind = SymKind.ATTRIBUTE
        adef.name.index = i
        adef.kind = _KIND[attr["kind"]]
        adef.isbag = (attr["kind"] == "bag")
        adef.suppressed = bool(attr.get("suppressed", False))
        names_mod.Names.append(adef)
        # seed declared symbolic values (as the .names parser would)
        for v in attr.get("values", []):
            names_mod.update_field_values(i, intern(str(v)))
    names_mod.mark_names_defined()


def _build_example(row, index):
    inst = []
    for i, adef in enumerate(names_mod.Names):
        v = row["values"][i]
        if adef.kind == AttrKind.IGNORE or v is None:
            inst.append(Aval(AttrKind.MISSING_VALUE))
        elif adef.kind == AttrKind.TEXT:
            # raw document string, preserved verbatim (no interning: values are
            # per-example unique and never enter the grammar/symbol machinery)
            inst.append(Aval(AttrKind.TEXT, text=str(v)))
        elif adef.kind == AttrKind.CONTINUOUS:
            inst.append(Aval(AttrKind.CONTINUOUS, num=float(v)))
        elif adef.kind == AttrKind.SYMBOL:
            sym = intern(str(v))
            names_mod.update_field_values(i, sym)
            inst.append(Aval(AttrKind.SYMBOL, sym=sym))
        elif adef.kind == AttrKind.TEXT:
            inst.append(Aval(AttrKind.TEXT, text=str(v)))
        else:  # SET
            syms = [intern(str(w)) for w in v]
            if adef.isbag:
                syms = _bagify(syms)
            for sym in syms:
                names_mod.update_field_values(i, sym)
            inst.append(Aval(AttrKind.SET, set=syms))
    lab = Atom(nom=intern(str(row["label"])))
    if config.Task_mode == "extract":
        # The label is a value, not a class: keep its repr verbatim (extract.gold
        # parses it, normalize.equiv compares it) and don't register it as a
        # class.  index 0 is the abstain slot, so stratify_and_shuffle_data --
        # the only reader of lab.nom.index left in extraction mode -- degenerates
        # to a plain shuffle.
        lab.nom.index = 0
    else:
        names_mod.update_class(lab)
    return Example(inst=inst, lab=lab, wt=float(row.get("weight", 1.0)), index=index)


def inst_to_dict(inst):
    """Convert an Example.inst (list of Aval) into the {attr_name: value} dict
    that a Python rule body sees (``covers(x)`` / ``extract(x)``).

    Values are str (symbolic or text), float (continuous), list[str] (set), or
    None (missing).  Attributes flagged ignore/suppressed are omitted so
    generated code can't condition on data the discrete learner would skip.
    The inverse of _build_example's inst construction, hence its place here.
    """
    out = {}
    for i, av in enumerate(inst):
        if names_mod.llm_hidden(i):
            continue
        adef = names_mod.Names[i]
        name = adef.name.name
        if av.kind == AttrKind.MISSING_VALUE:
            out[name] = None
        elif av.kind == AttrKind.SYMBOL:
            out[name] = av.sym.name if av.sym is not None else None
        elif av.kind == AttrKind.CONTINUOUS:
            out[name] = av.num
        elif av.kind == AttrKind.SET:
            out[name] = [s.name for s in (av.set or [])]
        elif av.kind == AttrKind.TEXT:
            out[name] = av.text
        else:
            out[name] = None
    return out


def _load_examples(rows):
    return [_build_example(row, i) for i, row in enumerate(rows or [])]


def load_dataset(path):
    """Load a dataset JSON file; return (train_data, test_data).  ``test_data``
    is None when the file has no "test" key.  Resets all global state first."""
    reset_all()
    with open(path) as fp:
        spec = json.load(fp)
    config.Task_mode = "extract" if spec.get("task") == "extract" else "classify"
    _setup_names(spec)
    train = _load_examples(spec.get("train"))
    test = _load_examples(spec["test"]) if "test" in spec and spec["test"] is not None else None
    return train, test
