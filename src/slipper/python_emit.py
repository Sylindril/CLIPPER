"""python_emit.py - emit a standalone Python classifier from a learned Concept.

When the slipper2 LLM path produces a Concept made entirely of PyRules, we
serialize the model as a self-contained Python file: each PyRule.source
becomes a module-level helper (``covers`` -> ``ruleN``), a ``classify``
function reproduces ``boosting.soft_classify``, and a ``__main__`` block
scores a JSON test set into a confusion matrix.  Mixed or grammar-only
hypotheses are skipped because the gsym forms have no direct Python
equivalent (would need a gsym -> Python translator).
"""

import textwrap

from . import config
from . import names as names_mod
from .concept import PyRule
from .errors import warning


_HEADER = '''\
"""Self-contained classifier emitted by slipper2.

Reproduces the learned soft ruleset as a working Python program with no
dependency on slipper.  Run with a STEM.json test file to get a confusion
matrix and accuracy on the test split.
"""

import json
import math
import sys

'''


_CLASSIFY_BOOST = '''\

def classify(x):
    """Return (predicted_class, confidence)."""
    sumcon = _alphah(_DEFAULT_NPOSX, _DEFAULT_NNEGX)
    for fn, conseq_is_pos, p, n in _RULES:
        try:
            covered = bool(fn(x))
        except Exception:
            covered = False
        if covered:
            if conseq_is_pos:
                sumcon += _alphah(p, n)
            else:
                sumcon += _alphah(n, p)
    return (NEGATIVE_CLASS if sumcon <= 0 else POSITIVE_CLASS), sumcon
'''

# --weighting cover / coverlist: an ordered list, first firing rule wins with
# its own consequent, default class otherwise.  Every cover rule carries the
# positive consequent, which makes this a disjunction; a coverlist rule may
# carry either.  The (p, n) columns are carried for reference only.
_CLASSIFY_COVER = '''\

def classify(x):
    """Return (predicted_class, confidence): first rule that fires wins."""
    for fn, conseq_is_pos, p, n in _RULES:
        try:
            covered = bool(fn(x))
        except Exception:
            covered = False
        if covered:
            if conseq_is_pos:
                return POSITIVE_CLASS, 1.0
            return NEGATIVE_CLASS, -1.0
    return NEGATIVE_CLASS, -1.0
'''

_RUNNER = '''\


def _row_to_dict(row, attributes):
    x = {}
    for adef, val in zip(attributes, row["values"]):
        if adef.get("kind") == "ignore" or adef.get("suppressed"):
            continue
        x[adef["name"]] = val
    return x


def _main():
    if len(sys.argv) != 2:
        print("usage: %s STEM.json" % sys.argv[0], file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1]) as fp:
        data = json.load(fp)
    rows = data.get("test") or []
    attrs = data["attributes"]
    confusion = {}
    correct = 0
    for row in rows:
        x = _row_to_dict(row, attrs)
        predicted, _ = classify(x)
        actual = row.get("label")
        key = (actual, predicted)
        confusion[key] = confusion.get(key, 0) + 1
        if predicted == actual:
            correct += 1
    classes = sorted({POSITIVE_CLASS, NEGATIVE_CLASS}
                     | {a for (a, _) in confusion}
                     | {p for (_, p) in confusion})
    col_w = max(8, max((len(c) for c in classes), default=0) + 2)
    print("confusion matrix (rows=actual, cols=predicted):")
    print(" " * col_w + "".join("%-*s" % (col_w, c) for c in classes))
    for a in classes:
        line = "%-*s" % (col_w, a)
        for p in classes:
            line += "%-*d" % (col_w, confusion.get((a, p), 0))
        print(line)
    if rows:
        print("accuracy: %.4f (%d / %d)"
              % (correct / len(rows), correct, len(rows)))
    else:
        print("accuracy: n/a (no test data)")


if __name__ == "__main__":
    _main()
'''


def _rename_covers(source: str, fname: str) -> str:
    """Rename the first ``def covers(`` in ``source`` to ``def <fname>(``.

    If the source doesn't define ``covers`` (shouldn't happen — build_pyrule
    rejects such outputs), wrap whatever the LLM gave us inside a stub
    function that returns False.
    """
    source = (source or "").strip()
    if "def covers(" in source:
        return source.replace("def covers(", "def " + fname + "(", 1)
    return "def %s(x):\n%s\n    return False" % (
        fname, textwrap.indent(source or "pass", "    "))


def _python_source(hyp) -> str:
    Classes = names_mod.Classes
    neg = hyp.def_.nom.name
    pos = next((c.nom.name for c in Classes if c.nom.name != neg), "?")

    parts = [_HEADER]
    parts.append("POSITIVE_CLASS = %r\n" % pos)
    parts.append("NEGATIVE_CLASS = %r\n" % neg)
    parts.append("EPSILON = %r\n" % config.Epsilon)
    parts.append("_DEFAULT_NPOSX = %r\n" % hyp.nposx)
    parts.append("_DEFAULT_NNEGX = %r\n\n\n" % hyp.nnegx)
    parts.append("def _alphah(p, n):\n")
    parts.append("    return 0.5 * math.log((p + EPSILON) / (n + EPSILON))\n\n\n")

    parts.append("# === learned rule helpers ===\n\n")
    for i, r in enumerate(hyp.rules, start=1):
        renamed = _rename_covers(r.source, "rule%d" % i)
        parts.append(renamed.rstrip() + "\n\n\n")

    parts.append("# === rule table (fn, conseq_is_positive, nposx, nnegx) ===\n\n")
    parts.append("_RULES = [\n")
    for i, r in enumerate(hyp.rules, start=1):
        parts.append("    (%s, %r, %r, %r),\n"
                     % ("rule%d" % i, r.conseq.name == pos, r.nposx, r.nnegx))
    parts.append("]\n")

    parts.append(_CLASSIFY_COVER if config.Weighting in ("cover", "coverlist")
                 else _CLASSIFY_BOOST)
    parts.append(_RUNNER)
    return "".join(parts)


def write_python(path: str, hyp):
    """Write a self-contained Python program implementing ``hyp`` to ``path``.

    Returns ``path`` on success, ``None`` if the hypothesis isn't emittable
    (no rules, or any non-PyRule entries).
    """
    if not hyp.rules:
        return None
    if config.Task_mode == "extract":
        return None            # no emitter for value-extraction hypotheses yet
    if any(not isinstance(r, PyRule) for r in hyp.rules):
        warning("python emit skipped (%s): hypothesis contains non-PyRule entries",
                path)
        return None
    # utf-8 explicitly: rule sources may quote raw document text (emoji etc.)
    # under --use-text, which the Windows default cp1252 cannot encode.
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_python_source(hyp))
    return path
