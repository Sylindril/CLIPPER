"""normalize.py - canonical form for extracted values (MedCalc extraction task).

The gold `Relevant Entities` column is a Python dict repr whose values come in four
shapes: `[value, unit]` 2-lists, bare bools, bare strings/numbers, and 3-lists (steroid
conversion).  Units and casing are unnormalized in the source data - Systolic BP carries
'mm hg' / 'mm Hg' / 'mmhg', Heart Rate 'beats per minute' / 'bpm' - so exact match on the
raw strings scores noise.

Both a rule's predicted value and the gold label pass through `canon` before comparison.
This lives here, outside the rule sandbox, so training and test are scored by the same
canonicalizer and a generated `extract()` cannot smuggle its own normalization in.

The alias table is spelling/casing only.  Never put a numeric conversion in it: celsius
and fahrenheit stay distinct, and so do mEq/L and mmol/L.
"""

import ast

# Tag for bools.  They must not collide with numbers, and `True == 1.0` in Python, so
# returning the bare bool is not enough: a bare-bool gold and a numeric one would vote
# together.  The sentinel cannot occur in parsed data.
BOOL = object()

# casefold(micro sign) is the Greek mu, so the keys are folded at import.
_UNIT_ALIASES = {k.casefold(): v for k, v in {
    "mm hg": "mm hg",
    "mmhg": "mm hg",
    "beats per minute": "bpm",
    "bpm": "bpm",
    "µL": "µL",
    "uL": "µL",
    "µmol/L": "µmol/L",
    "umol/L": "µmol/L",
    "degrees celsius": "degrees celsius",
    "celsius": "degrees celsius",
    "degrees fahrenheit": "degrees fahrenheit",
    "fahrenheit": "degrees fahrenheit",
}.items()}


def parse_value(s):
    """Parse one repr'd value.  Gold labels are always literals; rule outputs may be
    plain strings, which come back stripped."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return s.strip()


def canon(v):
    """Map a parsed value to a hashable canonical form."""
    if isinstance(v, bool):
        return (BOOL, v)               # before the number test: bool subclasses int
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, str):
        s = v.casefold().strip()
        try:
            return round(float(s), 6)  # a rule returning "48" matches gold 48
        except ValueError:
            return _UNIT_ALIASES.get(s, s)
    if isinstance(v, (list, tuple)):
        return tuple(canon(x) for x in v)
    return v


def equiv(pred, gold):
    return canon(pred) == canon(gold)
