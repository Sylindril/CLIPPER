"""extract.py - value-extraction rules and scoring.

The learner is unchanged: boosting still scores every candidate by a weighted
(p, n) pair.  Only the meaning of the pair moves -- a rule now returns a
*value* instead of a bool, so

    p = weight where the rule fires and normalize.equiv(value, gold)
    n = weight where it fires and the value is wrong
    abstain (None) contributes to neither

which leaves _alphah / _value / _loss / the reweighting update / internal-CV
round selection structurally intact (extract_evaluate replaces
counting.count_examples, extract_error_rate replaces soft_error_rate).

Two rule kinds, both scored through a ``.extract(x) -> value | None`` method:

    ExtractRule - sibling of concept.PyRule; LLM-written source defining
                  ``extract(x: dict)``, run in the same smolagents sandbox.
    RegexRule   - design A': pure data (pattern, flags, template), no sandbox.

There is no consequent (``conseq`` stays None) and no default rule: the
default is to abstain, so a prediction is the alpha-weighted argmax over the
canonical values the firing rules returned, and no firing rule means no answer.

normalize.py is the only comparator -- nothing here canonicalizes on its own.
"""

import json
import re

from . import config, dataset
from .errors import fatal
from .normalize import canon, equiv, parse_value
from .symbols import fmt_g

_RE_FLAGS = {"i": re.I, "m": re.M, "s": re.S, "x": re.X, "a": re.A}

# Parsed --rule-pool files, keyed by path.  Cleared by dataset.reset_all.
_pool_cache = {}


def reset():
    _pool_cache.clear()


def _alphah(p, n):
    from .boosting import _alphah as f      # one definition of the confidence
    return f(p, n)


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

class ExtractRule:
    """A rule whose body is Python returning a value (or None to abstain).

    Mirrors concept.PyRule field for field; ``source`` is the generated text
    defining ``def extract(x: dict)`` and ``fn`` the sandbox callable.
    ``conseq`` has no meaning when the prediction is the returned value -- it
    is kept (as None) only so the shared boosting/printing paths can read it.
    """
    __slots__ = ("conseq", "source", "fn", "nposx", "nnegx")

    def __init__(self, source="", fn=None, nposx=0.0, nnegx=0.0):
        self.conseq = None
        self.source = source
        self.fn = fn
        self.nposx = nposx
        self.nnegx = nnegx

    @property
    def alpha(self):
        return _alphah(self.nposx, self.nnegx)

    def extract(self, x):
        return self.fn(x)

    def shyp_line(self):
        return ("exrule %s %s SOURCE %s .\n"
                % (fmt_g(self.nposx), fmt_g(self.nnegx), json.dumps(self.source)))


def sandbox_rule(source, nposx=0.0, nnegx=0.0):
    """An ExtractRule whose body runs in the PyRule sandbox: the same
    LocalPythonExecutor, bound to ``extract`` instead of ``covers`` and with
    the bool coercion dropped (llm_rule_builder._SandboxRule)."""
    from .llm_rule_builder import _SandboxRule
    return ExtractRule(source=source,
                       fn=_SandboxRule(source, entry="extract", coerce=None),
                       nposx=nposx, nnegx=nnegx)


class RegexRule:
    """``re.search(pattern, x["NOTE"], flags)``, with the value built from
    ``template``: int items are match-group indices, str items are literals.

    ``[1, "years"]`` on ``(\\d+)-year-old`` gives ``["87", "years"]``; a
    one-item template gives the bare item; template None means group 1.  No
    match means abstain.  Numeric strings are coerced by normalize.canon, so a
    rule never converts anything itself.
    """
    __slots__ = ("conseq", "pattern", "flags", "template", "rx", "nposx", "nnegx")

    def __init__(self, pattern, flags="", template=None, nposx=0.0, nnegx=0.0):
        self.conseq = None
        self.pattern = pattern
        self.flags = flags or ""
        self.template = template
        f = 0
        for ch in self.flags:
            if ch not in _RE_FLAGS:
                fatal("unknown regex flag %r in rule pool (known: %s)",
                      ch, "".join(sorted(_RE_FLAGS)))
            f |= _RE_FLAGS[ch]
        self.rx = re.compile(pattern, f)
        self.nposx = nposx
        self.nnegx = nnegx

    @property
    def alpha(self):
        return _alphah(self.nposx, self.nnegx)

    @property
    def source(self):
        """The canonical spec; also the dedup key (see boosting._rule_cmp)."""
        return json.dumps(self.spec(), sort_keys=True)

    def spec(self):
        return {"pattern": self.pattern, "flags": self.flags,
                "template": self.template}

    def extract(self, x):
        m = self.rx.search(x["NOTE"])
        if m is None:
            return None
        if self.template is None:
            return m.group(1)
        out = [m.group(t) if isinstance(t, int) else t for t in self.template]
        return out[0] if len(out) == 1 else out

    def shyp_line(self):
        return ("rerule %s %s SPEC %s .\n"
                % (fmt_g(self.nposx), fmt_g(self.nnegx), self.source))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def gold(ex):
    """The example's gold value: lab carries the raw repr (see dataset.py)."""
    return parse_value(ex.lab.nom.name)


def inst_dict(inst):
    # dataset, not llm_rule_builder: the --rule-pool path must not drag in the
    # LLM stack.  (Module-level `from . import dataset` even though dataset
    # imports us back for reset_all -- neither body touches the other at import
    # time, so the partially-initialized module binding is enough.)
    return dataset.inst_to_dict(inst)


def rule_extract(rule, x):
    """``rule.extract(x)``, with a crashing rule abstaining -- the extraction
    counterpart of boosting._rule_covers' exception tally."""
    from . import boosting
    try:
        return rule.extract(x)
    except Exception as e:
        boosting._round_exceptions[(type(e).__name__, str(e))] += 1
        return None


def rule_y(rule, ex):
    """The boosting label for one example: +1 fired and right, -1 fired and
    wrong, 0 abstained (no weight update)."""
    v = rule_extract(rule, inst_dict(ex.inst))
    if v is None:
        return 0
    return 1 if equiv(v, gold(ex)) else -1


def extract_evaluate(rule, data):
    """(p, n) under the live example weights: right/wrong where it fires."""
    p = n = 0.0
    for ex in data:
        y = rule_y(rule, ex)
        if y > 0:
            p += ex.wt
        elif y < 0:
            n += ex.wt
    return p, n


def extract_classify(c, x):
    """Return (value, confidence): sum alpha per distinct canonical value over
    every firing rule, argmax.  Nothing fires -> (None, 0.0), i.e. abstain --
    the default rule is abstain, so there is no default-value competition."""
    votes = {}
    for r in c.rules:
        v = rule_extract(r, x)
        if v is None:
            continue
        k = canon(v)
        acc = votes.get(k)
        if acc is None:
            votes[k] = [r.alpha, v]
        else:
            acc[0] += r.alpha
    if not votes:
        return None, 0.0
    best = max(votes.values(), key=lambda a: a[0])
    return best[1], best[0]


def extract_error_rate(c, dataset):
    """Weighted 1 - exact match after canonicalization; abstaining is wrong.
    Same signature as boosting.soft_error_rate (crossval / cli need no edit)."""
    err = tot = 0.0
    for ex in dataset:
        tot += ex.wt
        v, _ = extract_classify(c, inst_dict(ex.inst))
        if v is None or not equiv(v, gold(ex)):
            err += ex.wt
    if tot == 0:
        return 0.0
    return err / tot


def extract_counts(c, dataset):
    """(n_exact, n_abstain, n) -- the unweighted counts cli.py reports below
    the error-rate lines."""
    exact = abstain = 0
    for ex in dataset:
        v, _ = extract_classify(c, inst_dict(ex.inst))
        if v is None:
            abstain += 1
        elif equiv(v, gold(ex)):
            exact += 1
    return exact, abstain, len(dataset)


# ---------------------------------------------------------------------------
# the rule pool (--rule-pool FILE): the deterministic, no-LLM rule builder
# ---------------------------------------------------------------------------

def load_pool(path):
    """Parse a JSON list of RegexRule specs; cached by path."""
    pool = _pool_cache.get(path)
    if pool is None:
        with open(path) as fp:
            specs = json.load(fp)
        if not isinstance(specs, list) or not specs:
            fatal("--rule-pool %s must hold a non-empty JSON list of rule specs",
                  path)
        pool = [RegexRule(s["pattern"], s.get("flags", ""), s.get("template"))
                for s in specs]
        _pool_cache[path] = pool
    return pool


def best_pool_rule(data):
    """Score every pool rule under the live weights and return (rule, p, n)
    for the best _value(p, n).  A fresh rule object each round, so the same
    pool entry picked twice gets its own (p, n) and compress_soft_rules can
    merge them by spec."""
    from .boosting import _value
    best = None
    for r in load_pool(config.Rule_pool):
        p, n = extract_evaluate(r, data)
        v = _value(p, n)
        if best is None or v > best[0]:
            best = (v, r, p, n)
    _, r, p, n = best
    return RegexRule(r.pattern, r.flags, r.template, nposx=p, nnegx=n), p, n
