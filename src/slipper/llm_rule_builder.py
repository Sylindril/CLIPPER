"""llm_rule_builder.py - LLM-based replacement for boosting.build_rule.

When ``config.Rule_llm_model`` is set, ``boosting.soft_fit`` calls
``build_pyrule`` instead of growing a rule with discrete grammar search.  The
LLM is given the attribute schema and a weighted sample of training examples
drawn from the current boosting distribution, and must return a Python snippet
defining ``def covers(x: dict) -> bool``.  The snippet is executed inside a
smolagents LocalPythonExecutor sandbox; the resulting predicate becomes a
``concept.PyRule``.

Layout:
    _render_schema / _render_examples                 - prompt content
                     (inst_to_dict, the instance -> dict conversion the
                      generated code sees, lives in dataset.py)
    _DEFAULT_PROMPT / _load_prompt                    - template plumbing
    _PROMPT_VARIANTS                                  - --prompt-variant task blocks
    _get_agent                                        - Pydantic AI agent cache
    _SandboxRule                                      - smolagents-backed eval
    _evaluate                                         - (p, n) over a dataset
    build_pyrule                                      - entry point
    ... value extraction ...                          - the extraction arms
    _EXTRACT_PROMPT_VARIANTS / build_extract_rule       (see extract.py)

The weighted sample is drawn by ``data.weighted_sample`` and passed in by
``boosting.soft_fit``; the same routine drives sampling in the discrete
rule-search path so the two are directly comparable.

On a ``"task": "extract"`` dataset the same plumbing builds a *value*-returning
rule instead: ``build_extract_rule`` reuses the rendering, agent, sandbox and
retry machinery below and differs only in the task definition, the structured
output model, and the fact that (p, n) comes from ``extract.extract_evaluate``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel
from pydantic_ai import RunContext   # imported here so get_type_hints can resolve
                                     # `RunContext[_ToolDeps]` on the tool function
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

# Per-HTTP-request timeout for tool-using runs.  Stops a stuck call from
# burning the 10-minute litellm default while the retry loop sits idle.
_PER_REQUEST_TIMEOUT_S = 60.0
# The tool-less agent makes one long request per rule (no tool round trips), so
# it gets its own, much longer limit: measured per-call means at n=250 run from
# 1.3s (3.1-flash-lite) to 74s (claude-sonnet-5, one cell averaging 152s).  It
# used to have none at all, and a single unanswered request hung a sonnet cell
# for over two hours.
_NO_TOOLS_REQUEST_TIMEOUT_S = 600.0


def _tool_limits(budget: int) -> UsageLimits:
    """N tool calls cost ~N+1 model requests (initial + N tool round-trips);
    leave a small margin for the final commit."""
    return UsageLimits(request_limit=budget + 5, tool_calls_limit=budget)


try:
    from pydantic_ai import ModelRetry as _ModelRetry
except Exception:                       # pragma: no cover - older pydantic_ai
    _ModelRetry = None

# The exception a tool raises on a bad model-supplied argument.  ModelRetry
# (when available) is caught by pydantic-ai and handed back to the model for
# self-correction within the run; otherwise ValueError aborts the attempt and
# our retry loop / grown-rule fallback takes over.  Exposed for tests.
_TOOL_ARG_EXC = _ModelRetry if _ModelRetry is not None else ValueError


def _tool_arg_error(msg: str) -> Exception:
    """A tool argument-validation error to *raise* (see ``_TOOL_ARG_EXC``)."""
    return _TOOL_ARG_EXC(msg)

from . import config, extract, optimize, record, rng, trace
from .retrieval import RetrievalSession
from . import names as names_mod
from .dataset import inst_to_dict      # instance -> {attr: value}; lives with
                                       # the loader (pure function over Names)
from .errors import LLMAccountError, fatal, warning
from .names import AttrKind
from .symbols import Operator



# Provider phrasings for "no money / no quota", matched case-insensitively
# against the exception text (which for pydantic-ai's ModelHTTPError includes
# the provider's response body).  Anthropic returns these as HTTP 400, OpenAI
# as a 429 or a stream error, OpenRouter as 402 -- so the status code alone is
# not enough, and a plain rate-limit 429 must stay retryable.
_ACCOUNT_ERROR_PHRASES = (
    "credit balance is too low",      # anthropic
    "no credits remaining",           # openai
    "insufficient_quota",             # openai
    "exceeded your current quota",    # openai
    "requires more credits",          # openrouter
    "insufficient credits",           # openrouter
    "payment required",               # any 402
)
_ACCOUNT_ERROR_STATUS = (401, 402, 403)   # bad key, unpaid, forbidden


def _account_error_reason(e: BaseException) -> Optional[str]:
    """A short reason if ``e`` (or anything in its cause chain) is an account
    or billing refusal rather than a problem with this particular request;
    None otherwise.  Keys on the HTTP status where the exception carries one
    (pydantic-ai's ModelHTTPError and litellm's exceptions both expose
    ``status_code``) and on provider phrasing in the message."""
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        text = str(e)
        status = getattr(e, "status_code", None)
        if status in _ACCOUNT_ERROR_STATUS:
            return "HTTP %s: %s" % (status, text[:300])
        low = text.lower()
        if any(p in low for p in _ACCOUNT_ERROR_PHRASES):
            return text[:300]
        e = e.__cause__ or e.__context__
    return None


_METER_INSTALLED = False


def _install_request_meter() -> None:
    """Wrap the litellm call that pydantic-ai-litellm makes, so every request
    reaches record.note_api_request -- including the requests of an attempt
    that later fails.  Without this, spend was only counted for successful
    run_sync attempts (_record_call), and conditions with many failed attempts
    under-reported cost several-fold.  A wrapper rather than a litellm
    callback: litellm's async success callbacks can run after pydantic-ai's
    event loop has closed, which would drop exactly these events.
    Idempotent."""
    global _METER_INSTALLED
    if _METER_INSTALLED:
        return
    try:
        import pydantic_ai_litellm.litellm_model as _plm
    except ImportError:                 # pragma: no cover - dependency missing
        return
    original = _plm.acompletion

    async def metered(*args, **kwargs):
        model = kwargs.get("model", "")
        try:
            response = await original(*args, **kwargs)
        except Exception as e:
            record.note_api_request(model, error=e)
            raise
        record.note_api_request(model, response=response)
        return response

    _plm.acompletion = metered
    _METER_INSTALLED = True

def _record_call(result, latency_s: float) -> None:
    """Feed one ``agent.run_sync`` result's usage + wall latency to any active
    ``record.recorder()`` (no-op when none is collecting)."""
    usage = None
    try:
        usage = result.usage()
    except Exception:
        pass
    record.note_llm_call(latency_s, config.Rule_llm_model, usage=usage)


# ---------------------------------------------------------------------------
# rendering: instances, schema, examples
# ---------------------------------------------------------------------------

def _render_schema() -> str:
    lines = ["| attribute | kind | domain |", "|---|---|---|"]
    for i, adef in enumerate(names_mod.Names):
        if names_mod.llm_hidden(i):
            continue
        if adef.kind == AttrKind.SYMBOL:
            kind = "symbolic"
            domain = ", ".join(v.name for v in adef.values) if adef.values else "(any)"
        elif adef.kind == AttrKind.CONTINUOUS:
            kind = "continuous"
            domain = "float"
        elif adef.kind == AttrKind.SET:
            kind = "set" if not adef.isbag else "bag"
            domain = ", ".join(v.name for v in adef.values) if adef.values else "(any)"
        elif adef.kind == AttrKind.TEXT:
            kind = "text"
            # No enumerable domain: the value is the raw string itself.
            domain = "free text"
        else:
            continue
        lines.append("| %s | %s | %s |" % (adef.name.name, kind, domain))
    return "\n".join(lines)


def _schema_block() -> str:
    """The prompt's attribute-schema section, gated by --schema-format.

    'full' emits an explicit attribute|kind|domain menu (with full symbolic
    domain enumerations); 'compressed' omits it entirely, leaving the model to
    read attributes off the sample plus the prompt's inline type legend.  On
    wide, interaction-heavy datasets the full menu is what lets the model find
    multi-attribute rules instead of a single marginal .

    The section header lives here (not the template) so 'compressed' leaves no
    dangling heading behind the empty {schema} slot.

    Extraction takes the compressed path unconditionally: its attributes are
    free text declared symbolic, so the note column's "domain" is one distinct
    value per note and the menu would dwarf the sample.  The example table's
    header and kind rows already name every key of `x`.
    """
    if config.Task_mode == "extract":
        return ""
    if getattr(config, "Schema_format", "full") == "compressed":
        return ""
    return ("Attribute schema (the full set of keys available in `x`, "
            "with domains):\n\n%s\n" % _render_schema())


def _text_note() -> str:
    """The prompt's text-attribute line in the type legend.

    Text attributes are always visible to the LLM.  With a text attribute in
    the schema the legend gains one line for it; with ``--use-text`` it also
    tells the model that regex is available.  Empty for purely tabular data, so
    those prompts are unchanged.
    """
    if getattr(config, "Use_text", False):
        return (
            "  - text       -> str: the raw document (or None for missing)\n"
            "\n"
            "For text attributes you may use regular expressions: `import re` is\n"
            "available in the sandbox.  Match on phrases, word order, or negation\n"
            "the bag-of-words attributes cannot express (e.g.\n"
            "re.search(r'\\bnot\\s+(good|great)\\b', x.get('TEXT') or '', re.I)).\n"
            "Prefer patterns that generalize; do not memorize whole documents.\n")
    if any(adef.kind == AttrKind.TEXT and not names_mod.llm_hidden(i)
           for i, adef in enumerate(names_mod.Names)):
        return "  - text       -> str, raw free text (or None for missing)\n"
    return ""


def _advice_block() -> str:
    """The prompt's task-advice section (--include-advice EXT).

    Empty unless cli.py loaded a STEM<EXT> file into ``config.Advice_text``.
    The header lives here (not the template) so runs without advice leave no
    dangling heading behind the empty {task_info} slot.
    """
    advice = getattr(config, "Advice_text", None)
    if not advice or not advice.strip():
        return ""
    return ("Background information about this prediction task:\n\n%s\n"
            % advice.strip())


def _render_examples(sampled) -> str:
    """One example per line as ``[label, weight]  {dict}``.

    The sample's .wt is normalized to sum to 1 (data.weighted_sample);
    rescaling by len(sampled) renders a uniformly-weighted sample as w=1.0
    per row, keeping the scale interpretable as boosting reweights rows.
    """
    rows = []
    n = len(sampled)
    for ex in sampled:
        d = inst_to_dict(ex.inst)
        rows.append("[%s w=%.4g] %r" % (ex.lab.nom.name, n * ex.wt, d))
    return "\n".join(rows)


def _attr_kind_map() -> Dict[str, str]:
    """name -> kind string ('symbolic'/'continuous'/'set'/'bag'/'text') for the
    same non-ignored attributes ``inst_to_dict`` exposes as keys of ``x``."""
    out: Dict[str, str] = {}
    for i, adef in enumerate(names_mod.Names):
        if names_mod.llm_hidden(i):
            continue
        if adef.kind == AttrKind.SYMBOL:
            out[adef.name.name] = "symbolic"
        elif adef.kind == AttrKind.CONTINUOUS:
            out[adef.name.name] = "continuous"
        elif adef.kind == AttrKind.SET:
            out[adef.name.name] = "bag" if adef.isbag else "set"
        elif adef.kind == AttrKind.TEXT:
            out[adef.name.name] = "text"
    return out


def _cell(v: Any) -> str:
    """Render one attribute value as a table cell.

    Default (``--cell-format repr``) ``repr()``-wraps every value, so strings
    read as ``'t'`` and are distinguishable from numbers (the dict format does
    the same via ``%r``).  ``--cell-format str`` renders bare values instead:
    missing as literal ``None`` — the value ``covers()`` actually receives, not
    a bare ``?`` (a ``?`` marker led the LLM to write unguarded ``< n`` compares
    on missing-heavy data) — floats via ``%g``, sets/bags as ``[a b]``, strings
    unquoted.  ``repr(None)`` is ``'None'``, so the missing marker is unchanged
    across both modes.

    Nothing here truncates: a ``text`` attribute's whole note reaches the cell.
    Both modes keep it on one physical line so the markdown row stays intact --
    ``repr`` because it escapes newlines as ``\\n``, ``str`` via the flattening
    below (which also makes cells pipe-safe, the one lossy step).
    """
    if getattr(config, "Cell_format", "str") == "repr":
        s = repr(v)
    elif v is None:
        return "None"
    elif isinstance(v, float):
        return "%g" % v
    elif isinstance(v, list):
        s = "[%s]" % " ".join(str(x) for x in v)
    else:
        s = str(v)
    return s.replace("|", "/").replace("\n", " ")  # keep cells pipe-safe


def _render_examples_table(sampled) -> str:
    """Combined schema+data as one markdown table (used by --example-format table).

    Header row is the exact keys of ``x``; the next row gives each column's
    kind; ``class``/``weight`` lead the table but are not keys of ``x``. Domains
    are left implicit — the LLM reads them off the sampled values, which is what
    makes this far more compact than the per-row dicts plus a separate schema.

    With ``config.Table_minibatch_size`` > 0 the rows are emitted as consecutive
    subtables of at most that many rows, each repeating the header and kind rows,
    so a cell is never far from the header it binds to.
    """
    n = len(sampled)
    keys = list(inst_to_dict(sampled[0].inst).keys()) if sampled else []
    kinds = _attr_kind_map()
    header = ["class", "weight"] + keys
    kindrow = ["—", "—"] + [kinds.get(k, "?") for k in keys]
    head = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        "| " + " | ".join(kindrow) + " |",
    ]
    rows = []
    for ex in sampled:
        d = inst_to_dict(ex.inst)
        row = [ex.lab.nom.name, "%.4g" % (n * ex.wt)] + [_cell(d.get(k)) for k in keys]
        rows.append("| " + " | ".join(row) + " |")
    r = getattr(config, "Table_minibatch_size", 0) or len(rows) or 1
    # Blank line between subtables: markdown needs it to start a new table.
    return "\n\n".join("\n".join(head + rows[i:i + r]) for i in range(0, len(rows), r)
                       ) or "\n".join(head)


# ---------------------------------------------------------------------------
# tool-call logging
# ---------------------------------------------------------------------------

def _short_result(result: Any) -> str:
    """One-line human summary of a tool result for the SUMM trace line."""
    if isinstance(result, dict):
        if "rows" in result and "remaining_rows" in result:
            return "returned=%s remaining=%s status=%s" % (
                result["returned"], result["remaining_rows"], result["status"])
        return " ".join("%s=%.4g" % (k, v) if isinstance(v, (int, float))
                        else "%s=%s" % (k, v) for k, v in result.items())
    if isinstance(result, list):
        return "[%s]" % ", ".join(
            "%s(p=%.2f,n=%.2f)" % (o.get("value"), o.get("p", 0.0), o.get("n", 0.0))
            for o in result[:3])
    return str(result)


class _ToolLog:
    """Records the ordered sequence of tool calls for one rule-build round.

    Each call is logged at ENTRY (via ``begin``) so calls that fail argument
    validation or hit a ModelRetry are captured too -- unlike the old
    success-only trace prints, which hid every rejected call.  ``end`` pairs
    with the preceding ``begin`` (tools run one at a time under the synchronous
    agent, so a single pending slot suffices), emits the ``-v1`` SUMM line, and
    appends the finished record.  ``flush`` writes the round's records as a JSON
    sidecar when ``-L``/``config.Log_stem`` is set.

    Inert until ``start`` is called, so the tool functions stay side-effect-free
    when invoked directly (e.g. from unit tests) outside an LLM rule build.
    """

    def __init__(self) -> None:
        self.active = False
        self.round = 0
        self.stage = ""
        self.calls: List[Dict[str, Any]] = []
        self._pending: Optional[Dict[str, Any]] = None

    def start(self, round: int, stage: str) -> None:
        self.active = True
        self.round = round
        self.stage = stage
        self.calls = []
        self._pending = None

    def begin(self, tool: str, args: Dict[str, Any]) -> None:
        if not self.active:
            return
        self._pending = {"seq": len(self.calls), "tool": tool, "args": args}

    def end(self, result: Any = None, error: Optional[BaseException] = None) -> None:
        if not self.active or self._pending is None:
            return
        rec = self._pending
        self._pending = None
        if error is not None:
            rec["error"] = "%s: %s" % (type(error).__name__, error)
        else:
            rec["result"] = result
        self.calls.append(rec)
        if trace.tracing(trace.SUMM):
            args = ", ".join("%s=%r" % (k, v) for k, v in rec["args"].items())
            head = "// tool[%s r%d #%d] %s(%s)" % (
                self.stage, self.round, rec["seq"], rec["tool"], args)
            tail = ("ERROR %s" % rec["error"] if "error" in rec
                    else _short_result(rec["result"]))
            print("%s -> %s" % (head, tail))

    def flush(self, log_stem: Optional[str]) -> None:
        """Write the round's records to ``STEM-NNN-STAGE.tools.json`` (if
        ``log_stem`` is set) and go inert until the next ``start``."""
        self.active = False
        if not log_stem:
            return
        path = "%s-%03d-%s.tools.json" % (log_stem, self.round, self.stage)
        payload = {
            "round": self.round,
            "stage": self.stage,
            "model": config.Rule_llm_model,
            "n_calls": len(self.calls),
            "calls": self.calls,
        }
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
        except OSError as e:
            warning("could not write tool-call log %s: %s", path, e)


_toollog = _ToolLog()


# ---------------------------------------------------------------------------
# tool implementation (count_coverage)
# ---------------------------------------------------------------------------

_SUPPORTED_OPS = ("==", "!=", ">=", "<=", ">", "<", "contains", "like")


def _matches(v: Any, op: str, value: Any) -> bool:
    """Evaluate a single condition; missing (None) never matches.

    Type / operator mismatches (e.g. >= on a string) return False rather than
    raising — the LLM should see "no examples satisfy this" rather than a
    tool-call retry storm.
    """
    if v is None:
        return False
    try:
        if op == "==":
            return v == value
        if op == "!=":
            return v != value
        if op == ">=":
            return v >= value
        if op == "<=":
            return v <= value
        if op == ">":
            return v > value
        if op == "<":
            return v < value
        if op == "contains":
            return value in v
        if op == "like":
            return isinstance(v, str) and value in v
    except Exception:
        return False
    return False


def _count_coverage(dataset, cl, field: str, op: str, value: Any) -> Tuple[float, float]:
    _toollog.begin("count_coverage", {"field": field, "op": op, "value": value})
    try:
        if op not in _SUPPORTED_OPS:
            # ModelRetry (when available) hands the message back to the model so
            # it fixes the call in-place, instead of a bare exception that aborts
            # the whole agent run and wastes the attempt.
            raise _tool_arg_error("unsupported op %r; use one of %s"
                                  % (op, ", ".join(_SUPPORTED_OPS)))
        p = n = 0.0
        for ex in dataset:
            d = inst_to_dict(ex.inst)
            if not _matches(d.get(field), op, value):
                continue
            if ex.lab.nom is cl:
                p += ex.wt
            else:
                n += ex.wt
    except Exception as e:
        _toollog.end(error=e)
        raise
    _toollog.end(result={"p": p, "n": n})
    return p, n


def _count_coverage_of_lambda_body(dataset, cl, body: str) -> Tuple[float, float]:
    """Compile ``lambda x: <body>`` inside the smolagents sandbox and count
    weighted positive/negative coverage on the full training set.

    Body must be a single-line Python expression in x (e.g. ``x.get('f')==1``
    or ``x.get('a') and x.get('b')``).  Per-example evaluation errors are
    treated as "doesn't cover" (mirrors _rule_covers / _evaluate semantics).
    """
    _toollog.begin("count_coverage_of_lambda_body", {"body": body})
    try:
        if "\n" in body:
            raise _tool_arg_error("body must be a single-line expression; got %r" % body)
        source = "covers = lambda x: %s\n" % body
        try:
            rule = _SandboxRule(source)
        except Exception as e:
            raise _tool_arg_error("failed to compile `lambda x: %s`: %s: %s"
                                  % (body, type(e).__name__, e))
        p = n = 0.0
        for ex in dataset:
            try:
                if rule(inst_to_dict(ex.inst)):
                    if ex.lab.nom is cl:
                        p += ex.wt
                    else:
                        n += ex.wt
            except Exception:
                continue
    except Exception as e:
        _toollog.end(error=e)
        raise
    _toollog.end(result={"p": p, "n": n})
    return p, n


# ---------------------------------------------------------------------------
# tool implementation (best_threshold / best_symbol) — full-data value pickers
# ---------------------------------------------------------------------------

def _attr_index(field: str) -> int:
    """Map a schema attribute name to its Example.inst index (skipping the
    ignored/suppressed fields inst_to_dict also hides)."""
    for i, adef in enumerate(names_mod.Names):
        if names_mod.llm_hidden(i):
            continue
        if adef.name.name == field:
            return i
    raise _tool_arg_error("unknown attribute %r (not in the schema)" % field)


class _Wild:
    """Minimal stand-in for a grammar wildcard: optimize.best_threshold /
    compute_field_stats only read ``.attr_index`` and ``.op``."""
    __slots__ = ("attr_index", "op")

    def __init__(self, attr_index: int, op) -> None:
        self.attr_index = attr_index
        self.op = op


def _best_threshold(dataset, cl, field: str, direction: str) -> Dict[str, float]:
    """Optimal cutpoint for continuous ``field`` over the full weighted data.

    Sweeps every candidate cutpoint and returns the one maximizing the boosting
    rule objective (``config.Value_function`` = sqrt(p)-sqrt(n)+1) with its
    weighted (p, n) coverage — ``optimize.best_threshold`` over the full weighted
    data, exposed as a tool so the LLM need not eyeball a cutpoint from the
    sampled rows.
    """
    _toollog.begin("best_threshold", {"field": field, "direction": direction})
    try:
        idx = _attr_index(field)
        if names_mod.Names[idx].kind != AttrKind.CONTINUOUS:
            raise _tool_arg_error("best_threshold: %r is not a continuous attribute" % field)
        d = direction.strip()
        if d in (">=", ">"):
            op = Operator.OPGE
        elif d in ("<=", "<"):
            op = Operator.OPLE
        else:
            raise _tool_arg_error("direction must be '>=' or '<=', got %r" % direction)
        th, pos, neg = optimize.best_threshold(_Wild(idx, op), cl, dataset, 0.0, 0.0, 0.0)
    except Exception as e:
        _toollog.end(error=e)
        raise
    result = {"threshold": th, "p": pos, "n": neg}
    _toollog.end(result=result)
    return result


def _best_symbol(dataset, cl, field: str, negate: bool = False,
                 top_k: int = 5) -> List[Dict[str, Any]]:
    """Top-``top_k`` values for symbolic/set ``field`` ranked by the boosting
    objective over the full weighted data, best first.

    Each entry is ``{"value", "p", "n"}`` for the condition ``x[field] ==
    value`` (or ``!= value`` when ``negate``); for set/bag fields the condition
    is membership.  Reuses the discrete learner's ``compute_field_stats`` so the
    ranking matches ``optimize.best_symbol``.
    """
    _toollog.begin("best_symbol", {"field": field, "negate": negate})
    try:
        idx = _attr_index(field)
        kind = names_mod.Names[idx].kind
        if kind == AttrKind.SYMBOL:
            op = Operator.OPNEQ if negate else Operator.OPEQ
        elif kind == AttrKind.SET:
            op = Operator.OPOUT if negate else Operator.OPIN
        else:
            raise _tool_arg_error("best_symbol: %r is not a symbolic/set attribute" % field)
        optimize.compute_field_stats(cl, idx, dataset)
        invert = op in (Operator.OPOUT, Operator.OPNEQ)
        ranked = []
        for i in range(optimize.n_visited_symbols()):
            sym = optimize.visited_symbol(i)
            p, n = optimize.neg_field_stat(sym) if invert else optimize.pos_field_stat(sym)
            ranked.append((config.Value_function(None, 0.0, 0.0, 0.0, p, n), sym.name, p, n))
        ranked.sort(key=lambda r: r[0], reverse=True)
        out = [{"value": name, "p": p, "n": n} for _, name, p, n in ranked[:max(1, top_k)]]
    except Exception as e:
        _toollog.end(error=e)
        raise
    _toollog.end(result=out)
    return out


# ---------------------------------------------------------------------------
# tool implementation (candidate_conditions) — precomputed counts + optimization
# ---------------------------------------------------------------------------

def _schema_attrs() -> List[Tuple[str, int, Any]]:
    """(name, inst-index, kind) for each attribute ``inst_to_dict`` exposes as a
    key of ``x`` (skips ignored/suppressed and the class column)."""
    out = []
    for i, adef in enumerate(names_mod.Names):
        if names_mod.llm_hidden(i):
            continue
        if adef.kind in (AttrKind.SYMBOL, AttrKind.CONTINUOUS, AttrKind.SET):
            out.append((adef.name.name, i, adef.kind))
    return out


def _compile_filter(body: str):
    """Predicate ``dict -> bool`` for a filter given as the body of
    ``lambda x: <body>`` (same convention as count_coverage_of_lambda_body).
    Per-example evaluation errors count as "does not pass" (mirrors _evaluate)."""
    if "\n" in body:
        raise _tool_arg_error("filter must be a single-line expression; got %r" % body)
    try:
        rule = _SandboxRule("covers = lambda x: %s\n" % body)
    except Exception as e:
        raise _tool_arg_error("failed to compile filter `lambda x: %s`: %s: %s"
                              % (body, type(e).__name__, e))

    def passes(x, _r=rule):
        try:
            return bool(_r(x))
        except Exception:
            return False
    return passes


def _av_matches(av, op, value) -> bool:
    """True iff attribute value ``av`` satisfies ``<op> value``; missing never
    matches.  Mirrors the coverage semantics of optimize.best_* / _count_threshold."""
    if av.kind == AttrKind.MISSING_VALUE:
        return False
    if op == Operator.OPLE:
        return av.kind == AttrKind.CONTINUOUS and av.num <= value
    if op == Operator.OPGE:
        return av.kind == AttrKind.CONTINUOUS and av.num >= value
    if op in (Operator.OPEQ, Operator.OPNEQ):
        name = av.sym.name if av.sym is not None else None
        return (name == value) if op == Operator.OPEQ else (name != value)
    if op in (Operator.OPIN, Operator.OPOUT):
        inside = value in {s.name for s in (av.set or [])}
        return inside if op == Operator.OPIN else not inside
    return False


def _cover_pn(examples, cl, idx, op, value) -> Tuple[float, float]:
    """Weighted (p, n) coverage of ``x[idx] <op> value`` over ``examples``."""
    p = n = 0.0
    for ex in examples:
        if _av_matches(ex.inst[idx], op, value):
            if ex.lab.nom is cl:
                p += ex.wt
            else:
                n += ex.wt
    return p, n


def _totals(examples, cl) -> Tuple[float, float]:
    """Weighted (p, n) over ``examples`` regardless of any condition."""
    p = n = 0.0
    for ex in examples:
        if ex.lab.nom is cl:
            p += ex.wt
        else:
            n += ex.wt
    return p, n


def _cond_str(field: str, op, value) -> str:
    """Render a condition as a one-line expression in ``x`` (a valid
    count_coverage_of_lambda_body body), kept pipe-safe for a markdown cell."""
    if op == Operator.OPLE:
        s = "x[%r] <= %g" % (field, value)
    elif op == Operator.OPGE:
        s = "x[%r] >= %g" % (field, value)
    elif op == Operator.OPEQ:
        s = "x[%r] == %r" % (field, value)
    elif op == Operator.OPIN:
        s = "%r in (x[%r] or [])" % (value, field)
    else:  # not produced below, but keep general
        s = "%s %s %r" % (field, op, value)
    return s.replace("|", "/")


def _best_condition(idx: int, kind, cl, filt_full):
    """Best single condition on attribute ``idx`` over ``filt_full`` (the
    filter-restricted full data).  Returns (op, value, full_p, full_n, value_score)
    or None when the attribute has no covered data.  Continuous attributes try
    both threshold directions and keep the higher-scoring one."""
    if kind == AttrKind.CONTINUOUS:
        best = None
        for op in (Operator.OPLE, Operator.OPGE):
            th, p, n = optimize.best_threshold(_Wild(idx, op), cl, filt_full,
                                               0.0, 0.0, 0.0)
            if p == 0.0 and n == 0.0:
                continue
            val = config.Value_function(None, 0.0, 0.0, 0.0, p, n)
            if best is None or val > best[4]:
                best = (op, th, p, n, val)
        return best
    op = Operator.OPEQ if kind == AttrKind.SYMBOL else Operator.OPIN
    sym, p, n = optimize.best_symbol(_Wild(idx, op), cl, filt_full, 0.0, 0.0, 0.0)
    if sym is None:
        return None
    return (op, sym.name, p, n, config.Value_function(None, 0.0, 0.0, 0.0, p, n))


def _candidate_conditions_table(dataset, sample, cl, filter_body: str = "") -> str:
    """Markdown table of the best single condition per attribute under a filter.

    ``filter_body`` is a count_coverage_of_lambda_body-style expression in ``x``
    (empty = the vacuous filter that passes every example).  The filter restricts
    BOTH the full training set and the prompt ``sample``; for each attribute the
    best condition (by sqrt(p)-sqrt(n)) is chosen over the filtered full data and
    its coverage reported over both populations.  Counts are *effective example
    counts* (weight x population size), so ``sample p`` is on the same per-row
    scale as the ``weight`` column of the rendered example table.  The leading
    row is the filter itself (all examples that pass it)."""
    body = (filter_body or "").strip()
    if body:
        passes = _compile_filter(body)
        filt_full = [ex for ex in dataset if passes(inst_to_dict(ex.inst))]
        filt_samp = [ex for ex in sample if passes(inst_to_dict(ex.inst))]
    else:
        filt_full = list(dataset)
        filt_samp = list(sample)
    nfull = len(dataset) or 1
    nsamp = len(sample) or 1

    rows = []
    for field, idx, kind in _schema_attrs():
        best = _best_condition(idx, kind, cl, filt_full)
        if best is None:
            continue
        op, value, fp, fn, val = best
        sp, sn = _cover_pn(filt_samp, cl, idx, op, value)
        rows.append((val, _cond_str(field, op, value),
                     sp * nsamp, sn * nsamp, fp * nfull, fn * nfull))
    rows.sort(key=lambda r: r[0], reverse=True)  # best candidate first

    tp, tn = _totals(filt_samp, cl)
    fp0, fn0 = _totals(filt_full, cl)
    head = "%s (%d full rows)" % (body or "(all examples)", len(filt_full))
    out = ["| condition | sample p | sample n | full p | full n |",
           "|---|---|---|---|---|",
           "| %s | %g | %g | %g | %g |" % (head.replace("|", "/"),
                                           tp * nsamp, tn * nsamp,
                                           fp0 * nfull, fn0 * nfull)]
    for _, cond, sp, sn, fp, fn in rows:
        out.append("| %s | %g | %g | %g | %g |" % (cond, sp, sn, fp, fn))
    return "\n".join(out)


def _candidate_conditions(dataset, sample, cl, filter_body: str = "") -> str:
    """Logged wrapper around _candidate_conditions_table (the tool entry point)."""
    _toollog.begin("candidate_conditions", {"filter": filter_body})
    try:
        table = _candidate_conditions_table(dataset, sample, cl, filter_body)
    except Exception as e:
        _toollog.end(error=e)
        raise
    _toollog.end(result={"chars": len(table)})
    return table


def _candidate_conditions_prompt(dataset, sample, cl) -> str:
    """The prompt's candidate-conditions section (--candidate-conditions), built
    for the vacuous filter.  Empty unless the flag is set; the header lives here
    so a disabled run leaves no dangling heading behind the empty slot."""
    if not getattr(config, "Candidate_conditions", False):
        return ""
    table = _candidate_conditions_table(dataset, sample, cl, "")
    return (
        "Candidate conditions — for each attribute, the single best condition "
        "(by sqrt(p)-sqrt(n)) with its accurate weighted coverage, computed for "
        "you over BOTH the sample above and the FULL training set.  Columns are "
        "effective example counts, so `sample p`/`sample n` are on the same "
        "per-row scale as the sample's `weight` column and `full p`/`full n` are "
        "counts over all training examples.  The first row is every example; "
        "trust these counts over tallying the sample yourself:\n\n%s\n" % table)


# ---------------------------------------------------------------------------
# prompt template
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = """{header}

{task_definition}
{grow_note}
{task_info}
{maximization_criterion}
The dict `x` has one key per non-ignored attribute, with these types:
  - symbolic   -> str (one of the listed domain values, or None for missing)
  - continuous -> float (or None for missing)
  - set / bag  -> list[str] (or None for missing)
{text_note}
{schema}
{sample_note}
{examples}
{candidate_conditions}
{tools_hint}
Rules:
  - Return ONLY the Python source defining `covers`.  No prose, no markdown
    fences in the source field; you may put reasoning in the reasoning field.
  - Handle missing values (None) explicitly; assume nothing about defaults.
  - Stick to standard library; do not import third-party packages.
  - Do not raise; return {noncover} on any case you don't want to cover.
"""


_DEFAULT_PROMPT_TABLE = """{header}

{task_definition}
{grow_note}
{task_info}
{maximization_criterion}
The dict `x` has one key per attribute.  Each attribute has a type, used as:
  - symbolic   -> str (or None for missing)
  - continuous -> float (or None for missing)
  - set / bag  -> list[str] (or None for missing)
{text_note}
{schema}
{sample_note}
{examples}
{candidate_conditions}
{tools_hint}
Rules:
  - Return ONLY the Python source defining `covers`.  No prose, no markdown
    fences in the source field; you may put reasoning in the reasoning field.
  - Handle missing values (None) explicitly; assume nothing about defaults.
  - Stick to standard library; do not import third-party packages.
  - Do not raise; return {noncover} on any case you don't want to cover.
"""


_MAXIMIZATION_CRITERION="""
When constructing a rule you should attempt to maximize the following
statistic.  Letting W_{posclass} be the number of {posclass} examples
covered by the rule, and W_{negclass} be similar, you should try to
maximize

    sqrt(W_{posclass}) - sqrt(W_{negclass})
"""


# Injected into the grow agent's prompt (--pruning-agent). A second agent will
# simplify/prune this rule afterward using the full training data, so the grow
# agent should reach for the whole positive cluster rather than fuss over exact
# cutpoints or over-narrow conditions.
_GROW_NOTE = """
NOTE: This is only the FIRST of two steps.  A second agent will afterward
simplify and prune this rule against the full training data -- dropping weak
conditions and resetting thresholds/symbols to their data-optimal values.  So
aim to CAPTURE THE POSITIVE CLUSTER (favor recall); don't agonize over exact
numeric cutpoints or trimming conditions -- the prune step handles that.
"""

#
# Task-definition variants (--prompt-variant).  Each variant pairs a
# {task_definition} block with the return value the templates' Rules tail
# tells the model to use for uncovered examples ({noncover}), so the two
# instructions can't contradict each other.  Register new variants in
# _PROMPT_VARIANTS; cli.py validates the flag against the same names.
#

# --prompt-variant task_variant_1: "meaningful cluster" framing; uncovered
# examples return None instead of False.
_TASK_DEFINITION_VARIANT_1 = """
Task: write a rule in the form of a Python function

    def covers(x: dict) -> bool | None: ...

that "fires" (returns True) for a meaningful cluster of examples of
class **{posclass}**.  Notes:

 - The identified cluster should be as large as possible, and should
   include far more {posclass} examples than {negclass} examples:
   i.e., high precision matters more than high recall.

 - For examples **not** in the identified cluster return None -
   hopefully some other rule in the ensemble will fire on those
   examples.

 - A recommended format for the function you learn is

   - Extract some attribute values needed from the example x

   - Optionally, compute some intermediate expressions in terms of the
     extracted attributes

   - Give a Python comment explaining the cluster, based on your prior
     knowledge of the problem, or based on your analysis of the
     examples

   - List code for one or more rules in the form 'if TEST1 and ... and
     TESTK: return True', where each TEST uses the cluster-relevant
     attributes.

"""


# --prompt-variant task_variant_2: like variant_1 but insists on exactly ONE
# conjunctive rule per round (no disjunctions) -- boosting adds more rules in
# later rounds for the examples this one misses.
_TASK_DEFINITION_VARIANT_2 = """
Task: write a rule in the form of a Python function

    def covers(x: dict) -> bool | None: ...

that "fires" (returns True) for a meaningful cluster of examples of
class **{posclass}**.  Notes:

 - The rule must be a SINGLE conjunction: exactly one test of the form

       if TEST1 and TEST2 and ... and TESTK: return True

   where each TEST constrains one cluster-relevant attribute.  Do NOT
   write more than one `if ... : return True`, and do NOT use `or` /
   disjunctions.  This is only one rule in a boosted ensemble; later
   rounds will add more rules for the {posclass} examples this rule
   does not cover, so you do not need to cover all of them here.

 - Within that single conjunction, the cluster should be as large as
   possible while including far more {posclass} than {negclass}
   examples: high precision matters more than high recall.

 - For examples **not** in the identified cluster return None -
   some other rule in the ensemble will fire on those examples.

 - A recommended format for the function you learn is

   - Extract the attribute values you need from the example x

   - Give a Python comment explaining the cluster, based on your prior
     knowledge of the problem or your analysis of the examples

   - The single `if TEST1 and ... and TESTK: return True` line

   - `return None` otherwise

"""


# --prompt-variant baseline (default): the original prompt's task text.
_TASK_DEFINITION_BASELINE = """
Task: write a Python function

    def covers(x: dict) -> bool: ...

that returns True for examples of class **{posclass}** and False for
examples of class **{negclass}**.  This is one rule in an ensemble; it
does not need to be perfect across the whole dataset.  A good rule is
one that fires on a useful subset of {posclass} examples while rarely
firing on {negclass} examples; high precision matters more than high
recall.
"""



# --- swappable framing -------------------------------------------------------
# The header and the sample note are per-variant because one-pass mode makes the
# defaults false: there is no ensemble being extended and no boosting
# distribution to read.  The DEFAULT strings below must stay byte-identical to
# what the templates hardcoded before they were parameterized -- every condition
# on the development suite was run with that exact wording, and a reworded prompt is
# a different condition.

_HEADER_DEFAULT = "You are building one rule for a SLIPPER-style boosted rule learner."

_HEADER_ONESHOT = ("You are building the COMPLETE rule set for a SLIPPER-style "
                   "rule learner, in a single pass.")

_SAMPLE_NOTE_DEFAULT = """Below is a weighted sample of {n_sampled} training examples drawn from the
current boosting distribution; higher-weight rows are the ones the previous
rules misclassify and that this new rule most needs to handle.
"""

_SAMPLE_NOTE_DEFAULT_TABLE = """Below is a weighted sample of {n_sampled} training examples drawn from the
current boosting distribution, as a table.  The **first row after the header
gives each column's type**.  The first two columns, `class` and `weight`, are
NOT keys of `x`: `class` is the true label and `weight` is the boosting weight
(higher-weight rows are the ones previous rules misclassify and that this new
rule most needs to handle).  Every other column header is exactly a key of `x`;
a `None` cell means that value is missing (`x.get(key)` is None) — guard for it
before any comparison.
"""

_SAMPLE_NOTE_ONESHOT = """Below is a sample of {n_sampled} training examples, drawn uniformly: every row
carries the same weight, because nothing has been learned yet and there are no
previous rules to correct.  Treat every row as equally important.
"""

_SAMPLE_NOTE_ONESHOT_TABLE = """Below is a sample of {n_sampled} training examples, as a table, drawn
uniformly: every row carries the same weight, because nothing has been learned
yet and there are no previous rules to correct.  The **first row after the
header gives each column's type**.  The first two columns, `class` and `weight`,
are NOT keys of `x`: `class` is the true label and `weight` is the (here
uniform) example weight.  Every other column header is exactly a key of `x`; a
`None` cell means that value is missing (`x.get(key)` is None) — guard for it
before any comparison.
"""

_TASK_DEFINITION_ONESHOT = """
Task: write a Python function

    def covers(x: dict) -> bool: ...

that is a COMPLETE rule set for class **{posclass}**, expressed as a
disjunction -- a sequence of independent rules, any one of which firing is
enough:

    def covers(x):
        if <rule 1 conditions>: return True
        if <rule 2 conditions>: return True
        ...
        return False

You get ONE attempt.  No further rules will be added and nothing will be
pruned, so this function alone decides the classification: anything it does not
fire on is classified **{negclass}**.  Balance covering {posclass} against
keeping {negclass} out, since both directions now cost you errors.
{rules_hint}"""


# name -> (task-definition template, {noncover} return value for the Rules tail)
_PROMPT_VARIANTS: Dict[str, Tuple[str, str]] = {
    "baseline": (_TASK_DEFINITION_BASELINE, "False"),
    "task_variant_1": (_TASK_DEFINITION_VARIANT_1, "None"),
    "task_variant_2": (_TASK_DEFINITION_VARIANT_2, "None"),
    "oneshot": (_TASK_DEFINITION_ONESHOT, "False"),
}

# How many rules to write.  Without a suggestion the model decides; with one
# (--oneshot plus -n K) it is told the budget, which is the one-pass analogue of
# the round budget cross-validation hands the iterative path.  Phrased as a
# budget, not a quota, so the model isn't pushed into padding a short rule set.
_RULES_HINT_FREE = """
Use as many rules as the data calls for -- one if a single condition separates
the classes, several if {posclass} has distinct clusters.
"""

_RULES_HINT_K = """
Aim for about {k} rules.  Treat that as a budget rather than a quota: fewer is
fine if fewer suffice, and do not pad the set to reach it.
"""


# Per-variant framing; anything not listed keeps the default wording above.
_HEADERS = {"oneshot": _HEADER_ONESHOT}
_SAMPLE_NOTES = {"oneshot": (_SAMPLE_NOTE_ONESHOT, _SAMPLE_NOTE_ONESHOT_TABLE)}


# Built in pieces so the second tool's description appears only when -b enables
# it (otherwise the LLM is told about a tool it doesn't have).

_TOOLS_HINT_INTRO_VARIANT1 = """
You also have {tooldesc} that query the FULL training set under the current
boosting weights.  Note the sample you have been given is a subset: the full
training set is about {ratio:.2f} times larger.
"""

_TOOLS_HINT_INTRO = """
You also have {tooldesc} that query the FULL training set under the current
boosting weights (the sample above is only a subset — ground your rule on these):
"""

_TOOLS_HINT_COUNT = """
  count_coverage(field: str, op: str, value) -> (p, n)
    Weighted positive/negative coverage of a single condition
    x.get(field) <op> value.  op is one of ==, !=, >=, <=, >, <, contains
    (value in set/bag), like (value substring in string).  Cheap.
"""

_TOOLS_HINT_LAMBDA_VARIANT1 = """
  count_coverage_of_lambda_body(body: str) -> (p, n) 
    Arbitrary one-line expression in x, wrapped as `lambda x: <body>`.
    Sandbox-executed once per training example, so it is slightly more
    expensive. Use for compound conditions (conjunctions, disjunctions,
    computed comparisons) the other tool can't express.
"""


_TOOLS_HINT_LAMBDA = """
  count_coverage_of_lambda_body(body: str) -> (p, n)
    Arbitrary one-line expression in x, wrapped as `lambda x: <body>` and
    sandbox-executed once per training example.  Use for compound conditions
    (conjunctions, disjunctions, computed comparisons) the other tool can't
    express.  Slower (sandboxed eval per example) — prefer count_coverage
    when a single (field, op, value) triple is enough.
"""

_TOOLS_HINT_VALUE_VARIANT1 = """
  best_threshold(field: str, direction: str) -> {threshold, p, n}
    Optimal numeric cutpoint for a continuous `field`, computed over the full
    data (not the sample).  direction is ">=" or "<="; returns the cutpoint
    maximizing sqrt(p)-sqrt(n) and its weighted coverage.  

    Use this **instead of guessing a threshold** from the sampled rows,
    especially when the sample is small compared to the full training
    set.

  best_symbol(field: str, negate: bool = False) -> [{value, p, n}, ...]
    Up to 5 best values for a symbolic/set `field` by the same objective, best
    first, for x[field] == value (negate=True scores x[field] != value; for
    set/bag fields the condition is membership).

    Use this **instead of guessing a value** from the sampled rows,
    especially when the sample is small compared to the full training
    set and/or there are many possible values for a field.
"""


_TOOLS_HINT_VALUE = """
  best_threshold(field: str, direction: str) -> {threshold, p, n}
    Optimal numeric cutpoint for a continuous `field`, computed over the full
    data (not the sample).  direction is ">=" or "<="; returns the cutpoint
    maximizing sqrt(p)-sqrt(n) and its weighted coverage.  Use this instead of
    guessing a threshold from the sampled rows.
  best_symbol(field: str, negate: bool = False) -> [{value, p, n}, ...]
    Up to 5 best values for a symbolic/set `field` by the same objective, best
    first, for x[field] == value (negate=True scores x[field] != value; for
    set/bag fields the condition is membership).
"""

_TOOLS_HINT_CANDIDATE = """
  candidate_conditions(filter: str = "") -> markdown table
    For every attribute, the single best condition (best sqrt(p)-sqrt(n)
    cutpoint/value), with accurate weighted coverage over both the sample and
    the FULL training set, restricted to examples passing `filter` (a
    count_coverage_of_lambda_body-style body in x; "" = all examples).  Use it
    to refine a partial rule: pass the conjunction you have so far as `filter`
    and read off the best condition to add next.
"""

_TOOLS_HINT_RETRIEVAL = """
  retrieve_examples(where: str = "", label: str = "either", k: int = 10)
    Retrieve unseen labeled training rows matching a Python expression in x
    (the same expression convention as the coverage tool). Empty where matches
    all rows. label is "positive", "negative", or "either", relative to the
    class this rule should cover. For example, retrieve negative rows covered
    by your candidate to find a useful refinement, or positives outside it.
"""

_TOOLS_HINT_RANDOM_ROWS = """
  sample_examples(label: str = "either", k: int = 10)
    Request additional unseen labeled training rows, sampled without a feature
    filter. label is "positive", "negative", or "either", relative to the
    class this rule should cover. Inspect these examples to refine a candidate.
"""

_TOOLS_HINT_ROWS_LIMITS = """
    Rows are drawn without replacement according to current boosting weights
    within the requested class/region. At most {per_call} rows per call and
    {max_rows} additional rows / {max_chars} JSON row characters in this whole
    proposal, including retries. Already displayed rows are excluded. Notes
    are returned whole. Each call shares the tool budget below with coverage.
    row_id, class, and boosting_weight are metadata, NOT features in x.
    boosting_weight is the original full-training weight, unlike the rescaled
    weights in the initial table. These selected examples do not estimate
    population coverage; use the coverage tool for that.
"""

# Appended when config.Require_tool_output.  pydantic-ai collects the rule via
# a function tool (`final_result`); a model that answers in prose instead never
# satisfies it, however correct the rule is.
_TOOL_OUTPUT_HINT = """
IMPORTANT: do not answer in prose, and do not put the rule in a markdown code
block.  Return your answer by calling the `final_result` tool, passing the
function source in its `source` field and your explanation in `reasoning`.
"""


_TOOLS_HINT_TAIL = """
Missing values (None) never match.

Budget: **at most {tool_budget} tool calls per rule** (hard cap; going over
ends your turn with no rule).  Verify the most promising candidates from the
sample above, then commit.  Reserve at least one call for the final commit.
"""


def _tools_hint(tool_budget: int, use_count_coverage: bool,
                use_lambda_body: bool, use_value_tools: bool,
                use_candidate: bool = False,
                captured: Optional[float] = None,
                retrieval_mode: str = "off") -> str:
    n = ((1 if use_count_coverage else 0) + (1 if use_lambda_body else 0)
         + (2 if use_value_tools else 0) + (1 if use_candidate else 0)
         + (1 if retrieval_mode != "off" else 0))
    tooldesc = "one tool," if n == 1 else "%d tools" % n
    variant1 = getattr(config, "Tools_prompt", "baseline") == "variant1"
    if variant1:
        # captured is this round's captured weight fraction; 1/captured is how
        # many times larger the full training set is than the rendered sample.
        ratio = (1.0 / captured) if captured else 1.0
        hint = _TOOLS_HINT_INTRO_VARIANT1.format(tooldesc=tooldesc, ratio=ratio)
    else:
        hint = _TOOLS_HINT_INTRO.format(tooldesc=tooldesc)
    if use_count_coverage:
        hint += _TOOLS_HINT_COUNT
    if use_lambda_body:
        hint += _TOOLS_HINT_LAMBDA_VARIANT1 if variant1 else _TOOLS_HINT_LAMBDA
    if use_value_tools:
        hint += _TOOLS_HINT_VALUE_VARIANT1 if variant1 else _TOOLS_HINT_VALUE
    if use_candidate:
        hint += _TOOLS_HINT_CANDIDATE
    if retrieval_mode != "off":
        hint += (_TOOLS_HINT_RETRIEVAL if retrieval_mode == "targeted"
                 else _TOOLS_HINT_RANDOM_ROWS)
        hint += _TOOLS_HINT_ROWS_LIMITS.format(
            per_call=config.Retrieval_per_call, max_rows=config.Retrieval_max_rows,
            max_chars=config.Retrieval_max_chars)
        if config.Retrieval_required:
            tool = "retrieve_examples" if retrieval_mode == "targeted" else "sample_examples"
            hint += ("\nBefore finalizing this rule, you MUST call %s at least once and inspect\n"
                     "the returned examples. Do this early enough to refine and check your\n"
                     "candidate. This query uses one of the same %d tool calls, leaving at\n"
                     "most %d for other queries. An empty result still satisfies the requirement.\n"
                     % (tool, tool_budget, max(0, tool_budget-1)))
    # The claimed budget is a prompt-only override: the enforced cap is still
    # tool_budget, via _tool_limits at the call site.
    claimed = getattr(config, "Claimed_tool_budget", None)
    return hint + _TOOLS_HINT_TAIL.format(
        tool_budget=tool_budget if claimed is None else claimed)


def _rules_hint(posname: str) -> str:
    """The '{rules_hint}' block of the one-pass task definition (unused by the
    per-round variants, which are handed it and ignore it)."""
    k = getattr(config, "Oneshot_rules", 0)
    if k and k > 0:
        return _RULES_HINT_K.format(k=k)
    return _RULES_HINT_FREE.format(posclass=posname)


def _load_prompt() -> str:
    if config.Prompt_file:
        with open(config.Prompt_file) as fp:
            return fp.read()
    if getattr(config, "Example_format", "dict") == "table":
        return _DEFAULT_PROMPT_TABLE
    return _DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Pydantic AI agent (via the pydantic-ai-litellm bridge)
# ---------------------------------------------------------------------------

class PyRuleSpec(BaseModel):
    reasoning: str
    source: str


@dataclass
class _ToolDeps:
    """Per-call deps passed to agent.run_sync when tools are enabled.

    The tool reads ``dataset`` (the full training set, with .wt at the current
    boosting distribution) and ``cl`` (the positive class symbol) — i.e. the
    same population the boosting algorithm is reweighting, not the sample
    rendered into the prompt.  ``sample`` is the rendered prompt sample, used
    only by candidate_conditions for its sample-side coverage columns.
    """
    dataset: List[Any]
    cl: Any
    sample: List[Any] = None
    retrieval: Optional[RetrievalSession] = None
    retrieval_calls_at_start: int = 0


_agents: Dict[Tuple, Any] = {}


def _retrieve_rows(deps, where, label, k, tool):
    args = dict(label=label, k=k)
    if tool == "retrieve_examples":
        args["where"] = where
    _toollog.begin(tool, args)
    try:
        if deps.retrieval is None:
            raise ValueError("retrieval is disabled for this proposal")
        result = deps.retrieval.fetch(where=where, label=label, k=k,
                                      compile_filter=_compile_filter,
                                      to_dict=inst_to_dict)
    except Exception as e:
        _toollog.end(error=e)
        if isinstance(e, ValueError):
            raise _tool_arg_error(str(e)) from e
        raise
    _toollog.end(result=result)
    return result


def _llm_settings(timeout: Optional[float] = _PER_REQUEST_TIMEOUT_S):
    """ModelSettings for a LiteLLMModel: a per-request ``timeout`` in seconds
    (falsy = none; the tool paths use _PER_REQUEST_TIMEOUT_S, the tool-less
    path the longer _NO_TOOLS_REQUEST_TIMEOUT_S), plus --reasoning-effort /
    --llm-extra-body as extra_body and --llm-max-tokens, all of which
    pydantic-ai-litellm forwards to litellm.  None when nothing applies."""
    kwargs: Dict[str, Any] = {}
    if timeout:
        kwargs["timeout"] = timeout
    extra: Dict[str, Any] = {}
    if config.Reasoning_effort:
        extra["reasoning_effort"] = config.Reasoning_effort
    if config.Llm_extra_body:
        try:
            extra.update(json.loads(config.Llm_extra_body))
        except ValueError as e:
            fatal("--llm-extra-body is not valid JSON: %s", e)
    if extra:
        kwargs["extra_body"] = extra
    if config.Llm_max_tokens:
        kwargs["max_tokens"] = config.Llm_max_tokens
    return ModelSettings(**kwargs) if kwargs else None


def _get_agent(model: str, use_tools: bool, use_count_coverage: bool = True,
               use_lambda_body: bool = False, use_value_tools: bool = False,
               use_candidate: bool = False, retrieval_mode: str = "off",
               require_retrieval: bool = False):
    _install_request_meter()
    key = (model, use_tools, use_count_coverage, use_lambda_body, use_value_tools,
           use_candidate, retrieval_mode, require_retrieval)
    if key in _agents:
        return _agents[key]
    try:
        from pydantic_ai import Agent
        from pydantic_ai_litellm import LiteLLMModel  # type: ignore
    except ImportError as e:
        fatal("LLM rule builder requires pydantic-ai + pydantic-ai-litellm: %s", e)
    system = ("You are a concise, careful Python programmer "
              "writing a single classification rule.")
    if use_tools:
        agent = Agent(
            LiteLLMModel(model, settings=_llm_settings()),
            output_type=PyRuleSpec,
            deps_type=_ToolDeps,
            system_prompt=system,
        )

        if use_count_coverage:
            @agent.tool
            def count_coverage(ctx: RunContext[_ToolDeps],
                               field: str, op: str, value: Any) -> Tuple[float, float]:
                """Weighted count of training examples that satisfy a single condition.

                Returns (p, n): summed Example.wt for positive-class and
                negative-class examples respectively where the condition holds.
                Weights are the live boosting distribution (full training set,
                not the sampled prompt rows).

                Args:
                    field: attribute name (must appear in the schema).
                    op: one of "==", "!=", ">=", "<=", ">", "<", "contains", "like".
                        - ==, !=, >=, <=, >, <: scalar comparison against x.get(field).
                        - contains: value in x.get(field) (for set/bag attributes).
                        - like: substring test, value in x.get(field) (string text).
                    value: comparison value; type must be compatible with op
                        (e.g. a number for >=/<=, a string for contains/like).

                Examples covered are counted under the *current* boosting weights;
                missing values (x.get(field) is None) never satisfy any operator.
                """
                return _count_coverage(ctx.deps.dataset, ctx.deps.cl,
                                       field, op, value)

        if use_lambda_body:
            @agent.tool
            def count_coverage_of_lambda_body(
                ctx: RunContext[_ToolDeps], body: str
            ) -> Tuple[float, float]:
                """Weighted count of training examples matched by an arbitrary expression.

                The body is wrapped as ``lambda x: <body>`` and executed in the
                same sandbox as the final rule, so anything legal there is legal
                here.  Use this to test compound conditions (e.g. conjunctions,
                disjunctions, computed comparisons) that count_coverage's single
                (field, op, value) form can't express.

                Args:
                    body: a single-line Python expression in `x` (the per-example
                        dict).  Examples:
                          "x.get('physician_fee_freeze') == 'y'"
                          "x.get('a') == '1' and x.get('b') == '1'"
                          "x.get('age') is not None and x.get('age') >= 30"
                        Per-example errors during evaluation count as "no cover";
                        a syntax/compile error raises ValueError so you can retry
                        with corrected source.

                Returns (p, n): summed positive- and negative-class weights under
                the live boosting distribution on the FULL training set.
                """
                return _count_coverage_of_lambda_body(
                    ctx.deps.dataset, ctx.deps.cl, body)

        if use_value_tools:
            @agent.tool
            def best_threshold(ctx: RunContext[_ToolDeps],
                               field: str, direction: str) -> Dict[str, float]:
                """Optimal numeric cutpoint for a continuous attribute, over the
                FULL training set under the current boosting weights.

                Sweeps every candidate cutpoint and returns the one maximizing
                the rule objective sqrt(p)-sqrt(n), so you don't have to eyeball
                a threshold from the sampled rows.

                Args:
                    field: a continuous attribute name (from the schema).
                    direction: ">=" (rule fires when x[field] >= threshold) or
                        "<=" (fires when x[field] <= threshold).

                Returns {"threshold": t, "p": pos_wt, "n": neg_wt}.  Write the
                condition as: x.get(field) is not None and x[field] >= t
                (or <= t).  Raises ValueError if field is not continuous.
                """
                return _best_threshold(ctx.deps.dataset, ctx.deps.cl,
                                       field, direction)

            @agent.tool
            def best_symbol(ctx: RunContext[_ToolDeps],
                            field: str, negate: bool = False
                            ) -> List[Dict[str, Any]]:
                """Best values for a symbolic/set attribute, ranked by the rule
                objective over the FULL training set under current boosting
                weights — instead of guessing which category separates classes.

                Args:
                    field: a symbolic or set/bag attribute name.
                    negate: if False (default), score x[field] == value; if
                        True, score x[field] != value.

                Returns up to 5 {"value": v, "p": pos_wt, "n": neg_wt} entries,
                best first.  For a symbolic field write x.get(field) == v; for a
                set/bag field write v in (x.get(field) or []).  Raises ValueError
                if field is not symbolic/set.
                """
                return _best_symbol(ctx.deps.dataset, ctx.deps.cl, field, negate)

        if use_candidate:
            @agent.tool
            def candidate_conditions(ctx: RunContext[_ToolDeps],
                                     filter: str = "") -> str:
                """Best single condition per attribute, with accurate weighted
                coverage over the sample and the FULL training set, for the
                examples passing ``filter`` — precomputed so you needn't count.

                Args:
                    filter: a count_coverage_of_lambda_body-style one-line
                        expression in ``x`` restricting which examples are
                        considered ("" = all examples).  Pass the conjunction of
                        conditions committed so far to see the best next
                        condition to add.

                Returns a markdown table, best candidate first; the leading row
                is the filter itself (its full coverage).  Counts are effective
                example counts (weight x population size).
                """
                return _candidate_conditions(ctx.deps.dataset, ctx.deps.sample,
                                             ctx.deps.cl, filter)
        if retrieval_mode == "targeted":
            @agent.tool(sequential=True)
            def retrieve_examples(ctx: RunContext[_ToolDeps], where: str = "",
                                  label: Literal["positive", "negative", "either"] = "either",
                                  k: int = 10) -> Dict[str, Any]:
                """Retrieve unseen labeled training rows matching an expression in x.

                where is a single-line Python Boolean expression, like the
                coverage tool's body; empty means all rows. label selects the
                rule's positive class, its negative class, or either. Rows are
                drawn using current boosting weights. k is limited by per-call,
                total-row, and character budgets. Returns whole feature dicts,
                class labels, original boosting weights and metadata row IDs.
                Use counterexamples to refine a candidate. This is a selected
                sample, not a coverage estimate. No test rows are accessible.
                """
                return _retrieve_rows(ctx.deps, where, label, k, "retrieve_examples")
        elif retrieval_mode == "random":
            @agent.tool(sequential=True)
            def sample_examples(ctx: RunContext[_ToolDeps],
                                label: Literal["positive", "negative", "either"] = "either",
                                k: int = 10) -> Dict[str, Any]:
                """Sample additional unseen labeled training rows without a feature filter.

                label selects the rule's positive class, its negative class, or
                either. Sample by current boosting weight without replacement.
                k is limited by per-call, total-row, and character budgets.
                Returns whole feature dicts, labels, original boosting weights
                and metadata row IDs. No test rows are accessible.
                """
                return _retrieve_rows(ctx.deps, "", label, k, "sample_examples")
        if retrieval_mode != "off" and require_retrieval:
            @agent.output_validator
            def check_retrieval(ctx: RunContext[_ToolDeps], output: PyRuleSpec) -> PyRuleSpec:
                if ctx.deps.retrieval.calls_made <= ctx.deps.retrieval_calls_at_start:
                    name = "retrieve_examples" if retrieval_mode == "targeted" else "sample_examples"
                    raise _tool_arg_error("Call %s at least once before finalizing the rule; "
                                          "it shares the existing tool allowance." % name)
                return output
    else:
        agent = Agent(
            LiteLLMModel(model, settings=_llm_settings(timeout=_NO_TOOLS_REQUEST_TIMEOUT_S)),
            output_type=PyRuleSpec,
            system_prompt=system,
        )
    _agents[key] = agent
    return agent


# ---------------------------------------------------------------------------
# prune agent (--pruning-agent): simplify a grown rule using tools only
# ---------------------------------------------------------------------------

@dataclass
class _PruneDeps:
    """Deps for the prune agent's tools.  Coverage tools query ``cover_data``
    (the grow sample the rule was built from); the value pickers query
    ``value_data`` (the full training set).  ``cl`` is the positive class."""
    cover_data: List[Any]
    value_data: List[Any]
    cl: Any


_PRUNE_PROMPT = """You are SIMPLIFYING one rule for a SLIPPER-style boosted rule learner.

Another agent grew this candidate rule to fire on class **{posclass}** (vs
**{negclass}**), from a small sample of examples:

{grown_source}

Return a SIMPLER `covers(x: dict)` that keeps essentially the same positive
cluster but generalizes better.  You MAY:
  - remove conditions that don't earn their keep,
  - loosen or tighten a numeric threshold,
  - change which symbolic value(s) a condition tests.
You may NOT introduce a condition on any attribute the grown rule does not
already use.  Fewer, higher-precision conditions are better.

{schema}
You are shown NO examples.  Investigate ONLY through the tools below, then
commit your simplified rule.
{tools_hint}
Rules:
  - Return ONLY the Python source defining `covers`.  No prose, no markdown
    fences in the source field; put reasoning in the reasoning field.
  - Handle missing values (None) explicitly; assume nothing about defaults.
  - Stick to standard library; do not import third-party packages.
  - Do not raise; return {noncover} on any case you don't want to cover.
"""

_PRUNE_TOOLS_INTRO = """
You have {tooldesc} to investigate the rule (you see no examples directly):
"""

_PRUNE_TOOLS_COUNT = """
  count_coverage(field: str, op: str, value) -> (p, n)
    Weighted positive/negative coverage of a single condition
    x.get(field) <op> value over the GROW sample the rule was built from.
    op is one of ==, !=, >=, <=, >, <, contains, like.
"""

_PRUNE_TOOLS_LAMBDA = """
  count_coverage_of_lambda_body(body: str) -> (p, n)
    (p, n) for `lambda x: <body>` over the GROW sample -- use it to score the
    whole rule or a compound condition before committing.
"""

_PRUNE_TOOLS_VALUE = """
  best_threshold(field: str, direction: str) -> {threshold, p, n}
    Optimal cutpoint for a continuous field over the FULL training set.
  best_symbol(field: str, negate: bool = False) -> [{value, p, n}, ...]
    Best values for a symbolic/set field over the FULL training set.
"""

_PRUNE_TOOLS_TAIL = """
Missing values (None) never match.  The coverage tools score the GROW sample
the rule was grown on; the best_* tools score the FULL training set (use them
to reset thresholds/symbols to their data-optimal values).

Budget: **at most {tool_budget} tool calls** (hard cap; going over ends your
turn with no rule).  Reserve at least one call for the final commit.
"""


def _prune_tools_hint(tool_budget: int, use_count_coverage: bool,
                      use_lambda_body: bool, use_value_tools: bool) -> str:
    n = ((1 if use_count_coverage else 0) + (1 if use_lambda_body else 0)
         + (2 if use_value_tools else 0))
    tooldesc = "one tool," if n == 1 else "%d tools" % n
    hint = _PRUNE_TOOLS_INTRO.format(tooldesc=tooldesc)
    if use_count_coverage:
        hint += _PRUNE_TOOLS_COUNT
    if use_lambda_body:
        hint += _PRUNE_TOOLS_LAMBDA
    if use_value_tools:
        hint += _PRUNE_TOOLS_VALUE
    return hint + _PRUNE_TOOLS_TAIL.format(tool_budget=tool_budget)


_prune_agents: Dict[Tuple[str, bool, bool, bool], Any] = {}


def _get_prune_agent(model: str, use_count_coverage: bool,
                     use_lambda_body: bool, use_value_tools: bool):
    """Agent for the prune stage.  Same tool set as the build agent, but the
    coverage tools read ``cover_data`` (grow sample) and the value pickers read
    ``value_data`` (full train) off ``_PruneDeps``."""
    _install_request_meter()
    key = (model, use_count_coverage, use_lambda_body, use_value_tools)
    if key in _prune_agents:
        return _prune_agents[key]
    try:
        from pydantic_ai import Agent
        from pydantic_ai_litellm import LiteLLMModel  # type: ignore
    except ImportError as e:
        fatal("LLM rule builder requires pydantic-ai + pydantic-ai-litellm: %s", e)
    system = ("You are a concise, careful Python programmer "
              "simplifying a single classification rule.")
    agent = Agent(
        LiteLLMModel(model, settings=_llm_settings()),
        output_type=PyRuleSpec,
        deps_type=_PruneDeps,
        system_prompt=system,
    )

    if use_count_coverage:
        @agent.tool
        def count_coverage(ctx: RunContext[_PruneDeps],
                           field: str, op: str, value: Any) -> Tuple[float, float]:
            """Weighted (p, n) coverage of a single condition over the GROW
            sample the rule was built from.

            Args:
                field: attribute name (must appear in the schema).
                op: one of "==", "!=", ">=", "<=", ">", "<", "contains", "like".
                value: comparison value compatible with op.

            Missing values (x.get(field) is None) never satisfy any operator.
            """
            return _count_coverage(ctx.deps.cover_data, ctx.deps.cl,
                                   field, op, value)

    if use_lambda_body:
        @agent.tool
        def count_coverage_of_lambda_body(
            ctx: RunContext[_PruneDeps], body: str
        ) -> Tuple[float, float]:
            """Weighted (p, n) for ``lambda x: <body>`` over the GROW sample.

            Args:
                body: a single-line Python expression in `x`, e.g.
                    "x.get('age') is not None and x.get('age') >= 30".
            Use it to score the whole rule or a compound condition.
            """
            return _count_coverage_of_lambda_body(
                ctx.deps.cover_data, ctx.deps.cl, body)

    if use_value_tools:
        @agent.tool
        def best_threshold(ctx: RunContext[_PruneDeps],
                           field: str, direction: str) -> Dict[str, float]:
            """Optimal numeric cutpoint for a continuous attribute over the
            FULL training set (not the grow sample).

            Args:
                field: a continuous attribute name (from the schema).
                direction: ">=" or "<=".
            Returns {"threshold": t, "p": pos_wt, "n": neg_wt}; write the
            condition as x.get(field) is not None and x[field] >= t (or <= t).
            """
            return _best_threshold(ctx.deps.value_data, ctx.deps.cl,
                                   field, direction)

        @agent.tool
        def best_symbol(ctx: RunContext[_PruneDeps],
                        field: str, negate: bool = False
                        ) -> List[Dict[str, Any]]:
            """Best values for a symbolic/set attribute, ranked over the FULL
            training set (not the grow sample).

            Args:
                field: a symbolic or set/bag attribute name.
                negate: if True score x[field] != value instead of ==.
            Returns up to 5 {"value", "p", "n"} entries, best first.
            """
            return _best_symbol(ctx.deps.value_data, ctx.deps.cl, field, negate)

    _prune_agents[key] = agent
    return agent


def prune_pyrule(cl, cover_data, value_data, grown_source, round: int = 0
                 ) -> Tuple[str, Callable[[Dict[str, Any]], bool], float, float]:
    """Second stage of --pruning-agent: hand the grown rule to a tool-using
    agent that SIMPLIFIES it (drop/loosen conditions, reset thresholds/symbols;
    never add a condition on a new attribute).  Returns (source, fn, p, n) with
    (p, n) evaluated on ``value_data`` (the full training set), so the tuple
    drops into ``PyRule`` exactly like build_pyrule's.

    Coverage tools score ``cover_data`` (the grow sample); value pickers score
    ``value_data``.  Fatal if no tools are enabled.  On repeated failure, falls
    back to the grown rule (evaluated on ``value_data``) rather than a no-op.
    """
    if not config.Rule_llm_model:
        fatal("prune_pyrule called without -R set")
    tool_budget = max(0, int(config.Tool_budget))
    use_tools = tool_budget > 0
    use_count_coverage = use_tools and bool(config.Count_coverage_tool)
    use_lambda_body = use_tools and bool(config.Lambda_body_tool)
    use_value_tools = use_tools and bool(config.Value_tools)
    if not (use_count_coverage or use_lambda_body or use_value_tools):
        fatal("--pruning-agent needs at least one tool for the prune agent; "
              "enable tools with -T N (add --value-tools / -b as desired)")
    agent = _get_prune_agent(config.Rule_llm_model, use_count_coverage,
                             use_lambda_body, use_value_tools)
    posname = cl.name
    Classes = names_mod.Classes
    negname = next((c.nom.name for c in Classes if c.nom is not cl), "?")
    variant = getattr(config, "Prompt_variant", "baseline")
    _, noncover = _PROMPT_VARIANTS.get(variant, (None, "False"))
    prompt = _PRUNE_PROMPT.format(
        posclass=posname,
        negclass=negname,
        grown_source=grown_source.strip(),
        schema=_schema_block(),
        tools_hint=_prune_tools_hint(tool_budget, use_count_coverage,
                                     use_lambda_body, use_value_tools),
        noncover=noncover,
    )
    run_kwargs = {"deps": _PruneDeps(cover_data=cover_data, value_data=value_data, cl=cl),
                  "usage_limits": _tool_limits(tool_budget)}
    _toollog.start(round, "prune")
    last_err: Optional[str] = None
    try:
        for attempt in range(3):
            try:
                _t0 = time.monotonic()
                result = agent.run_sync(prompt, **run_kwargs)
                _record_call(result, time.monotonic() - _t0)
                spec: PyRuleSpec = result.output if hasattr(result, "output") else result.data
                if trace.tracing(trace.SUMM):
                    print("// === prune agent output (attempt %d) ===" % (attempt + 1))
                    try:
                        usage = result.usage()
                        print("// usage: %d requests / %d tool calls"
                              % (usage.requests, usage.tool_calls))
                    except Exception:
                        pass
                    print("// reasoning:")
                    for line in (spec.reasoning or "").splitlines() or [""]:
                        print("//   " + line)
                    print("// source (raw):")
                    for line in (spec.source or "").splitlines() or [""]:
                        print("//   " + line)
                source = _strip_fences(spec.source)
                if "def covers" not in source:
                    last_err = "prune output did not define covers()"
                    if trace.tracing(trace.SUMM):
                        print("// REJECTED: %s" % last_err)
                    record.note_failed_attempt()
                    continue
                fn = _SandboxRule(source)
                fn(inst_to_dict(value_data[0].inst))   # self-check: must not raise
                p, n = _evaluate(fn, cl, value_data)
                if trace.tracing(trace.SUMM):
                    print("// pruned rule (attempt %d): p=%.4g n=%.4g" % (attempt + 1, p, n))
                return source, fn, p, n
            except Exception as e:
                reason = _account_error_reason(e)
                if reason:
                    raise LLMAccountError(reason) from e
                last_err = "%s: %s" % (type(e).__name__, e)
                record.note_failed_attempt()
                warning("prune attempt %d failed: %s", attempt + 1, last_err)

        warning("all prune attempts failed (%s); keeping the grown rule", last_err)
        try:
            fn = _SandboxRule(grown_source)
            p, n = _evaluate(fn, cl, value_data)
            return grown_source, fn, p, n
        except Exception:
            noop = "def covers(x):\n    return False\n"
            return noop, (lambda x: False), 0.0, 0.0
    finally:
        _toollog.flush(config.Log_stem)


# ---------------------------------------------------------------------------
# sandboxed predicate
# ---------------------------------------------------------------------------

class _SandboxRule:
    """A callable wrapping a smolagents-sandbox-executed ``covers`` function.

    LocalPythonExecutor maintains state across .send_variables/.execute calls,
    so the source is exec'd once at construction time to bind ``covers`` in
    the executor's namespace.  Subsequent __call__ invocations push one
    instance into the sandbox and evaluate ``covers(__instance)``.

    ``entry``/``coerce`` are the extraction hooks (see extract.ExtractRule):
    an extraction rule binds ``extract`` instead and keeps the returned value
    as it is -- ``coerce=bool`` here is the only place the sandbox insists on
    a predicate.
    """

    def __init__(self, source: str, entry: str = "covers", coerce=bool):
        self.source = source
        self.entry = entry
        self.coerce = coerce
        try:
            from smolagents.local_python_executor import LocalPythonExecutor
        except ImportError as e:
            fatal("LLM rule builder requires smolagents: %s", e)
        # No additional imports needed: smolagents' BASE_BUILTIN_MODULES already
        # authorizes `re` (plus math/itertools/...), so regex rules under
        # --use-text work unchanged -- and the capability envelope is identical
        # to every earlier set-of-words run (re was always importable; those
        # rules simply had no string to apply it to).
        self._executor = LocalPythonExecutor(additional_authorized_imports=[])
        # Bind tools (none) then exec the function definition.
        if hasattr(self._executor, "send_tools"):
            self._executor.send_tools({})
        self._executor(source)

    def __call__(self, instance: Dict[str, Any]) -> Any:
        self._executor.send_variables({"__instance": instance})
        result = self._executor("%s(__instance)" % self.entry)
        # smolagents 1.x returns a CodeOutput dataclass with .output; earlier
        # versions returned a (value, logs, is_final) tuple.  Cover both.
        if hasattr(result, "output"):
            out = result.output
        elif isinstance(result, tuple) and result:
            out = result[0]
        else:
            out = result
        return self.coerce(out) if self.coerce else out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _evaluate(fn: Callable[[Dict[str, Any]], bool], cl, dataset) -> Tuple[float, float]:
    """Return (p, n): weighted #positives correctly covered, #negatives
    incorrectly covered, with weights read from Example.wt."""
    p = n = 0.0
    for ex in dataset:
        try:
            if fn(inst_to_dict(ex.inst)):
                if ex.lab.nom is cl:
                    p += ex.wt
                else:
                    n += ex.wt
        except Exception:
            continue
    return p, n


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def build_pyrule(cl, dataset, sample, round: int = 0,
                 captured: Optional[float] = None,
                 grow_mode: bool = False) -> Tuple[str, Callable[[Dict[str, Any]], bool], float, float]:
    """Ask the LLM for a covers() snippet, sandbox-compile it, evaluate
    (p, n) on the full weighted ``dataset``.  Returns (source, fn, p, n).

    ``sample`` is the weighted prompt sample (drawn by ``boosting.soft_fit``
    via ``data.weighted_sample``); pass the full dataset to skip sampling.
    ``round`` is the current boosting round (0-indexed); used only to name
    the per-round prompt log when ``config.Log_stem`` is set.
    ``captured`` is this round's captured weight fraction; when
    ``config.Count_coverage_max_capmean`` is set, count_coverage is offered only
    on rounds whose captured fraction is at or below that threshold (a per-round
    gate).
    ``grow_mode`` (the --pruning-agent grow stage) forces tools OFF regardless of
    -T and injects the grow-note advice; a separate prune agent simplifies the
    result afterward.

    Retries up to 2 times on a sandbox/runtime failure; if a third attempt
    still fails, falls back to an always-False covers (which gives (0, 0)
    and lets soft_fit pick the default-rule update instead).
    """
    if not config.Rule_llm_model:
        fatal("build_pyrule called without -R set")
    tool_budget = 0 if grow_mode else max(0, int(config.Tool_budget))
    use_tools = tool_budget > 0
    use_count_coverage = use_tools and bool(config.Count_coverage_tool)
    # Per-round capMean gate: drop count_coverage on rounds whose sample already
    # captures more than the threshold of the boosting mass.
    max_cap = config.Count_coverage_max_capmean
    if use_count_coverage and max_cap is not None and captured is not None:
        use_count_coverage = captured <= max_cap
        if trace.tracing(trace.SUMM):
            print("// count_coverage gate: captured=%.4f %s %.4f -> %s"
                  % (captured, "<=" if use_count_coverage else ">", max_cap,
                     "on" if use_count_coverage else "OFF"))
    use_lambda_body = use_tools and bool(config.Lambda_body_tool)
    use_value_tools = use_tools and bool(config.Value_tools)
    use_candidate = use_tools and bool(config.Candidate_conditions_tool)
    retrieval_mode = config.Retrieval_mode if use_tools else "off"
    if retrieval_mode not in ("off", "targeted", "random"):
        fatal("unknown retrieval mode %r", retrieval_mode)
    if use_tools and not (use_count_coverage or use_lambda_body or use_value_tools
                          or use_candidate or retrieval_mode != "off"):
        warning("tool budget -T%d set but every tool is disabled", tool_budget)
    agent = _get_agent(config.Rule_llm_model, use_tools, use_count_coverage,
                       use_lambda_body, use_value_tools, use_candidate, retrieval_mode,
                       config.Retrieval_required)
    posname = cl.name
    Classes = names_mod.Classes
    negname = next((c.nom.name for c in Classes if c.nom is not cl), "?")
    if getattr(config, "Example_format", "dict") == "table":
        examples_block = _render_examples_table(sample)
    else:
        examples_block = _render_examples(sample)
    schema_block = _schema_block()
    if config.Log_stem:
        log_path = "%s-%03d.txt" % (config.Log_stem, round)
        try:
            with open(log_path, "w", encoding="utf-8") as fp:
                fp.write(examples_block + "\n")
        except OSError as e:
            warning("could not write LLM-prompt log %s: %s", log_path, e)
    tools_hint = (_tools_hint(tool_budget, use_count_coverage,
                              use_lambda_body, use_value_tools, use_candidate,
                              captured=captured, retrieval_mode=retrieval_mode)
                  if use_tools else "")
    candidate_block = _candidate_conditions_prompt(dataset, sample, cl)
    maximization = (_MAXIMIZATION_CRITERION.format(posclass=posname, negclass=negname)
                    if config.Maximize_criterion else "")
    variant = getattr(config, "Prompt_variant", "baseline")
    if variant not in _PROMPT_VARIANTS:
        fatal("unknown prompt variant %r; known: %s",
              variant, ", ".join(sorted(_PROMPT_VARIANTS)))
    task_template, noncover = _PROMPT_VARIANTS[variant]
    table = getattr(config, "Example_format", "dict") == "table"
    note = _SAMPLE_NOTES.get(
        variant, (_SAMPLE_NOTE_DEFAULT, _SAMPLE_NOTE_DEFAULT_TABLE))[1 if table else 0]
    prompt = _load_prompt().format(
        posclass=posname,
        negclass=negname,
        header=_HEADERS.get(variant, _HEADER_DEFAULT),
        sample_note=note.format(n_sampled=len(sample)),
        task_definition=task_template.format(
            posclass=posname, negclass=negname,
            rules_hint=_rules_hint(posname)),
        grow_note=(_GROW_NOTE if grow_mode else ""),
        noncover=noncover,
        task_info=_advice_block(),
        maximization_criterion=maximization,
        text_note=_text_note(),
        schema=schema_block,
        examples=examples_block,
        candidate_conditions=candidate_block,
        n_sampled=len(sample),
        tools_hint=tools_hint,
    )
    if config.Require_tool_output:
        prompt += _TOOL_OUTPUT_HINT
    if config.Log_stem:
        prompt_path = "%s-%03d.prompt.txt" % (config.Log_stem, round)
        try:
            with open(prompt_path, "w") as fp:
                fp.write(prompt)
        except OSError as e:
            warning("could not write LLM-prompt log %s: %s", prompt_path, e)

    retrieval = (RetrievalSession(dataset, cl, sample, rng.derived_np_rng(0x524554, round),
                                 config.Retrieval_max_rows, config.Retrieval_per_call,
                                 config.Retrieval_max_chars)
                 if retrieval_mode != "off" else None)
    run_kwargs = (
        {"deps": _ToolDeps(dataset=dataset, cl=cl, sample=sample, retrieval=retrieval),
         "usage_limits": _tool_limits(tool_budget)}
        if use_tools else {}
    )
    if use_tools:
        _toollog.start(round, "build")
    last_err: Optional[str] = None
    try:
        for attempt in range(3):
            try:
                if retrieval is not None:
                    run_kwargs["deps"].retrieval_calls_at_start = retrieval.calls_made
                _t0 = time.monotonic()
                result = agent.run_sync(prompt, **run_kwargs)
                _record_call(result, time.monotonic() - _t0)
                spec: PyRuleSpec = result.output if hasattr(result, "output") else result.data
                if trace.tracing(trace.SUMM):
                    print("// === LLM output (attempt %d) ===" % (attempt + 1))
                    if use_tools:
                        try:
                            usage = result.usage()
                            print("// usage: %d requests / %d tool calls"
                                  % (usage.requests, usage.tool_calls))
                        except Exception:
                            pass
                    print("// reasoning:")
                    for line in (spec.reasoning or "").splitlines() or [""]:
                        print("//   " + line)
                    print("// source (raw):")
                    for line in (spec.source or "").splitlines() or [""]:
                        print("//   " + line)
                source = _strip_fences(spec.source)
                if "def covers" not in source:
                    last_err = "LLM output did not define covers()"
                    if trace.tracing(trace.SUMM):
                        print("// REJECTED: %s" % last_err)
                    record.note_failed_attempt()
                    continue
                if trace.tracing(trace.SUMM) and source != spec.source.strip():
                    print("// source (post-strip):")
                    for line in source.splitlines():
                        print("//   " + line)
                fn = _SandboxRule(source)
                # quick self-check: must run on the first example without raising
                fn(inst_to_dict(dataset[0].inst))
                p, n = _evaluate(fn, cl, dataset)
                if trace.tracing(trace.SUMM):
                    print("// LLM rule built (attempt %d): p=%.4g n=%.4g" % (attempt + 1, p, n))
                return source, fn, p, n
            except Exception as e:
                reason = _account_error_reason(e)
                if reason:
                    raise LLMAccountError(reason) from e
                last_err = "%s: %s" % (type(e).__name__, e)
                record.note_failed_attempt()
                warning("LLM rule attempt %d failed: %s", attempt + 1, last_err)

        warning("all LLM rule attempts failed (%s); emitting a no-op rule for this round",
                last_err)
        record.note_lost_round()
        noop_source = "def covers(x):\n    return False\n"
        return noop_source, (lambda x: False), 0.0, 0.0
    finally:
        _toollog.flush(config.Log_stem if use_tools else None)


# ---------------------------------------------------------------------------
# value extraction (a "task": "extract" dataset -- see extract.py)
#
# Same prompt assembly, agent cache, sandbox and retry loop as above; what
# changes is the task definition, the structured-output model, and the scoring
# (extract.extract_evaluate, not _evaluate).  Two arms, picked by
# --rule-body / config.Rule_body: "python" (design A -- the model writes
# `def extract(x: dict)`) and "regex" (design A' -- the model fills in a
# RegexRule spec).  There is no class in extraction, so nothing takes a `cl`.
# ---------------------------------------------------------------------------

_HEADER_EXTRACT = ("You are building one value-extraction rule for a "
                   "SLIPPER-style boosted rule learner.")

_SAMPLE_NOTE_EXTRACT = """Below is a weighted sample of {n_sampled} training examples drawn from the
current boosting distribution, one per line as `[GOLD w=WEIGHT] {{dict}}`.  GOLD
is that row's gold value, written exactly as the scorer parses it, and the dict
is the `x` your rule receives; higher-weight rows are the ones the previous
rules get wrong or abstain on, and that this new rule most needs to handle.
"""

_SAMPLE_NOTE_EXTRACT_TABLE = """Below is a weighted sample of {n_sampled} training examples drawn from the
current boosting distribution, as a table.  The **first row after the header
gives each column's type**.  The first two columns are NOT keys of `x`: `class`
is that row's GOLD VALUE, written exactly as the scorer parses it (a
`[value, unit]` list, or a bare string/number/bool), and `weight` is the
boosting weight (higher-weight rows are the ones the previous rules get wrong
or abstain on, and that this new rule most needs to handle).  Every other
column header is exactly a key of `x` -- typically `NOTE` (the free-text note),
`CALC_ID` (which calculator the note belongs to) and `ENTITY` (which value is
being asked for).  A `None` cell means that value is missing (`x.get(key)` is
None) -- guard for it before any comparison.
"""

_EXTRACT_PROMPT = """{header}

{task_definition}
{task_info}
The dict `x` has one key per attribute.  Each attribute has a type, used as:
  - symbolic   -> str (or None for missing)
  - continuous -> float (or None for missing)
  - set / bag  -> list[str] (or None for missing)

{schema}
{sample_note}
{examples}
"""


# --rule-body python (default): design A -- free Python returning the value.
_TASK_DEFINITION_EXTRACT_PYTHON = """
Task: write a rule in the form of a Python function

    def extract(x: dict): ...

that returns the VALUE asked for by `x['ENTITY']`, read out of the note text in
`x['NOTE']`, for a meaningful cluster of examples -- and returns None on every
other example.  Notes:

 - Return the value in the SAME SHAPE as the gold values in the sample: a
   two-item `[value, unit]` list where the gold is a list, and a bare string,
   number or bool where the gold is bare.  Copy the shape off the rows your
   rule fires on.

 - Return None to ABSTAIN.  Abstaining is free -- it scores nothing either way
   -- while a wrong value costs exactly what a right one earns.  This is one
   rule in a boosted ensemble: later rounds add rules for the examples you
   skip, so fire only where you are confident, and never guess or return a
   default.

 - Do NOT normalize the value: no unit conversion, no case folding, no
   reformatting, no rounding, no stripping of units.  Your value and the gold
   value are put through the SAME canonicalizer before they are compared, so
   'mm Hg' already matches 'mmhg' and '48' already matches 48.  Normalizing it
   yourself can only lose information and cost you matches.

 - That canonicalizer folds CASE, SPACING and UNIT SPELLING ONLY -- it does NOT
   map synonyms.  Returning 'woman' never matches a gold of 'Female', and
   'gentleman' never matches 'Male'.  So when the gold is a fixed token rather
   than text lifted out of the note, match the note's synonym but RETURN THE
   GOLD'S OWN TOKEN as a literal: `if re.search(r'\\bwoman\\b', note): return
   'Female'` -- never `return m.group(1)` on the synonym.

 - `re` is importable inside the sandbox that runs your code -- `import re` at
   the top of the function body is fine, and a regex over `x['NOTE']` is
   usually the shortest way to write the rule.  No third-party import is
   available.

 - A recommended format for the function you learn is

   - Read the text you need out of x (usually x['NOTE']), guarding for None

   - Give a Python comment saying which examples the rule is for, based on your
     prior knowledge of the task or your analysis of the sample

   - One or a few `re.search` / string tests; on a match, build the value in
     the gold's shape and return it

   - `return None` otherwise

Rules:
  - Return ONLY the Python source defining `extract`.  No prose, no markdown
    fences in the source field; you may put reasoning in the reasoning field.
  - Handle missing values (None) explicitly; assume nothing about defaults.
  - Stick to the standard library; do not import third-party packages.
  - Do not raise; return None on any case you don't want to answer.
"""


# --rule-body regex: design A' -- a RegexRule spec as structured output, no code.
_TASK_DEFINITION_EXTRACT_REGEX = """
Task: write a rule as a REGEX SPEC -- structured fields, not code.  The learner
runs it as

    m = re.search(pattern, x['NOTE'], flags)

and builds the value from `template`, so fill in these three fields:

  - pattern:  a Python regular expression, with a capturing group around each
              piece of the value you want.
  - flags:    a string of single-letter `re` flags -- "i" (IGNORECASE), "m"
              (MULTILINE), "s" (DOTALL), "x" (VERBOSE), "a" (ASCII) -- "" for
              none, concatenated for several (e.g. "is").
  - template: a list saying how to assemble the value from the match.  An INT
              item is a capture-group number (1 is the first group); a STRING
              item is a literal.  `[1, "years"]` on `(\\d+)-year-old` gives
              `["87", "years"]`.  A one-item list gives the bare item, so `[1]`
              gives `"87"` and `["Female"]` gives `"Female"`.  Null means
              "group 1", the same as `[1]`.

Notes:

 - Match the SHAPE of the gold values in the sample: a two-item template where
   the gold is a `[value, unit]` list, a one-item template where the gold is
   bare.  Copy the unit literal verbatim off the gold column.

 - No match means ABSTAIN, and abstaining is free -- it scores nothing either
   way -- while a wrong value costs exactly what a right one earns.  This is
   one rule in a boosted ensemble and later rounds add rules for the notes this
   pattern misses, so prefer a precise pattern that matches fewer notes over a
   loose one that captures the wrong number.

 - Do NOT normalize the value: no unit conversion, no case folding, no
   reformatting, no rounding.  The captured text and the gold value are put
   through the SAME canonicalizer before they are compared, so 'mm Hg' already
   matches 'mmhg' and '48' already matches 48.  Write the unit literal the way
   the gold column writes it and let the scorer do the rest.

 - That canonicalizer folds CASE, SPACING and UNIT SPELLING ONLY -- it does NOT
   map synonyms.  A group that captures 'woman' will never match a gold of
   'Female', and one that captures 'gentleman' will never match 'Male'; only
   the words the gold column itself uses can match it.  So whenever the gold is
   a fixed token rather than text lifted verbatim out of the note, let the
   PATTERN detect the note's synonym and put the gold's own token in the
   TEMPLATE as a literal.  For a gold of 'Female' that is
   pattern `\\bwoman\\b` with template ["Female"] -- never pattern
   `\\b(woman)\\b` with template [1].

 - You write NO code, so there is no place to post-process the match: every
   part of the value has to come out of the pattern's groups and the template's
   literals.

Rules:
  - Fill in pattern, flags and template only; put your reasoning in the
    reasoning field.  No prose and no markdown fences in the pattern.
  - The pattern is searched against x['NOTE'] only, not the whole of `x`.
  - Every int in the template must be a group the pattern actually captures.
"""


class ExtractRuleSpec(BaseModel):
    """Structured output for the python arm: PyRuleSpec, renamed for the entry
    point it defines (``extract``, not ``covers``)."""
    reasoning: str
    source: str


class RegexRuleSpec(BaseModel):
    """Structured output for the regex arm: the ``extract.RegexRule`` fields
    themselves, so the model returns a spec rather than code."""
    reasoning: str
    pattern: str
    flags: str = ""
    template: Optional[List[Union[int, str]]] = None


# variant name -> (task-definition block, structured-output model).
_EXTRACT_PROMPT_VARIANTS: Dict[str, Tuple[str, Any]] = {
    "extract_python": (_TASK_DEFINITION_EXTRACT_PYTHON, ExtractRuleSpec),
    "extract_regex": (_TASK_DEFINITION_EXTRACT_REGEX, RegexRuleSpec),
}

# config.Rule_body (--rule-body) -> variant name.  cli.py validates the flag
# against these keys, the way --prompt-variant is validated against
# _PROMPT_VARIANTS.
_RULE_BODIES: Dict[str, str] = {
    "python": "extract_python",
    "regex": "extract_regex",
}

# The fallback body: an extract() that abstains everywhere, so the rule never
# fires and scores (0, 0) -- which loses to the abstain baseline in soft_fit.
_EXTRACT_NOOP_SOURCE = "def extract(x):\n    return None\n"


def _extract_variant() -> str:
    """The prompt variant selected by ``config.Rule_body``."""
    body = getattr(config, "Rule_body", "python")
    if body not in _RULE_BODIES:
        fatal("unknown --rule-body %r; known: %s",
              body, ", ".join(sorted(_RULE_BODIES)))
    return _RULE_BODIES[body]


_extract_agents: Dict[Tuple[str, str], Any] = {}


def _get_extract_agent(model: str, variant: str):
    """Agent cache for the extraction arms (sibling of ``_get_agent``).

    No tools, in either arm and whatever -T says: every tool in this module
    scores (p, n) by class membership (``ex.lab.nom is cl``), which has no
    meaning once a rule returns a value instead of a bool, so extraction gets
    the tool-less agent rather than an adapted tool set.  The cache key is
    (model, variant) because the two arms differ in output_type.
    """
    key = (model, variant)
    if key in _extract_agents:
        return _extract_agents[key]
    try:
        from pydantic_ai import Agent
        from pydantic_ai_litellm import LiteLLMModel  # type: ignore
    except ImportError as e:
        fatal("LLM rule builder requires pydantic-ai + pydantic-ai-litellm: %s", e)
    system = ("You are a concise, careful Python programmer writing a single "
              "value-extraction rule."
              if variant == "extract_python" else
              "You are a concise, careful engineer writing a single "
              "regular-expression value-extraction rule.")
    agent = Agent(
        LiteLLMModel(model),
        output_type=_EXTRACT_PROMPT_VARIANTS[variant][1],
        system_prompt=system,
    )
    _extract_agents[key] = agent
    return agent


def _rule_from_extract_spec(variant: str, spec):
    """One structured response -> an unscored extraction rule.

    Bad model output *raises* (the retry loop catches it and re-asks); it must
    not be fatal(), which is also why the regex flags are validated here rather
    than letting ``extract.RegexRule``'s fatal() escape the run.
    """
    if variant == "extract_python":
        source = _strip_fences(spec.source)
        if "def extract" not in source:
            raise ValueError("LLM output did not define extract()")
        return extract.sandbox_rule(source)
    pattern = _strip_fences(spec.pattern or "")
    if not pattern:
        raise ValueError("LLM output gave an empty regex pattern")
    flags = (spec.flags or "").strip()
    bad = [ch for ch in flags if ch not in extract._RE_FLAGS]
    if bad:
        raise ValueError("unknown regex flag(s) %r (known: %s)"
                         % ("".join(bad), "".join(sorted(extract._RE_FLAGS))))
    template = spec.template
    if template is not None and not template:
        raise ValueError("regex template must be null or a non-empty list")
    return extract.RegexRule(pattern, flags, template)


def _print_extract_spec(variant: str, spec, attempt: int) -> None:
    """The -v1 echo of one attempt's structured output (sibling of the
    ``// === LLM output ===`` block in build_pyrule)."""
    print("// === LLM extraction output (attempt %d) ===" % attempt)
    print("// reasoning:")
    for line in (spec.reasoning or "").splitlines() or [""]:
        print("//   " + line)
    if variant == "extract_python":
        print("// source (raw):")
        for line in (spec.source or "").splitlines() or [""]:
            print("//   " + line)
    else:
        print("// spec: pattern=%r flags=%r template=%r"
              % (spec.pattern, spec.flags, spec.template))


def _abstain_rule():
    """The never-firing fallback rule.  Built without the sandbox: the sandbox
    is exactly what may have just failed."""
    return extract.ExtractRule(source=_EXTRACT_NOOP_SOURCE, fn=lambda x: None)


def build_extract_rule(dataset, sample, round: int = 0):
    """Ask the LLM for one value-extraction rule; return (rule, p, n).

    That is the triple ``extract.best_pool_rule`` returns, so boosting.soft_fit's
    extraction branch only has to pick the builder: ``config.Rule_llm_model``
    set -> here, else the deterministic --rule-pool path.

    ``config.Rule_body`` selects the arm -- "python" has the model write
    ``def extract(x: dict)``, built with ``extract.sandbox_rule`` (design A);
    "regex" has it fill in a pattern/flags/template spec, built as an
    ``extract.RegexRule`` (design A').  ``sample`` is the round's weighted
    prompt sample (drawn by ``boosting.soft_fit``); (p, n) is always
    ``extract.extract_evaluate`` over the full weighted ``dataset``, so it is on
    the same scale as the pool path's.  ``round`` names the per-round prompt log
    when ``config.Log_stem`` is set.

    Tools are off in both arms -- see ``_get_extract_agent``.  Retries up to 2
    times on a bad answer or a sandbox/runtime failure; if a third attempt still
    fails, falls back to a rule that abstains everywhere, whose (0, 0) loses to
    the abstain baseline so the round adds nothing.
    """
    if not config.Rule_llm_model:
        fatal("build_extract_rule called without -R set")
    variant = _extract_variant()
    if round == 0 and config.Tool_budget:
        warning("extraction mode offers the LLM no tools (-T%d ignored): the "
                "coverage tools score class membership, which a value-returning "
                "rule has none of", int(config.Tool_budget))
    task_template = _EXTRACT_PROMPT_VARIANTS[variant][0]
    agent = _get_extract_agent(config.Rule_llm_model, variant)
    table = getattr(config, "Example_format", "dict") == "table"
    examples_block = (_render_examples_table(sample) if table
                      else _render_examples(sample))
    if config.Log_stem:
        log_path = "%s-%03d.txt" % (config.Log_stem, round)
        try:
            with open(log_path, "w") as fp:
                fp.write(examples_block + "\n")
        except OSError as e:
            warning("could not write LLM-prompt log %s: %s", log_path, e)
    note = _SAMPLE_NOTE_EXTRACT_TABLE if table else _SAMPLE_NOTE_EXTRACT
    prompt = _EXTRACT_PROMPT.format(
        header=_HEADER_EXTRACT,
        task_definition=task_template,
        task_info=_advice_block(),
        schema=_schema_block(),
        sample_note=note.format(n_sampled=len(sample)),
        examples=examples_block,
    )
    if config.Log_stem:
        prompt_path = "%s-%03d.prompt.txt" % (config.Log_stem, round)
        try:
            with open(prompt_path, "w") as fp:
                fp.write(prompt)
        except OSError as e:
            warning("could not write LLM-prompt log %s: %s", prompt_path, e)

    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            _t0 = time.monotonic()
            result = agent.run_sync(prompt)
            _record_call(result, time.monotonic() - _t0)
            spec = result.output if hasattr(result, "output") else result.data
            if trace.tracing(trace.SUMM):
                _print_extract_spec(variant, spec, attempt + 1)
            rule = _rule_from_extract_spec(variant, spec)
            # quick self-check: must run on the first example without raising
            rule.extract(extract.inst_dict(dataset[0].inst))
            p, n = extract.extract_evaluate(rule, dataset)
            rule.nposx = p
            rule.nnegx = n
            if trace.tracing(trace.SUMM):
                print("// LLM extraction rule built (attempt %d): p=%.4g n=%.4g"
                      % (attempt + 1, p, n))
            return rule, p, n
        except Exception as e:
            last_err = "%s: %s" % (type(e).__name__, e)
            warning("LLM extraction rule attempt %d failed: %s",
                    attempt + 1, last_err)

    warning("all LLM extraction rule attempts failed (%s); abstaining this round",
            last_err)
    return _abstain_rule(), 0.0, 0.0
