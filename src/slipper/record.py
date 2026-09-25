"""record.py - lightweight side-information recorder for LLM runs.

A ``recorder()`` context manager collects aggregate side information (latency,
cost, token counts) about the LLM inference that happens inside the block::

    from slipper import record
    with record.recorder() as records:
        ... run learning / LLM inference ...
    # records is a dict of totals: latency_s, cost_usd, input_tokens, ...

Recorders nest: an event is added to every active recorder on the stack, so an
outer recorder sees everything an inner one does.  Instrumented code (the LLM
rule builder) calls ``note_llm_call(...)``; when no recorder is active it is a
no-op, so the instrumentation is free outside a ``with`` block.

The stack is a plain module global, not thread-local: a single slipper run is
synchronous, and the experiment harness isolates runs in separate processes.
"""

from contextlib import contextmanager
from typing import Any, Dict, List

# Active record dicts, innermost last.
_stack: List[Dict[str, Any]] = []

# Integer counters plus the two floats handled specially in _fresh().
#
# Two accountings run side by side.  The original fields (input_tokens ...
# n_llm_calls, cost_usd) count only *successful* agent attempts: they are fed
# by note_llm_call after a run_sync returns, so an attempt that raises --
# UsageLimitExceeded, a structured-output failure, an HTTP error -- adds
# nothing, even though every request it made was billed.  The api_* fields
# count every request that reached the provider (note_api_request, fed by a
# wrapper around the litellm call pydantic-ai-litellm makes), successful
# attempt or not; they are the ones to trust for spend.  failed_attempts and
# lost_rounds (note_failed_attempt / note_lost_round) record the failures
# themselves, so they no longer have to be inferred from llm_calls < bestj.
_INT_FIELDS = ("input_tokens", "output_tokens", "total_tokens",
               "requests", "tool_calls", "n_llm_calls",
               "api_requests", "api_failures", "api_input_tokens",
               "api_output_tokens", "failed_attempts", "lost_rounds")


def _fresh() -> Dict[str, Any]:
    d: Dict[str, Any] = {k: 0 for k in _INT_FIELDS}
    d["latency_s"] = 0.0
    d["cost_usd"] = 0.0
    d["cost_unavailable"] = False   # True if any call's model was unpriced
    d["api_cost_usd"] = 0.0
    d["api_cost_unavailable"] = False
    return d


@contextmanager
def recorder():
    """Context manager yielding a dict of accumulated LLM side information."""
    d = _fresh()
    _stack.append(d)
    try:
        yield d
    finally:
        _stack.pop()


def active() -> bool:
    """True if a recorder is currently collecting."""
    return bool(_stack)


def note_llm_call(latency_s: float, model: str, usage: Any = None,
                  input_tokens: int = 0, output_tokens: int = 0,
                  cost_usd=None) -> None:
    """Add one LLM call's side info to every active recorder (no-op if none).

    ``usage`` is a pydantic-ai usage object (``input_tokens`` / ``output_tokens``
    / ``requests`` / ``tool_calls``); its token/request counts are read when
    given, else the ``input_tokens`` / ``output_tokens`` args are used.
    ``cost_usd`` overrides the computed cost; when None the cost is derived from
    the token counts via litellm's price map (0.0 + ``cost_unavailable`` if the
    model isn't priced).
    """
    if not _stack:
        return
    it = ot = req = tc = 0
    if usage is not None:
        it = getattr(usage, "input_tokens", 0) or 0
        ot = getattr(usage, "output_tokens", 0) or 0
        req = getattr(usage, "requests", 0) or 0
        tc = getattr(usage, "tool_calls", 0) or 0
    it = it or input_tokens
    ot = ot or output_tokens
    unavailable = False
    if cost_usd is None:
        cost_usd, unavailable = _cost(model, it, ot)
    for d in _stack:
        d["latency_s"] += latency_s
        d["cost_usd"] += cost_usd
        d["input_tokens"] += it
        d["output_tokens"] += ot
        d["total_tokens"] += it + ot
        d["requests"] += req
        d["tool_calls"] += tc
        d["n_llm_calls"] += 1
        if unavailable:
            d["cost_unavailable"] = True


def _cost(model: str, input_tokens: int, output_tokens: int):
    """(cost_usd, unavailable): dollar cost via litellm's price map, or
    (0.0, True) when the model isn't priced / litellm is unavailable."""
    if not model or (input_tokens == 0 and output_tokens == 0):
        return 0.0, False
    try:
        import litellm
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=input_tokens,
            completion_tokens=output_tokens)
        return float(prompt_cost) + float(completion_cost), False
    except Exception:
        return 0.0, True


def note_api_request(model: str, response: Any = None,
                     error: BaseException = None) -> None:
    """Add one provider request to every active recorder (no-op if none).

    Called for every request, whether or not the surrounding agent attempt
    succeeds.  A completed request contributes its tokens and its cost from
    ``litellm.completion_cost`` (which reads the response's own model and usage,
    so cached and reasoning tokens are priced as the provider bills them); a
    request that raised (``error``) counts as an api_failure with no cost --
    refused requests are not billed.
    """
    if not _stack:
        return
    it = ot = 0
    cost, unavailable = 0.0, False
    if error is None and response is not None:
        usage = getattr(response, "usage", None)
        it = getattr(usage, "prompt_tokens", 0) or 0
        ot = getattr(usage, "completion_tokens", 0) or 0
        try:
            import litellm
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:                 # unpriced: fall back to the price map
            cost, unavailable = _cost(model, it, ot)
    for d in _stack:
        d["api_requests"] += 1
        if error is not None:
            d["api_failures"] += 1
            continue
        d["api_input_tokens"] += it
        d["api_output_tokens"] += ot
        d["api_cost_usd"] += cost
        if unavailable:
            d["api_cost_unavailable"] = True


def note_failed_attempt() -> None:
    """One agent attempt that produced no usable rule (it raised, or its code
    was rejected)."""
    for d in _stack:
        d["failed_attempts"] += 1


def note_lost_round() -> None:
    """One boosting round whose every attempt failed, so a no-op rule was
    emitted in its place."""
    for d in _stack:
        d["lost_rounds"] += 1
