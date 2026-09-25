"""record.recorder() side-information collection (latency / cost / tokens)."""
import pytest

from slipper import record


class _FakeUsage:
    def __init__(self, i, o, req=1, tc=0):
        self.input_tokens, self.output_tokens = i, o
        self.requests, self.tool_calls = req, tc


def test_noop_when_inactive():
    # No active recorder: note_llm_call is a silent no-op (instrumentation is
    # free outside a `with` block).
    assert not record.active()
    record.note_llm_call(1.0, "m", input_tokens=10, output_tokens=5, cost_usd=0.1)
    assert not record.active()


def test_accumulates_totals():
    with record.recorder() as r:
        assert record.active()
        record.note_llm_call(1.5, "m", usage=_FakeUsage(100, 20, req=2, tc=3), cost_usd=0.01)
        record.note_llm_call(0.5, "m", usage=_FakeUsage(50, 10), cost_usd=0.02)
    assert not record.active()
    assert r["n_llm_calls"] == 2
    assert r["input_tokens"] == 150
    assert r["output_tokens"] == 30
    assert r["total_tokens"] == 180
    assert r["requests"] == 3
    assert r["tool_calls"] == 3
    assert r["latency_s"] == pytest.approx(2.0)
    assert r["cost_usd"] == pytest.approx(0.03)
    assert r["cost_unavailable"] is False


def test_falls_back_to_explicit_token_args():
    with record.recorder() as r:
        record.note_llm_call(1.0, "m", input_tokens=7, output_tokens=3, cost_usd=0.0)
    assert (r["input_tokens"], r["output_tokens"], r["total_tokens"]) == (7, 3, 10)


def test_nesting_outer_sees_inner():
    with record.recorder() as outer:
        record.note_llm_call(1.0, "m", input_tokens=10, output_tokens=1, cost_usd=0.1)
        with record.recorder() as inner:
            record.note_llm_call(2.0, "m", input_tokens=20, output_tokens=2, cost_usd=0.2)
        assert inner["n_llm_calls"] == 1
        assert inner["input_tokens"] == 20
        assert inner["latency_s"] == pytest.approx(2.0)
    assert outer["n_llm_calls"] == 2          # outer saw both calls
    assert outer["input_tokens"] == 30
    assert outer["latency_s"] == pytest.approx(3.0)
    assert outer["cost_usd"] == pytest.approx(0.3)


def test_cost_zero_tokens_is_free():
    assert record._cost("some/model", 0, 0) == (0.0, False)


def test_cost_unpriced_model_is_unavailable():
    cost, unavailable = record._cost("no/such-model-xyz-123", 100, 50)
    assert cost == 0.0 and unavailable is True


def test_unpriced_model_flags_recorder():
    with record.recorder() as r:
        # cost_usd=None -> compute via litellm; bogus model -> unavailable
        record.note_llm_call(1.0, "no/such-model-xyz-123",
                             input_tokens=100, output_tokens=50)
    assert r["cost_usd"] == 0.0
    assert r["cost_unavailable"] is True
    assert r["n_llm_calls"] == 1
