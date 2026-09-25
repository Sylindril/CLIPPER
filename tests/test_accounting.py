"""Cost accounting: every provider request is metered (not just successful
agent attempts), and failed attempts / lost rounds are recorded as data.

Before this, record.py only saw attempts whose run_sync returned, so a
condition with many failed attempts under-reported spend several-fold and its
no-op rounds were invisible."""
import json
import os
import sys
import types

import pytest

from slipper import record


def _resp(pt, ct):
    return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=pt,
                                                              completion_tokens=ct))


def test_note_api_request_counts_every_request_and_prices_from_the_response(monkeypatch):
    import litellm
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.25)
    with record.recorder() as outer:
        with record.recorder() as inner:
            record.note_api_request("m", response=_resp(1000, 50))
            record.note_api_request("m", error=RuntimeError("HTTP 500"))
    for d in (outer, inner):                         # recorders nest
        assert (d["api_requests"], d["api_failures"]) == (2, 1)
        assert (d["api_input_tokens"], d["api_output_tokens"]) == (1000, 50)
        assert d["api_cost_usd"] == 0.25             # the failed request is not billed
        assert d["n_llm_calls"] == 0                 # the old accounting saw nothing


def test_failure_counters():
    with record.recorder() as d:
        record.note_failed_attempt(); record.note_failed_attempt()
        record.note_lost_round()
    assert (d["failed_attempts"], d["lost_rounds"]) == (2, 1)


def test_request_meter_records_success_and_failure(monkeypatch):
    import litellm
    import pydantic_ai_litellm.litellm_model as plm
    from slipper import llm_rule_builder as L
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.1)
    outcomes = iter([_resp(10, 2), RuntimeError("boom")])

    async def fake(*a, **k):
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o
    monkeypatch.setattr(plm, "acompletion", fake)
    monkeypatch.setattr(L, "_METER_INSTALLED", False)
    L._install_request_meter()
    import asyncio
    with record.recorder() as d:
        asyncio.run(plm.acompletion(model="m", messages=[]))
        with pytest.raises(RuntimeError):
            asyncio.run(plm.acompletion(model="m", messages=[]))
    assert (d["api_requests"], d["api_failures"], d["api_input_tokens"]) == (2, 1, 10)
    assert d["api_cost_usd"] == 0.1


def test_cli_prints_accounting_even_when_every_attempt_fails(tmp_path, monkeypatch, capsys):
    """The case that used to vanish: no attempt succeeds, so "LLM side info" is
    never printed -- the accounting line must still report the failures."""
    from pydantic_ai.exceptions import ModelHTTPError
    from slipper import cli, config, dataset
    from slipper import llm_rule_builder as L
    ds = {"classes": ["pos", "neg"],
          "attributes": [{"name": "age", "kind": "continuous"}],
          "train": [{"values": [float(i)], "label": "pos" if i < 3 else "neg"} for i in range(6)],
          "test": [{"values": [1.0], "label": "pos"}, {"values": [5.0], "label": "neg"}]}
    stem = tmp_path / "toy"
    (tmp_path / "toy.json").write_text(json.dumps(ds))

    class Fake:
        def run_sync(self, *a, **k):
            raise ModelHTTPError(status_code=429, model_name="m", body="Rate limit reached")
    monkeypatch.setattr(L, "_get_agent", lambda *a, **k: Fake())
    dataset.reset_all()
    cli.main(["slipper", "-R", "anthropic/test", "-T0", "-i0", "-n1", str(stem)])
    out = capsys.readouterr().out
    assert "LLM side info" not in out
    assert "LLM accounting:" in out and "3 failed attempts, 1 lost rounds" in out


SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)
import run_learning_curve as R


def test_runner_parses_the_accounting_line():
    out = ("LLM side info:     2 calls, 4.0s latency, 46343 in / 1059 out tokens, $0.0051\n"
           "LLM accounting:    6 requests (1 failed), 139000 in / 3100 out tokens, $0.0152, "
           "4 failed attempts, 1 lost rounds\n")
    assert R.parse_accounting(out) == {"api_requests": "6", "api_failures": "1",
                                       "api_cost_usd": "0.0152", "failed_attempts": "4",
                                       "lost_rounds": "1"}
    assert R.parse_accounting("Test error rate: 4.86%") == {}


def test_ensure_header_extends_an_old_header_in_place(tmp_path):
    p = tmp_path / "results.csv"
    old = R._HEADER[:12]
    p.write_text(",".join(old) + "\nx,ds,250,1,2,4.86,1,2,4.0,100,10,0.01\n")
    R._ensure_header(str(p))
    lines = p.read_text().splitlines()
    assert lines[0] == ",".join(R._HEADER)
    assert lines[1] == "x,ds,250,1,2,4.86,1,2,4.0,100,10,0.01"   # rows untouched
    R._ensure_header(str(p))                                      # idempotent
    assert p.read_text().splitlines()[0] == ",".join(R._HEADER)
