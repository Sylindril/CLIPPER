"""Training isolation, exposure limits and independent retrieval randomness."""
from types import SimpleNamespace

import numpy as np
import pytest

from slipper import config, rng
from slipper import llm_rule_builder as L
from slipper.retrieval import RetrievalSession


@pytest.fixture
def population():
    pos, neg = SimpleNamespace(name="yes"), SimpleNamespace(name="no")
    rows = [SimpleNamespace(index=i, inst={"age": 20 + i, "NOTE": "note " + str(i)},
                            lab=SimpleNamespace(nom=pos if i % 2 else neg), wt=(i+1)/210)
            for i in range(20)]
    return rows, pos


def fetch(session, where="", label="either", k=10):
    return session.fetch(where=where, label=label, k=k,
                         compile_filter=L._compile_filter, to_dict=lambda x: x)


def test_counterexamples_are_unseen_training_negatives_and_weights_are_unchanged(population):
    rows, pos = population
    original = [(r.index, r.wt, dict(r.inst)) for r in rows]
    session = RetrievalSession(rows, pos, rows[:5], np.random.default_rng(3))
    result = fetch(session, "x['age'] >= 30", "negative")
    assert {r["row_id"] for r in result["rows"]} == {10, 12, 14, 16, 18}
    assert all(r["class"] == "no" and r["x"]["age"] >= 30 for r in result["rows"])
    assert all(r["boosting_weight"] == rows[r["row_id"]].wt for r in result["rows"])
    assert [(r.index, r.wt, r.inst) for r in rows] == original
    assert not fetch(session, "x['age'] >= 30", "negative")["rows"]


def test_per_call_and_cumulative_limits_share_seen_state(population):
    rows, pos = population
    session = RetrievalSession(rows, pos, rows[:2], np.random.default_rng(7), 5, 3)
    a, b, c = [fetch(session, k=999) for _ in range(3)]
    assert [len(r["rows"]) for r in (a, b, c)] == [3, 2, 0]
    assert len({r["row_id"] for s in (a, b) for r in s["rows"]}) == 5
    assert c["status"] == "retrieval_budget_exhausted"


def test_no_notes_are_truncated_to_fit_character_budget(population):
    rows, pos = population
    session = RetrievalSession(rows, pos, [], np.random.default_rng(2), max_chars=1)
    result = fetch(session)
    assert result["rows"] == [] and session.seen == set()
    assert result["status"] == "row_exceeds_remaining_character_budget"


def test_invalid_query_does_not_consume_rows(population):
    rows, pos = population
    session = RetrievalSession(rows, pos, [], np.random.default_rng(2))
    for args in [dict(where="x["), dict(where="True\nFalse"), dict(k=0), dict(label="test")]:
        with pytest.raises((ValueError, L._TOOL_ARG_EXC)):
            fetch(session, **args)
        assert session.rows_used == 0


def test_separate_proposals_and_seeds_are_reproducible_without_advancing_learner_rng(population):
    rows, pos = population
    rng.seed(42)
    expected = rng.np_rng().random(8)
    rng.seed(42)
    a = fetch(RetrievalSession(rows, pos, [], rng.derived_np_rng(9, 2)))
    b = fetch(RetrievalSession(rows, pos, [], rng.derived_np_rng(9, 2)))
    assert a == b
    assert np.array_equal(rng.np_rng().random(8), expected)
    rng.seed(43)
    assert fetch(RetrievalSession(rows, pos, [], rng.derived_np_rng(9, 2))) != a
    rng.seed()


def test_sampling_uses_boosting_weights(population):
    rows, pos = population
    for row in rows:
        row.wt = 0.0
    rows[9].wt = 1.0
    session = RetrievalSession(rows, pos, [], np.random.default_rng(3))
    assert [r["row_id"] for r in fetch(session)["rows"]] == [9]


def test_modes_have_distinct_tools_and_share_the_existing_call_budget(monkeypatch):
    monkeypatch.setattr(config, "Claimed_tool_budget", None)
    for mode, name in [("off", None), ("targeted", "retrieve_examples"), ("random", "sample_examples")]:
        agent = L._get_agent("test", True, False, True, False, False, mode)
        names = set(agent._function_toolset.tools)
        assert names == ({"count_coverage_of_lambda_body", name} if name else
                         {"count_coverage_of_lambda_body"})
        hint = L._tools_hint(6, False, True, False, retrieval_mode=mode)
        assert "at most 6 tool calls" in hint
        if name:
            assert name in hint and "20 additional rows" in hint
    assert L._tool_limits(6).tool_calls_limit == 6


def test_tool_log_records_retrieved_ids(population, monkeypatch, tmp_path):
    rows, pos = population
    monkeypatch.setattr(L, "inst_to_dict", lambda x: x)
    deps = L._ToolDeps(rows, pos, [], RetrievalSession(rows, pos, [], np.random.default_rng(1)))
    L._toollog.start(0, "build")
    result = L._retrieve_rows(deps, "", "positive", 2, "retrieve_examples")
    try:
        assert L._toollog.calls[-1]["result"] == result
        assert all(row["row_id"] % 2 for row in result["rows"])
    finally:
        L._toollog.flush(None)


def test_required_retrieval_rejects_a_final_rule_until_the_tool_is_used(population, monkeypatch):
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    rows, pos = population
    monkeypatch.setattr(L, "inst_to_dict", lambda x: x)
    deps = L._ToolDeps(rows, pos, [], RetrievalSession(rows, pos, [], np.random.default_rng(3)))
    turns = []

    def respond(messages, info):
        turns.append(messages)
        if len(turns) == 2:
            return ModelResponse(parts=[ToolCallPart("retrieve_examples", {"where": "", "k": 1})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
            {"reasoning": "test", "source": "def covers(x): return True"})])

    agent = L._get_agent("test", True, False, True, False, False, "targeted", True)
    agent.run_sync("test", model=FunctionModel(respond), deps=deps, usage_limits=L._tool_limits(6))
    assert len(turns) == 3 and deps.retrieval.calls_made == 1 and deps.retrieval.rows_used == 1
