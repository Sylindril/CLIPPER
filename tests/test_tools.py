"""LLM-agent tools: _matches predicate + _count_coverage tallying."""
import pytest

from slipper import llm_rule_builder as L
from slipper.data import Example


# ---------------------------------------------------------------------------
# _matches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,op,value,want", [
    ("y", "==", "y", True),
    ("y", "==", "n", False),
    ("y", "!=", "n", True),
    ("y", "!=", "y", False),
    (5, ">=", 3, True),
    (5, ">=", 10, False),
    (3, "<=", 5, True),
    (10, "<=", 5, False),
    (["a", "b"], "contains", "a", True),
    (["a", "b"], "contains", "c", False),
    ("hello world", "like", "world", True),
    ("hello world", "like", "xyz", False),
    (["hello"], "like", "hello", False),   # like only matches strings
    ("text", ">=", 3, False),              # type mismatch returns False, not raise
    (None, "==", "y", False),              # None never matches
    (None, "contains", "x", False),
    ("x", "??", "x", False),               # unknown op via _matches returns False
])
def test_matches(v, op, value, want):
    assert L._matches(v, op, value) is want


# ---------------------------------------------------------------------------
# _count_coverage
# ---------------------------------------------------------------------------

class _FakeNom:
    def __init__(self, name): self.name = name


class _FakeLab:
    def __init__(self, nom): self.nom = nom


def _ex(d, cls, wt):
    """Build an Example whose .inst already IS the dict the tool will see
    (we monkeypatch inst_to_dict to identity for these tests)."""
    return Example(inst=d, lab=_FakeLab(cls), wt=wt)


@pytest.fixture
def passthrough_inst_to_dict(monkeypatch):
    monkeypatch.setattr(L, "inst_to_dict", lambda inst: inst)


def test_count_coverage_invalid_op_raises():
    with pytest.raises(L._TOOL_ARG_EXC, match="unsupported op"):
        L._count_coverage([], None, "f", "??", "v")


def test_count_coverage_eq(passthrough_inst_to_dict):
    pos, neg = _FakeNom("pos"), _FakeNom("neg")
    data = [
        _ex({"f": "y"}, pos, 0.4),
        _ex({"f": "y"}, neg, 0.1),
        _ex({"f": "n"}, pos, 0.2),
        _ex({"f": None}, neg, 0.3),
    ]
    p, n = L._count_coverage(data, pos, "f", "==", "y")
    assert p == pytest.approx(0.4)
    assert n == pytest.approx(0.1)


def test_count_coverage_missing_field_is_zero(passthrough_inst_to_dict):
    pos, neg = _FakeNom("pos"), _FakeNom("neg")
    data = [
        _ex({"f": None}, pos, 1.0),
        _ex({}, neg, 1.0),
    ]
    p, n = L._count_coverage(data, pos, "f", "==", "y")
    assert p == 0.0 and n == 0.0


def test_count_coverage_contains_on_set(passthrough_inst_to_dict):
    pos, neg = _FakeNom("pos"), _FakeNom("neg")
    data = [
        _ex({"words": ["speaker", "today"]}, pos, 0.3),
        _ex({"words": ["other"]},            neg, 0.2),
        _ex({"words": ["speaker", "x"]},     neg, 0.5),
    ]
    p, n = L._count_coverage(data, pos, "words", "contains", "speaker")
    assert p == pytest.approx(0.3)
    assert n == pytest.approx(0.5)


def test_count_coverage_like_substring(passthrough_inst_to_dict):
    pos, neg = _FakeNom("pos"), _FakeNom("neg")
    data = [
        _ex({"body": "Title: hello"}, pos, 0.5),
        _ex({"body": "no header"},    pos, 0.2),
        _ex({"body": "Title: world"}, neg, 0.3),
    ]
    p, n = L._count_coverage(data, pos, "body", "like", "Title:")
    assert p == pytest.approx(0.5)
    assert n == pytest.approx(0.3)


def test_count_coverage_numeric(passthrough_inst_to_dict):
    pos, neg = _FakeNom("pos"), _FakeNom("neg")
    data = [
        _ex({"x": 1.0}, pos, 0.1),
        _ex({"x": 5.0}, pos, 0.2),
        _ex({"x": 8.0}, neg, 0.4),
        _ex({"x": 3.0}, neg, 0.3),
    ]
    p, n = L._count_coverage(data, pos, "x", ">=", 5.0)
    assert p == pytest.approx(0.2)
    assert n == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _count_coverage_of_lambda_body  (uses smolagents sandbox; slower)
# ---------------------------------------------------------------------------

def test_lambda_body_simple_eq(passthrough_inst_to_dict):
    pos, neg = _FakeNom("pos"), _FakeNom("neg")
    data = [
        _ex({"f": "y"}, pos, 0.4),
        _ex({"f": "y"}, neg, 0.1),
        _ex({"f": "n"}, pos, 0.2),
    ]
    p, n = L._count_coverage_of_lambda_body(data, pos, "x.get('f') == 'y'")
    assert p == pytest.approx(0.4)
    assert n == pytest.approx(0.1)


def test_lambda_body_conjunction(passthrough_inst_to_dict):
    pos, neg = _FakeNom("pos"), _FakeNom("neg")
    data = [
        _ex({"a": "1", "b": "1"}, pos, 0.3),  # both -> covered
        _ex({"a": "1", "b": "0"}, neg, 0.5),  # only a -> not covered
        _ex({"a": "1", "b": "1"}, neg, 0.2),  # both -> covered
    ]
    p, n = L._count_coverage_of_lambda_body(
        data, pos, "x.get('a') == '1' and x.get('b') == '1'")
    assert p == pytest.approx(0.3)
    assert n == pytest.approx(0.2)


def test_lambda_body_per_example_error_counts_as_no_cover(passthrough_inst_to_dict):
    pos = _FakeNom("pos")
    data = [
        _ex({"f": None}, pos, 1.0),   # x.get('f').lower() -> AttributeError
        _ex({"f": "Y"}, pos, 0.5),    # 'Y'.lower() == 'y' -> True
    ]
    p, n = L._count_coverage_of_lambda_body(data, pos, "x.get('f').lower() == 'y'")
    assert p == pytest.approx(0.5)
    assert n == pytest.approx(0.0)


def test_lambda_body_rejects_multiline():
    with pytest.raises(L._TOOL_ARG_EXC, match="single-line"):
        L._count_coverage_of_lambda_body([], None, "x.get('f') == 'y'\nor x.get('g')")


def test_lambda_body_compile_error_raises():
    # `1 + + 1` is a SyntaxError when wrapped as `lambda x: 1 + + 1`? actually
    # `+ +1` is parseable; use a real syntax error instead.
    with pytest.raises(L._TOOL_ARG_EXC, match="failed to compile"):
        L._count_coverage_of_lambda_body([], None, "x.get('f' ==")


# ---------------------------------------------------------------------------
# best_threshold / best_symbol  (full-data value pickers, --value-tools)
#
# These need the real names/symbols machinery (Aval instances, the symbol
# table), so we load a tiny synthetic dataset through load_dataset rather than
# faking inst_to_dict.  Layout (age continuous, color symbolic):
#
#   age  color  class     age<=30 is pure-pos; color==red is pure-pos
#   10   red    pos
#   20   red    pos
#   30   blue   pos
#   40   blue   neg
#   50   blue   neg
#
# objective is sqrt(p)-sqrt(n)+1 (newboost_value), all weights 1.0.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny(tmp_path):
    import json
    from slipper import dataset, config, value
    from slipper import names as names_mod

    ds = {
        "classes": ["pos", "neg"],
        "attributes": [
            {"name": "age", "kind": "continuous"},
            {"name": "color", "kind": "symbolic", "values": ["red", "blue"]},
        ],
        "train": [
            {"values": [10.0, "red"], "label": "pos"},
            {"values": [20.0, "red"], "label": "pos"},
            {"values": [30.0, "blue"], "label": "pos"},
            {"values": [40.0, "blue"], "label": "neg"},
            {"values": [50.0, "blue"], "label": "neg"},
        ],
        "test": [],
    }
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(ds))
    dataset.reset_all()
    train, _ = dataset.load_dataset(str(p))
    for ex in train:
        ex.wt = 1.0
    config.Value_function = value.newboost_value
    cl = next(c.nom for c in names_mod.Classes if c.nom.name == "pos")
    return train, cl


def test_best_threshold_le_is_argmax(tiny):
    train, cl = tiny
    r = L._best_threshold(train, cl, "age", "<=")
    assert r["threshold"] == pytest.approx(30.0)   # age<=30 covers all 3 pos, 0 neg
    assert r["p"] == pytest.approx(3.0)
    assert r["n"] == pytest.approx(0.0)


def test_best_threshold_ge_is_argmax(tiny):
    train, cl = tiny
    r = L._best_threshold(train, cl, "age", ">=")
    assert r["threshold"] == pytest.approx(10.0)   # age>=10 (all): 3 pos, 2 neg is best >=
    assert r["p"] == pytest.approx(3.0)
    assert r["n"] == pytest.approx(2.0)


def test_best_threshold_gt_lt_aliases(tiny):
    train, cl = tiny
    assert L._best_threshold(train, cl, "age", ">")["threshold"] == pytest.approx(10.0)
    assert L._best_threshold(train, cl, "age", "<")["threshold"] == pytest.approx(30.0)


def test_best_symbol_ranks_by_objective(tiny):
    train, cl = tiny
    r = L._best_symbol(train, cl, "color")
    assert [e["value"] for e in r] == ["red", "blue"]      # red: 2/0 beats blue: 1/2
    assert (r[0]["p"], r[0]["n"]) == pytest.approx((2.0, 0.0))
    assert (r[1]["p"], r[1]["n"]) == pytest.approx((1.0, 2.0))


def test_best_symbol_negate_scores_complement(tiny):
    train, cl = tiny
    r = L._best_symbol(train, cl, "color", negate=True)
    assert r[0]["value"] == "blue"                          # color!=blue == red-only
    assert (r[0]["p"], r[0]["n"]) == pytest.approx((2.0, 0.0))


def test_best_symbol_top_k_limit(tiny):
    train, cl = tiny
    assert len(L._best_symbol(train, cl, "color", top_k=1)) == 1


def test_best_threshold_rejects_symbolic(tiny):
    train, cl = tiny
    with pytest.raises(L._TOOL_ARG_EXC, match="not a continuous"):
        L._best_threshold(train, cl, "color", ">=")


def test_best_threshold_rejects_bad_direction(tiny):
    train, cl = tiny
    with pytest.raises(L._TOOL_ARG_EXC, match="direction"):
        L._best_threshold(train, cl, "age", "!=")


def test_best_symbol_rejects_continuous(tiny):
    train, cl = tiny
    with pytest.raises(L._TOOL_ARG_EXC, match="not a symbolic"):
        L._best_symbol(train, cl, "age")


def test_value_tools_reject_unknown_field(tiny):
    train, cl = tiny
    with pytest.raises(L._TOOL_ARG_EXC, match="unknown attribute"):
        L._best_threshold(train, cl, "nope", ">=")


# ---------------------------------------------------------------------------
# candidate_conditions  (best condition per attribute; sample vs full-data)
#
# Reuses the `tiny` dataset but renormalizes weights to sum to 1 (0.2 each) so
# the effective example counts (weight x population size) equal plain example
# counts.  Here sample == full training set, so both column pairs match.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_norm(tiny):
    train, cl = tiny
    for ex in train:
        ex.wt = 1.0 / len(train)   # sum to 1, mirroring the live distribution
    return train, cl


def _parse_table(md):
    """Parse the candidate_conditions markdown into [(condition, sp, sn, fp, fn)]
    (data rows only, skipping header + separator)."""
    rows = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[1] in ("sample p", "---"):
            continue
        cond, sp, sn, fp, fn = cells
        rows.append((cond, float(sp), float(sn), float(fp), float(fn)))
    return rows


def test_candidate_conditions_vacuous(tiny_norm):
    train, cl = tiny_norm
    rows = _parse_table(L._candidate_conditions_table(train, train, cl, ""))
    # leading row = all examples (3 pos, 2 neg), same on both scales
    assert rows[0][0].startswith("(all examples)")
    assert rows[0][1:] == pytest.approx((3.0, 2.0, 3.0, 2.0))
    # best candidate first: age<=30 (3 pos, 0 neg) beats color==red (2 pos, 0 neg)
    by_cond = {r[0]: r[1:] for r in rows[1:]}
    assert "x['age'] <= 30" in by_cond
    assert by_cond["x['age'] <= 30"] == pytest.approx((3.0, 0.0, 3.0, 0.0))
    assert by_cond["x['color'] == 'red'"] == pytest.approx((2.0, 0.0, 2.0, 0.0))
    assert rows[1][0] == "x['age'] <= 30"   # sorted by objective, best first


def test_candidate_conditions_filtered(tiny_norm):
    train, cl = tiny_norm
    # restrict to blue: {(30,pos),(40,neg),(50,neg)}
    rows = _parse_table(
        L._candidate_conditions_table(train, train, cl, "x['color'] == 'blue'"))
    assert rows[0][0].startswith("x['color'] == 'blue'")
    # filter total: 1 pos, 2 neg (effective counts, filter restricts both scales)
    assert rows[0][1:] == pytest.approx((1.0, 2.0, 1.0, 2.0))
    by_cond = {r[0]: r[1:] for r in rows[1:]}
    # within blue, age<=30 isolates the single positive
    assert by_cond["x['age'] <= 30"] == pytest.approx((1.0, 0.0, 1.0, 0.0))
    # only blue is present in the filtered data, so that's color's best value
    assert by_cond["x['color'] == 'blue'"] == pytest.approx((1.0, 2.0, 1.0, 2.0))


def test_candidate_conditions_sample_vs_full_differ(tiny_norm):
    train, cl = tiny_norm
    # sample = the two red rows only; full = all five.  age<=30 covers 2 in the
    # sample but 3 over the full data -> the two column pairs must diverge.
    sample = [ex for ex in train if ex.inst[1].sym.name == "red"]
    rows = _parse_table(L._candidate_conditions_table(train, sample, cl, ""))
    by_cond = {r[0]: r[1:] for r in rows[1:]}
    sp, sn, fp, fn = by_cond["x['age'] <= 30"]
    assert (fp, fn) == pytest.approx((3.0, 0.0))    # full data: 3 positives
    assert sn == pytest.approx(0.0)
    assert sp < fp                                   # sample undercounts


def test_candidate_conditions_bad_filter_raises(tiny_norm):
    train, cl = tiny_norm
    with pytest.raises(L._TOOL_ARG_EXC, match="single-line"):
        L._candidate_conditions_table(train, train, cl, "x['a']\nand x['b']")


def test_candidate_conditions_prompt_gated_off(tiny_norm, monkeypatch):
    train, cl = tiny_norm
    from slipper import config
    monkeypatch.setattr(config, "Candidate_conditions", False, raising=False)
    assert L._candidate_conditions_prompt(train, train, cl) == ""
    monkeypatch.setattr(config, "Candidate_conditions", True, raising=False)
    block = L._candidate_conditions_prompt(train, train, cl)
    assert "Candidate conditions" in block and "x['age'] <= 30" in block


# ---------------------------------------------------------------------------
# _llm_settings (--reasoning-effort)
# ---------------------------------------------------------------------------

def test_llm_settings_unset_keeps_existing_behaviour(monkeypatch):
    """Empty Reasoning_effort must leave every existing condition unchanged:
    the timeout call sites keep their timeout, the bare one stays settings-less."""
    monkeypatch.setattr(L.config, "Reasoning_effort", "")
    assert L._llm_settings() == {"timeout": L._PER_REQUEST_TIMEOUT_S}
    assert L._llm_settings(timeout=False) is None


def test_llm_settings_merges_extra_body_with_reasoning_effort(monkeypatch):
    """--llm-extra-body is the provider escape hatch; it shares extra_body with
    --reasoning-effort rather than clobbering it.  Added for Qwen served by
    mlx_lm.server, which spends its whole budget thinking unless told not to."""
    monkeypatch.setattr(L.config, "Llm_max_tokens", 0)
    monkeypatch.setattr(L.config, "Reasoning_effort", "")
    monkeypatch.setattr(L.config, "Llm_extra_body", "")
    assert "extra_body" not in L._llm_settings()
    monkeypatch.setattr(L.config, "Llm_extra_body",
                        '{"chat_template_kwargs": {"enable_thinking": false}}')
    assert L._llm_settings()["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setattr(L.config, "Reasoning_effort", "none")
    assert L._llm_settings()["extra_body"] == {
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False}}


def test_tool_output_hint_is_opt_in_and_names_the_tool():
    """--require-tool-output must ride the always-included prompt, not the
    tools hint: the structured output is a function tool even at -T0."""
    import inspect
    src = inspect.getsource(L.build_pyrule)
    assert "config.Require_tool_output" in src and "_TOOL_OUTPUT_HINT" in src
    assert "final_result" in L._TOOL_OUTPUT_HINT
    assert L._TOOL_OUTPUT_HINT not in L._TOOLS_HINT_TAIL


def test_llm_settings_sends_max_tokens(monkeypatch):
    """--llm-max-tokens: some servers default far too low (mlx_lm.server stops
    at 512, truncating the structured output mid-JSON).  Unset sends nothing."""
    monkeypatch.setattr(L.config, "Reasoning_effort", "")
    monkeypatch.setattr(L.config, "Llm_max_tokens", 0)
    assert "max_tokens" not in L._llm_settings()
    monkeypatch.setattr(L.config, "Llm_max_tokens", 4096)
    assert L._llm_settings() == {"timeout": L._PER_REQUEST_TIMEOUT_S, "max_tokens": 4096}
    assert L._llm_settings(timeout=False) == {"max_tokens": 4096}


def test_llm_settings_sends_reasoning_effort_as_extra_body(monkeypatch):
    # openai gpt-5.6-* 400s on function tools + its default effort, and
    # pydantic-ai's structured output is a function tool (so -T0 doesn't help).
    monkeypatch.setattr(L.config, "Reasoning_effort", "none")
    assert L._llm_settings() == {"timeout": L._PER_REQUEST_TIMEOUT_S,
                                 "extra_body": {"reasoning_effort": "none"}}
    assert L._llm_settings(timeout=False) == {"extra_body": {"reasoning_effort": "none"}}


# ---------------------------------------------------------------------------
# account / billing errors: stop, don't retry into a no-op rule
# ---------------------------------------------------------------------------

def _http(status, body):
    from pydantic_ai.exceptions import ModelHTTPError
    return ModelHTTPError(status_code=status, model_name="m", body=body)


@pytest.mark.parametrize("exc,fatal", [
    # the exact body anthropic returned when the balance ran out mid-suite
    (_http(400, {"type": "error", "error": {"type": "invalid_request_error",
     "message": "Your credit balance is too low to access the Anthropic API."}}), True),
    (_http(402, "Payment Required"), True),
    (_http(401, "invalid x-api-key"), True),
    (_http(429, {"error": {"code": "insufficient_quota"}}), True),
    (_http(429, "Rate limit reached, please retry"), False),   # transient: retry
    (_http(400, "tool_use ids must be unique"), False),         # a bad request, not billing
    (ValueError("LLM output did not define covers()"), False),
])
def test_account_error_reason_classifies(exc, fatal):
    assert bool(L._account_error_reason(exc)) is fatal


def test_account_error_reason_follows_the_cause_chain():
    try:
        try:
            raise _http(400, "Your credit balance is too low")
        except Exception as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert L._account_error_reason(outer)


class _FakeAgent:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def run_sync(self, *a, **k):
        self.calls += 1
        raise self.exc


def _patch_llm(monkeypatch, agent):
    monkeypatch.setattr(L.config, "Rule_llm_model", "anthropic/test")
    monkeypatch.setattr(L.config, "Tool_budget", 0)
    monkeypatch.setattr(L, "_get_agent", lambda *a, **k: agent)


def test_build_pyrule_raises_on_credit_exhaustion_without_retrying(tiny, monkeypatch):
    from slipper.errors import LLMAccountError
    train, cl = tiny
    agent = _FakeAgent(_http(400, "Your credit balance is too low to access the Anthropic API."))
    _patch_llm(monkeypatch, agent)
    with pytest.raises(LLMAccountError, match="credit balance"):
        L.build_pyrule(cl, train, train)
    assert agent.calls == 1                      # no retries, no no-op rule


def test_build_pyrule_still_absorbs_ordinary_failures(tiny, monkeypatch):
    train, cl = tiny
    agent = _FakeAgent(_http(429, "Rate limit reached, please retry"))
    _patch_llm(monkeypatch, agent)
    source, fn, p, n = L.build_pyrule(cl, train, train)
    assert agent.calls == 3 and "return False" in source and (p, n) == (0.0, 0.0)


def test_console_main_maps_account_error_to_its_exit_code(monkeypatch):
    from slipper import cli
    from slipper.errors import ACCOUNT_ERROR_EXIT, LLMAccountError

    def boom(argv):
        raise LLMAccountError("credit balance is too low")
    monkeypatch.setattr(cli, "main", boom)
    with pytest.raises(SystemExit) as e:
        cli.console_main()
    assert e.value.code == ACCOUNT_ERROR_EXIT


def test_no_tools_agent_gets_a_long_but_finite_request_timeout():
    """The tool-less agent had no timeout, and one unanswered request hung a
    claude-sonnet-5 cell for 2+ hours.  It can't share the tool paths' 60s:
    sonnet averages ~74s per tool-less call (one cell averaged 152s)."""
    import inspect
    src = inspect.getsource(L._get_agent)
    assert "_llm_settings(timeout=_NO_TOOLS_REQUEST_TIMEOUT_S)" in src
    assert "timeout=False" not in src
    assert L._NO_TOOLS_REQUEST_TIMEOUT_S >= 3 * 152     # ~3.9x the worst cell average
    assert L._llm_settings(timeout=L._NO_TOOLS_REQUEST_TIMEOUT_S)["timeout"] == \
        L._NO_TOOLS_REQUEST_TIMEOUT_S
