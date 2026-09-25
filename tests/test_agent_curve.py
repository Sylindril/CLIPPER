"""scripts/run_agent_curve.py: result parsing, side info, sandboxed scoring,
workspace contents, and the sandy invocation.  No Docker or network needed."""
import argparse
import json
import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
sys.path.insert(0, SCRIPTS)
import run_agent_curve as A


# Shapes taken from real sandy -p runs: [sandy] log lines, then (claude) a
# terminal escape sequence glued to one-line JSON, or (gemini) indented JSON.
CLAUDE_OUT = (
    "[sandy] Launching Claude Code sandbox...\n"
    "[sandy]   Network:           Egress proxy (strict)\n"
    "\n"
    "\x1b]4;4;rgb:61/8f/ff\x1b\\"
    + json.dumps({"type": "result", "num_turns": 19, "total_cost_usd": 0.37,
                  "result": "done {braces} in text",
                  "usage": {"input_tokens": 26, "cache_creation_input_tokens": 100,
                            "cache_read_input_tokens": 1000, "output_tokens": 7580}})
    + "\n")

GEMINI_RES = {"session_id": "s", "response": "done",
              "stats": {"models": {
                  "gemini-3.5-flash": {"api": {"totalRequests": 12},
                                            "tokens": {"prompt": 1000, "candidates": 50,
                                                       "thoughts": 10, "cached": 600}},
                  "gemini-other": {"api": {"totalRequests": 3},
                                   "tokens": {"prompt": 200, "candidates": 5}}}}}
GEMINI_OUT = "[sandy] Launching Gemini CLI sandbox...\n" + json.dumps(GEMINI_RES, indent=2)


CODEX_EVENTS = [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "command_execution", "command": "ls {x}"}},
    {"type": "item.completed", "item": {"type": "command_execution", "command": "cat"}},
    {"type": "item.completed", "item": {"type": "reasoning"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
    {"type": "turn.completed", "usage": {"input_tokens": 21543, "cached_input_tokens": 9883,
                                         "output_tokens": 1898,
                                         "reasoning_output_tokens": 168}},
]
CODEX_OUT = ("[sandy] Launching Codex CLI sandbox...\n"
             + "\n".join(json.dumps(e) for e in CODEX_EVENTS) + "\n")


def test_parse_result_codex_event_stream():
    assert A.parse_result(CODEX_OUT) == {"events": CODEX_EVENTS}


def test_side_info_codex_sums_turn_usage_and_counts_steps(monkeypatch):
    import litellm
    seen = []

    def fake_cost(model, prompt_tokens, completion_tokens, cache_read_input_tokens):
        seen.append((model, prompt_tokens, completion_tokens, cache_read_input_tokens))
        return 0.5, 0.25

    monkeypatch.setattr(litellm, "cost_per_token", fake_cost)
    side = A.side_info("codex", A.parse_result(CODEX_OUT), 3.0, "gpt-5.6-luna")
    assert seen == [("gpt-5.6-luna", 21543, 1898, 9883)]
    assert side == dict(llm_calls=3, latency_s=3.0, in_tok=21543, out_tok=1898,
                        cost_usd=0.75)


def test_side_info_codex_failed_turn_has_no_usage(monkeypatch):
    import litellm
    monkeypatch.setattr(litellm, "cost_per_token", lambda **kw: (0.0, 0.0))
    res = A.parse_result('{"type": "turn.started"}\n{"type": "turn.failed"}\n')
    side = A.side_info("codex", res, 1.0, "gpt-5.6-luna")
    assert (side["llm_calls"], side["in_tok"], side["out_tok"]) == (0, 0, 0)


def test_parse_result_claude_skips_log_lines_and_escape_sequence():
    res = A.parse_result(CLAUDE_OUT)
    assert res["num_turns"] == 19 and res["result"] == "done {braces} in text"


def test_parse_result_gemini_multiline_json():
    assert A.parse_result(GEMINI_OUT) == GEMINI_RES


@pytest.mark.parametrize("out", ["", "[sandy] only log lines\n", '[sandy] x\n{"truncated": '])
def test_parse_result_missing_or_broken_json_is_empty(out):
    assert A.parse_result(out) == {}


def test_side_info_claude_sums_all_input_token_kinds():
    side = A.side_info("claude", A.parse_result(CLAUDE_OUT), 12.34)
    assert side == dict(llm_calls=19, latency_s=12.3, in_tok=1126, out_tok=7580,
                        cost_usd=0.37)
    assert set(side) == set(A.rlc._SIDE_COLS)


def test_side_info_gemini_sums_models_and_prices_with_litellm(monkeypatch):
    import litellm
    seen = []

    def fake_cost(model, prompt_tokens, completion_tokens, cache_read_input_tokens):
        seen.append((model, prompt_tokens, completion_tokens, cache_read_input_tokens))
        return 0.001 * prompt_tokens, 0.01 * completion_tokens

    monkeypatch.setattr(litellm, "cost_per_token", fake_cost)
    side = A.side_info("gemini", GEMINI_RES, 5.0)
    assert seen == [("gemini/gemini-3.5-flash", 1000, 60, 600),
                    ("gemini/gemini-other", 200, 5, 0)]
    assert side == dict(llm_calls=15, latency_s=5.0, in_tok=1200, out_tok=65,
                        cost_usd=round(1.0 + 0.6 + 0.2 + 0.05, 6))


def test_side_info_gemini_unpriced_model_leaves_cost_blank(monkeypatch):
    import litellm

    def unpriced(**kw):
        raise ValueError("model not mapped")

    monkeypatch.setattr(litellm, "cost_per_token", unpriced)
    assert A.side_info("gemini", GEMINI_RES, 1.0)["cost_usd"] == ""


def test_result_models_reports_what_the_agent_actually_used():
    # gemini CLI 0.60.0 silently runs an unknown model as its own default
    assert A.result_models("gemini", GEMINI_RES) == ["gemini-3.5-flash", "gemini-other"]
    assert A.result_models("claude", A.parse_result(CLAUDE_OUT)) == []
    assert A.result_models("claude", {"modelUsage": {"claude-sonnet-5": {}}}) == \
        ["claude-sonnet-5"]
    assert A.result_models("codex", A.parse_result(CODEX_OUT)) == []


SPEC = {
    "classes": ["pos", "neg"],
    "attributes": [{"name": "age", "kind": "continuous"},
                   {"name": "color", "kind": "symbolic", "values": ["red", "blue"]}],
    "train": [{"values": [70.0, "red"], "label": "pos"},
              {"values": [20.0, None], "label": "neg"}],
    "test": [{"values": [70.0, "red"], "label": "pos"},
             {"values": [65.0, "blue"], "label": "pos"},
             {"values": [30.0, "red"], "label": "neg"},
             {"values": [None, "blue"], "label": "neg"}],
}

RULES = '''
def classify(x):
    age = x["age"] if x["age"] is not None else 0.0
    if age >= 60 and x["color"] == "red":
        return "pos"
    if age >= 90:
        return "pos"
    return "neg"
'''


def test_score_error_and_rule_count_ignores_if_expressions():
    err, rules, imports = A.score(RULES, SPEC)
    assert (err, rules, imports) == (25.0, 2, 0)      # misses the 65/blue positive


def test_score_example_that_raises_counts_as_error():
    src = 'def classify(x):\n    return "pos" if x["age"] > 50 else "neg"\n'
    err, rules, _ = A.score(src, SPEC)       # rows 1-3 right; None > 50 raises on row 4
    assert err == 25.0 and rules == 0


def test_score_counts_authorized_imports():
    src = 'import math\ndef classify(x):\n    return "neg"\n'
    assert A.score(src, SPEC) == (50.0, 0, 1)


def test_score_sandbox_rejects_unauthorized_import():
    src = 'import os\ndef classify(x):\n    return "neg"\n'
    with pytest.raises(Exception):
        A.score(src, SPEC)


def test_build_workspace_holds_only_training_data(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "rules.py").write_text("stale")
    A.build_workspace(str(ws), SPEC, max_turns=7)
    assert sorted(os.listdir(ws)) == [".gemini", "TASK.md", "schema.json", "train.csv"]
    assert (ws / "train.csv").read_text().splitlines() == [
        "age,color,label", "70.0,red,pos", "20.0,,neg"]
    assert json.loads((ws / "schema.json").read_text()) == {
        "classes": SPEC["classes"], "attributes": SPEC["attributes"]}
    task = (ws / "TASK.md").read_text()
    assert "holds 2 labeled" in task and "one of `pos`, `neg`" in task
    assert "65.0" not in task
    gem = json.loads((ws / ".gemini" / "settings.json").read_text())
    assert gem["model"]["maxSessionTurns"] == 7


def test_build_workspace_forbid_learners_appends_instruction(tmp_path):
    """--forbid-learners is the agent_gpt56_sol_norun condition: sol otherwise
    pip-installs sklearn and transcribes a fitted CART."""
    off, on = tmp_path / "off", tmp_path / "on"
    A.build_workspace(str(off), SPEC, max_turns=5)
    A.build_workspace(str(on), SPEC, max_turns=5, forbid_learners=True)
    base, forbidden = (off / "TASK.md").read_text(), (on / "TASK.md").read_text()
    assert "should not run any existing learning algorithm" not in base
    assert forbidden.startswith(base)
    flat = " ".join(forbidden.split())          # the template hard-wraps
    for phrase in ("should not run any existing learning algorithm",
                   "Helper script purpose:", "will be discarded"):
        assert phrase in flat


def _args(**kw):
    base = dict(agent="claude", model="m", effort="high", budget_usd=1.5, max_turns=9,
                cpus=2, mem="4g")
    return argparse.Namespace(**{**base, **kw})


def test_agent_invocation_claude(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS_ROOT", str(tmp_path / "cells"))
    cmd, env = A.agent_invocation(_args())
    assert cmd[:4] == ["sandy", "--new", "-p", A.PROMPT]
    for flag, val in [("--setting-sources", "project,local"), ("--max-budget-usd", "1.5"),
                      ("--max-turns", "9"), ("--disallowedTools", "WebSearch,WebFetch")]:
        assert cmd[cmd.index(flag) + 1] == val
    assert env["SANDY_CLAUDE_AUTH"] == "api_key" and env["SANDY_MODEL"] == "m"
    assert env["SANDY_EGRESS_STRICT"] == "1" and env["SANDY_EFFORT"] == "high"
    nogh = env["PATH"].split(os.pathsep)[0]
    assert nogh == str(tmp_path / "nogh")
    assert os.access(os.path.join(nogh, "gh"), os.X_OK)


FAKE_SANDY = """\
#!/bin/sh
echo "setup-start $(python3 -c 'import time;print(time.time())')" >> "$LOG"
sleep 0.4
echo "setup-end $(python3 -c 'import time;print(time.time())')" >> "$LOG"
echo "[sandy] Launching Fake sandbox..."
sleep "$AGENT_SECS"
echo '{"num_turns": 1}'
echo "egress summary" >&2
"""


@pytest.fixture
def fake_sandy(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS_ROOT", str(tmp_path / "cells"))
    monkeypatch.setattr(A.subprocess, "run",
                        lambda *a, **k: argparse.Namespace(stdout="", returncode=0))
    script = tmp_path / "sandy"
    script.write_text(FAKE_SANDY)
    script.chmod(0o755)
    log = tmp_path / "setup.log"

    def run(name, agent_secs, timeout=60):
        ws = tmp_path / name
        ws.mkdir()
        env = dict(os.environ, LOG=str(log), AGENT_SECS=str(agent_secs))
        return A.run_sandy(str(ws), [str(script)], env, timeout)
    return run, log


def test_run_sandy_serializes_setup_but_not_agent_runs(fake_sandy):
    import threading
    import time
    run, log = fake_sandy
    results = []
    t0 = time.time()
    threads = [threading.Thread(target=lambda n=n: results.append(run(n, 1.5)))
               for n in ("ws1", "ws2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    for rc, out, err, _ in results:
        assert rc == 0 and A.parse_result(out) == {"num_turns": 1}
        assert "egress summary" in err
    stamps = [float(l.split()[1]) for l in log.read_text().splitlines()]
    (s1, e1, s2, e2) = sorted(stamps)
    assert e1 <= s2                                    # setup phases never overlap
    assert wall < 2 * (0.4 + 1.5)                      # agent phases do


def test_run_sandy_timeout_kills_and_reports(fake_sandy):
    run, _ = fake_sandy
    rc, out, err, wall = run("slow", agent_secs=30, timeout=1)
    assert rc != 0 and wall < 15
    assert "timed out after 1s" in err and not A._live


def test_agent_invocation_codex(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS_ROOT", str(tmp_path / "cells"))
    cmd, env = A.agent_invocation(_args(agent="codex"))
    assert cmd == ["sandy", "--new", "-p", A.PROMPT, "--json",
                   "--config", 'web_search="disabled"',
                   "--config", 'model_reasoning_effort="high"']
    assert "-c" not in cmd                          # sandy drops -c as --continue
    assert env["SANDY_CODEX_AUTH"] == "api_key" and env["CODEX_MODEL"] == "m"
    assert env["SANDY_AGENT"] == "codex" and "SANDY_CLAUDE_AUTH" not in env
    cmd, _ = A.agent_invocation(_args(agent="codex", effort=""))
    assert not any("reasoning_effort" in c for c in cmd)


def test_agent_invocation_gemini_has_no_claude_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS_ROOT", str(tmp_path / "cells"))
    cmd, env = A.agent_invocation(_args(agent="gemini"))
    assert cmd == ["sandy", "--new", "-p", A.PROMPT, "--output-format", "json"]
    assert env["GEMINI_MODEL"] == "m" and env["SANDY_AGENT"] == "gemini"
    assert "SANDY_CLAUDE_AUTH" not in env
