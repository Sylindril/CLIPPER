"""Execution deadlines and exact-prompt recovery for long-running experiments."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("medcalc_regex_runtime", REPO / "scripts/medcalc_regex_runtime.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_deadline_interrupts_backtracking_and_restores_signal_state():
    code = """
import signal, sys
sys.path.insert(0, 'scripts')
from medcalc_regex_runtime import install_limit, RegexExecutionTimeout
from slipper.extract import RegexRule
previous = signal.getsignal(signal.SIGALRM)
guard = install_limit(.03)
assert RegexRule(r'height (\\d+) (cm)', template=[1, 2]).extract({'NOTE':'height 172 cm'}) == ['172','cm']
rule = RegexRule(r'^(a+)+$', template=[1])
for _ in range(2):
    try:
        rule.extract({'NOTE':'a' * 10000 + '!'})
        raise AssertionError('pathological regex was not bounded')
    except RegexExecutionTimeout:
        pass
assert guard['timeout_events'] == 1
assert guard['cached_timeout_events'] == 1
assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
assert signal.getsignal(signal.SIGALRM) == previous
guard['restore']()
"""
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, capture_output=True, timeout=5)


def test_repr_parser_preserves_patterns_with_field_like_text():
    pattern = r"(?: flags='i' template=[1]|height (\d+))"
    text = f"pattern={pattern!r} flags={'im'!r} template={[1, 'cm']!r}"
    assert R.spec_fields(text) == {"pattern":pattern, "flags":"im", "template":[1,"cm"]}


def test_python_deadline_interrupts_regex_inside_the_sandbox():
    code = """
import signal, sys
sys.path.insert(0, 'scripts')
from medcalc_regex_runtime import install_python_limit as install_limit, PythonExecutionTimeout
from slipper.extract import sandbox_rule
previous = signal.getsignal(signal.SIGALRM)
guard = install_limit(.03)
rule = sandbox_rule("def extract(x):\\n    import re\\n    return re.search(r'^(a+)+$', x['NOTE'])\\n")
for _ in range(2):
    try:
        rule.extract({'NOTE':'a' * 10000 + '!'})
        raise AssertionError('sandbox regex was not bounded')
    except Exception as error:
        assert 'exceeded' in str(error)
assert guard['timeout_events'] == 1
assert guard['cached_timeout_events'] == 1
assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
assert signal.getsignal(signal.SIGALRM) == previous
guard['restore']()
"""
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, capture_output=True, timeout=8)


def test_replay_requires_the_identical_prompt_and_does_not_bill_twice(tmp_path, monkeypatch):
    from slipper import llm_rule_builder as builder
    prompt = "Original weighted training sample"
    (tmp_path / "rule-000.prompt.txt").write_text(prompt)
    (tmp_path / "stdout.txt").write_text(
        "// boost round   0 rules   0  train 1.0000\n"
        "// === LLM extraction output (attempt 1) ===\n"
        "// spec: pattern='height (\\\\d+)' flags='i' template=[1]\n")
    new_calls, billed = [], []
    fresh = SimpleNamespace(output="new output")

    class Agent:
        def run_sync(self, text):
            new_calls.append(text)
            return fresh

    monkeypatch.setattr(builder, "_get_extract_agent", lambda *args: Agent())
    monkeypatch.setattr(builder, "_record_call", lambda result, latency: billed.append(result))
    state = R.install_replay(tmp_path)
    agent = builder._get_extract_agent("unused", "extract_regex")
    # A changed weighted sample must not consume an old proposal.
    assert agent.run_sync("different sample") is fresh
    result = agent.run_sync(prompt)
    assert result.output.pattern == r"height (\d+)"
    builder._record_call(result, 0)
    builder._record_call(fresh, 0)
    assert new_calls == ["different sample"] and billed == [fresh]
    assert state == {"replayed_calls":1, "new_calls":1}
    recovered = R.read_proposals(tmp_path)
    assert recovered[0]["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
