"""Bound pure regex execution and recover recorded extraction proposals.

Used by the repeat harness, without editing the frozen learner implementation.
An audit of every proposed regex on every available input establishes whether
the execution limit changes a completed fit. Regex outputs depend only on NOTE,
not on the current boosting weights.
"""
import ast
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import sys
import tokenize
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class RegexExecutionTimeout(TimeoutError):
    pass


def install_limit(seconds=1.0):
    """Install once per isolated worker; preserve re's matching semantics."""
    from slipper.extract import RegexRule
    original = RegexRule.extract
    timed_out = set()
    state = {"timeout_events": 0, "cached_timeout_events": 0}

    def expired(signum, frame):
        raise RegexExecutionTimeout(f"Regex evaluation exceeded {seconds:g} seconds")

    def bounded(self, x):
        key = (self.pattern, self.flags, x["NOTE"])
        if key in timed_out:
            state["cached_timeout_events"] += 1
            raise RegexExecutionTimeout(f"Regex evaluation exceeded {seconds:g} seconds")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("Regex deadline would replace an active process timer")
        prior = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return original(self, x)
        except RegexExecutionTimeout:
            state["timeout_events"] += 1
            timed_out.add(key)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)

    RegexRule.extract = bounded
    state["restore"] = lambda: setattr(RegexRule, "extract", original)
    return state



class PythonExecutionTimeout(TimeoutError):
    pass


def install_python_limit(seconds=1.0):
    """Bound each sandboxed Python extraction call; same caching as install_limit."""
    from slipper.extract import ExtractRule
    original = ExtractRule.extract
    timed_out = set()
    state = {"timeout_events": 0, "cached_timeout_events": 0}

    def bounded(self, instance):
        key = (self.source, json.dumps(instance, sort_keys=True))
        if key in timed_out:
            state["cached_timeout_events"] += 1
            raise PythonExecutionTimeout(f"Python extraction exceeded {seconds:g} seconds")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("Python deadline would replace an active process timer")
        expired_here = False

        def expired(signum, frame):
            nonlocal expired_here
            expired_here = True
            timed_out.add(key)
            state["timeout_events"] += 1
            raise PythonExecutionTimeout(f"Python extraction exceeded {seconds:g} seconds")

        previous = signal.signal(signal.SIGALRM, expired)
        # smolagents' thread timeout waits for its worker during executor
        # shutdown, and a C regex can retain the GIL indefinitely. Execute the
        # same interpreter synchronously so the process alarm can interrupt it.
        executor = getattr(self.fn, "_executor", None)
        previous_timeout = getattr(executor, "timeout_seconds", None)
        if executor is not None:
            executor.timeout_seconds = None
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            value = original(self, instance)
            if expired_here:
                raise PythonExecutionTimeout(f"Python extraction exceeded {seconds:g} seconds")
            return value
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if executor is not None:
                executor.timeout_seconds = previous_timeout

    ExtractRule.extract = bounded
    state["restore"] = lambda: setattr(ExtractRule, "extract", original)
    return state


def spec_fields(text):
    """Read repr fields without evaluating generated text as Python code."""
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    starts = [t.start[1] for t in tokens if t.type == tokenize.NAME and t.string in ("flags", "template")]
    assert len(starts) == 2
    parts = [text[:starts[0]], text[starts[0]:starts[1]], text[starts[1]:]]
    node = ast.parse("spec(" + ",".join(parts) + ")", mode="eval").body
    assert isinstance(node, ast.Call) and not node.args
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def read_proposals(attempt):
    """Recover all structured outputs, including rejected and stalled rules."""
    attempt = Path(attempt)
    lines = (attempt / "stdout.txt").read_text().splitlines()
    out, round_number, attempt_number = [], None, None
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match(r"// boost round\s+(\d+) rules", line)
        if match:
            round_number = int(match.group(1))
        match = re.match(r"// === LLM extraction output \(attempt (\d+)\) ===", line)
        if match:
            attempt_number = int(match.group(1))
        fields = None
        if line.startswith("// spec: "):
            fields = spec_fields(line[len("// spec: "):])
        elif line == "// source (raw):":
            source = []
            index += 1
            while index < len(lines) and lines[index].startswith("//   "):
                source.append(lines[index][len("//   "):])
                index += 1
            fields = {"source": "\n".join(source)}
            index -= 1
        if fields is not None:
            assert round_number is not None and attempt_number is not None
            prompt = (attempt / f"rule-{round_number:03d}.prompt.txt").read_text()
            out.append(dict(round=round_number, attempt=attempt_number, output=fields,
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        index += 1
    return out


def install_replay(attempt):
    """Replay saved responses only when the newly rendered prompt is identical."""
    from slipper import llm_rule_builder as builder
    by_prompt = defaultdict(list)
    for proposal in read_proposals(attempt):
        by_prompt[proposal["prompt_sha256"]].append(proposal["output"])
    state = {"replayed_calls": 0, "new_calls": 0}
    original_get = builder._get_extract_agent
    original_record = builder._record_call

    class ReplayAgent:
        def __init__(self, model, variant):
            self.model, self.variant = model, variant

        def run_sync(self, prompt):
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if by_prompt[key]:
                fields = by_prompt[key].pop(0)
                output_type = builder._EXTRACT_PROMPT_VARIANTS[self.variant][1]
                state["replayed_calls"] += 1
                return SimpleNamespace(output=output_type(reasoning="Recovered original proposal", **fields),
                                       _medcalc_replayed=True)
            state["new_calls"] += 1
            return original_get(self.model, self.variant).run_sync(prompt)

    def record_call(result, latency):
        if not getattr(result, "_medcalc_replayed", False):
            original_record(result, latency)

    builder._get_extract_agent = lambda model, variant: ReplayAgent(model, variant)
    builder._record_call = record_call
    return state


def audit_cell(task):
    parent, cell, seconds = task
    from slipper import llm_rule_builder as builder
    parent = Path(parent)
    result_path = parent / "cells" / cell["id"] / "result.json"
    result = json.loads(result_path.read_text())
    attempt = parent / result["attempt"]
    if cell.get("pool"):
        proposals = [dict(output=p) for p in json.loads((parent / cell["pool"]).read_text())]
    else:
        proposals = read_proposals(attempt)
        assert proposals, cell["id"]
    bundle = json.loads((parent / cell["input"]).read_text())
    notes = sorted({row["values"][0] for split in ("train", "dev", "final") for row in bundle[split]})
    guard = install_limit(seconds)
    checked, invalid, errors = 0, 0, 0
    seen = set()
    try:
        for proposal in proposals:
            output = proposal["output"]
            signature = json.dumps(output, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            try:
                spec = builder.RegexRuleSpec(reasoning="Audit", **output)
                rule = builder._rule_from_extract_spec("extract_regex", spec)
            except Exception:
                invalid += 1
                continue
            for note in notes:
                try:
                    rule.extract({"NOTE": note})
                except RegexExecutionTimeout:
                    # This fit needs replay under the amended execution policy.
                    return dict(id=cell["id"], passed=False, checked=checked,
                                timeout_events=guard["timeout_events"])
                except Exception:
                    errors += 1  # Same exceptions already abstained in the original runtime.
                checked += 1
    finally:
        guard["restore"]()
    return dict(id=cell["id"], passed=True, proposed=len(proposals), unique=len(seen),
                checked=checked, invalid_specs=invalid, ordinary_exceptions=errors, timeout_events=0)


def audit(parent, report, jobs=4, seconds=1.0):
    parent = Path(parent)
    manifest = json.loads((parent / "manifest.json").read_text())
    tasks = [(str(parent), cell, seconds) for cell in manifest["cells"]
             if cell["kind"] == "extraction"
             and (cell["condition"].startswith("extract_regex") or cell["condition"] == "extract_pool")
             and (parent / "cells" / cell["id"] / "result.json").exists()]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(audit_cell, tasks))
    payload = dict(parent_identity=manifest["identity"], seconds=seconds, results=results)
    Path(report).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Audited {len(results)} completed regex/pool fits; "
          f"{sum(r['passed'] for r in results)} unchanged, "
          f"{sum(not r['passed'] for r in results)} require replay", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    audit(args.parent.resolve(), args.report, args.jobs)
