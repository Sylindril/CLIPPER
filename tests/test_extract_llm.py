"""The LLM rule builder's extraction arms (design A / A').

No network and no API key: ``_get_extract_agent`` is replaced by a fake that
records the prompts it is handed and replays canned structured outputs, the way
tests/test_prompt.py monkeypatches the prompt machinery.  What is checked is
the prompt text of both variants, the rule each canned response builds, the
retry-then-abstain fallback, and the CLI's flag validation (which exits before
any model is contacted).
"""
import os
import subprocess
import sys

import pytest

from slipper import config, dataset, extract
from slipper import llm_rule_builder as L

from test_extract import SPEC, _write

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _reset_config_globals():
    """Extraction mode and the rule-builder knobs are config globals; don't
    leak them between tests."""
    saved = {k: getattr(config, k) for k in (
        "Task_mode", "Rule_pool", "Rule_body", "Rule_llm_model", "Epsilon",
        "Sample_size", "Tool_budget", "Log_stem", "Example_format",
        "N_boost", "Slipper_folds",
    )}
    yield
    for k, v in saved.items():
        setattr(config, k, v)
    dataset.reset_all()
    L._extract_agents.clear()


# ---------------------------------------------------------------------------
# the fake agent
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, output):
        self.output = output

    def usage(self):
        return None


class _FakeAgent:
    """Stands in for the pydantic-ai agent: records every prompt (and every
    run_sync kwarg, so a test can assert no tool deps were passed) and replays
    canned outputs, the last one repeating for further attempts."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.kwargs = []

    def run_sync(self, prompt, **kw):
        self.prompts.append(prompt)
        self.kwargs.append(kw)
        out = self.outputs[min(len(self.prompts) - 1, len(self.outputs) - 1)]
        if isinstance(out, Exception):
            raise out
        return _FakeResult(out)


@pytest.fixture
def ages(tmp_path):
    """The synthetic ages dataset from test_extract, loaded in extraction mode
    with the LLM path selected."""
    train, test = dataset.load_dataset(_write(tmp_path, "ages.json", SPEC))
    config.Epsilon = 0.5 / len(train)
    config.Rule_llm_model = "fake/model"
    config.Rule_pool = None
    return train, test


@pytest.fixture
def agent(monkeypatch):
    """Install a fake agent; the test fills in its canned outputs."""
    holder = {}

    def install(*outputs):
        holder["agent"] = _FakeAgent(*outputs)
        monkeypatch.setattr(L, "_get_extract_agent",
                            lambda model, variant: holder["agent"])
        return holder["agent"]
    return install


def _py(source, reasoning="because"):
    return L.ExtractRuleSpec(reasoning=reasoning, source=source)


def _rx(pattern, flags="", template=None, reasoning="because"):
    return L.RegexRuleSpec(reasoning=reasoning, pattern=pattern, flags=flags,
                           template=template)


YEARS_SOURCE = ("def extract(x):\n"
                "    import re\n"
                "    m = re.search(r'(\\d+)-year-old', x['NOTE'] or '')\n"
                "    return [m.group(1), 'years'] if m else None\n")


# ---------------------------------------------------------------------------
# prompt text
# ---------------------------------------------------------------------------

def _prompt(ages, agent, body, out):
    train, _ = ages
    config.Rule_body = body
    a = agent(out)
    L.build_extract_rule(train, train[:4])
    return a.prompts[0]


def test_python_variant_states_the_contract(ages, agent):
    prompt = _prompt(ages, agent, "python", _py(YEARS_SOURCE))
    # the signature and the abstain contract
    assert "def extract(x: dict): ..." in prompt
    assert "Return None to ABSTAIN" in prompt
    assert "Abstaining is free" in prompt
    assert "fire only where you are confident" in prompt
    assert "never guess or return a\n   default" in prompt
    # the no-normalization warning, and why
    assert "Do NOT normalize the value" in prompt
    assert "SAME canonicalizer before they are compared" in prompt
    # the sandbox's one importable module
    assert "`re` is importable inside the sandbox" in prompt
    # shape of the returned value
    assert "SAME SHAPE as the gold values" in prompt
    assert "Return ONLY the Python source defining `extract`" in prompt


def test_regex_variant_states_the_spec_fields(ages, agent):
    prompt = _prompt(ages, agent, "regex", _rx(r"(\d+)-year-old",
                                               template=[1, "years"]))
    assert "REGEX SPEC -- structured fields, not code" in prompt
    assert "m = re.search(pattern, x['NOTE'], flags)" in prompt
    for field in ("  - pattern:", "  - flags:", "  - template:"):
        assert field in prompt
    # the template's int/str convention, with the worked example
    assert "An INT\n              item is a capture-group number" in prompt
    assert "a STRING\n              item is a literal" in prompt
    assert '`[1, "years"]` on `(\\d+)-year-old` gives' in prompt
    # the same abstain contract and normalization warning as the python arm
    assert "No match means ABSTAIN" in prompt
    assert "abstaining is free" in prompt
    assert "Do NOT normalize the value" in prompt
    assert "SAME canonicalizer before they are compared" in prompt
    # no code to post-process the match
    assert "You write NO code" in prompt


def test_the_prompt_carries_the_sample_table_and_gold_values(ages, agent):
    prompt = _prompt(ages, agent, "python", _py(YEARS_SOURCE))
    # the rendered example table: the keys of x are its column headers ...
    assert "| class | weight | NOTE | CALC_ID | ENTITY |" in prompt
    # ... the gold label repr is the `class` column ...
    assert "| [87, 'years'] |" in prompt
    assert "a 87-year-old patient was admitted" in prompt
    # ... and the note explains that `class` is the gold value, not a label
    assert "`class`\nis that row's GOLD VALUE" in prompt
    assert "`ENTITY` (which value is\nbeing asked for)" in prompt


def test_the_dict_format_renders_the_other_note(ages, agent, monkeypatch):
    monkeypatch.setattr(config, "Example_format", "dict")
    prompt = _prompt(ages, agent, "python", _py(YEARS_SOURCE))
    assert "one per line as `[GOLD w=WEIGHT] {dict}`" in prompt
    assert "[[87, 'years'] w=" in prompt


def test_the_free_text_schema_menu_is_suppressed(ages, agent):
    """NOTE is declared symbolic, so the full schema block would enumerate one
    'domain value' per note and dwarf the sample."""
    assert config.Schema_format == "full"
    prompt = _prompt(ages, agent, "python", _py(YEARS_SOURCE))
    assert "Attribute schema" not in prompt
    assert "| attribute | kind | domain |" not in prompt


# ---------------------------------------------------------------------------
# rule construction
# ---------------------------------------------------------------------------

def test_python_arm_builds_a_working_sandboxed_rule(ages, agent):
    pytest.importorskip("smolagents")
    train, _ = ages
    config.Rule_body = "python"
    agent(_py(YEARS_SOURCE))
    rule, p, n = L.build_extract_rule(train, train)
    assert isinstance(rule, extract.ExtractRule)
    assert rule.source == YEARS_SOURCE.strip()
    assert rule.extract({"NOTE": "a 87-year-old man"}) == ["87", "years"]
    assert rule.extract({"NOTE": "a 6-month-old"}) is None
    # 8 rows right, the decoy row wrong, the month row abstains -- as for the
    # equivalent pool rule (tests/test_extract.py)
    assert (p, n) == (8.0, 1.0)
    assert (rule.nposx, rule.nnegx) == (8.0, 1.0)
    assert rule.shyp_line().startswith("exrule ")


def test_regex_arm_builds_a_regex_rule_from_the_spec(ages, agent):
    train, _ = ages
    config.Rule_body = "regex"
    agent(_rx(r"(\d+)-year-old", flags="i", template=[1, "years"]))
    rule, p, n = L.build_extract_rule(train, train)
    assert isinstance(rule, extract.RegexRule)
    assert rule.spec() == {"pattern": r"(\d+)-year-old", "flags": "i",
                           "template": [1, "years"]}
    assert rule.extract({"NOTE": "a 87-YEAR-OLD man"}) == ["87", "years"]
    assert (p, n) == (8.0, 1.0)
    assert (rule.nposx, rule.nnegx) == (8.0, 1.0)
    assert rule.shyp_line().startswith("rerule ")


def test_regex_arm_accepts_a_null_template(ages, agent):
    train, _ = ages
    config.Rule_body = "regex"
    agent(_rx(r"(\d+)-year-old"))
    rule, _p, _n = L.build_extract_rule(train, train)
    assert rule.template is None
    assert rule.extract({"NOTE": "a 87-year-old man"}) == "87"


def test_markdown_fences_are_stripped_from_both_arms(ages, agent):
    pytest.importorskip("smolagents")
    train, _ = ages
    config.Rule_body = "python"
    agent(_py("```python\n" + YEARS_SOURCE + "```"))
    rule, _p, _n = L.build_extract_rule(train, train)
    assert rule.source == YEARS_SOURCE.strip()
    config.Rule_body = "regex"
    agent(_rx("```\n(\\d+)-year-old\n```", template=[1, "years"]))
    rule, _p, _n = L.build_extract_rule(train, train)
    assert rule.pattern == r"(\d+)-year-old"


def test_no_tools_are_offered_in_either_arm(ages, agent):
    """Every tool in the module counts (p, n) by class membership, which a
    value-returning rule has none of -- so extraction runs tool-less whatever
    -T says, and the prompt promises no tools."""
    train, _ = ages
    config.Tool_budget = 6
    a = agent(_rx(r"(\d+)-year-old", template=[1, "years"]))
    config.Rule_body = "regex"
    L.build_extract_rule(train, train)
    assert a.kwargs == [{}]                    # no deps, no usage_limits
    assert "tool" not in a.prompts[0].lower()


# ---------------------------------------------------------------------------
# retry and fallback
# ---------------------------------------------------------------------------

def test_a_bad_response_is_retried_then_the_rule_abstains(ages, agent):
    train, _ = ages
    config.Rule_body = "python"
    a = agent(_py("def covers(x):\n    return True\n"))   # wrong entry point
    rule, p, n = L.build_extract_rule(train, train)
    assert len(a.prompts) == 3                            # 3 attempts, then give up
    assert (p, n) == (0.0, 0.0)
    assert rule.source == L._EXTRACT_NOOP_SOURCE
    assert extract.extract_evaluate(rule, train) == (0.0, 0.0)
    for ex in train:                                      # fires nowhere
        assert rule.extract(extract.inst_dict(ex.inst)) is None


def test_a_retry_that_succeeds_keeps_the_second_answer(ages, agent):
    train, _ = ages
    config.Rule_body = "regex"
    a = agent(_rx(""),                                    # empty pattern: rejected
              _rx(r"(\d+)-year-old", template=[1, "years"]))
    rule, p, n = L.build_extract_rule(train, train)
    assert len(a.prompts) == 2
    assert isinstance(rule, extract.RegexRule) and (p, n) == (8.0, 1.0)


def test_a_bad_regex_flag_is_a_retry_not_a_fatal(ages, agent):
    """extract.RegexRule calls fatal() on an unknown flag; the builder has to
    validate first, or one bad model answer would end the whole run."""
    train, _ = ages
    config.Rule_body = "regex"
    agent(_rx(r"(\d+)-year-old", flags="z", template=[1, "years"]))
    rule, p, n = L.build_extract_rule(train, train)       # no SystemExit
    assert (p, n) == (0.0, 0.0) and rule.source == L._EXTRACT_NOOP_SOURCE


def test_an_uncompilable_pattern_is_a_retry(ages, agent):
    train, _ = ages
    config.Rule_body = "regex"
    agent(_rx(r"(unclosed", template=[1]))
    rule, p, n = L.build_extract_rule(train, train)
    assert (p, n) == (0.0, 0.0) and rule.source == L._EXTRACT_NOOP_SOURCE


def test_an_agent_exception_is_retried(ages, agent):
    train, _ = ages
    config.Rule_body = "regex"
    a = agent(RuntimeError("model is down"),
              _rx(r"(\d+)-year-old", template=[1, "years"]))
    rule, p, n = L.build_extract_rule(train, train)
    assert len(a.prompts) == 2 and (p, n) == (8.0, 1.0)


def test_build_without_a_model_is_fatal(ages, agent):
    train, _ = ages
    config.Rule_llm_model = ""
    with pytest.raises(SystemExit):
        L.build_extract_rule(train, train)


def test_an_unknown_rule_body_is_fatal(ages, agent):
    train, _ = ages
    config.Rule_body = "lisp"
    agent(_py(YEARS_SOURCE))
    with pytest.raises(SystemExit):
        L.build_extract_rule(train, train)


# ---------------------------------------------------------------------------
# the boosting dispatch (soft_fit's extraction branch picks the builder)
# ---------------------------------------------------------------------------

def test_soft_fit_boosts_with_the_llm_extraction_builder(ages, agent, monkeypatch):
    """-R on an extraction dataset routes soft_fit to build_extract_rule; the
    pool path is not touched (Rule_pool is None, so it would crash if used)."""
    from slipper import boosting
    train, _ = ages
    config.Rule_body = "regex"
    config.Slipper_folds = 0
    config.N_boost = 2
    a = agent(_rx(r"(\d+)-year-old", template=[1, "years"]),
              _rx(r"(\d+)-month-old", template=[1, "months"]))
    monkeypatch.setattr(config, "Sample_size", 4)
    hyp = boosting.soft_model(train, None)
    assert len(a.prompts) == 2                       # one LLM call per round
    assert [r.spec()["template"] for r in hyp.rules] == [[1, "years"],
                                                         [1, "months"]]
    # the two rules together get 9 of 10 rows; only the decoy row stays wrong
    assert extract.extract_counts(hyp, train) == (9, 0, 10)


def test_soft_fit_adds_nothing_when_the_llm_only_abstains(ages, agent, monkeypatch):
    """The fallback's (0, 0) loses to the abstain baseline, so a round that
    exhausts its retries adds no rule -- but, the builder being stochastic, it
    does not end training either: every round of the -n budget still runs."""
    from slipper import boosting
    train, _ = ages
    config.Rule_body = "regex"
    config.Slipper_folds = 0
    config.N_boost = 3
    a = agent(_rx(""))                               # never valid
    monkeypatch.setattr(config, "Sample_size", 4)
    hyp = boosting.soft_model(train, None)
    assert hyp.rules == []
    assert len(a.prompts) == 9                       # 3 rounds x 3 attempts
    assert extract.extract_counts(hyp, train) == (0, 10, 10)


def test_a_losing_llm_round_is_discarded_and_the_loop_continues(ages, agent,
                                                                monkeypatch):
    """A rule that fires and is wrong everywhere loses to the abstain baseline.
    On the --rule-pool path that ends training (the same loser would lose
    forever); on the stochastic LLM path the round is discarded and the next
    round still gets its call -- otherwise one bad answer truncates the whole
    ensemble."""
    from slipper import boosting
    train, _ = ages
    config.Rule_body = "regex"
    config.Slipper_folds = 0
    config.N_boost = 2
    a = agent(_rx(r"(\d+)-year-old", template=[1, "furlongs"]),   # 0 right, 9 wrong
              _rx(r"(\d+)-year-old", template=[1, "years"]))      # 8 right, 1 wrong
    monkeypatch.setattr(config, "Sample_size", 4)
    hyp = boosting.soft_model(train, None)
    assert len(a.prompts) == 2                       # both rounds ran
    assert [r.spec()["template"] for r in hyp.rules] == [[1, "years"]]
    assert extract.extract_counts(hyp, train) == (8, 1, 10)


# ---------------------------------------------------------------------------
# the registry and the config knob
# ---------------------------------------------------------------------------

def test_both_arms_are_registered_with_their_output_model():
    assert set(L._RULE_BODIES) == {"python", "regex"}
    assert set(L._RULE_BODIES.values()) == set(L._EXTRACT_PROMPT_VARIANTS)
    assert L._EXTRACT_PROMPT_VARIANTS["extract_python"][1] is L.ExtractRuleSpec
    assert L._EXTRACT_PROMPT_VARIANTS["extract_regex"][1] is L.RegexRuleSpec


def test_the_extract_template_carries_every_field():
    """A missing field would raise KeyError mid-run, against a live model."""
    out = L._EXTRACT_PROMPT.format(
        header="H", task_definition="TD", task_info="TI", schema="SCH",
        sample_note="SN", examples="EX")
    for v in ("H", "TD", "SCH", "EX", "SN"):
        assert v in out


def test_rule_body_is_a_serialized_config_knob(tmp_path):
    d = config.config_dict()
    assert d["Rule_body"] == "python"
    config.Rule_body = "regex"
    p = str(tmp_path / "c.yaml")
    config.save_yaml(p)
    config.Rule_body = "python"
    assert config.load_yaml(p)["Rule_body"] == "regex"
    assert config.Rule_body == "regex"


# ---------------------------------------------------------------------------
# CLI flag validation (each of these exits before any model is contacted)
# ---------------------------------------------------------------------------

def _run(*args, cwd):
    return subprocess.run([sys.executable, "-m", "slipper", *args],
                          capture_output=True, text=True, cwd=cwd, timeout=180)


def test_cli_rejects_an_unknown_rule_body(tmp_path):
    stem = _write(tmp_path, "ages.json", SPEC)
    proc = _run("--rule-body", "lisp", "-R", "fake/model", "-i0", "-n1", stem,
                cwd=REPO_ROOT)
    assert proc.returncode != 0
    assert "--rule-body must be one of python, regex" in proc.stdout + proc.stderr


@pytest.mark.parametrize("body", ["python", "regex", "REGEX"])
def test_cli_accepts_both_rule_bodies(tmp_path, body):
    """The flag parses and echoes (case-insensitively) without reaching the
    learner: --rule-pool with it is the error, so the run stops at validation."""
    stem = _write(tmp_path, "ages.json", SPEC)
    pool = _write(tmp_path, "pool.json",
                  [{"pattern": r"(\d+)-year-old", "template": [1, "years"]}])
    proc = _run("--rule-body", body, "--rule-pool", pool, "-R", "fake/model",
                "-i0", "-n1", stem, cwd=REPO_ROOT)
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "option: extraction rule body = %s" % body.lower() in out
    assert "pass one, not both" in out


def test_cli_extraction_still_needs_one_rule_builder(tmp_path):
    stem = _write(tmp_path, "ages.json", SPEC)
    proc = _run("--rule-body", "regex", "-i0", "-n1", stem, cwd=REPO_ROOT)
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "needs --rule-pool" in out and "-R MODEL" in out


def test_cli_help_documents_the_flag(tmp_path):
    proc = _run("-h", cwd=REPO_ROOT)
    assert proc.returncode == 0
    assert "--rule-body=NAME" in proc.stdout


def test_cli_deterministic_path_stdout_is_unchanged(tmp_path):
    """The --rule-body default must not add a line to the no-LLM path."""
    stem = _write(tmp_path, "ages.json", SPEC)
    pool = _write(tmp_path, "pool.json",
                  [{"pattern": r"(\d+)-year-old", "template": [1, "years"]}])
    proc = _run("--rule-pool", pool, "-i0", "-n3", stem, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "rule body" not in proc.stdout
    assert "Train exact match: 8 of 10" in proc.stdout
