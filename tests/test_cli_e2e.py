"""End-to-end CLI smoke tests: drive `python -m slipper` as a subprocess and
verify it converges on vote1 within a reasonable error budget."""
import os
import re
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
VOTE1_JSON = os.path.join(REPO_ROOT, "pdata", "vote1.json")


def needs_api(env_var):
    """Skip a test unless the named API-key env var is set and non-empty."""
    return pytest.mark.skipif(
        not os.environ.get(env_var),
        reason=f"requires {env_var} in the environment",
    )


# Signatures of "the API did not answer", as opposed to "the model answered
# badly".  Only the former justifies skipping -- an accuracy miss is a result.
_API_DOWN_RE = re.compile(
    r"AuthenticationError|PermissionDenied|InvalidAPIKey|API key not valid"
    r"|RateLimitError|ResourceExhausted|quota|ServiceUnavailable|Timeout"
    r"|ConnectionError|InternalServerError|503|429|401|403",
    re.IGNORECASE,
)
_LLM_CALLS_RE = re.compile(r"^LLM side info:\s+(\d+) calls", re.MULTILINE)


def _skip_if_api_unavailable(proc, rule_model):
    """Skip (don't fail) when the LLM path never got a usable answer.

    A missing *env var* is caught by @needs_api, but a key that is present and
    broken -- revoked, out of quota, rate-limited, unreachable -- gets past that
    and would otherwise surface as a bogus accuracy failure, since slipper
    degrades quietly: a call that fails yields a no-op rule rather than an error.
    Skipping only on an explicit API-failure signature (or on zero calls having
    been made at all) keeps a genuine accuracy regression a failure.
    """
    if not rule_model:
        return
    blob = (proc.stdout or "") + (proc.stderr or "")
    m = _LLM_CALLS_RE.search(blob)
    if m and int(m.group(1)) == 0:
        pytest.skip("LLM path made 0 calls -- API unavailable, not a regression")
    hit = _API_DOWN_RE.search(blob)
    if hit:
        pytest.skip("LLM API unavailable (%s) -- not a regression" % hit.group(0))


def _run_slipper(rule_model, error_budget_pct, timeout=180):
    cmd = [
        sys.executable, "-m", "slipper",
        "-n", "5", "-i", "0", "-R", rule_model, "-M", "50", "-L", "",
        "-v", "1", "--seed", "1",
        "pdata/vote1",
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=timeout,
    )
    _skip_if_api_unavailable(proc, rule_model)
    assert proc.returncode == 0, (
        f"slipper exited {proc.returncode}\nstderr:\n{proc.stderr}"
        f"\nstdout tail:\n{proc.stdout[-2000:]}"
    )
    m = re.search(r"^Test error rate:\s+([\d.]+)%", proc.stdout, re.MULTILINE)
    assert m, f"no 'Test error rate' line in stdout:\n{proc.stdout[-2000:]}"
    err_pct = float(m.group(1))
    assert err_pct < error_budget_pct, (
        f"test error {err_pct:.2f}% exceeds {error_budget_pct}% budget"
        f"\nstdout tail:\n{proc.stdout[-2000:]}"
    )


@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cli_vote1_discrete_under_5pct():
    _run_slipper(rule_model="", error_budget_pct=5.0)


@needs_api("GEMINI_API_KEY")
@pytest.mark.skipif(not os.path.exists(VOTE1_JSON), reason="needs pdata/vote1.json")
def test_cli_vote1_gemini_under_15pct():
    # Much looser budget than the discrete test.  --seed 1 pins slipper's RNG and
    # the weighted sample, but NOT the model's own sampling, so the error rate
    # moves run to run: observed 8.15% and 11.85% on the same command.  This is a
    # smoke test that the LLM path produces usable rules, not an accuracy
    # regression test -- a tight budget here only buys flaky failures.
    _run_slipper(rule_model="gemini/gemini-3.1-flash-lite",
                 error_budget_pct=15.0)


def test_oneshot_requires_a_model():
    """--oneshot is an LLM-only preset; without -R it must fail loudly rather
    than silently fall through to the discrete search."""
    proc = subprocess.run(
        [sys.executable, "-m", "slipper", "--oneshot", "pdata/vote1"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
    assert proc.returncode != 0
    assert "--oneshot requires an LLM model" in (proc.stdout + proc.stderr)


def _options_echo(*args):
    """Parse-time `option:` lines only -- run with a stem that fails to load, so
    nothing is learned and no LLM is contacted."""
    proc = subprocess.run(
        [sys.executable, "-m", "slipper", *args, "/nonexistent-stem"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
    return [l for l in proc.stdout.splitlines() if l.startswith("option:")]


def test_oneshot_still_presets_its_prompt_variant():
    """--oneshot installs the `oneshot` variant unless --prompt-variant is given.
    Regression: the preset used to test `Prompt_variant == "baseline"`, which
    silently stopped firing once task_variant_2 became the default."""
    echo = "\n".join(_options_echo("-R", "fake/model", "--oneshot"))
    assert "prompt variant oneshot" in echo


def test_oneshot_does_not_override_an_explicit_variant():
    echo = "\n".join(_options_echo(
        "-R", "fake/model", "--oneshot", "--prompt-variant", "task_variant_2"))
    assert "prompt variant task_variant_2" in echo


def test_count_coverage_can_be_switched_back_on():
    """Every changed default stays reachable from the command line."""
    assert any("add the base count_coverage tool" in l
               for l in _options_echo("-R", "fake/model", "--count-coverage"))
    assert any("drop the count_coverage_of_lambda_body tool" in l
               for l in _options_echo("-R", "fake/model", "--no-lambda-body-tool"))
