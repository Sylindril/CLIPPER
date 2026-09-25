"""scripts/run_learning_curve.py: an LLM account error (no credit, bad key)
stops the suite instead of producing plausible-looking no-op rows."""
import csv
import os
import sys
import types

import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts")
sys.path.insert(0, SCRIPTS)
import run_learning_curve as R
from slipper.errors import ACCOUNT_ERROR_EXIT


def _proc(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_cell_raises_suite_abort_on_account_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "subsample_train", lambda spec, ds, size, seed: {"train": []})
    monkeypatch.setattr(R, "load_source", lambda ds: {})
    calls = []

    def fake_run(flags, celljson):
        calls.append(flags)
        if "-i0" not in flags:          # step 1: discrete CV for bestj
            return _proc(stdout="// best result 0.1 after 5 rounds")
        return _proc(ACCOUNT_ERROR_EXIT, stderr="FATAL: LLM account error, stopping: "
                     "credit balance is too low")
    monkeypatch.setattr(R, "_run", fake_run)
    with pytest.raises(R.SuiteAbort, match="credit balance"):
        R.run_cell(str(tmp_path), "lab", ["-R", "x"], "ds", 250, 1)
    assert len(calls) == 2


def test_run_suite_stops_writes_no_row_and_exits_with_account_code(tmp_path, monkeypatch):
    """First abort cancels the queue; cells already done keep their rows, so a
    rerun of the same command resumes exactly the missing ones."""
    monkeypatch.setattr(R, "load_source", lambda ds: {"train": [0] * 1000})
    monkeypatch.setattr(R.SUITE, "datasets", ["a", "b", "c"], raising=False)
    ran = []

    def fake_cell(griddir, label, extra, ds, size, seed, *rest):
        ran.append(ds)
        if ds == "b":
            raise R.SuiteAbort("b size=250 seed=1: credit balance is too low")
        return 5, 12.5, 3, {}
    monkeypatch.setattr(R, "run_cell", fake_cell)
    with pytest.raises(SystemExit) as e:
        R.run_suite(str(tmp_path), "lab", ["-R", "x"], [250], 1, 1, ["a", "b", "c"])
    assert e.value.code == ACCOUNT_ERROR_EXIT
    assert ran == ["a", "b"]                       # "c" was cancelled, never ran
    rows = list(csv.DictReader(open(tmp_path / "results.csv")))
    assert [r["dataset"] for r in rows] == ["a"]    # no row for the failed cell
