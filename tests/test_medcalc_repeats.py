"""Offline checks for training isolation, repeat identity, and pooled scoring."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("medcalc_repeats", REPO / "scripts/run_medcalc_repeats.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_top_keys_use_only_train_counts_with_stable_ties():
    counts = {"b": {"train": 10, "final": 1000}, "a": {"train": 10, "final": 0},
              "c": {"train": 9, "final": 9999}}
    assert R.choose_keys(counts, 2) == ["a", "b"]


def test_frozen_inputs_cannot_be_silently_reused_after_a_change(tmp_path):
    path = tmp_path / "data.json"
    R.write_json(path, {"train": [1]})
    manifest = {"inputs": {"data.json": R.file_hash(path)}, "sources": {}}
    manifest["identity"] = R.digest(manifest)
    R.verify_inputs(tmp_path, manifest)
    R.write_json(path, {"train": [2]})
    with pytest.raises(AssertionError, match="frozen file changed"):
        R.verify_inputs(tmp_path, manifest)


def test_worker_trains_without_evaluation_symbols_and_matches_cli(tmp_path, monkeypatch):
    from slipper import boosting, config, dataset, names
    saved = config.config_dict()
    cell = dict(id="classification/slipper_full/cap1/seed03/calc_2", kind="classification",
                condition="slipper_full", seed=3, cap=1, target="2", input="bundle.json")
    bundle = {"attributes": [{"name": "word", "kind": "set"}],
              "train": [{"values": [["body", v]]} for v in (["alpha"] * 8 + ["beta"] * 12)],
              "train_gold": [2] * 8 + [3] * 12,
              "dev": [{"values": [["heldout-only"]]}, {"values": [["body", "alpha"]]}], "dev_gold": [2, 2],
              "final": [{"values": [["body", "beta"]]}], "final_gold": [3]}
    R.write_json(tmp_path / "bundle.json", bundle)
    manifest = {"conditions": R.make_conditions("unused"), "cells": [cell], "identity": "test",
                "inputs": {"bundle.json": R.file_hash(tmp_path / "bundle.json")}}
    R.write_json(tmp_path / "manifest.json", manifest)
    train_model = boosting.soft_model
    calls = []

    def checked(train, test):
        assert test is None
        assert {x.name for x in names.Names[0].values} == {"body", "alpha", "beta"}
        calls.append(len(train))
        return train_model(train, test)

    monkeypatch.setattr(boosting, "soft_model", checked)
    attempt = tmp_path / "attempt"
    try:
        R.worker(tmp_path, cell["id"], attempt)
        assert calls == [20]
        assert {x.name for x in names.Names[0].values} == {"body", "alpha", "beta"}
        result = R.read_json(attempt / "result.json")
        assert result["splits"]["dev"]["n"] == 2
        assert result["splits"]["final"]["n"] == 1
        reference = tmp_path / "reference.json"
        R.write_json(reference, R.training_spec(bundle, cell))
        subprocess.run([sys.executable, "-m", "slipper", "--config", str(attempt / "resolved_config.yaml"),
                        "--seed", "3", str(reference)], cwd=REPO, capture_output=True, text=True, check=True)
        assert (attempt / "model.shyp").read_text() == (tmp_path / "reference.shyp").read_text()
    finally:
        for key,value in saved.items(): setattr(config,key,value)
        dataset.reset_all()


def test_extraction_join_pools_values_before_computing_f1():
    from score_extraction import score
    bundles = {
        "age": {"dev": [{"values": ["note", "calc=2", "age"], "label": "48"}]},
        "sodium": {"dev": [{"values": ["note", "calc=2", "Sodium"], "label": "140"},
                            {"values": ["another", "calc=3", "Sodium"], "label": "130"}]}}
    rows = R.extraction_rows(bundles, {"age": [48], "sodium": [140, None]}, "dev")
    metrics = score(rows)
    assert metrics["rows"] == 2
    assert (metrics["tp"],metrics["fp"],metrics["fn"]) == (2,0,1)
    assert metrics["micro_f1"] == pytest.approx(0.8)
    assert metrics["exact_set"] == 0.5
