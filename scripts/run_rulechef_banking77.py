"""Run RuleChef's own BANKING77 benchmark on one of our training subsamples.

Their script (vendor/rulechef/benchmark_banking77.py) runs unchanged.  We patch
only its loader, so it receives our n training rows and the fixed 200 test rows;
its sampler, refinement loop and metric code all execute as published.  Their
sampler takes 5 rows per intent for synthesis and leaves the other n-25 rows as
the refinement pool.

We also record, without changing behaviour:
  - each test prediction, by wrapping rulechef.evaluation.evaluate_dataset;
  - the token usage of every chat-completion request.

Uncovered test rows are scored as the training subsample's majority intent, as
SLIPPER's default rule would score them (`err`); their own metric counts an
uncovered row as wrong (`err_strict`).

Runs in RuleChef's own interpreter (see vendor/rulechef/README.md):

    python scripts/run_rulechef_banking77.py --input cell.json --seed 1 --out DIR
"""
import argparse
import collections
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "rulechef"))
import benchmark_banking77 as bench  # their script, vendored unmodified

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# The configuration recorded in RuleChef's published BANKING77 results.
THEIR_CONFIG = ["--shots", "5", "--format", "regex", "--max-rules", "100", "--max-samples", "200",
                "--max-iterations", "15", "--agentic"]


def record_usage(calls):
    from openai.resources.chat.completions import Completions
    create = Completions.create

    def counted(self, *args, **kwargs):
        start = time.time()
        try:
            response = create(self, *args, **kwargs)
        except Exception as e:
            calls.append(dict(error=f"{type(e).__name__}: {e}", seconds=time.time() - start))
            raise
        usage = response.usage
        calls.append(dict(prompt_tokens=usage.prompt_tokens if usage else None,
                          completion_tokens=usage.completion_tokens if usage else None,
                          seconds=time.time() - start))
        return response

    Completions.create = counted


def record_test_predictions(predictions):
    import rulechef.evaluation as evaluation
    evaluate = evaluation.evaluate_dataset

    def recorded(rules, dataset, apply_rules_fn, *args, **kwargs):
        if dataset.name == "banking77_test":
            for item in dataset.get_all_training_data():
                out = apply_rules_fn(rules, item.input, dataset.task.type, dataset.task.text_field)
                predictions.append(dict(text=item.input["text"], gold=item.expected_output["label"],
                                        pred=(out or {}).get("label") or None))
        return evaluate(rules, dataset, apply_rules_fn, *args, **kwargs)

    evaluation.evaluate_dataset = recorded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    args = ap.parse_args()
    cell = json.loads(args.input.read_text())
    train, test, classes = cell["train"], cell["test"], cell["classes"]
    bench.load_banking77 = lambda: (train, test, classes)
    calls, predictions = [], []
    record_usage(calls)
    record_test_predictions(predictions)
    os.environ["OPENAI_API_KEY"] = os.environ["GEMINI_API_KEY"]
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "rulechef.json"
    sys.argv = ["benchmark_banking77.py", *THEIR_CONFIG, "--classes", ",".join(classes),
                "--test-limit", str(len(test)), "--seed", str(args.seed), "--model", args.model,
                "--base-url", GEMINI_OPENAI_URL, "--output", str(output)]
    start = time.time()
    bench.main()
    theirs = json.loads(output.read_text())
    assert len(predictions) == len(test), "test predictions were not recorded"
    majority = collections.Counter(r["label"] for r in train).most_common(1)[0][0]
    wrong = sum((p["pred"] or majority) != p["gold"] for p in predictions)
    result = dict(
        argv=sys.argv, model=args.model, seed=args.seed, n_train=len(train), fallback=majority,
        rulechef_version=importlib.metadata.version("rulechef"),
        err=100 * wrong / len(predictions),
        err_strict=100 * (1 - theirs["results"]["accuracy"]),
        coverage=100 * theirs["results"]["coverage"],
        rules=theirs["results"]["num_rules"],
        rules_by_iteration=[m["num_rules"] for m in theirs["iteration_metrics"]],
        llm_calls=sum("error" not in c for c in calls),
        llm_errors=sum("error" in c for c in calls),
        prompt_tokens=sum(c.get("prompt_tokens") or 0 for c in calls),
        completion_tokens=sum(c.get("completion_tokens") or 0 for c in calls),
        elapsed_s=time.time() - start)
    (args.out / "predictions.json").write_text(json.dumps(predictions, indent=1))
    (args.out / "calls.json").write_text(json.dumps(calls, indent=1))
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("err", "err_strict", "coverage", "rules",
                                             "rules_by_iteration", "llm_calls", "llm_errors")}))


if __name__ == "__main__":
    main()
