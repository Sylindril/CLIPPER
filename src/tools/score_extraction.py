"""score_extraction.py - set-level scoring for the MedCalc extraction suite.

``build_medcalc_extraction_suite.py`` shreds each patient note into one instance
per ``(row, entity key)`` pair and files those instances **by entity key**, so a
run of the learner produces one ``<slug>.pred`` per key and nothing anywhere
records which note a prediction came from.  MedCalc's own metrics, though, are
per *note*: the set-level precision/recall/F1 of MedMCP-Calc's EA formula and
the strict all-keys-right criterion of MedRaC (and the
MedCalc-Bench evaluation guidance).  This tool joins the shards back.

**The join is positional, and needs no ids.**  ``make_instances`` walks the rows
of a split in order and appends one instance per schema key, so within one
entity file the instances of a given calculator appear in row order.  The k-th
instance carrying ``CALC_ID = "calc=C"`` in *any* entity file is therefore the
k-th test row of calculator C, and ``(calc_id, occurrence_index)`` names that
note-row across every file.  ``cli.py`` writes ``.pred`` lines in test order, so
line i of ``<slug>.pred`` pairs with test instance i of ``<slug>.json``; the gold
column is re-checked against the dataset on every line rather than trusted.

Omitted keys are **not** treated as negatives: the gold set for a note-row is
exactly its calculator's schema keys, every one of which is a positive.  A key
whose pool was skipped therefore costs recall (it is a missing prediction), and
an abstaining rule costs recall but not precision -- abstention is not a wrong
answer, only a missing one.

``normalize.canon`` is the only comparator, the same one the learner scored
with, so units and number formats are folded identically on both sides.

    uv run python src/tools/score_extraction.py DATASET_DIR PRED_DIR
"""

import argparse
import ast
import collections
import glob
import json
import os

from slipper.normalize import canon


def parse_pred_line(line):
    """``predicted_repr TAB confidence TAB 0 TAB gold_repr`` -> (pred, gold).

    Both columns are ``repr`` output (cli.py:601); an unparseable one falls back
    to the raw string, which then simply fails to match, rather than crashing
    the whole scoring run on one malformed line."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 4:
        raise ValueError("malformed .pred line (want 4 tab-separated fields): %r" % line)
    return _lit(parts[0]), _lit(parts[3])


def _lit(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return s.strip()


MISSING = object()          # no .pred file for this key (distinct from abstain)


def read_entity(ds_path, pred_path):
    """One entity file -> [(row_id, key, gold, pred_or_MISSING)] in test order.

    ``pred_path`` may be absent (a key whose pool was skipped, or that was not
    run): the gold instances are still emitted, with no prediction."""
    with open(ds_path) as f:
        test = json.load(f)["test"]
    preds = None
    if pred_path and os.path.exists(pred_path):
        with open(pred_path) as f:
            preds = [parse_pred_line(l) for l in f if l.strip()]
        if len(preds) != len(test):
            raise SystemExit("%s has %d lines but %s has %d test rows"
                             % (pred_path, len(preds), ds_path, len(test)))
    seen = collections.Counter()
    out = []
    for i, row in enumerate(test):
        calc = row["values"][1]
        key = row["values"][2]
        seen[calc] += 1
        gold = _lit(row["label"])
        pred = MISSING
        if preds is not None:
            p, g = preds[i]
            if canon(g) != canon(gold):
                raise SystemExit("%s line %d gold %r does not match dataset gold %r "
                                 "-- the .pred is stale or misaligned"
                                 % (pred_path, i + 1, g, gold))
            pred = p
        out.append(((calc, seen[calc]), key, gold, pred))
    return out


def walk(ds_dir, pred_dir):
    """Yield (slug, was_run, [(row_id, key, gold, pred)]) per entity file.

    The entity slug is the file name; the ``key`` on an instance is the schema's
    display name for it, so a caller that wants to attribute an instance back to
    its entity file (per-key EM%, say) needs the slug this yields rather than the
    key ``collect`` files rows under."""
    for ds_path in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
        slug = os.path.basename(ds_path)[:-5]
        pred_path = os.path.join(pred_dir, slug + ".pred")
        yield slug, os.path.exists(pred_path), read_entity(ds_path, pred_path)


def collect(ds_dir, pred_dir):
    """row_id -> {key: (gold, pred)} over every entity file in the dataset dir."""
    rows = collections.defaultdict(dict)
    keys_run = set()
    for slug, was_run, instances in walk(ds_dir, pred_dir):
        if was_run:
            keys_run.add(slug)
        for row_id, key, gold, pred in instances:
            rows[row_id][key] = (gold, pred)
    return rows, keys_run


def score(rows):
    """Per-row P/R/F1 (macro), pooled P/R/F1 (micro), and strict set match."""
    tp = fp = fn = 0
    macro = [0.0, 0.0, 0.0]
    exact = 0
    for keys in rows.values():
        g = {(k, canon(gold)) for k, (gold, _) in keys.items()}
        p = {(k, canon(pred)) for k, (_, pred) in keys.items()
             if pred is not MISSING and pred is not None}
        hit = len(g & p)
        tp += hit
        fp += len(p) - hit
        fn += len(g) - hit
        pr = hit / len(p) if p else 0.0
        rc = hit / len(g) if g else 0.0
        macro[0] += pr
        macro[1] += rc
        macro[2] += (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0
        exact += (g == p)
    n = len(rows) or 1
    return {"rows": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "macro_p": macro[0] / n, "macro_r": macro[1] / n, "macro_f1": macro[2] / n,
            "micro_p": tp / (tp + fp) if tp + fp else 0.0,
            "micro_r": tp / (tp + fn) if tp + fn else 0.0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "exact_set": exact / n}


def per_calculator(rows):
    """The same strict/micro numbers split by calculator, for the table."""
    by = collections.defaultdict(dict)
    for (calc, occ), keys in rows.items():
        by[calc][(calc, occ)] = keys
    return {calc: score(sub) for calc, sub in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="dir of <slug>.json extraction datasets (dev/ or final/)")
    ap.add_argument("pred_dir", help="dir holding the <slug>.pred files of a run")
    ap.add_argument("--by-calculator", action="store_true",
                    help="also break the metrics out per calculator")
    ns = ap.parse_args()

    rows, keys_run = collect(ns.dataset_dir, ns.pred_dir)
    if not rows:
        raise SystemExit("no instances found under %s" % ns.dataset_dir)
    n_keys = len(glob.glob(os.path.join(ns.dataset_dir, "*.json")))
    s = score(rows)

    print("dataset  %s" % ns.dataset_dir)
    print("preds    %s   (%d of %d entity keys have a .pred)"
          % (ns.pred_dir, len(keys_run), n_keys))
    print("note-rows joined: %d   gold (key, value) pairs: %d"
          % (s["rows"], s["tp"] + s["fn"]))
    print()
    print("  %-28s %8s %8s %8s" % ("", "P", "R", "F1"))
    print("  %-28s %7.1f%% %7.1f%% %7.1f%%"
          % ("per-row, macro-averaged", 100 * s["macro_p"], 100 * s["macro_r"],
             100 * s["macro_f1"]))
    print("  %-28s %7.1f%% %7.1f%% %7.1f%%"
          % ("pooled, micro-averaged", 100 * s["micro_p"], 100 * s["micro_r"],
             100 * s["micro_f1"]))
    print()
    print("  %-28s %7.1f%%  (%d of %d note-rows, all keys right)"
          % ("strict exact set match", 100 * s["exact_set"],
             round(s["exact_set"] * s["rows"]), s["rows"]))
    print("  %-28s tp %d   fp %d   fn %d" % ("pooled counts", s["tp"], s["fp"], s["fn"]))

    if ns.by_calculator:
        print()
        print("  %-10s %6s %7s %7s %7s" % ("calc", "rows", "micro-P", "micro-R", "strict"))
        for calc, cs in sorted(per_calculator(rows).items(),
                              key=lambda kv: int(kv[0].split("=")[-1])):
            print("  %-10s %6d %6.1f%% %6.1f%% %6.1f%%"
                  % (calc, cs["rows"], 100 * cs["micro_p"], 100 * cs["micro_r"],
                     100 * cs["exact_set"]))


if __name__ == "__main__":
    main()
