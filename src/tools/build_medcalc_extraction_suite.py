"""build_medcalc_extraction_suite.py - MedCalc `Relevant Entities` -> extraction JSON.

The classification suite (``build_medcalc_suite.py``) learns ``note -> calculator``.
This one builds the *other half* of the pipeline: ``(note, entity key) -> value``,
one instance per ``(row, entity)`` pair, so the gold value rides on the example
label and neither ``Example`` nor ``Aval`` has to grow a slot.

``Relevant Entities`` is a **Python dict repr string**, not JSON (``json.loads``
fails on the single quotes and ``True``/``False``), so it is parsed with
``ast.literal_eval``; the build starts by parsing every row of both splits and
printing the value-shape histogram, which is the cheapest tripwire for a silent
upstream format change.

The column is a *silver* gold standard: for the rule-based calculators, negative
criteria are sometimes listed as ``False`` and sometimes just omitted, so the key
set varies row to row and "missing" has no defined meaning.  This build therefore
keeps only the calculators whose **every** row (train and test) shares one
identical key set.  The count is asserted, not
inferred: if it moves, the build stops rather than silently rescoping the eval.

Labels are ``repr()`` of the parsed value, **raw and uncanonicalized** -- unit
strings, casing and number formats exactly as shipped (dataset gotcha
f).  Canonicalization belongs in a separate scoring module so that train and test
go through the same normalizer and the learner cannot smuggle normalization into
its rule bodies.

Rows are split by ``split_train`` from the classification suite on the full
55-class gold list and only *then* filtered to the fixed-schema calculators, so a
note in the extraction train pool is in the classification train pool too.

Output is one file per entity key (not per calculator): ``<entity_slug>.json``
holds every instance of that key, pooled across every calculator whose schema
contains it, which is the unit the learner is boosted on.  ``dev/`` and
``final/`` share a train set and differ only in the test set, exactly as in the
one-vs-rest suite.

Run from the repo root::

    uv run --group medcalc python src/tools/build_medcalc_extraction_suite.py
"""

import argparse
import ast
import collections
import hashlib
import json
import os
import re
import statistics

from build_medcalc_suite import HF_REPO, HF_REVISION, split_train

# 33, not the 34 of an earlier count: that count was computed on the test
# split alone.  The one difference is calc 15 (Child-Pugh), whose 20 test rows do
# share a key set while its train rows do not -- 17 of 82 omit `Encephalopathy`
# and 12 omit `Ascites`.  Admitting it would mean inventing gold values for those
# omitted keys, which is the exact noise this restriction exists to exclude.
# A mismatch here is a scope change, not a rounding difference: the build stops.
EXPECT_FIXED = 33

NONALNUM = re.compile(r"[^a-z0-9]+")

ATTRIBUTES = [{"name": "NOTE", "kind": "symbol"},
              {"name": "CALC_ID", "kind": "symbol"},
              {"name": "ENTITY", "kind": "symbol"}]


def slugify(key):
    """Filename-safe form of an entity key: casefolded, non-alphanumerics collapsed."""
    return NONALNUM.sub("_", key.casefold()).strip("_")


def value_shape(v):
    """The dataset report's shape vocabulary for one gold value (bool before number)."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, list):
        return "%d-list" % len(v)
    if isinstance(v, str):
        return "str"
    return "number"


def parse_entities(column):
    """literal_eval every row; an unparseable row raises here rather than being skipped."""
    return [ast.literal_eval(s) for s in column]


def key_sets(ents, gold):
    """Calculator -> the set of distinct entity-key tuples its rows use."""
    sets = collections.defaultdict(set)
    for e, cid in zip(ents, gold):
        sets[cid].add(tuple(sorted(e)))
    return sets


def make_instances(notes, gold, ents, ids, schema):
    """Entity key -> instances, one per (row, key) over the given row indices."""
    out = collections.defaultdict(list)
    for i in ids:
        if gold[i] in schema:
            for key in schema[gold[i]]:
                out[key].append({"values": [notes[i], "calc=%d" % gold[i], key],
                                 "label": repr(ents[i][key]), "weight": 1.0})
    return dict(out)


def content_hash(per_key):
    """Digest of every instance in key order -- the manifest's build fingerprint."""
    h = hashlib.sha256()
    for key in sorted(per_key):
        for r in per_key[key]:
            h.update(json.dumps([r["values"], r["label"]]).encode())
            h.update(b"\n")
    return h.hexdigest()


def emit(out_dir, train, test):
    """Write one file per entity key, holding that key's train and test instances."""
    os.makedirs(out_dir, exist_ok=True)
    for key in sorted(train):
        with open(os.path.join(out_dir, "%s.json" % slugify(key)), "w") as f:
            json.dump({"task": "extract", "attributes": ATTRIBUTES,
                       "train": train[key], "test": test[key]}, f)


def describe(name, per_key):
    """One summary line: total instances and the per-key-file min/median/max."""
    n = sorted(len(v) for v in per_key.values())
    print("  %-10s %6d instances   per key file: min %d  median %g  max %d"
          % (name, sum(n), n[0], statistics.median(n), n[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="datasets/medcalc_extract",
                    help="output directory")
    ap.add_argument("--cap", type=int, default=150, help="max training rows per calculator")
    ap.add_argument("--eval-per-class", type=int, default=10,
                    help="dev holdout rows per calculator (balanced, like the final test)")
    ap.add_argument("--seed", type=int, default=1,
                    help="RNG seed for the split and the cap; must match the classification suite")
    ap.add_argument("--force", action="store_true", help="rebuild even if a matching manifest exists")
    ns = ap.parse_args()

    manifest_path = os.path.join(ns.out, "manifest.json")
    args = {"revision": HF_REVISION, "seed": ns.seed, "cap": ns.cap,
            "eval_per_class": ns.eval_per_class}
    if not ns.force and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            if json.load(f)["args"] == args:
                print("%s is already built with these args; pass --force to rebuild" % ns.out)
                return

    # imported late so --help and the already-built exit above work without the
    # optional `medcalc` group installed.
    import datasets
    data = datasets.load_dataset(HF_REPO, revision=HF_REVISION)
    gold = {s: [int(c) for c in data[s]["Calculator ID"]] for s in ("train", "test")}
    ents = {s: parse_entities(data[s]["Relevant Entities"]) for s in ("train", "test")}

    shapes = {s: collections.Counter(value_shape(v) for e in ents[s] for v in e.values())
              for s in ents}
    print("value shapes (%d train rows, %d test rows, 0 unparseable):"
          % (len(ents["train"]), len(ents["test"])))
    for shape in sorted(set(shapes["train"]) | set(shapes["test"])):
        print("  %-8s train %6d   test %5d" % (shape, shapes["train"][shape], shapes["test"][shape]))

    sets = key_sets(ents["train"] + ents["test"], gold["train"] + gold["test"])
    schema = {cid: next(iter(ks)) for cid, ks in sorted(sets.items()) if len(ks) == 1}
    if len(schema) != EXPECT_FIXED:
        names = dict(zip(gold["test"], data["test"]["Calculator Name"]))
        print("fixed-schema calculators over train+test: %d, expected %d" % (len(schema), EXPECT_FIXED))
        for cid in sorted(sets):
            print("  calc %-3d %2d key set(s)  %s" % (cid, len(sets[cid]), names[cid]))
        raise SystemExit("scope changed; not building -- decide the restriction first")

    pool_ids, eval_ids = split_train(gold["train"], ns.cap, ns.eval_per_class, ns.seed)
    train = make_instances(data["train"]["Patient Note"], gold["train"], ents["train"],
                           pool_ids, schema)
    splits = {"dev": make_instances(data["train"]["Patient Note"], gold["train"], ents["train"],
                                    eval_ids, schema),
              "final": make_instances(data["test"]["Patient Note"], gold["test"], ents["test"],
                                      range(len(gold["test"])), schema)}
    slugs = {slugify(k): k for k in train}
    assert len(slugs) == len(train), "entity-key slugs collide"
    for name, test in splits.items():
        emit(os.path.join(ns.out, name), train, test)

    manifest = {
        "repo": HF_REPO, "args": args, "task": "extract",
        "calculators": sorted(schema),
        "schemas": {str(c): list(schema[c]) for c in sorted(schema)},
        "excluded": {str(cid): len(sets[cid]) for cid in sorted(sets) if cid not in schema},
        "excluded_note": "value = distinct entity-key sets over train+test. "
                         "says 34 because it counted on test alone; the extra one is calc 15 "
                         "(Child-Pugh), fixed on test but ragged in train (17 rows omit "
                         "Encephalopathy, 12 omit Ascites).",
        "slugs": slugs,
        "shape_histogram": {s: dict(sorted(shapes[s].items())) for s in shapes},
        "instance_counts": {slug: {"train": len(train[key]),
                                   "dev": len(splits["dev"][key]),
                                   "final": len(splits["final"][key])}
                            for slug, key in sorted(slugs.items())},
        "content_hashes": {"train": content_hash(train),
                           "dev_test": content_hash(splits["dev"]),
                           "final_test": content_hash(splits["final"])},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print("%s: %d calculators, %d entity keys" % (ns.out, len(schema), len(train)))
    describe("train", train)
    describe("dev test", splits["dev"])
    describe("final test", splits["final"])


if __name__ == "__main__":
    main()
