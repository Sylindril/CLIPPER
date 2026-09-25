"""build_medcalc_suite.py - convert MedCalc-Bench-Verified to one-vs-rest slipper JSON.

MedCalc-Bench's official task is ``(note, question) -> answer``.  What we want
instead is the *derived* selection task ``Patient Note -> Calculator ID``: 55
classes, note-only.  ``Question`` is excluded by default because it uniquely
determines the calculator for all 116 distinct question strings, which would
reduce the task to a string lookup (``--include_question`` builds the degenerate
variant anyway, into a sibling ``*_q`` directory, as a ceiling check).

slipper2 is a two-class learner, so 55 classes become **55 one-vs-rest files per
directory**, emitted from one shared split: every file holds the same rows in
the same order and differs only in ``classes`` and each row's ``label``.  That
is what lets the downstream argmax runner join the 55 ``.pred`` files on row
order alone.

Two directories are built:

* ``dev/``   - train = capped training pool, test = a balanced holdout of the
  official train split (``--eval-per-class`` rows per calculator, mirroring the
  final test's balance so dev iteration predicts final behavior).  Iterate here.
* ``final/`` - the same train, test = the official balanced 1,100-row test split
  (20 rows per calculator).  Touch only for final numbers.

The note is featurized as a single lowercased bag-of-words ``WORDS`` attribute
of kind ``set`` -- the shape of the one existing text dataset, ``tdata/talks.json``,
which the loader, the schema rendering and the ``contains`` grammar ops already
support.  The vocabulary (document-frequency floor, then a top-N cap) is
computed on the training pool alone; eval and test are filtered with it, so no
test token ever reaches the learner's search space.

``--include-note`` additionally emits the untokenized note as a ``NOTE``
attribute of kind ``text`` (into a sibling ``*_text`` directory).  Only the LLM
path can read it -- the discrete grammar builds no conditions over a text
column -- so the two paths run off the same file: the grammar sees ``WORDS``,
the model sees the prose it was handicapped by ``WORDS`` for.

The train split is severely imbalanced (16-927 rows per calculator), so the
training pool is capped at ``--cap`` rows per calculator: rare classes keep
everything, the giants get subsampled, and discrete rule search stays tractable.

Run from the repo root::

    uv run --group medcalc python src/tools/build_medcalc_suite.py
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re

HF_REPO = "nsk7153/MedCalc-Bench-Verified"
# HF has no immutable version tags, so pin the commit.  This is the revision
# last modified 2026-04-22, i.e. the v1.0.7 data (v1.0.7 changed only the README
# and the prompt code on top of v1.0.6, so the parquet is the v1.0.6/v1.0.7 one).
HF_REVISION = "591157b3343b4dda247294f9d929da4c75026fa8"

WORD = re.compile(r"[a-z]+")


def tokenize(text):
    """Lowercased alphabetic tokens of length >= 2, deduplicated and sorted."""
    return sorted({t for t in WORD.findall(text.lower()) if len(t) >= 2})


def build_vocab(docs, df_min, vocab_max):
    """Tokens with document frequency >= df_min, capped at the vocab_max most frequent."""
    df = collections.Counter(t for doc in docs for t in doc)
    keep = [(-n, t) for t, n in df.items() if n >= df_min]
    keep.sort()
    return {t for _, t in keep[:vocab_max]}


def split_train(gold, cap, eval_n, seed):
    """Split the official train split into a capped training pool and a
    *balanced* dev holdout of exactly ``eval_n`` rows per calculator.

    Returns ``(pool_ids, eval_ids)`` as row indices in original dataset order.
    The holdout is balanced so dev iteration predicts behavior on the balanced
    official test (an imbalanced holdout rewards base-rate priors the final
    test then punishes).  The cost is real for rare classes: the smallest
    (FeverPAIN, 16 rows) keeps only ``16 - eval_n`` training rows.
    """
    by_class = collections.defaultdict(list)
    for i, cid in enumerate(gold):
        by_class[cid].append(i)

    rng = random.Random(seed)
    pool, held = [], []
    for cid in sorted(by_class):          # sorted -> stable across runs
        ids = by_class[cid][:]
        if len(ids) <= eval_n:
            raise SystemExit("calculator %d has only %d rows; cannot hold out %d"
                             % (cid, len(ids), eval_n))
        rng.shuffle(ids)
        held += ids[:eval_n]
        pool += ids[eval_n:][:cap]
    return sorted(pool), sorted(held)


def make_rows(split, ids, columns, vocabs, raw_columns):
    """One slipper example per row index, with a placeholder label.

    ``columns`` are vocabulary-filtered token lists (kind ``set``);
    ``raw_columns`` are the source strings verbatim (kind ``text``), appended
    after them so the value order matches the attribute order in ``emit``.
    """
    text = {attr: split[col] for attr, col in columns.items()}
    raw = {attr: split[col] for attr, col in raw_columns.items()}
    return [{"values": ([[t for t in tokenize(text[a][i]) if t in vocabs[a]] for a in columns]
                        + [raw[a][i] for a in raw_columns]),
             "label": None, "weight": 1.0}
            for i in ids]


def content_hash(rows):
    """Digest of a split's row content in order -- the manifest's build fingerprint."""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r["values"]).encode())
        h.update(b"\n")
    return h.hexdigest()


def emit(out_dir, attributes, train, train_gold, test, test_gold, calculators):
    """Write the 55 one-vs-rest files plus the runner's join metadata.

    The row dicts are shared and relabelled in place between dumps, so all 55
    files are byte-identical apart from ``classes`` and the ``label`` fields.
    """
    os.makedirs(out_dir, exist_ok=True)
    for cid in sorted(calculators):
        classes = ["calc=%d" % cid, "calc=rest"]
        for rows, gold in ((train, train_gold), (test, test_gold)):
            for row, g in zip(rows, gold):
                row["label"] = classes[0] if g == cid else classes[1]
        with open(os.path.join(out_dir, "calc_%d.json" % cid), "w") as f:
            json.dump({"classes": classes, "attributes": attributes,
                       "train": train, "test": test}, f)

    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(test_gold, f)
    with open(os.path.join(out_dir, "calculators.json"), "w") as f:
        json.dump(calculators, f, indent=1, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="datasets/medcalc_suite",
                    help="output directory")
    ap.add_argument("--cap", type=int, default=150, help="max training rows per calculator")
    ap.add_argument("--eval-per-class", type=int, default=10,
                    help="dev holdout rows per calculator (balanced, like the final test)")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for the 85/15 split and the cap")
    ap.add_argument("--df-min", type=int, default=5, help="drop tokens in fewer than this many training notes")
    ap.add_argument("--vocab-max", type=int, default=5000, help="keep at most this many tokens, most frequent first")
    ap.add_argument("--include_question", "--include-question", action="store_true",
                    help="also emit a QUESTION_WORDS attribute (into OUT_q); the question "
                         "determines the calculator outright, so this is a ceiling check only")
    ap.add_argument("--include_note", "--include-note", action="store_true",
                    help="also emit a NOTE attribute of kind text -- the raw, untokenized "
                         "Patient Note (into OUT_text); readable only by the LLM path")
    ap.add_argument("--note_only", "--note-only", action="store_true",
                    help="emit ONLY the NOTE text attribute, no WORDS (into OUT_noteonly). "
                         "Given both, the LLM writes WORDS rules and never reads the prose "
                         "(measured); this variant forces the representation question")
    ap.add_argument("--force", action="store_true", help="rebuild even if a matching manifest exists")
    ns = ap.parse_args()

    root = (ns.out + ("_q" if ns.include_question else "")
            + ("_noteonly" if ns.note_only else "_text" if ns.include_note else ""))
    manifest_path = os.path.join(root, "manifest.json")
    args = {"revision": HF_REVISION, "seed": ns.seed, "cap": ns.cap,
            "eval_per_class": ns.eval_per_class, "df_min": ns.df_min,
            "vocab_max": ns.vocab_max, "include_question": ns.include_question,
            "include_note": ns.include_note, "note_only": ns.note_only}
    if not ns.force and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            if json.load(f)["args"] == args:
                print("%s is already built with these args; pass --force to rebuild" % root)
                return

    # imported late so --help and the already-built exit above work without the
    # optional `medcalc` group installed.  HF caches the download under
    # ~/.cache/huggingface, so repeated builds don't re-fetch.
    import datasets
    data = datasets.load_dataset(HF_REPO, revision=HF_REVISION)
    gold = {s: [int(c) for c in data[s]["Calculator ID"]] for s in ("train", "test")}
    pool_ids, eval_ids = split_train(gold["train"], ns.cap, ns.eval_per_class, ns.seed)

    # vocabularies come from the training pool only -- eval and test are filtered
    # with them, never the other way round.
    columns = {} if ns.note_only else {"WORDS": "Patient Note"}
    if ns.include_question:
        columns["QUESTION_WORDS"] = "Question"
    raw_columns = {"NOTE": "Patient Note"} if ns.include_note or ns.note_only else {}
    vocabs = {}
    for attr, col in columns.items():
        text = data["train"][col]
        vocabs[attr] = build_vocab([tokenize(text[i]) for i in pool_ids], ns.df_min, ns.vocab_max)

    train = make_rows(data["train"], pool_ids, columns, vocabs, raw_columns)
    train_gold = [gold["train"][i] for i in pool_ids]
    splits = {
        "dev": (make_rows(data["train"], eval_ids, columns, vocabs, raw_columns),
                [gold["train"][i] for i in eval_ids]),
        "final": (make_rows(data["test"], range(len(gold["test"])), columns, vocabs,
                            raw_columns),
                  gold["test"]),
    }

    calculators = {int(cid): {"name": name, "category": cat}
                   for cid, name, cat in zip(data["test"]["Calculator ID"],
                                             data["test"]["Calculator Name"],
                                             data["test"]["Category"])}
    attributes = ([{"name": a, "kind": "set"} for a in columns]
                  + [{"name": a, "kind": "text"} for a in raw_columns])
    for name, (test, test_gold) in splits.items():
        emit(os.path.join(root, name), attributes, train, train_gold, test, test_gold, calculators)

    counts = collections.Counter(train_gold)
    manifest = {
        "repo": HF_REPO, "args": args,
        "vocab_sizes": {a: len(v) for a, v in vocabs.items()},
        "splits": {"train": len(train), "dev_test": len(splits["dev"][0]),
                   "final_test": len(splits["final"][0])},
        "content_hashes": {"train": content_hash(train),
                           "dev_test": content_hash(splits["dev"][0]),
                           "final_test": content_hash(splits["final"][0])},
        "train_class_counts": {str(c): counts[c] for c in sorted(counts)},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    print("%s: %d calculators, %d train (%d-%d per class), %d dev test, %d final test"
          % (root, len(calculators), len(train), min(counts.values()), max(counts.values()),
             len(splits["dev"][0]), len(splits["final"][0])))
    for a, v in vocabs.items():
        print("  %-14s vocab %d (df >= %d, top %d)" % (a, len(v), ns.df_min, ns.vocab_max))


if __name__ == "__main__":
    main()
