"""csv_to_json.py - convert a raw CSV dataset to the slipper JSON format.

Companion to ``to_json.py`` (which handles legacy ``.names``/``.data``).  This
one reads an arbitrary CSV, collapses the chosen target column into a *two-class*
(one-vs-rest) problem, infers per-column attribute types, and writes the
``classes``/``attributes``/``train``/``test`` JSON that ``dataset.load_dataset``
expects.

One-vs-rest is the point: ``--positive VAL`` names the target value that becomes
the positive class; every other target value collapses into a single negative
class (``--neg-label``, default ``rest``).  This keeps things in the two-class
``slipper2`` world even when the source target has many values.

Column references (``--target``/``--drop``/``--keep``/``--continuous``/
``--symbolic``) accept either a header name or a 0-based integer index.  With
``--no-header`` the file has no header row and columns are named ``col0``,
``col1`` ... unless ``--names`` supplies names.

Type inference: a kept feature column is ``continuous`` iff every non-missing
value parses as a float, otherwise ``symbolic`` (with its observed value domain
recorded).  Override with ``--continuous`` / ``--symbolic``.

Example::

    python src/tools/csv_to_json.py test_suite/telco_churn.csv \\
        --target Churn --positive Yes --drop customerID \\
        --test-frac 0.33 --seed 1 -o test_suite/telco_churn.json
"""

import argparse
import csv
import json
import random
import sys

DEFAULT_MISSING = {"", "?", "na", "n/a", "nan", "null", "none"}


def _is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _resolve(ref, header):
    """Resolve a column reference (name or int-like string) to an index."""
    if ref in header:
        return header.index(ref)
    try:
        idx = int(ref)
    except (TypeError, ValueError):
        raise SystemExit("unknown column %r (have: %s)" % (ref, ", ".join(header)))
    if idx < 0:
        idx += len(header)
    if not 0 <= idx < len(header):
        raise SystemExit("column index %r out of range (%d columns)" % (ref, len(header)))
    return idx


def _split_cols(arg, header):
    if not arg:
        return []
    return [_resolve(c.strip(), header) for c in arg.split(",") if c.strip()]


def convert(rows, header, target_idx, positive, *, pos_label, neg_label,
            drop=(), keep=None, force_cont=(), force_sym=(), missing=DEFAULT_MISSING,
            test_frac=0.0, seed=0):
    drop = set(drop) | {target_idx}
    if keep is not None:
        feat_idx = [i for i in keep if i not in drop]
    else:
        feat_idx = [i for i in range(len(header)) if i not in drop]
    force_cont, force_sym = set(force_cont), set(force_sym)

    def norm(v):
        v = v.strip()
        return None if v.lower() in missing else v

    # --- type inference per kept feature column ---
    attributes = []
    col_kind = {}
    for i in feat_idx:
        vals = [norm(r[i]) for r in rows if i < len(r)]
        nonmiss = [v for v in vals if v is not None]
        if i in force_sym:
            kind = "symbolic"
        elif i in force_cont:
            kind = "continuous"
        else:
            kind = "continuous" if nonmiss and all(_is_float(v) for v in nonmiss) else "symbolic"
        col_kind[i] = kind
        attr = {"name": header[i], "kind": kind}
        if kind == "symbolic":
            attr["values"] = sorted(set(nonmiss))
        attributes.append(attr)

    # --- build examples ---
    examples = []
    for r in rows:
        if target_idx >= len(r):
            continue
        tgt = r[target_idx].strip()
        label = pos_label if tgt == positive else neg_label
        values = []
        for i in feat_idx:
            v = norm(r[i]) if i < len(r) else None
            if v is not None and col_kind[i] == "continuous":
                v = float(v)
            values.append(v)
        examples.append({"values": values, "label": label})

    # class labels share slipper's symbol namespace with attribute values, so a
    # label that also appears as a feature value is rejected by the loader.
    symvals = set()
    for a in attributes:
        symvals |= set(a.get("values", []))
    if pos_label == neg_label:
        raise SystemExit("positive and negative labels are both %r" % pos_label)
    for lab in (pos_label, neg_label):
        if lab in symvals:
            raise SystemExit(
                "class label %r collides with an attribute value; "
                "pass a distinct --pos-label/--neg-label" % lab)

    classes = [pos_label, neg_label]
    out = {"classes": classes, "attributes": attributes}
    if test_frac and test_frac > 0:
        # Stratified holdout: draw test_frac from within each class so the
        # positive rate is preserved in both splits (reduces test-error variance,
        # esp. on the imbalanced datasets). Deterministic given `seed`.
        rng = random.Random(seed)
        by_label = {}
        for i, e in enumerate(examples):
            by_label.setdefault(e["label"], []).append(i)
        test_ids = set()
        for lab in sorted(by_label):          # sorted -> stable across runs
            ids = by_label[lab][:]
            rng.shuffle(ids)
            ntest = int(round(len(ids) * test_frac))
            test_ids.update(ids[:ntest])
        out["train"] = [examples[i] for i in range(len(examples)) if i not in test_ids]
        out["test"] = [examples[i] for i in range(len(examples)) if i in test_ids]
    else:
        out["train"] = examples
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--target", required=True, help="target column (name or 0-based index)")
    ap.add_argument("--positive", required=True, help="target value -> positive class")
    ap.add_argument("--pos-label", help="positive class label (default: TARGET=POSITIVE)")
    ap.add_argument("--neg-label", help="label for the aggregated rest (default: TARGET=rest)")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    ap.add_argument("--no-header", action="store_true", help="file has no header row")
    ap.add_argument("--names", help="comma-separated column names (for --no-header)")
    ap.add_argument("--drop", help="columns to drop (names/indices, comma-separated)")
    ap.add_argument("--keep", help="keep ONLY these feature columns (names/indices)")
    ap.add_argument("--continuous", help="force these columns continuous")
    ap.add_argument("--symbolic", help="force these columns symbolic")
    ap.add_argument("--missing", help="extra missing-value tokens (comma-separated)")
    ap.add_argument("--test-frac", type=float, default=0.0, help="fraction held out as test (default 0)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the train/test split")
    ap.add_argument("-o", "--out", help="output JSON path (default: CSV with .json)")
    ns = ap.parse_args(argv[1:])

    with open(ns.csv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter=ns.delimiter))
    rows = [r for r in rows if r]  # drop blank lines

    if ns.no_header:
        ncol = max(len(r) for r in rows)
        if ns.names:
            header = [n.strip() for n in ns.names.split(",")]
            if len(header) != ncol:
                ap.error("--names has %d names but CSV has %d columns" % (len(header), ncol))
        else:
            header = ["col%d" % i for i in range(ncol)]
    else:
        header = [h.strip() for h in rows[0]]
        rows = rows[1:]

    target_idx = _resolve(ns.target, header)
    missing = set(DEFAULT_MISSING)
    if ns.missing:
        missing |= {m.strip().lower() for m in ns.missing.split(",")}
    keep = _split_cols(ns.keep, header) if ns.keep else None

    tname = header[target_idx]
    pos_label = ns.pos_label or "%s=%s" % (tname, ns.positive)
    neg_label = ns.neg_label or "%s=rest" % tname

    out = convert(
        rows, header, target_idx, ns.positive,
        pos_label=pos_label, neg_label=neg_label,
        drop=_split_cols(ns.drop, header), keep=keep,
        force_cont=_split_cols(ns.continuous, header),
        force_sym=_split_cols(ns.symbolic, header),
        missing=missing, test_frac=ns.test_frac, seed=ns.seed,
    )

    out_path = ns.out or (ns.csv_path.rsplit(".", 1)[0] + ".json")
    with open(out_path, "w") as f:
        json.dump(out, f)

    npos = sum(1 for e in out["train"] + out.get("test", []) if e["label"] == out["classes"][0])
    ntot = len(out["train"]) + len(out.get("test", []))
    print("%s -> %s : %d attrs, %d train, %d test, pos=%r %d/%d (%.1f%%)"
          % (ns.csv_path, out_path, len(out["attributes"]), len(out["train"]),
             len(out.get("test", [])), out["classes"][0], npos, ntot, 100.0 * npos / ntot))


if __name__ == "__main__":
    main(sys.argv)
