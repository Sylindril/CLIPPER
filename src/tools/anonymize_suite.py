"""anonymize_suite.py - copy a suite's STEM.json files with meaningless names.

For the natural-vs-anonymized comparison: the LLM sees the same examples but
no semantic hints from the schema.  Per dataset:

  * the problem name (file stem) becomes ``task_<6 hex>`` (md5 of the original
    stem, so reruns are stable);
  * attribute names become ``a01``, ``a02``, ... in their original order;
  * the two classes become ``y=1`` (positive, i.e. ``classes[0]``) and
    ``y=rest``, and every example label is rewritten to match -- the original
    class names (``income=>50K``, ``Churn=Yes``) name the target;
  * symbolic values become ``v1``, ``v2``, ... per attribute, numbered in the
    declared ``values`` order, then any undeclared value in order of appearance.

Numbers, missing values (null) and train/test row order are left untouched, so a harness keyed on the original stem draws the
same subsets (see ``sample_key`` in scripts/suite.py).  Sidecars (.info,
advice files) are not copied.  ``MAPPING.json`` in the output dir records
anon stem -> {original stem, attribute names, class names, symbolic values}.

Run from the repo root::

    uv run python src/tools/anonymize_suite.py                       # test_suite -> anon_suite
    uv run python src/tools/anonymize_suite.py tuning_suite anon_tuning_suite
"""

import hashlib
import json
import os
import sys


def anon_stem(stem):
    return "task_" + hashlib.md5(stem.encode()).hexdigest()[:6]


def anonymize(spec):
    """Return (anonymized spec, mapping info) for one dataset spec."""
    classes = spec["classes"]
    if len(classes) != 2:
        sys.exit("expected a two-class dataset, got classes %r" % classes)
    new_classes = ["y=1", "y=rest"]
    relabel = dict(zip(classes, new_classes))
    width = max(2, len(str(len(spec["attributes"]))))
    attrs = [dict(a, name="a%0*d" % (width, i + 1))
             for i, a in enumerate(spec["attributes"])]
    splits = [spec.get(k) or [] for k in ("train", "test")]

    # per symbolic attribute index: original value -> vN
    vmaps = {}
    for i, a in enumerate(spec["attributes"]):
        if a["kind"] != "symbolic":
            continue
        seen = [str(v) for v in a.get("values", [])]
        seen += [str(r["values"][i]) for rows in splits for r in rows
                 if r["values"][i] is not None]
        vmaps[i] = {v: "v%d" % (k + 1) for k, v in enumerate(dict.fromkeys(seen))}
        if "values" in a:
            attrs[i]["values"] = [vmaps[i][str(v)] for v in a["values"]]

    def row(r):
        vals = [vmaps[i][str(v)] if i in vmaps and v is not None else v
                for i, v in enumerate(r["values"])]
        return dict(r, values=vals, label=relabel[r["label"]])

    def rows(split):
        if split is None:
            return None
        return [row(r) for r in split]

    out = dict(spec, classes=new_classes, attributes=attrs,
               train=rows(spec.get("train")))
    if "test" in spec:
        out["test"] = rows(spec["test"])
    info = {"attributes": {a["name"]: o["name"]
                           for a, o in zip(attrs, spec["attributes"])},
            "classes": relabel,
            "values": {attrs[i]["name"]: {n: v for v, n in m.items()}
                       for i, m in vmaps.items()}}
    return out, info


def main(argv):
    src = argv[1] if len(argv) > 1 else "test_suite"
    dst = argv[2] if len(argv) > 2 else "anon_suite"
    os.makedirs(dst, exist_ok=True)
    mapping = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-len(".json")]
        with open(os.path.join(src, fn)) as f:
            spec, info = anonymize(json.load(f))
        new = anon_stem(stem)
        with open(os.path.join(dst, new + ".json"), "w") as f:
            json.dump(spec, f)
        mapping[new] = {"original": stem, **info}
        print("%s -> %s" % (stem, new))
    with open(os.path.join(dst, "MAPPING.json"), "w") as f:
        json.dump(mapping, f, indent=2)


if __name__ == "__main__":
    main(sys.argv)
