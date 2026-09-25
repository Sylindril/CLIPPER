"""build_test_suite.py - convert the test_suite/ CSVs to slipper2 JSON.

Drives ``csv_to_json.convert`` over the eight benchmark datasets in
``test_suite/`` (adopted from Springer BISE s12599-024-00922-2; see
``test_suite/README.md``).  Each is turned into a *two-class* one-vs-rest
problem with the **event class as positive** -- i.e. the outcome the dataset's
prediction question asks about (churn=Yes, stroke=1, income>50K, ...), which is
the minority class in 7 of 8 and the concept slipper2 learns by default anyway.

Per-dataset config below records the target column, the positive (event) value,
ID columns dropped, and a couple of leakage / paper-alignment choices (noted
inline).  A fixed seed gives a deterministic train/test split so model
comparisons are reproducible.

Run from the repo root::

    python src/tools/build_test_suite.py            # writes test_suite/*.json
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv_to_json as c2j  # noqa: E402

TEST_FRAC = 0.25
SEED = 1

# Adult has no header row; UCI column order:
ADULT_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]

# ProPublica COMPAS: keep a standard, non-leaky feature subset (drops the COMPAS
# score columns decile_score/score_text/is_recid/... which leak the outcome).
COMPAS_KEEP = [
    "sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "c_charge_degree",
]

CONFIG = [
    # file, delimiter, header, names, target, positive(event), drop, keep
    dict(file="college.csv", target="will_go_to_college", positive="True"),
    dict(file="water_potability.csv", target="Potability", positive="1"),
    dict(file="stroke.csv", target="stroke", positive="1", drop=["id"]),
    dict(file="telco_churn.csv", target="Churn", positive="Yes", drop=["customerID"]),
    dict(file="compas.csv", target="two_year_recid", positive="1", keep=COMPAS_KEEP),
    dict(file="heloc.csv", target="RiskPerformance", positive="Good"),
    # Adult: drop native-country -> 6 numeric + 7 categorical (paper's count).
    dict(file="adult.csv", header=False, names=ADULT_NAMES, target="income",
         positive=">50K", drop=["native-country"]),
    # Bank: ';'-delimited; drop 'duration' (well-known leakage: known only after
    # the call) -> 6 numeric + 9 categorical (paper's count).
    dict(file="bank_marketing.csv", delimiter=";", target="y", positive="yes",
         drop=["duration"]),
]


def build_one(suite_dir, cfg):
    path = os.path.join(suite_dir, cfg["file"])
    delim = cfg.get("delimiter", ",")
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=delim) if r]

    if cfg.get("header", True):
        header = [h.strip() for h in rows[0]]
        rows = rows[1:]
    else:
        header = cfg["names"]

    resolve = lambda refs: [c2j._resolve(r, header) for r in refs]
    target_idx = c2j._resolve(cfg["target"], header)
    keep = resolve(cfg["keep"]) if cfg.get("keep") else None

    out = c2j.convert(
        rows, header, target_idx, cfg["positive"],
        pos_label="%s=%s" % (cfg["target"], cfg["positive"]),
        neg_label="%s=rest" % cfg["target"],
        drop=resolve(cfg.get("drop", [])), keep=keep,
        test_frac=TEST_FRAC, seed=SEED,
    )

    out_path = os.path.splitext(path)[0] + ".json"
    with open(out_path, "w") as f:
        json.dump(out, f)

    npos = sum(1 for e in out["train"] + out.get("test", []) if e["label"] == out["classes"][0])
    ntot = len(out["train"]) + len(out["test"])
    print("%-22s -> %-20s %2d attrs  %5d tr  %5d te  pos=%-22s %5d (%.1f%%)"
          % (cfg["file"], os.path.basename(out_path), len(out["attributes"]),
             len(out["train"]), len(out["test"]), out["classes"][0], npos,
             100.0 * npos / ntot))


def main(argv):
    suite_dir = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "test_suite")
    suite_dir = os.path.normpath(suite_dir)
    for cfg in CONFIG:
        build_one(suite_dir, cfg)


if __name__ == "__main__":
    main(sys.argv)
