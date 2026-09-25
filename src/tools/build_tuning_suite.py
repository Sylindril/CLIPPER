"""build_tuning_suite.py - convert the tuning_suite/ raw datasets to slipper2 JSON.

A *second* benchmark suite, distinct from ``test_suite/`` so that slopper (the
LLM rule-builder, ``-R``) can be tuned without overfitting to the datasets used
for the main model comparison.  Every dataset here was kept for one reason:
**it has meaningful attribute names**, which is what the LLM prompt leans on.
See ``datasets/README.md`` for the suitability report and the datasets that
were located but rejected (crx, splice-junction, bupa liver, ripper_synthetic).

Each raw source (UCI ``.data``/``.names``, a public CSV mirror, or the in-repo
``move1``) is turned into a *two-class* one-vs-rest problem with the **event
class as positive** (die, malignant, band, surgical, hypothyroid, diabetic,
survived) -- the concept the question actually asks about, and the minority
class in every case.  Conversion is delegated to ``csv_to_json.convert``; this
file is just the per-dataset config (column names, target, drops, and the
type/format quirks each source needs).

Run from the repo root::

    python src/tools/build_tuning_suite.py          # writes tuning_suite/*.json
"""

import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import csv_to_json as c2j  # noqa: E402
import to_json  # noqa: E402

SUITE = os.path.join(_HERE, "..", "..", "tuning_suite")
TEST_FRAC = 0.25   # held-out fraction for sources without a native split
SEED = 1

# Sentinel target values written by a `binarize` rule (threshold/set targets).
POS, NEG = "__pos__", "__neg__"

# Meaningful column names for the headerless UCI/CSV sources.  Order matches the
# raw file exactly; the class column is included and named by `target`.
HYPOTHYROID = ["class", "age", "sex", "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_medication", "thyroid_surgery", "query_hypothyroid",
    "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium", "goitre",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG"]

BCW = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
    "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses", "class"]

BANDS = ["timestamp", "cylinder_number", "customer", "job_number",
    "grain_screened", "ink_color", "proof_on_ctd_ink", "blade_mfg",
    "cylinder_division", "paper_type", "ink_type", "direct_steam",
    "solvent_type", "type_on_cylinder", "press_type", "press", "unit_number",
    "cylinder_size", "paper_mill_location", "plating_tank", "proof_cut",
    "viscosity", "caliper", "ink_temperature", "humidity", "roughness",
    "blade_pressure", "varnish_pct", "press_speed", "ink_pct", "solvent_pct",
    "esa_voltage", "esa_amperage", "wax", "hardener", "roller_durometer",
    "current_density", "anode_space_ratio", "chrome_content", "band_type"]

PIMA = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age", "class"]

ABALONE = ["sex", "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight", "rings"]

# German credit: raw file is headerless with coded categorical values (A11, A34…).
# The class column (21st) is 1=good / 2=bad.  Names + value decodings are from
# tuning-suite/german.doc (UCI dataset 144); decoding lifts this into the
# "meaningful names AND values" tier the suite is selected for.
GERMAN = ["checking_account_status", "duration_months", "credit_history",
    "purpose", "credit_amount", "savings_account", "employment_since",
    "installment_rate_pct", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years", "other_installment_plans",
    "housing", "existing_credits_count", "job", "num_dependents", "telephone",
    "foreign_worker", "credit_risk"]

# Every code is globally unique (A<attr><val>), so one flat map decodes any cell.
GERMAN_DECODE = {
    "A11": "checking_lt_0DM", "A12": "checking_0_200DM",
    "A13": "checking_ge_200DM", "A14": "no_checking",
    "A30": "no_credits_or_paid", "A31": "all_paid_this_bank",
    "A32": "paid_duly_till_now", "A33": "past_payment_delay",
    "A34": "critical_or_other_credits",
    "A40": "purpose_car_new", "A41": "purpose_car_used",
    "A42": "purpose_furniture", "A43": "purpose_radio_tv",
    "A44": "purpose_appliances", "A45": "purpose_repairs",
    "A46": "purpose_education", "A47": "purpose_vacation",
    "A48": "purpose_retraining", "A49": "purpose_business",
    "A410": "purpose_other",
    "A61": "savings_lt_100DM", "A62": "savings_100_500DM",
    "A63": "savings_500_1000DM", "A64": "savings_ge_1000DM",
    "A65": "savings_unknown_none",
    "A71": "unemployed", "A72": "employed_lt_1yr", "A73": "employed_1_4yr",
    "A74": "employed_4_7yr", "A75": "employed_ge_7yr",
    "A91": "male_divorced", "A92": "female_div_or_married",
    "A93": "male_single", "A94": "male_married_widowed", "A95": "female_single",
    "A101": "debtors_none", "A102": "debtors_co_applicant",
    "A103": "debtors_guarantor",
    "A121": "property_real_estate", "A122": "property_life_insurance",
    "A123": "property_car_other", "A124": "property_unknown_none",
    "A141": "other_plans_bank", "A142": "other_plans_stores",
    "A143": "other_plans_none",
    "A151": "housing_rent", "A152": "housing_own", "A153": "housing_free",
    "A171": "job_unskilled_nonresident", "A172": "job_unskilled_resident",
    "A173": "job_skilled", "A174": "job_management_selfemployed",
    "A191": "phone_none", "A192": "phone_yes",
    "A201": "foreign_yes", "A202": "foreign_no",
}

OBESITY_POS = {"Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"}

# Per-dataset config.  `header` is either a name list (raw file is headerless)
# or "file" (take row 0).  `sep` is None for whitespace-delimited.  `test` names
# a native test split; otherwise TEST_FRAC is held out (stratified).  `force_sym`
# lists columns whose values are integer *codes* (categorical), which type
# inference would otherwise mistake for continuous.  `value_map` decodes coded
# cell values (german).  `binarize` maps a raw target string -> positive(bool)
# for threshold/set targets (abalone rings, student G3, obesity BMI-class); when
# present, `pos`/`neg` name the two classes and `positive` is ignored.
DATASETS = [
    dict(stem="hypothyroid", data="hypothyroid.data", sep=",", header=HYPOTHYROID,
         target="class", positive="hypothyroid", pos="hypothyroid", neg="negative"),
    dict(stem="breast_cancer_wisconsin", data="breast_cancer_wisconsin.data",
         sep=",", header=BCW, target="class", positive="4",
         pos="malignant", neg="benign", drop=["id"]),
    dict(stem="cylinder_bands", data="cylinder_bands.data", sep=",", header=BANDS,
         target="band_type", positive="band", pos="band", neg="noband",
         lower=True, ncols=40,  # drop the 2 malformed rows (18/23 cols)
         drop=["timestamp", "cylinder_number", "job_number"],
         force_sym=["press", "unit_number", "plating_tank"]),
    dict(stem="pima_diabetes", data="pima_diabetes.data", sep=",", header=PIMA,
         target="class", positive="1", pos="diabetic", neg="nondiabetic"),
    dict(stem="titanic", data="titanic.csv", sep=",", header="file",
         target="Survived", positive="1", pos="survived", neg="died",
         drop=["PassengerId", "Name", "Ticket", "Cabin"], force_sym=["Pclass"]),
    # --- new (2026-07-05) ---
    dict(stem="abalone", data="abalone.data", sep=",", header=ABALONE,
         target="rings", pos="old", neg="young",
         binarize=lambda t: int(t) >= 10),  # rings>=10 (age>=11.5yr) ~= 50/50
    dict(stem="german_credit", data="german.data", sep=None, header=GERMAN,
         target="credit_risk", positive="2", pos="bad", neg="good",
         value_map=GERMAN_DECODE),          # decode A-codes to meaningful values
    dict(stem="obesity", data="obesity.csv", sep=",", header="file",
         target="NObeyesdad", pos="obese", neg="not_obese",
         drop=["Weight", "Height"],         # BMI = Weight/Height^2 leaks the label
         binarize=lambda t: t in OBESITY_POS),
    dict(stem="student_por", data="student-por.csv", sep=";", header="file",
         target="G3", pos="fail", neg="pass",
         drop=["G1", "G2"],                 # prior-period grades leak the outcome
         rename={"health": "health_status"},  # 'health' also a Mjob/Fjob value
         binarize=lambda t: int(t) < 10),   # G3 < 10 -> fail (event)
]


def read_rows(path, sep, has_header):
    if sep is None:  # whitespace-delimited (horse-colic)
        with open(path) as f:
            rows = [ln.split() for ln in f if ln.split()]
    else:
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f, delimiter=sep) if r]
    header = None
    if has_header:
        header, rows = [h.strip() for h in rows[0]], rows[1:]
    return header, rows


def idx(names, header):
    return [header.index(n) if isinstance(n, str) else n for n in (names or [])]


def build(cfg):
    header = cfg["header"]
    has_header = header == "file"
    fhdr, rows = read_rows(os.path.join(SUITE, cfg["data"]), cfg["sep"], has_header)
    header = fhdr if has_header else header
    if cfg.get("rename"):  # avoid attr-name/value collisions in slipper's shared namespace
        header = [cfg["rename"].get(h, h) for h in header]
    test_rows = []
    if cfg.get("test"):
        _, test_rows = read_rows(os.path.join(SUITE, cfg["test"]), cfg["sep"], has_header)
    if cfg.get("ncols"):  # keep only well-formed rows (bands has a couple ragged ones)
        n = cfg["ncols"]
        rows, test_rows = [r for r in rows if len(r) == n], [r for r in test_rows if len(r) == n]
    if cfg.get("lower"):
        low = lambda r: [c.lower() for c in r]
        rows, test_rows = [low(r) for r in rows], [low(r) for r in test_rows]

    target_i = header.index(cfg["target"])

    if cfg.get("value_map"):  # decode coded cell values (german A-codes -> meanings)
        vm = cfg["value_map"]
        decode = lambda r: [vm.get(c, c) for c in r]
        rows, test_rows = [decode(r) for r in rows], [decode(r) for r in test_rows]

    positive = cfg["positive"] if "positive" in cfg else POS
    if cfg.get("binarize"):  # threshold/set target -> POS/NEG sentinel column
        fn = cfg["binarize"]
        def binz(rws):
            for r in rws:
                if target_i < len(r):
                    r[target_i] = POS if fn(r[target_i]) else NEG
            return rws
        rows, test_rows = binz(rows), binz(test_rows)

    kw = dict(pos_label=cfg["pos"], neg_label=cfg["neg"],
              drop=idx(cfg.get("drop"), header),
              force_sym=idx(cfg.get("force_sym"), header),
              force_cont=idx(cfg.get("force_cont"), header))

    if test_rows:  # native split: infer one schema over the union, then slice
        out = c2j.convert(rows + test_rows, header, target_i, positive, **kw)
        ntrain = len(rows)
        out["train"], out["test"] = out["train"][:ntrain], out["train"][ntrain:]
    else:
        out = c2j.convert(rows, header, target_i, positive,
                          test_frac=TEST_FRAC, seed=SEED, **kw)

    path = os.path.join(SUITE, cfg["stem"] + ".json")
    with open(path, "w") as f:
        json.dump(out, f)
    npos = sum(1 for e in out["train"] + out.get("test", []) if e["label"] == cfg["pos"])
    ntot = len(out["train"]) + len(out.get("test", []))
    print("%-24s %2d attrs  %4d train  %4d test   pos=%s %d/%d (%.0f%%)"
          % (cfg["stem"], len(out["attributes"]), len(out["train"]),
             len(out.get("test", [])), cfg["pos"], npos, ntot, 100.0 * npos / ntot))


def build_move1():
    # move1 is already in RIPPER/C4.5 .names/.data/.test form; to_json handles it.
    # Its raw files ship in tuning_suite/ (no public download); skip if absent.
    if not os.path.exists(os.path.join(SUITE, "move1.data")):
        print("move1: tuning_suite/move1.{names,data,test} not found; skipped")
        return
    data = to_json.convert(os.path.join(SUITE, "move1"))
    with open(os.path.join(SUITE, "move1.json"), "w") as f:
        json.dump(data, f)
    ntot = len(data["train"]) + len(data.get("test", []))
    npos = sum(1 for e in data["train"] + data.get("test", []) if e["label"] == "expert")
    print("%-24s %2d attrs  %4d train  %4d test   pos=%s %d/%d (%.0f%%)"
          % ("move1", len(data["attributes"]), len(data["train"]),
             len(data.get("test", [])), "expert", npos, ntot, 100.0 * npos / ntot))


def main():
    for cfg in DATASETS:
        build(cfg)
    build_move1()


if __name__ == "__main__":
    main()
