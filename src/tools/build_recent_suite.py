"""build_recent_suite.py - convert the recent_suite/ raw downloads to slipper2 JSON.

The **recent suite** groups the *post-LLM-cutoff* datasets: every dataset in
``test_suite/``, ``tuning_suite/`` and ``text_suite/`` predates the January-2025
knowledge cutoff of the gemini-3.1 models used in our experiments (they are
long-public UCI/Kaggle benchmarks), so LLM contamination cannot be ruled out
there.  The two datasets here were first published *after* that cutoff --
their labels cannot have been memorized -- and exist to show that the CLIPPER
results are not an artifact of data contamination.

  inflation_abstracts  UCI #1125, donated 2025-02-16.  Text classification:
                       1138 economics-inflation research abstracts, label 1 if
                       the abstract uses ML/AI methodologies (pos ml_ai, 20.9%).
                       Encoded like the text suite (WORDS set + raw TEXT).
  paddy_yield          UCI paddy dataset (post-cutoff per provenance check),
                       2789 paddy (rice) cultivation records from Tamil Nadu
                       agriblocks: field size, variety, soil, fertilizer
                       schedule, per-growth-phase weather.  Tabular, like the
                       development suite.

Paddy design decisions :

  * The raw target ``Paddy yield(in Kg)`` is 99.5% correlated with field size
    (``Hectares``), so an above-median split on raw yield degenerates to "is
    the field big" (Hectares>=4 alone scores 90.8%).  The target here is
    **yield per hectare** above/below the full-data median (5964 kg/ha) ->
    classes ``high_yield`` / ``low_yield`` (~50/50).
  * Dropped columns: ``Paddy yield(in Kg)`` (defines the target) and
    ``Trash(in bundles)`` (harvest byproduct -- post-outcome leak).
  * The ~10 input-schedule columns that are exact multiples of Hectares
    (seedrate = 25xH, DAP = 40xH, ...) are kept but redundant; disclosed in
    the data card.  Weather is deterministic per (agriblock, season) and the data
    looks synthetic/aggregated -- also disclosed.

Splits follow the house rule: deterministic stratified 25% holdout, seed 1
(same code as build_text_suite; helpers are imported from it so the two suites
cannot drift).

Run from the repo root::

    python src/tools/build_recent_suite.py            # writes recent_suite/*.json (inflation_abstracts)
"""

import csv
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from build_text_suite import stratified_split, tokenize_words  # noqa: E402

SUITE = os.path.join(_HERE, "..", "..", "recent_suite")
RAW = os.path.join(SUITE, "raw")


# --------------------------------------------------------------------------- #
# inflation_abstracts (text; moved here from build_text_suite)
# --------------------------------------------------------------------------- #
def build_inflation():
    """UCI #1125 (donated 2025-02-16): economics-inflation research abstracts,
    labeled 1 if the abstract uses ML/AI methodologies.  All 1138 rows kept,
    including 13 duplicate abstracts (no-drop policy).  Output is identical to
    the text-suite encoding: WORDS (set) + TEXT (raw string)."""
    with open(os.path.join(RAW, "inflation", "classified_abstracts.json"),
              encoding="utf-8") as fp:
        recs = json.load(fp)
    rows = [(r.get("Abstract") or "", "ml_ai" if str(r.get("Label")) == "1"
             else "non_ml") for r in recs]
    classes = ["ml_ai", "non_ml"]      # positive = uses-ML/AI (minority, ~21%)
    train, test = stratified_split(rows, classes)

    def pack(rows):
        return [{"values": [tokenize_words(text), text], "label": lab}
                for text, lab in rows]

    tr, te = pack(train), pack(test)
    out = {"classes": classes,
           "attributes": [{"name": "WORDS", "kind": "set"},
                          {"name": "TEXT", "kind": "text"}],
           "train": tr, "test": te}
    _emit("inflation_abstracts", out, pos=classes[0])


# --------------------------------------------------------------------------- #
# paddy_yield (tabular)
# --------------------------------------------------------------------------- #
# (raw CSV column, clean attribute name, kind).  Order matters: it defines the
# values[] layout.  "DAI" is read as applied irrigation (assumption noted in
# the data card).  Dropped entirely: 'Paddy yield(in Kg)' (target), 'Trash(in
# bundles)' (post-outcome).
_PADDY_COLS = [
    ("Hectares ",                          "hectares",                "continuous"),
    ("Agriblock",                          "agriblock",               "symbolic"),
    ("Variety",                            "variety",                 "symbolic"),
    ("Soil Types",                         "soil_type",               "symbolic"),
    ("Seedrate(in Kg)",                    "seedrate_kg",             "continuous"),
    ("LP_Mainfield(in Tonnes)",            "lp_mainfield_tonnes",     "continuous"),
    ("Nursery",                            "nursery",                 "symbolic"),
    ("Nursery area (Cents)",               "nursery_area_cents",      "continuous"),
    ("LP_nurseryarea(in Tonnes)",          "lp_nursery_tonnes",       "continuous"),
    ("DAP_20days",                         "dap_20days_kg",           "continuous"),
    ("Weed28D_thiobencarb",                "weed_28d_thiobencarb",    "continuous"),
    ("Urea_40Days",                        "urea_40days_kg",          "continuous"),
    ("Potassh_50Days",                     "potash_50days_kg",        "continuous"),
    ("Micronutrients_70Days",              "micronutrients_70days",   "continuous"),
    ("Pest_60Day(in ml)",                  "pesticide_60day_ml",      "continuous"),
    ("30DRain( in mm)",                    "rain_d1_30_mm",           "continuous"),
    ("30DAI(in mm)",                       "irrigation_d1_30_mm",     "continuous"),
    ("30_50DRain( in mm)",                 "rain_d30_50_mm",          "continuous"),
    ("30_50DAI(in mm)",                    "irrigation_d30_50_mm",    "continuous"),
    ("51_70DRain(in mm)",                  "rain_d51_70_mm",          "continuous"),
    ("51_70AI(in mm)",                     "irrigation_d51_70_mm",    "continuous"),
    ("71_105DRain(in mm)",                 "rain_d71_105_mm",         "continuous"),
    ("71_105DAI(in mm)",                   "irrigation_d71_105_mm",   "continuous"),
    ("Min temp_D1_D30",                    "min_temp_d1_30",          "continuous"),
    ("Max temp_D1_D30",                    "max_temp_d1_30",          "continuous"),
    ("Min temp_D31_D60",                   "min_temp_d31_60",         "continuous"),
    ("Max temp_D31_D60",                   "max_temp_d31_60",         "continuous"),
    ("Min temp_D61_D90",                   "min_temp_d61_90",         "continuous"),
    ("Max temp_D61_D90",                   "max_temp_d61_90",         "continuous"),
    ("Min temp_D91_D120",                  "min_temp_d91_120",        "continuous"),
    ("Max temp_D91_D120",                  "max_temp_d91_120",        "continuous"),
    ("Inst Wind Speed_D1_D30(in Knots)",   "wind_speed_d1_30_knots",  "continuous"),
    ("Inst Wind Speed_D31_D60(in Knots)",  "wind_speed_d31_60_knots", "continuous"),
    ("Inst Wind Speed_D61_D90(in Knots)",  "wind_speed_d61_90_knots", "continuous"),
    ("Inst Wind Speed_D91_D120(in Knots)", "wind_speed_d91_120_knots","continuous"),
    ("Wind Direction_D1_D30",              "wind_dir_d1_30",          "symbolic"),
    ("Wind Direction_D31_D60",             "wind_dir_d31_60",         "symbolic"),
    ("Wind Direction_D61_D90",             "wind_dir_d61_90",         "symbolic"),
    ("Wind Direction_D91_D120",            "wind_dir_d91_120",        "symbolic"),
    ("Relative Humidity_D1_D30",           "humidity_d1_30",          "continuous"),
    ("Relative Humidity_D31_D60",          "humidity_d31_60",         "continuous"),
    ("Relative Humidity_D61_D90",          "humidity_d61_90",         "continuous"),
    ("Relative Humidity_D91_D120",         "humidity_d91_120",        "continuous"),
]


def build_paddy():
    path = os.path.join(RAW, "paddy", "paddydataset.csv")
    with open(path, encoding="utf-8-sig", newline="") as fp:
        recs = list(csv.DictReader(fp))
    yield_ha = [float(r["Paddy yield(in Kg)"]) / float(r["Hectares "])
                for r in recs]
    med = statistics.median(yield_ha)

    rows = []
    for r, yh in zip(recs, yield_ha):
        vals = []
        for col, _, kind in _PADDY_COLS:
            v = (r.get(col) or "").strip()
            vals.append(float(v) if kind == "continuous" else v)
        rows.append((vals, "high_yield" if yh > med else "low_yield"))
    classes = ["high_yield", "low_yield"]   # positive = above-median yield/ha
    train, test = stratified_split(rows, classes)

    def pack(rows):
        return [{"values": vals, "label": lab} for vals, lab in rows]

    out = {"classes": classes,
           "attributes": [{"name": name, "kind": kind}
                          for _, name, kind in _PADDY_COLS],
           "train": pack(train), "test": pack(test)}
    _emit("paddy_yield", out, pos=classes[0],
          extra="median yield/ha=%.0f" % med)


# --------------------------------------------------------------------------- #
def _emit(name, out, pos, extra=""):
    path = os.path.join(SUITE, name + ".json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp)
    tr, te = out["train"], out["test"]
    npos = sum(1 for ex in tr if ex["label"] == pos)
    print("%-20s -> %-26s %5d train / %4d test  pos=%s %4.1f%%  %s"
          % (name, name + ".json", len(tr), len(te), pos,
             100.0 * npos / max(len(tr), 1), extra))


DATASETS = {
    "inflation_abstracts": build_inflation,
    "paddy_yield": build_paddy,
}


def main(argv):
    which = argv[1:] or list(DATASETS)
    for name in which:
        if name not in DATASETS:
            sys.exit("unknown dataset %r (choices: %s)"
                     % (name, ", ".join(DATASETS)))
        DATASETS[name]()


if __name__ == "__main__":
    main(sys.argv)
