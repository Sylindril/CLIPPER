#!/usr/bin/env python3
"""imodels_baselines.py - run interpretable-model baselines from imodels
(https://github.com/csinva/imodels) on the test_suite datasets and tabulate test
error, so the numbers sit next to the slipper2/slopper conditions.

Each dataset is the *same* two-class one-vs-rest problem slipper2 learns (event
class = ``classes[0]`` = positive) evaluated on the *same* fixed 25%-holdout test
split baked into ``test_suite/STEM.json`` by ``build_test_suite.py``.  So the
``Test err %`` column here is directly comparable to sweep.py's.

A **baseline** is a named sklearn-compatible estimator (see ``MODELS`` below).
The features slipper consumes symbolically/continuously are turned into a numeric
matrix with a fixed encoder (median-imputed continuous columns + one-hot
symbolic columns, fit on train only), which every model shares -- so differences
are the model, not the preprocessing.  Text datasets (a ``"kind": "text"``
attribute) instead use the TF-IDF pre-processor
(``make_text_encoder``): TfidfVectorizer (English stop words removed; vocab
filtered min_df=3 / max_df=0.8 / max_features=800) on the raw document ->
sparse matrix -> DataFrame with vocabulary-term columns.  Add a model by
dropping one line in ``MODELS``.

Install the deps first (and pass ``--group baselines`` to every ``uv run`` --
plain ``uv run`` re-syncs the env *without* the optional group)::

    uv sync --group baselines

Examples (from the repo root)::

    uv run --group baselines python imodels_baselines.py            # every model x dataset
    uv run --group baselines python imodels_baselines.py -c imodels_rulefit -c imodels_figs
    uv run --group baselines python imodels_baselines.py --smallsuite --seed 1
    uv run --group baselines python imodels_baselines.py --log baselines.md  # markdown summary
    uv run --group baselines python imodels_baselines.py --list

Metrics mirror sweep.py: ``Test err %`` / ``Train err %`` are 100*(1-accuracy);
``Rules`` is the model's ``complexity_`` (rule/split count) where exposed, else
``?``; ``Time`` is fit seconds.
"""

import argparse
import glob
import inspect
import json
import os
import signal
import sys
import time
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(HERE, "test_suite")  # overridable via --suite
TIMEOUT = None  # per-fit wall-clock cap in seconds (--timeout); None = no cap


class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds):
    """SIGALRM-based wall-clock cap around a fit (main thread, sequential sweep)."""
    if not seconds:
        yield
        return

    def _handler(signum, frame):
        raise _Timeout("exceeded %ds" % seconds)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

# Same order/spelling as sweep.py DATASETS so the two tables line up.
DATASETS = [
    "college", "water_potability", "stroke", "telco_churn",
    "compas", "heloc", "adult", "bank_marketing",
]
# --smallsuite drops the slowest (adult) and the degenerate one (stroke, n=0),
# matching sweep.py.
SMALL_DROP = {"adult", "stroke"}


def _ripper_class():
    """wittgenstein's RIPPER, wrapped to the shape the rest of the harness
    assumes, or None if wittgenstein isn't installed.

    Three gaps to close: its ``fit`` needs ``pos_class`` (the harness encodes
    the event class as 1, slipper's convention, and wittgenstein would
    otherwise pick one itself); ``predict`` returns a list of bools; and the
    rule count lives on ``ruleset_`` rather than in ``complexity_``.
    """
    try:
        import wittgenstein as lw
    except ImportError:
        return None

    class RipperClassifier(lw.RIPPER):
        def fit(self, X, y, **kw):
            kw.setdefault("pos_class", 1)
            super().fit(X, y, **kw)
            self.complexity_ = self.ruleset_.count_rules()
            return self

        def predict(self, X):
            return np.asarray(super().predict(X), dtype=int)

    return RipperClassifier


def _make_models():
    """name -> factory(seed) -> a fresh sklearn-compatible estimator.

    Imported lazily so ``--list`` / ``-h`` work without the baselines group
    installed.  ``random_state`` is passed only to models whose ``__init__``
    accepts it (checked by signature), so seeding is reproducible where the
    model supports it and silently skipped where it doesn't.
    """
    from imodels import (
        BayesianRuleListClassifier,
        BayesianRuleSetClassifier,
        BoostedRulesClassifier,
        C45TreeClassifier,
        FIGSClassifier,
        FPLassoClassifier,
        FPSkopeClassifier,
        GreedyRuleListClassifier,
        HSTreeClassifierCV,
        OneRClassifier,
        OptimalTreeClassifier,
        RuleFitClassifier,
        SkopeRulesClassifier,
        SlipperClassifier,
        TaoTreeClassifier,
    )
    # RandomForestPlusClassifier isn't re-exported at the imodels top level.
    from imodels.importance import RandomForestPlusClassifier
    _patch_hstree_log_loss()
    # Plain CART -- the trivial small-depth tree baseline (from sklearn, not imodels).
    from sklearn.tree import DecisionTreeClassifier
    # CORELS -- certifiably-optimal rule lists (not imodels; vendored wheel, see
    # vendor/README.md). Binary-only, `features=` fit kwarg, no complexity_.
    # The wheel is macOS-only (marked darwin-only in pyproject), so the model
    # is skipped where the import fails rather than sinking the whole registry.
    try:
        from corels import CorelsClassifier
    except ImportError:
        CorelsClassifier = None
    Ripper = _ripper_class()

    def seeded(cls, **kw):
        takes_rs = "random_state" in inspect.signature(cls.__init__).parameters

        def factory(seed):
            k = dict(kw)
            if takes_rs and seed is not None:
                k["random_state"] = seed
            return cls(**k)

        factory.desc = "%s(%s)" % (
            cls.__name__, ", ".join("%s=%r" % kv for kv in kw.items()))
        return factory

    models = {
        # imodels' own reimplementation of the Cohen & Singer rule booster this
        # repo ports -- the closest apples-to-apples baseline.
        "imodels_slipper": seeded(SlipperClassifier),
        # Rules mined from a tree ensemble + sparse linear fit.
        "imodels_rulefit": seeded(RuleFitClassifier),
        # Greedy sum-of-trees (Fast Interpretable Greedy-Sum-of-Trees).
        "imodels_figs":    seeded(FIGSClassifier),
        # --- imodels' remaining rule classifiers ---
        # AdaBoost over single-condition rule stumps.
        "imodels_boosted_rules": seeded(BoostedRulesClassifier),
        # Greedy CN2-style ordered rule list.
        "imodels_greedy_rule_list": seeded(GreedyRuleListClassifier),
        # 1R: the single best one-attribute rule.
        "imodels_oner": seeded(OneRClassifier),
        # Bayesian Rule Sets (unordered) and Bayesian Rule Lists (ordered).
        # brs never terminates even binary-encoded -> in DEFAULT_SKIP below
        # (registered so `-c imodels_brs` still works, just not in default sweeps).
        "imodels_brs": seeded(BayesianRuleSetClassifier),
        "imodels_brl": seeded(BayesianRuleListClassifier),
        # --- additional 'recommended' models ---
        # Hierarchical-shrinkage tree (shrinkage strength picked by CV).
        "imodels_hstree": seeded(HSTreeClassifierCV),
        # Random Forest Plus (RF + per-tree generalized linear read-out).
        "imodels_rf_plus": seeded(RandomForestPlusClassifier),
        # --- more rule/tree classifiers (near-zero-effort additions) ---
        # High-precision rules distilled from a tree ensemble.
        "imodels_skope_rules": seeded(SkopeRulesClassifier),
        # Classic C4.5 decision tree.
        "imodels_c45": seeded(C45TreeClassifier),
        # Frequent-pattern rule sets: FP-growth itemsets + Lasso / SkopeRules head.
        # Mine itemsets, so they want the all-binary encoder (see BINARY_MODELS).
        "imodels_fplasso": seeded(FPLassoClassifier),
        "imodels_fpskope": seeded(FPSkopeClassifier),
        # Optimal decision tree (GOSDT) and TAO-optimized tree -- both binary-only.
        "imodels_optimal_tree": seeded(OptimalTreeClassifier),
        "imodels_tao_tree": seeded(TaoTreeClassifier),
        # Plain small-depth CART -- the trivial sklearn tree baseline.
        "imodels_cart": seeded(DecisionTreeClassifier, max_depth=3),
    }
    if Ripper is not None:
        # RIPPER itself, via wittgenstein -- a maintained Python
        # reimplementation of the algorithm this repo's learner builds on
        # (Cohen, 1995).
        models["imodels_ripper"] = seeded(Ripper)
    if CorelsClassifier is not None:
        # Certifiably-optimal rule list (CORELS): provably-optimal ordered rule
        # list, the deterministic optimal counterpart to the MCMC-sampled `brl`.
        # verbosity=[] silences its per-fit rule-list dump to stdout.
        models["imodels_corels"] = seeded(CorelsClassifier, verbosity=[])
    return models


def load_xy(dataset):
    """Load test_suite/STEM.json -> (X_train, y_train, X_test, y_test, attrs).

    X_* are pandas DataFrames of raw attribute values (continuous as float w/
    NaN for missing, symbolic as object w/ ``None`` for missing); y_* are 0/1
    with 1 == the positive/event class (``classes[0]``, slipper's convention).
    """
    import pandas as pd

    with open(os.path.join(SUITE, dataset + ".json")) as f:
        data = json.load(f)
    attrs = data["attributes"]
    positive = data["classes"][0]
    names = [a["name"] for a in attrs]

    def to_frame(rows):
        df = pd.DataFrame([r["values"] for r in rows], columns=names)
        for a in attrs:                       # keep continuous numeric (NaN=missing)
            if a["kind"] == "continuous":
                df[a["name"]] = pd.to_numeric(df[a["name"]], errors="coerce")
        y = np.array([1 if r["label"] == positive else 0 for r in rows])
        return df, y

    Xtr, ytr = to_frame(data["train"])
    Xte, yte = to_frame(data.get("test", []))
    return Xtr, ytr, Xte, yte, attrs


def make_encoder(attrs):
    """Shared preprocessing: median-impute continuous, one-hot symbolic (unseen
    test categories ignored).  Fit on train only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cont = [a["name"] for a in attrs if a["kind"] == "continuous"]
    sym = [a["name"] for a in attrs if a["kind"] != "continuous"]
    cont_pipe = SimpleImputer(strategy="median")
    sym_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("cont", cont_pipe, cont), ("sym", sym_pipe, sym)])


def make_binary_encoder(attrs):
    """All-binary preprocessing for the Bayesian rule models (BRS/BRL), which
    mine frequent itemsets and require 0/1 features -- left as raw floats,
    continuous columns make itemset enumeration blow up and never terminate.
    Continuous columns are median-imputed then quantile-binned into one-hot
    indicators; symbolic columns are one-hot as usual.  Fit on train only.

    This deliberately breaks the "one shared encoder" invariant the other
    baselines hold to -- it's the only way BRS/BRL run at all -- so their rows
    are *not* strictly preprocessing-comparable to the rest of the table."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

    cont = [a["name"] for a in attrs if a["kind"] == "continuous"]
    sym = [a["name"] for a in attrs if a["kind"] != "continuous"]
    cont_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("bin", KBinsDiscretizer(n_bins=4, encode="onehot-dense",
                                 strategy="quantile")),
    ])
    sym_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("cont", cont_pipe, cont), ("sym", sym_pipe, sym)])


def _text_attr(attrs):
    """Name of the raw-document attribute (``"kind": "text"``, an attribute
    kind only text datasets carry), or None for tabular data."""
    for a in attrs:
        if a["kind"] == "text":
            return a["name"]
    return None


# Vocabulary filtering for the text encoders (settled 2026-08-19): English
# stop words, drop the singleton/rare tail (min_df) and corpus-dominant terms
# (max_df), cap the budget at the 800 highest-count terms.  The cap only binds
# on big-vocabulary datasets (sms_spam 7192 -> 800, inflation_abstracts 8772
# -> 800); the short-document sets fall under it via min_df alone (~290-680).
_TEXT_VOCAB_FILTER = dict(stop_words="english", min_df=3, max_df=0.8,
                          max_features=800)


class TextTfidfEncoder:
    """Alternate pre-processor for text datasets: TF-IDF over the raw text
    attribute, fit on train only like the tabular encoders.

    ``fit_transform``/``transform`` take the raw ``load_xy`` frame, vectorize
    its text column with ``sklearn.feature_extraction.text.TfidfVectorizer``
    (English stop words removed, so rule thresholds land on content words
    rather than document-length proxies like ``the``/``is``; vocabulary
    filtered to ``min_df=3, max_df=0.8, max_features=800`` -- see __init__)
    into a sparse matrix, then densify that into a plain pandas DataFrame
    whose columns are the vocabulary terms -- the term names keep the rule
    models' output readable (``free > 0.12``, not ``x_2041 > 0.12``).  The bag-of-words
    ``WORDS`` column is ignored: TF-IDF re-derives its own vocabulary from the
    raw text with sklearn's default analyzer.  Extra kwargs pass through to
    TfidfVectorizer verbatim.
    """

    def __init__(self, text_col, **tfidf_kw):
        from sklearn.feature_extraction.text import TfidfVectorizer
        for k, v in _TEXT_VOCAB_FILTER.items():
            tfidf_kw.setdefault(k, v)
        self.text_col = text_col
        self.vec = TfidfVectorizer(**tfidf_kw)

    def _docs(self, X):
        return X[self.text_col].fillna("").astype(str)

    def _frame(self, mat):
        import pandas as pd
        return pd.DataFrame(mat.toarray(),
                            columns=self.vec.get_feature_names_out())

    def fit_transform(self, X, y=None):
        return self._frame(self.vec.fit_transform(self._docs(X)))

    def transform(self, X):
        return self._frame(self.vec.transform(self._docs(X)))

    def get_feature_names_out(self):
        return self.vec.get_feature_names_out()


def make_text_encoder(attrs, **tfidf_kw):
    """TF-IDF pre-processor over the dataset's text attribute (see
    TextTfidfEncoder).  Used automatically by run_one for any dataset that
    has one -- the tabular encoders can't run there (the set-valued ``WORDS``
    column holds Python lists, which one-hot encoding rejects)."""
    return TextTfidfEncoder(_text_attr(attrs), **tfidf_kw)


class TextPresenceEncoder(TextTfidfEncoder):
    """0/1 term-presence variant of the text pre-processor, for the binary /
    itemset models (CORELS, BRS/BRL, FP-growth) that reject float features.
    Same vocabulary filtering as TF-IDF (`_TEXT_VOCAB_FILTER`), but
    ``CountVectorizer(binary=True)`` -- a dense 0/1 DataFrame, informationally
    the same presence signal as slipper's set encoding (over the filtered
    vocabulary)."""

    def __init__(self, text_col, **cv_kw):
        from sklearn.feature_extraction.text import CountVectorizer
        for k, v in _TEXT_VOCAB_FILTER.items():
            cv_kw.setdefault(k, v)
        self.text_col = text_col
        self.vec = CountVectorizer(binary=True, **cv_kw)


def make_binary_text_encoder(attrs, **cv_kw):
    """Presence-features pre-processor over the text attribute (see
    TextPresenceEncoder); run_one's text path uses it for BINARY_MODELS."""
    return TextPresenceEncoder(_text_attr(attrs), **cv_kw)


def _ascii_names(names):
    """pycorels' C extension encodes feature names as ASCII and crashes on
    anything else (youtube's vocabulary has accented/non-Latin terms) --
    transliterate and de-collide.  Display labels only; column order is
    what binds names to features."""
    import unicodedata
    out, seen = [], set()
    for i, n in enumerate(names):
        a = (unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
             or "f%d" % i)
        while a in seen:
            a += "_%d" % i
        seen.add(a)
        out.append(a)
    return out


# Models whose fit does numpy fancy indexing on X (HSTreeClassifierCV's
# internal CV splits with X[idx, :], which a DataFrame rejects with
# InvalidIndexError) -- run_one hands them a bare ndarray on the text path
# (the tabular encoders already produce arrays).
NUMPY_MODELS = {"imodels_hstree"}

# Models that need the all-binary encoder above (imodels' Bayesian rule learners
# + the FP-growth rule sets, which mine frequent itemsets and reject non-0/1
# columns; CORELS also wants 0/1 rules).  Everything else uses make_encoder.
BINARY_MODELS = {"imodels_brs", "imodels_brl", "imodels_fplasso", "imodels_fpskope",
                 "imodels_corels"}

# The BINARY_MODELS above take the column names as a `feature_names=` fit kwarg;
# CORELS is the odd one out -- it wants a 0/1 *int* matrix, spells the kwarg
# `features=`, and exposes its rule count via `model.rl().rules` (no complexity_).
CORELS_MODELS = {"imodels_corels"}

# Registered but skipped in default sweeps (still runnable via explicit -c).
# imodels_brs's MCMC never terminates on this suite even binary-encoded.
DEFAULT_SKIP = {"imodels_brs"}


def _patch_hstree_log_loss():
    """Make HSTreeClassifierCV survive sklearn >= 1.7's stricter log_loss checks.

    `HSTreeClassifierCV.fit` picks its shrinkage strength by scoring internal CV
    folds with `log_loss(y_out, est.predict_proba(X_out))`.  Two things break
    with a modern sklearn, neither of which says anything about the model:

      1. Hierarchical shrinkage can push a probability a few ulps below zero
         (~-2e-19).  sklearn used to clip; it now raises "y_prob contains values
         lower than 0".
      2. On a heavily imbalanced dataset a CV fold can hold only one class, and
         sklearn now demands an explicit `labels` for that case ("y_true
         contains only one label").  Hit by `stroke` (4.9% events) at n=100.

    Both are fixed by clipping to [0,1] and pinning labels=[0,1] for 0/1 targets.
    Patches the name imodels bound at import (`from sklearn.metrics import
    log_loss`), so it only affects that module's scorer -- not the sweep's own
    metric, which is plain accuracy via `model.predict`.
    """
    from imodels.tree import hierarchical_shrinkage as hs
    if getattr(hs.log_loss, "_slipper_patched", False):
        return
    inner = hs.log_loss

    def safe_log_loss(y_true, y_pred, **kw):
        y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
        if "labels" not in kw and set(np.unique(y_true)) <= {0, 1}:
            kw["labels"] = [0, 1]
        return inner(y_true, y_pred, **kw)

    safe_log_loss._slipper_patched = True
    hs.log_loss = safe_log_loss


def run_one(name, factory, dataset, seed):
    """Fit one model on one dataset; return a metrics dict (with 'status')."""
    try:
        Xtr, ytr, Xte, yte, attrs = load_xy(dataset)
        binary = name in BINARY_MODELS
        if _text_attr(attrs):
            # Text dataset: TF-IDF over the raw document string; the
            # binary/itemset models get 0/1 term-presence features instead
            # (same filtered vocabulary).
            enc = (make_binary_text_encoder(attrs) if binary
                   else make_text_encoder(attrs))
        elif binary:
            enc = make_binary_encoder(attrs)
        else:
            enc = make_encoder(attrs)
        Xtr_e = enc.fit_transform(Xtr)
        Xte_e = enc.transform(Xte) if len(Xte) else Xte
        if name in NUMPY_MODELS:
            Xtr_e = getattr(Xtr_e, "values", Xtr_e)
            Xte_e = getattr(Xte_e, "values", Xte_e)
        model = factory(seed)
        t0 = time.time()
        with _time_limit(TIMEOUT):
            if name in CORELS_MODELS:
                # Contiguous 0/1 uint8 ndarrays + `features=` kwarg: pycorels
                # does np.array(x, copy=False) internally, which NumPy 2
                # forbids when a copy is needed (e.g. from the text path's
                # DataFrame) -- so hand it arrays that need no copy.
                Xtr_e = np.ascontiguousarray(np.asarray(Xtr_e, dtype=np.uint8))
                Xte_e = (np.ascontiguousarray(np.asarray(Xte_e, dtype=np.uint8))
                         if len(Xte) else Xte_e)
                model.fit(Xtr_e, ytr,
                          features=_ascii_names(list(enc.get_feature_names_out())))
            elif binary:  # BRS/BRL want the binary column names for their rules
                model.fit(Xtr_e, ytr, feature_names=list(enc.get_feature_names_out()))
            else:
                model.fit(Xtr_e, ytr)
        elapsed = time.time() - t0
        train_err = 100.0 * (1.0 - _acc(model, Xtr_e, ytr))
        test_err = 100.0 * (1.0 - _acc(model, Xte_e, yte)) if len(yte) else float("nan")
        rules = getattr(model, "complexity_", None)
        if name in CORELS_MODELS:  # no complexity_; count rules in the rule list
            rules = len(model.rl().rules)
        return {
            "status": "ok", "test": test_err, "train": train_err,
            "rules": rules, "time": elapsed,
        }
    except _Timeout as e:
        return {"status": "TIMEOUT", "err": str(e)}
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        return {"status": "ERR", "err": "%s: %s" % (type(e).__name__, e)}


def _acc(model, X, y):
    if len(y) == 0:
        return float("nan")
    return float(np.mean(model.predict(X) == y))


def cell(res):
    return "%.2f" % res["test"] if res["status"] == "ok" else res["status"]


def write_log(path, models, dsets, results, seed):
    """Markdown summary: per model, the estimator spec and a table."""
    import datetime
    out = ["# imodels baselines — %s" % datetime.date.today().isoformat(), ""]
    out.append("Test split: same 25% holdout as `test_suite/*.json` (seed 1)."
               " Metric matches sweep.py (`Test err %` = 100*(1-accuracy)).")
    out.append("Encoder: median-impute continuous + one-hot symbolic, fit on train.")
    if any(m in BINARY_MODELS for m in models):
        out.append("Exception: %s use an all-binary encoder (continuous columns "
                   "quantile-binned into one-hot indicators) -- they mine itemsets "
                   "and can't run on raw floats -- so those rows aren't strictly "
                   "preprocessing-comparable to the rest." %
                   ", ".join("`%s`" % m for m in models if m in BINARY_MODELS))
    out.append("" if seed is None else "Model `random_state`: %d (where supported)." % seed)
    out.append("")
    for m in models:
        out += ["## Baseline: `%s`" % m, "",
                "model: `%s`" % getattr(MODELS[m], "desc", m), ""]
        out.append("| Dataset | Train err % | Test err % | Rules | Time (s) |")
        out.append("|---|---:|---:|---:|---:|")
        for d in dsets:
            res = results[(d, m)]
            if res["status"] == "ok":
                rules = res["rules"] if res["rules"] is not None else "?"
                out.append("| %s | %.2f | %.2f | %s | %.1f |" % (
                    d, res["train"], res["test"], rules, res["time"]))
            else:
                out.append("| %s | — | %s | — | — |" % (d, res["status"]))
        out.append("")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


MODELS = {}  # populated in main() (lazy import); keys drive --list/-c


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--condition", action="append", metavar="NAME",
                    dest="models", help="model(s) to run (repeatable; default: all).")
    ap.add_argument("-d", "--dataset", action="append", metavar="NAME",
                    help="dataset(s) to run (repeatable; default: all).")
    ap.add_argument("--smallsuite", action="store_true",
                    help="drop the slow/degenerate datasets (%s)" % ", ".join(sorted(SMALL_DROP)))
    ap.add_argument("--suite", metavar="DIR",
                    help="directory of STEM.json datasets (default: test_suite/). "
                         "When given without -d, every *.json in DIR is run.")
    ap.add_argument("--seed", type=int, default=None,
                    help="model random_state where supported (default: none)")
    ap.add_argument("--timeout", type=float, default=None, metavar="SECS",
                    help="per-fit wall-clock cap; a model exceeding it is marked TIMEOUT")
    ap.add_argument("--log", metavar="FILE", help="write a markdown summary to FILE")
    ap.add_argument("--list", action="store_true", help="list models/datasets and exit")
    ns = ap.parse_args(argv[1:])

    global SUITE, TIMEOUT
    TIMEOUT = ns.timeout
    if ns.suite:
        SUITE = os.path.abspath(ns.suite)

    global MODELS
    try:
        MODELS = _make_models()
    except ImportError as e:
        ap.error("baselines deps missing (%s). Run: uv sync --group baselines" % e)

    if ns.list:
        print("models:")
        for name, f in MODELS.items():
            skip = "  [skipped in default sweeps]" if name in DEFAULT_SKIP else ""
            print("  %-18s %s%s" % (name, getattr(f, "desc", ""), skip))
        print("datasets:")
        for d in DATASETS:
            print("  ", d)
        return 0

    models = ns.models or [m for m in MODELS if m not in DEFAULT_SKIP]
    if ns.dataset:
        dsets = ns.dataset
    elif ns.suite:  # no -d + custom suite: run every *.json in the dir
        dsets = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(SUITE, "*.json")))
    else:
        dsets = [d for d in DATASETS if not (ns.smallsuite and d in SMALL_DROP)]
    for m in models:
        if m not in MODELS:
            ap.error("unknown model %r (see --list)" % m)
    for d in dsets:
        if not os.path.exists(os.path.join(SUITE, d + ".json")):
            ap.error("no %s.json in %s" % (d, SUITE))

    results = {}
    total = len(dsets) * len(models)
    done = 0
    for d in dsets:
        for m in models:
            res = run_one(m, MODELS[m], d, ns.seed)
            results[(d, m)] = res
            done += 1
            if res["status"] == "ok":
                detail = "train=%.2f%% test=%.2f%% (%s rules, %.1fs)" % (
                    res["train"], res["test"],
                    res["rules"] if res["rules"] is not None else "?", res["time"])
            else:
                detail = "%s (%s)" % (res["status"], res.get("err", ""))
            print("[%*d/%d] %-16s %-18s %s" % (
                len(str(total)), done, total, d, m, detail), flush=True)

    seedmsg = ("seed=%d" % ns.seed) if ns.seed is not None else "no seed"
    print("\n==== test error %% (%s) ====" % seedmsg)
    w = max(14, max(len(d) for d in dsets) + 1)
    print("%-*s %s" % (w, "dataset", "  ".join("%18s" % m for m in models)))
    for d in dsets:
        cells = "  ".join("%18s" % cell(results[(d, m)]) for m in models)
        print("%-*s %s" % (w, d, cells))

    if ns.log:
        write_log(ns.log, models, dsets, results, ns.seed)
        print("\nwrote summary to %s" % ns.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
