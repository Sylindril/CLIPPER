"""Shared benchmark-suite metadata + an isolated slipper runner, used by the
scripts/ experiment harnesses (run_bestj.py, run_format_grid.py,
run_learning_curve.py, run_baseline_curve.py).

Five suites live here.  `tuning` is the 10-dataset development suite the
prompt/tool configuration was selected on; `test` is the 8-dataset tabular
evaluation suite; `text` is the 5-dataset text-classification suite
(src/tools/build_text_suite.py); `recent` holds a text dataset whose labels
were published after the model's training cutoff; `anon` is the test suite with meaningless
names (src/tools/anonymize_suite.py).  Harnesses take `--suite NAME` and default to
`tuning`.

BESTJ = the internal-CV round count of default SLIPPER on each development
dataset: the fixed `-n` budget that reproduces default-level accuracy without
paying for internal CV.  One source of truth so the harnesses can't drift.
Only the tuning suite has one -- the learning-curve harnesses recompute bestj
per cell on the subsample, so `test` needs none.
"""
import datetime
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BESTJ = {
    "hypothyroid": 1, "breast_cancer_wisconsin": 53, "cylinder_bands": 76,
    "pima_diabetes": 32, "titanic": 96, "abalone": 41, "german_credit": 45,
    "obesity": 97, "student_por": 6, "move1": 95,
}
DATASETS = list(BESTJ.keys())

# Test suite, in increasing train-split size (the order used in the results
# tables).  No bestj map: the learning-curve harnesses derive it per cell.
TEST_DATASETS = [
    "college", "water_potability", "stroke", "telco_churn",
    "compas", "heloc", "adult", "bank_marketing",
]


class Suite:
    """A benchmark suite: where its `STEM.json` files live, which datasets it
    holds, the optional fixed-budget `bestj` map, and the default output dir
    its harnesses write to (so two suites never share a results.csv)."""

    def __init__(self, name, dirname, datasets, bestj=None, sizes=(),
                 sample_key=None):
        self.name = name
        self.dirname = dirname
        self.datasets = list(datasets)
        self.bestj = bestj or {}
        self.sizes = list(sizes)        # default learning-curve size axis
        # stem -> name the train subsample is seeded on (default: the stem), so
        # an anonymized copy draws the same rows as its original
        self.sample_key = sample_key or {}

    def path(self, ds, ext=""):
        return os.path.join(REPO, self.dirname, ds + ext)

    def out(self, kind):
        """Default `scripts/<...>-out` dir for harness `kind` ('lc', 'bestj',
        ...).  The tuning suite keeps its historical names so existing
        results.csv files stay in place."""
        stem = {"lc": "lc-out", "bestj": "bestj-out"}.get(kind, kind + "-out")
        if self.name != "tuning":
            stem = "%s-%s" % (self.name, stem)
        return os.path.join(REPO, "scripts", stem)


TUNING = Suite("tuning", "tuning_suite", DATASETS, BESTJ,
               sizes=[100, 200, 300, 400, 800])
# The test-suite curve is deliberately coarser and reaches further: every
# dataset but college (750 train rows) runs the full range, so D=8/8/8/7/7.
TEST = Suite("test", "test_suite", TEST_DATASETS,
             sizes=[100, 250, 600, 1200, 2000])
# Text-classification suite (not reported in the paper).
TEXT = Suite("text", "text_suite",
             ["sentiment_amazon", "sentiment_imdb", "sentiment_yelp",
              "sms_spam", "youtube_spam"],
             sizes=[50, 100, 200, 300, 400, 500, 750, 1000])
# Post-cutoff text dataset (labels published after the model's training cutoff).
RECENT = Suite("recent", "recent_suite", ["inflation_abstracts"],
               sizes=[50, 100, 200, 300, 400, 500, 750, 1000])
# The test suite with gensym stems/attributes/classes/values (src/tools/anonymize_suite.py).
# Subsets are seeded on the original stem, so cells pair with `test`.
def _anon_suite(name, dirname, orig):
    path = os.path.join(REPO, dirname, "MAPPING.json")
    if not os.path.exists(path):
        return None
    import json
    with open(path) as f:
        key = {k: v["original"] for k, v in json.load(f).items()}
    inv = {v: k for k, v in key.items()}
    return Suite(name, dirname, [inv[d] for d in orig.datasets if d in inv],
                 sizes=orig.sizes, sample_key=key)


ANON = _anon_suite("anon", "anon_suite", TEST)
SUITES = {s.name: s for s in (TUNING, TEST, TEXT, RECENT, ANON) if s}


def add_suite_arg(ap):
    """Register the shared `--suite` flag on an argparse parser."""
    ap.add_argument("--suite", default="tuning", choices=sorted(SUITES),
                    help="benchmark suite to run (default: tuning)")
    return ap


def get(name):
    if name not in SUITES:
        sys.exit("unknown suite %r; known: %s" % (name, ", ".join(sorted(SUITES))))
    return SUITES[name]

_TEST_ERR_RE = re.compile(r"^Test error rate:\s+([\d.]+)%", re.MULTILINE)


def _advice_ext(flags):
    """The extension passed to --include-advice EXT (as `--include-advice .ext`
    or `--include-advice=.ext`), normalized with a leading dot; None if absent.
    Mirrors cli.py so the matching STEM<EXT> file gets linked into the cell."""
    for i, f in enumerate(flags):
        ext = None
        if f == "--include-advice" and i + 1 < len(flags):
            ext = flags[i + 1]
        elif f.startswith("--include-advice="):
            ext = f.split("=", 1)[1]
        if ext:
            return ext if ext.startswith(".") else "." + ext
    return None


def run_slipper_isolated(celldir, ds, flags, suite=TUNING, stem=None):
    """Run ``slipper <flags> <ds>`` inside its own dir with a symlink to the
    dataset json, so concurrent runs never race on the shared STEM.shyp /
    STEM.pyhyp.py and the committed <suite>/*.shyp references stay clean.

    ``flags`` is the full slipper argument list *except* the stem (e.g.
    ["--seed", "3", "-R", "", "-i0", "-n", "41"]).  Returns the test error
    rate (percent) or None on failure.  Every artifact the run writes (.shyp,
    .pred with ``-p``, ...) lands on ``os.path.join(celldir, ds)``.

    ``stem`` overrides the ``suite``/``ds`` lookup with an explicit source stem
    (``/path/to/foo`` for ``/path/to/foo.json``), for datasets that live outside
    a registered Suite; sidecars are then resolved next to it.
    """
    os.makedirs(celldir, exist_ok=True)
    # abspath because a relative stem would be resolved relative to the symlink's
    # own directory (the cell), not the caller's cwd.
    src = os.path.abspath(stem) if stem else suite.path(ds)
    # Sidecars the run may need, linked so the flags work inside isolated cells:
    # .info (task background) whenever present, and STEM<EXT> for --include-advice.
    exts = [".json"] + ([".info"] if os.path.exists(src + ".info") else [])
    adv = _advice_ext(flags)
    if adv:
        if os.path.exists(src + adv):
            if adv not in exts:
                exts.append(adv)
        else:
            sys.stderr.write(f"[WARN] {ds}{adv} not found; "
                             f"--include-advice will have no advice for {ds}\n")
    for ext in exts:
        link = os.path.join(celldir, ds + ext)
        if not os.path.exists(link):
            os.symlink(src + ext, link)
    cmd = ["uv", "run", "slipper", *flags, os.path.join(celldir, ds)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                          encoding="utf-8", errors="replace")
    m = _TEST_ERR_RE.search(proc.stdout)
    if not m:
        sys.stderr.write(f"[FAIL] {ds} flags={flags} rc={proc.returncode}\n"
                         f"{(proc.stderr or proc.stdout)[-500:]}\n")
        return None
    return float(m.group(1))


def write_run_config(out, name, config):
    """Write ``<out>/<name>_config.yaml`` capturing the run parameters plus a
    reproducibility header (local timestamp + short git commit), so every
    xxx-out directory records exactly what produced its results.  ``config`` is
    a dict of scalars/lists/dicts.  Rewritten on each (re)launch; returns path.
    """
    import yaml
    os.makedirs(out, exist_ok=True)
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or None
    except Exception:
        sha = None
    header = {
        "run_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": sha,
    }
    path = os.path.join(out, "%s_config.yaml" % name)
    with open(path, "w") as f:
        yaml.safe_dump({**header, **config}, f, sort_keys=False, default_flow_style=False)
    return path
