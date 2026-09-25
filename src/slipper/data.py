"""data.py - examples and dataset utilities.

An Aval is a tagged attribute value; an Example is a labeled instance made
up of Avals.  The utility functions here are the dataset operations the
learner needs: shallow copies, in-place coverage filtering, and the
stratified shuffle / partition routines used for cross-validation and for
growing rules on a holdout split.

A dataset is just a Python ``list`` of Example objects.
"""

import os
import sys

import numpy as np

from . import counting, rng
from . import names as names_mod
from .names import AttrKind

# --- sampling-drilldown instrumentation ----------------------------------------
# When SLIPPER_TRACE_SAMPLE_WT is set, weighted_sample emits one stderr line
# per draw reporting the captured weight fraction (how much of the current
# boosting distribution's mass the M-example sample actually holds).  Env-gated
# so it's a no-op in normal runs and easy to remove once the study is done.
_TRACE_SAMPLE_WT = bool(os.environ.get("SLIPPER_TRACE_SAMPLE_WT"))
_sample_draw_ctr = 0


class Aval:
    __slots__ = ("kind", "sym", "num", "set", "text")

    def __init__(self, kind, sym=None, num=0.0, set=None, text=None):
        self.kind = kind
        self.sym = sym      # Symbol, for SYMBOL
        self.num = num      # float, for CONTINUOUS
        self.set = set      # list[Symbol], for SET
        self.text = text    # str, for TEXT (kept out of the symbol table: a
                            # raw note is not a domain value the grammar can
                            # enumerate, and interning one per row would bloat
                            # the oblist)


class Example:
    __slots__ = ("inst", "lab", "wt", "index")

    def __init__(self, inst, lab, wt=1.0, index=0):
        self.inst = inst    # list[Aval]
        self.lab = lab      # Atom (lab.nom is the class Symbol)
        self.wt = wt
        self.index = index


# ---------------------------------------------------------------------------
# basic utilities
# ---------------------------------------------------------------------------

def new_data(m=0):
    """Make an empty dataset (the capacity hint is ignored in Python)."""
    return []


def copy_data(data):
    """Return a shallow copy: a new list sharing the same Example objects.

    The C ``copy_data`` shares backing storage so that in-place filtering of
    the copy permutes the original's array; because weight bookkeeping keys
    off ``Example.index`` (not array position), an independent shallow copy
    is behaviourally equivalent and safer.
    """
    return list(data)


def swap_out_example(data, i):
    """Mark the i-th example removed without destroying data (swap to end, drop)."""
    last = len(data) - 1
    data[i], data[last] = data[last], data[i]
    data.pop()


def remove_covered_examples(form, data):
    i = 0
    while i < len(data):
        if counting.form_covers(form, data[i].inst):
            swap_out_example(data, i)
        else:
            i += 1


def remove_uncovered_examples(form, data):
    i = 0
    while i < len(data):
        if not counting.form_covers(form, data[i].inst):
            swap_out_example(data, i)
        else:
            i += 1


# ---------------------------------------------------------------------------
# routines used for cross-validation
# ---------------------------------------------------------------------------

def stratify_and_shuffle_data(data, splits):
    """Randomise the order of ``data`` (in place) so class frequencies are
    preserved across the ``splits`` contiguous partitions.

    Note the deliberate quirk that the per-class shuffle below swaps over
    the range [k, n) rather than within the class block.
    """
    n = len(data)
    nclasses = len(names_mod.Classes)
    if n == 0:
        return

    # count #examples of each class
    cnt = [0] * nclasses
    for ex in data:
        cnt[ex.lab.nom.index] += 1
    # class_tmp[i] = offset at which to store examples of class i
    offset = [0] * nclasses
    for i in range(nclasses):
        offset[i] = sum(cnt[0:i])

    # copy in the data, sorting by class
    tmp = [None] * n
    pos = list(offset)
    for ex in data:
        ci = ex.lab.nom.index
        tmp[pos[ci]] = ex
        pos[ci] += 1

    # shuffle each class block (note the [k, n) swap range, see docstring above)
    i = 0
    while i < n:
        classi = tmp[i].lab.nom
        j = i
        while j < n and tmp[j].lab.nom is classi:
            j += 1
        if j < n:
            for k in range(i, j):
                r = k + rng.below(n - k)
                tmp[k], tmp[r] = tmp[r], tmp[k]
        i = j

    # store back, interleaving by splits
    data.clear()
    for j in range(splits):
        for i in range(j, n, splits):
            data.append(tmp[i])


def ith_stratified_partition(data, i, folds, data1, data2):
    """Put the i-th of ``folds`` partitions in data2 and the rest in data1
    (assumes ``data`` has been shuffled)."""
    n = len(data)
    m = n // folds
    data1.clear()
    data2.clear()
    lo = i * m
    hi = n if i == folds - 1 else (i + 1) * m
    for j in range(n):
        if lo <= j < hi:
            data2.append(data[j])
        else:
            data1.append(data[j])


def weighted_sample(dataset, k):
    """Return a fresh, renormalized example list drawn from ``dataset``.

    The returned list is always a list of *new* Example wrappers (sharing
    inst/lab/index with the originals), with .wt rescaled so the returned
    weights sum to 1.  Callers can therefore treat the result as a
    self-contained dataset that satisfies every "weights sum to 1" invariant
    the codebase relies on (see weighted_partition, _simplify_fct sf=3), and
    can mutate the sample's .wt without disturbing the parent dataset.

    If ``k`` <= 0 or ``k`` >= len(dataset), no sampling is done — every example
    is included (still freshly wrapped + renormalized).  Otherwise k examples
    are drawn without replacement, weighted by the input .wt (which boosting
    keeps normalized over the full training set).  Uses a separate seeded
    numpy Generator (rng.np_rng) so its draws don't perturb the RNG used
    elsewhere; two runs at the same seed agree.
    """
    n = len(dataset)
    if k <= 0 or k >= n:
        chosen = dataset
    else:
        weights = np.fromiter((ex.wt for ex in dataset), dtype=float, count=n)
        live = [i for i in range(n) if weights[i] > 0.0]
        if len(live) <= k:
            # Fewer candidates than the sample size: take them all rather than
            # letting numpy raise.  Only reachable under --weighting cover, which
            # zeroes covered examples; boosting weights are always positive.
            chosen = [dataset[i] for i in live]
        else:
            idx = rng.np_rng().choice(n, size=k, replace=False, p=weights)
            chosen = [dataset[i] for i in idx]
    total = sum(ex.wt for ex in chosen) or 1.0
    if _TRACE_SAMPLE_WT:
        # total (pre-renormalization) is the captured weight fraction: the input
        # .wt sums to 1 over the full training set, so this is the share of the
        # boosting distribution's mass held by the drawn sample.
        global _sample_draw_ctr
        _sample_draw_ctr += 1
        print("SAMPLEWT draw=%d n=%d k=%d nchosen=%d captured=%.6f"
              % (_sample_draw_ctr, n, (k if 0 < k < n else n), len(chosen), total),
              file=sys.stderr, flush=True)
    scale = 1.0 / total
    return [Example(ex.inst, ex.lab, ex.wt * scale, ex.index) for ex in chosen]


def stratified_weighted_sample(dataset, k, posc):
    """Like ``weighted_sample`` but class-balanced: draw ~k/2 positive and ~k/2
    negative examples (weighted, without replacement) instead of k drawn from
    the pooled distribution.

    ``posc`` is the positive-class nom (an example is positive iff
    ``ex.lab.nom is posc``).  Positives get ``k // 2`` slots, negatives the
    remainder (so an odd k gives the extra slot to the majority-safe negative
    side).  If a class has fewer than its target available, all of it is taken
    (no back-fill from the other class — the point is balance, not hitting
    exactly k).  The returned list is freshly wrapped and renormalized to sum
    to 1, identical in every other respect to ``weighted_sample`` — so it
    satisfies the same "weights sum to 1" invariants and its parent-weight sum
    is still the captured fraction.

    k <= 0 or k >= len(dataset) means "no sampling" (full, renormalized), same
    as ``weighted_sample``.
    """
    n = len(dataset)
    if k <= 0 or k >= n:
        chosen = dataset
    else:
        pos = [ex for ex in dataset if ex.lab.nom is posc]
        neg = [ex for ex in dataset if ex.lab.nom is not posc]
        n_pos = min(len(pos), k // 2)
        n_neg = min(len(neg), k - k // 2)

        def _draw(pool, m):
            if m <= 0 or not pool:
                return []
            if m >= len(pool):
                return list(pool)
            w = np.fromiter((ex.wt for ex in pool), dtype=float, count=len(pool))
            tot = w.sum()
            w = w / tot if tot > 0 else None  # tot==0 -> uniform draw
            idx = rng.np_rng().choice(len(pool), size=m, replace=False, p=w)
            return [pool[i] for i in idx]

        chosen = _draw(pos, n_pos) + _draw(neg, n_neg)
    total = sum(ex.wt for ex in chosen) or 1.0
    if _TRACE_SAMPLE_WT:
        global _sample_draw_ctr
        _sample_draw_ctr += 1
        npos = sum(1 for ex in chosen if ex.lab.nom is posc)
        print("SAMPLEWT draw=%d n=%d k=%d nchosen=%d npos=%d captured=%.6f (stratified)"
              % (_sample_draw_ctr, n, (k if 0 < k < n else n), len(chosen), npos, total),
              file=sys.stderr, flush=True)
    scale = 1.0 / total
    return [Example(ex.inst, ex.lab, ex.wt * scale, ex.index) for ex in chosen]


def stratified_partition(data, splits, data1, data2):
    """Split ``data`` as for k-fold CV, putting the first k-1 partitions in
    data1 and the last in data2."""
    stratify_and_shuffle_data(data, splits)
    n = len(data)
    cut = int(n * (splits - 1.0) / splits)
    data1.clear()
    data1.extend(data[0:cut])
    data2.clear()
    data2.extend(data[cut:])
