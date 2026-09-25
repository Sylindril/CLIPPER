"""optimize.py - choose optimal thresholds / symbols for wildcard tests.

``best_threshold`` finds the numeric cutpoint for a ``<=``/``>=`` wildcard;
``best_symbol`` finds the best value for a ``=``/``!=``/``~``/``!~`` wildcard.
Both score candidates with the installed ``config.Value_function`` and return
the chosen value together with its (pos, neg) weighted coverage, as a tuple.
"""

from . import config
from . import names as names_mod
from .names import AttrKind
from .symbols import Operator

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# numeric thresholds
# ---------------------------------------------------------------------------

def _count_threshold(gsym, cl, data, th):
    """Weighted (pos, neg) coverage of ``gsym`` at cutpoint ``th`` over ``data``.

    Matches the coverage semantics of the sweep below: OPLE covers av <= th,
    OPGE covers av >= th; missing values are never covered.
    """
    index = gsym.attr_index
    le = gsym.op == Operator.OPLE
    pos = neg = 0.0
    for exi in data:
        av = exi.inst[index]
        if av.kind == AttrKind.MISSING_VALUE:
            continue
        if av.num <= th if le else av.num >= th:
            if exi.lab.nom is cl:
                pos += exi.wt
            else:
                neg += exi.wt
    return pos, neg


def best_threshold(gsym, cl, data, old_info, oldp, oldn):
    """Return (best_threshold, pos, neg) for the wildcard cutpoint of gsym."""
    index = gsym.attr_index
    pairs = []
    num_pos = num_neg = 0.0
    for exi in data:
        av = exi.inst[index]
        if av.kind != AttrKind.MISSING_VALUE:
            lab = (exi.lab.nom is cl)
            pairs.append((av.num, exi.wt, lab))
            if lab:
                num_pos += exi.wt
            else:
                num_neg += exi.wt
    pairs.sort(key=lambda p: p[0])

    # with no negative data, pick the extreme value
    if num_neg == 0:
        if num_pos == 0:
            return 0.0, num_pos, num_neg
        if gsym.op == Operator.OPGE:
            return pairs[0][0], num_pos, num_neg
        return pairs[-1][0], num_pos, num_neg

    # general case: sweep thresholds in increasing order
    num_pos_lt = num_neg_lt = 0.0
    num_pos_eq = num_neg_eq = 0.0
    num_pos_gt = num_pos
    num_neg_gt = num_neg
    best_val = _NEG_INF
    best_th = 0.0
    pos = neg = 0.0
    n = len(pairs)
    i = 0
    while i < n:
        th = pairs[i][0]
        num_pos_lt += num_pos_eq
        num_neg_lt += num_neg_eq
        num_pos_eq = num_neg_eq = 0.0
        while i < n and pairs[i][0] == th:
            wti = pairs[i][1]
            if pairs[i][2]:
                num_pos_eq += wti
            else:
                num_neg_eq += wti
            i += 1
        # subtracting partial sums in floating point can drift the running
        # gt-counters to ~-1e-17; clamp so the values fed to sqrt-based value
        # functions (e.g. newboost_value) stay non-negative.
        num_pos_gt = max(0.0, num_pos_gt - num_pos_eq)
        num_neg_gt = max(0.0, num_neg_gt - num_neg_eq)
        if gsym.op == Operator.OPLE:
            ptmp = num_pos_lt + num_pos_eq
            ntmp = num_neg_lt + num_neg_eq
        else:  # OPGE
            ptmp = num_pos_gt + num_pos_eq
            ntmp = num_neg_gt + num_neg_eq
        val = config.Value_function(None, old_info, oldp, oldn, ptmp, ntmp)
        if val >= best_val:   # ties favour the larger threshold
            best_val = val
            best_th = th
            pos = ptmp
            neg = ntmp
    return best_th, pos, neg


# ---------------------------------------------------------------------------
# symbolic / set-valued wildcards
# ---------------------------------------------------------------------------

# per-field statistics accumulated while scanning candidate wildcard values
_Nump = None
_Numn = None
_Lastx = None
_Visited = []
_Allp = 0.0
_Alln = 0.0


def n_visited_symbols():
    return len(_Visited)


def visited_symbol(i):
    return _Visited[i]


def pos_field_stat(s):
    return _Nump[s.index], _Numn[s.index]


def neg_field_stat(s):
    return _Allp - _Nump[s.index], _Alln - _Numn[s.index]


def compute_field_stats(cl, attr_index, data):
    """Count, for class ``cl``, the weighted pos/neg occurrences of each
    distinct value of field ``attr_index`` over ``data``."""
    global _Nump, _Numn, _Lastx, _Visited, _Allp, _Alln
    n = names_mod.n_symbolic_values()

    if _Nump is None or len(_Nump) < n:
        _Nump = [0.0] * n
        _Numn = [0.0] * n
        _Lastx = [-1] * n
        _Visited = []
    else:
        for sym in _Visited:
            sx = sym.index
            _Nump[sx] = _Numn[sx] = 0.0
            _Lastx[sx] = -1

    _Allp = _Alln = 0.0
    _Visited = []
    symbolic = names_mod.symbolic_field(attr_index)
    is_set = names_mod.set_field(attr_index)

    for i, exi in enumerate(data):
        av = exi.inst[attr_index]
        if av.kind == AttrKind.MISSING_VALUE:
            continue
        posi = (exi.lab.nom is cl)
        if posi:
            _Allp += exi.wt
        else:
            _Alln += exi.wt
        if symbolic:
            sym = av.sym
            sx = sym.index
            if posi:
                _Nump[sx] += exi.wt
            else:
                _Numn[sx] += exi.wt
            if _Lastx[sx] == -1:
                _Visited.append(sym)
                _Lastx[sx] = i
        elif is_set:
            for sym in av.set:
                sx = sym.index
                if _Lastx[sx] != i:
                    if _Lastx[sx] == -1:
                        _Visited.append(sym)
                    _Lastx[sx] = i
                    if posi:
                        _Nump[sx] += exi.wt
                    else:
                        _Numn[sx] += exi.wt
        else:
            raise AssertionError("compute_field_stats on non-symbol/set field")


def best_symbol(gsym, cl, data, old_info, oldp, oldn):
    """Return (best_symbol, pos, neg) for the wildcard value of gsym."""
    compute_field_stats(cl, gsym.attr_index, data)
    invert = gsym.op in (Operator.OPOUT, Operator.OPNEQ)

    best_val = _NEG_INF
    best_sym = None
    for i in range(n_visited_symbols()):
        sym = visited_symbol(i)
        if invert:
            ptmp, ntmp = neg_field_stat(sym)
        else:
            ptmp, ntmp = pos_field_stat(sym)
        val = config.Value_function(None, old_info, oldp, oldn, ptmp, ntmp)
        if val > best_val:    # ties favour the first-visited symbol
            best_val = val
            best_sym = sym

    if best_sym is not None:
        if invert:
            pos, neg = neg_field_stat(best_sym)
        else:
            pos, neg = pos_field_stat(best_sym)
    else:
        pos = neg = 0.0
    return best_sym, pos, neg
