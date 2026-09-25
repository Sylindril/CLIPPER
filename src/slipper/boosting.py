"""boosting.py - the SLIPPER confidence-rated boosting core (Cohen & Singer, 1999).

``soft_model`` is the entry point: optionally cross-validate to pick the
number of boosting rounds, then run ``soft_fit`` for that many rounds.  Each
round grows the best positive rule (``build_rule`` -> ``soft_refine`` ->
``soft_simplify``) or, if a default-rule update scores better, adjusts the
default; example weights are then updated multiplicatively.  Finally
``compress_soft_rules`` merges duplicate rules and sorts by confidence.

Classification (``soft_classify``) sums each matching rule's confidence
alpha = 0.5*ln((p+eps)/(n+eps)); the sign picks the class.

Value extraction (``_extracting()``, from a "task": "extract" dataset) reuses
this loop unchanged apart from four seams: candidates are scored by
extract.extract_evaluate, the default rule is abstain (so the default-update
branch adds nothing -- an early stop on the fixed --rule-pool, a discarded
round under the stochastic LLM builder), reweighting reads extract.rule_y, and
the error rate is 1 - exact match.  See extract.py.
"""

import math
import os
import sys
import textwrap
from collections import Counter
from functools import cmp_to_key

from tqdm import tqdm

from . import config, counting, data, extract, model, refine, trace, value
from . import names as names_mod
from .concept import Concept, PyRule, Rule, Strategy, fshow_concept
from .grammar import fprint_gsym
from .errors import fatal
from .symbols import sym_str

_MAXREAL = math.inf

# PyRule exceptions caught in _rule_covers during the current boosting round.
# Reset at the top of each round in soft_fit; reported at the end.
_round_exceptions = Counter()

# When SLIPPER_PROGRESS is set (e.g. by sweep.py), emit a cheap "@progress t n"
# line to stderr each boosting round so a parent process can drive a per-run bar.
_PROGRESS = os.environ.get("SLIPPER_PROGRESS")


def _emit_progress(t, n):
    if _PROGRESS:
        sys.stderr.write("@progress %d %d\n" % (t, n))
        sys.stderr.flush()


# --- polymorphic rule predicates (Rule | PyRule) ---

def _rule_covers(r, inst):
    """True iff rule ``r`` covers instance ``inst``.

    For a grammar Rule this defers to counting.form_covers; for a PyRule it
    evaluates the LLM-built ``fn`` against a {attr_name: value} dict.  A
    sandbox/runtime error makes the rule not cover the example, so a buggy
    LLM-generated predicate never crashes classification; the (type, msg) is
    tallied in _round_exceptions for the surrounding boosting round.
    """
    if isinstance(r, PyRule):
        from . import llm_rule_builder
        try:
            return bool(r.fn(llm_rule_builder.inst_to_dict(inst)))
        except Exception as e:
            _round_exceptions[(type(e).__name__, str(e))] += 1
            return False
    return counting.form_covers(r.antec, inst)


# --- the alpha / value / loss quantities from the SLIPPER objective ---

def _covering():
    """True under either separate-and-conquer weighting (see config.Weighting).

    "cover" and "coverlist" share all of the machinery: no default reweighting,
    the recover_dist cover step, the acceptance gate and its patience, and the
    first-match-wins classifier.  They differ only in what a rule may conclude.
    """
    return config.Weighting in ("cover", "coverlist")


def _coverlist():
    """True when --weighting coverlist is in force: each accepted rule carries
    the consequent its own (p, n) chose, so the hypothesis is an ordered
    decision list rather than a positive-only disjunction."""
    return config.Weighting == "coverlist"


def _extracting():
    """True when the loaded dataset is a value-extraction task (see extract.py).

    Rules return values, not bools: (p, n) comes from extract_evaluate, the
    default rule is abstain (so the default-update branch is off) and error is
    1 - exact match.  Everything else in this file is unchanged.
    """
    return config.Task_mode == "extract"


def _alphah(p, n):
    eps = config.Epsilon
    return 0.5 * math.log((p + eps) / (n + eps))


def _value(p, n):
    return math.sqrt(p) - math.sqrt(n) + 1


def _loss(p, n):
    return (1.0 - p - n) + p * math.exp(_alphah(n, p)) + n * math.exp(_alphah(p, n))


# ---------------------------------------------------------------------------
# soft_model: learn a ruleset, choosing #rounds by internal cross-validation
# ---------------------------------------------------------------------------

def soft_model(train_data, test_data):
    Classes = names_mod.Classes
    if _extracting():
        # No classes to count or order: the single class is the abstain default
        # (dataset.ABSTAIN), and posclass is only carried through soft_fit.
        posclass = negclass = Classes[0]
    else:
        if len(Classes) != 2:
            fatal("only two-class version is implemented!")
        model.count_classes(train_data)
        model.reorder_classes(train_data)
        posclass = Classes[0]
        negclass = Classes[1]

    hyp = Concept()
    hyp.def_ = negclass
    hyp.rules = []
    config.Epsilon = 0.5 / len(train_data)

    n = len(train_data)
    dist = [0.0] * n
    initw = [0.0] * n

    # no cross-validation: just boost for the full N_boost rounds
    if config.Slipper_folds == 0:
        make_dist(train_data, dist, initw)
        soft_fit(posclass.nom, negclass.nom, train_data, dist, test_data, None, hyp)
        load_weights(train_data, initw)
        compress_soft_rules(hyp)
        return hyp

    # total_errors[t] accumulates holdout error after t rounds across folds
    total_errors = [0.0] * config.N_boost
    if config.Slipper_folds == -1:
        # single 1/3 holdout
        data1 = data.new_data()
        data2 = data.new_data()
        data.stratified_partition(train_data, 3, data1, data2)
        make_dist(data1, dist, initw)
        soft_fit(posclass.nom, negclass.nom, data1, dist, data2, total_errors, hyp)
        hyp.rules.clear()
        load_weights(data1, initw)
    else:
        # k-fold cross-validation
        data1 = data.new_data()
        data2 = data.new_data()
        data.stratify_and_shuffle_data(train_data, config.Slipper_folds)
        for i in tqdm(range(config.Slipper_folds), desc="xval folds", unit="fold",
                      disable=trace.tracing(trace.SUMM) or None):
            if trace.tracing(trace.SUMM):
                print("// running slipper on fold %d of internal cross-validation" % i)
            data.ith_stratified_partition(train_data, i, config.Slipper_folds, data1, data2)
            make_dist(data1, dist, initw)
            soft_fit(posclass.nom, negclass.nom, data1, dist, data2, total_errors, hyp)
            hyp.rules.clear()
            load_weights(data1, initw)

    # pick the number of rounds with lowest average holdout error
    if trace.tracing(trace.SUMM):
        print("// summary of cross-validation\n// T\tavg error")
    lowerr = _MAXREAL
    div = 1 if config.Slipper_folds == -1 else config.Slipper_folds
    bestj = 0
    for j in range(config.N_boost):
        if total_errors[j] < lowerr:
            lowerr = total_errors[j]
            bestj = j
        if trace.tracing(trace.SUMM):
            print("// %d\t%.4f" % (j, total_errors[j] / div))
    if trace.tracing(trace.SUMM):
        print("// best result (%g) after %d rounds" % (lowerr / config.Slipper_folds, bestj))

    # re-run for the chosen number of rounds on the full training set
    old_nboost = config.N_boost
    config.N_boost = bestj
    make_dist(train_data, dist, initw)
    soft_fit(posclass.nom, negclass.nom, train_data, dist, test_data, None, hyp)
    load_weights(train_data, initw)
    config.N_boost = old_nboost

    compress_soft_rules(hyp)
    return hyp


def soft_fit(posc, negc, train_data, dist, test_data, tot_errs, hyp):
    """Grow up to N_boost rules into ``hyp`` (side-effecting ``dist`` and, if
    given, ``tot_errs``)."""
    # The rule-search objective has to agree with what the hypothesis does with
    # the rule.  Boosting sums confidences, so it scores refinements with
    # newboost_value (sqrt(p) - sqrt(n) + 1).  A cover hypothesis just ORs its
    # rules, so it grows rules by FOIL gain (value.foil_gain) -- bits about the
    # class per surviving positive -- and sizes them with the matching holdout
    # objective (-S5, defaulted in cli.py).  --cover-objective linear swaps the
    # growth objective for value.cover_value (p - n + 1), which optimizes exactly
    # what the p > n acceptance gate below measures.  A coverlist rule may
    # conclude either class, so it grows on the two-sided form of whichever
    # objective is selected -- a majority-negative region scores as well as the
    # mirror-image positive one instead of below the universal rule.
    if _covering():
        linear = config.Cover_objective == "linear"
        if _coverlist():
            config.Value_function = (value.two_sided_cover_value if linear
                                     else value.two_sided_foil_gain)
        else:
            config.Value_function = value.cover_value if linear else value.foil_gain
    else:
        config.Value_function = value.newboost_value

    # construct and apply the default rule.  Under --weighting cover there is no
    # distribution to shift: the weights stay uniform over the uncovered set, and
    # the default is just the fallback class, so only its stats are recorded.
    # Extraction has no default class -- the default is to abstain, a rule that
    # never fires, so its stats are (0, 0) and there is nothing to reweight for.
    allp, alln = (0.0, 0.0) if _extracting() else count_weight(posc, train_data, dist)
    if not _covering() and not _extracting():
        reweight_default(posc, train_data, dist, allp, alln)
    hyp.nposx = allp
    hyp.nnegx = alln
    if trace.tracing(trace.SUMM):
        print("// weights for default: %g/%g, confidence %g"
              % (hyp.nposx, hyp.nnegx, _alphah(hyp.nposx, hyp.nnegx)))

    t = 0
    min_err = _MAXREAL
    err = 0.0
    upward_trend_ctr = 0
    cover_fails = 0
    bar = tqdm(total=config.N_boost, desc="boost", unit="round", leave=False,
               disable=trace.tracing(trace.SUMM) or None)
    _emit_progress(0, config.N_boost)
    while t < config.N_boost:
        _round_exceptions.clear()
        stop_reason = None
        if test_data:
            err = soft_error_rate(hyp, test_data)
            if tot_errs is not None:
                tot_errs[t] += err
            if err <= min_err + config.Epsilon:
                min_err = err
                upward_trend_ctr = 0
            elif err > min_err:
                upward_trend_ctr += 1
                if config.Upward_trend_max != 0 and upward_trend_ctr >= config.Upward_trend_max:
                    if tot_errs is not None:
                        if trace.tracing(trace.SUMM):
                            print("// aborting CV-boosting (round %d)" % t)
                        t += 1
                        while t < config.N_boost:
                            tot_errs[t] = _MAXREAL
                            t += 1
                        bar.close()
                        return
        if trace.tracing(trace.SUMM):
            line = "// boost round %3d rules %3d " % (t, len(hyp.rules))
            if test_data:
                line += "trend %1d test %8.4f " % (upward_trend_ctr, err)
            line += " train %.4f" % soft_error_rate(hyp, train_data)
            print(line)

        # find the best positive rule
        load_weights(train_data, dist)
        # Per-round weighted sample drawn from the current boosting distribution.
        # Sample_size <= 0 or >= N means "use the full training set" — same
        # semantics for both rule-builder paths so they can be compared fairly.
        # weighted_sample renormalizes weights to sum to 1 within the sample.
        # --stratify forces the *first* round's sample class-balanced (k/2 pos +
        # k/2 neg); later rounds self-stratify as boosting up-weights the hard
        # (misclassified) examples, so the gate is t == 0 only.
        if config.Stratify and t == 0:
            sample = data.stratified_weighted_sample(train_data, config.Sample_size, posc)
        else:
            sample = data.weighted_sample(train_data, config.Sample_size)
        if _extracting():
            # Two rule builders, same (rule, p, n) triple, both already scored
            # on the full training set (like the discrete path's rescoring
            # below): the LLM one when -R is set, else the deterministic pool.
            if config.Rule_llm_model:
                # design A (a sandboxed extract()) or A' (a regex spec), per
                # --rule-body; see llm_rule_builder.build_extract_rule.
                from . import llm_rule_builder
                new_rule, p, n = llm_rule_builder.build_extract_rule(
                    train_data, sample, t)
            else:
                # Score the whole --rule-pool under the live weights, keep the
                # best.
                new_rule, p, n = extract.best_pool_rule(train_data)
        elif config.Rule_llm_model:
            from . import llm_rule_builder
            if config.Pruning_agent:
                # Two-stage path: a tool-less grow agent proposes a rule from the
                # M-sample (== the "grow set"), then a prune agent simplifies it.
                # Coverage tools score the grow sample; value tools score full
                # train; final (p, n) is on full train (below).
                g_source, _g_fn, _, _ = llm_rule_builder.build_pyrule(
                    posc, sample, sample, t, grow_mode=True)
                source, fn, p, n = llm_rule_builder.prune_pyrule(
                    posc, sample, train_data, g_source, t)
            else:
                # Captured weight fraction of this round's sample: the drawn
                # examples' share of the L1-normalized boosting distribution.
                # train_data still carries those weights (weighted_sample
                # renormalizes only its own fresh wrappers), so it's just the
                # parent-weight sum over the sampled indices; 1.0 when no
                # subsampling happened. Drives the per-round count_coverage gate
                # (--count-coverage-max-capmean).
                if len(sample) < len(train_data):
                    _wt = {ex.index: ex.wt for ex in train_data}
                    captured = sum(_wt[s.index] for s in sample)
                else:
                    captured = 1.0
                source, fn, p, n = llm_rule_builder.build_pyrule(
                    posc, train_data, sample, t, captured=captured)
            new_rule = PyRule(conseq=posc, source=source, fn=fn, nposx=p, nnegx=n)
        else:
            new_form = []
            new_deriv = []
            refine.make_the_universe(posc, new_form)
            build_rule(new_form, new_deriv, posc, sample)
            # build_rule's (p, n) are on the sample's renormalized scale; rescore
            # on the full training set so pval and defval live on the same scale.
            p, n = counting.count_examples(new_form, posc, train_data)
            new_rule = Rule(conseq=posc, antec=new_form, nposx=p, nnegx=n, deriv=new_deriv)
        pval = _value(p, n)
        if not isinstance(new_rule, Rule):
            if trace.tracing(trace.SUMM):
                print("// best rule (val %g: %g/%g): py:" % (pval, p, n))
                for line in (new_rule.source or "").splitlines() or [""]:
                    print("//   " + line)
        elif trace.tracing(trace.LONG):
            print("// best rule (val %g: %g/%g):%s" % (pval, p, n, refine.form_str(new_rule.antec)))

        # compare against updating the default rule
        if _extracting():
            # The default is abstain: it fires on nothing, so its (p, n) is
            # (0, 0) and defval is _value(0, 0) == 1 -- the do-nothing baseline.
            # A candidate beats it exactly when sqrt(p) > sqrt(n); when none
            # does there is no default weight to move instead, so we stop.
            allp = alln = 0.0
            defval = _value(0.0, 0.0)
        else:
            allp, alln = count_weight(posc, train_data, dist)
            defval = max(_value(allp, alln), _value(alln, allp))
        if _covering():
            # Cover mode needs an acceptance test in place of the default-weight
            # update (it has no default weight to move).  It isn't defval: that
            # comparison includes flipping the default class, which a covering
            # learner can't do.  In a disjunction every covered example is
            # predicted positive, so adding this rule moves weighted training
            # error by (n - p) -- accept exactly while p > n, which also rejects
            # the universal rule whenever what's left is majority-negative.
            # Without a test the last round always adds it and the disjunction
            # then fires on everything.  Rounds after a stop score identically,
            # so the rest of tot_errs gets that error -- not _MAXREAL -- keeping
            # folds that stop at different rounds comparable in soft_model.
            #
            # Only Cover_patience consecutive failures stop the run.  At the
            # default of 1 that is the legacy hard stop on the first failure;
            # above it a failed gate skips the round instead -- nothing is added,
            # so weighted training error is unchanged, and the next round draws a
            # fresh sample.  Worth raising on the stochastic paths (-M
            # subsampling, the LLM builder), where one unlucky sample used to
            # kill the whole run.
            #
            # Under coverlist the rule chooses its own consequent, so the same
            # arithmetic gives `p != n`: whichever side it leans to, predicting
            # that side moves weighted training error by -|p - n|.  Only a tie
            # is undecidable, and a tie is the failure case -- it feeds the
            # patience counter exactly as `p <= n` does above.
            if (p == n) if _coverlist() else (p <= n):
                cover_fails += 1
                if cover_fails >= config.Cover_patience:
                    stop_reason = "no rule reduces error on the uncovered set"
                elif trace.tracing(trace.SUMM):
                    print("// - skipping round %d (%g/%g fails the gate; %d of %d)"
                          % (t, p, n, cover_fails, config.Cover_patience))
            else:
                cover_fails = 0
                if _coverlist() and n > p:
                    # The rule covers a majority-negative region; in a decision
                    # list that is a negative rule, not a rejected one.  (p, n)
                    # stay measured against posc so every rule's stats read on
                    # the same axis as the default's.
                    new_rule.conseq = negc
                hyp.rules.append(new_rule)
                if trace.tracing(trace.SUMM):
                    print("// + r%d (cover %g/%g): %s"
                          % (len(hyp.rules), p, n, _rule_str(new_rule)))
                if not recover_dist(new_rule, posc, train_data, dist):
                    stop_reason = "no uncovered positives left"
        elif defval >= pval and _extracting():
            # The default is abstain, so there is no default weight to move
            # instead: this round adds nothing either way.  Whether it also ends
            # training depends on the rule builder.  --rule-pool is a fixed
            # finite set, scored exhaustively under the live weights every
            # round, so a candidate that loses to abstaining now would lose
            # again forever -- stop.  The LLM builder is stochastic: the next
            # call sees the same weights and can still answer differently (and a
            # round that exhausts its retries abstains by construction), so one
            # bad round must not truncate the whole ensemble -- consume the
            # round, append nothing, and keep going to the -n budget.
            if not config.Rule_llm_model:
                stop_reason = "no candidate rule beats abstaining"
            elif trace.tracing(trace.SUMM):
                print("// round %d: no rule beat abstaining, discarded" % t)
        elif defval >= pval:
            reweight_default(posc, train_data, dist, allp, alln)
            hyp.nposx = (math.exp(2 * _alphah(hyp.nposx, hyp.nnegx) + 2 * _alphah(allp, alln))
                         * (hyp.nnegx + config.Epsilon) - config.Epsilon)
            if hyp.nposx < 0:
                hyp.nposx = 0
            if trace.tracing(trace.SUMM):
                print("// + weights for default: %g/%g, confidence %g"
                      % (hyp.nposx, hyp.nnegx, _alphah(hyp.nposx, hyp.nnegx)))
        else:
            reweight_with_rule(new_rule, posc, train_data, dist, p, n)
            hyp.rules.append(new_rule)
            if trace.tracing(trace.SUMM):
                print("// + r%d (loss %g): %s" % (len(hyp.rules), _loss(p, n), _rule_str(new_rule)))
        if trace.tracing(trace.SUMM):
            scale = len(train_data)
            print("// weights after round %d: min=%g max=%g (scaled by N=%d)"
                  % (t, scale * min(dist), scale * max(dist), scale))
        if _round_exceptions:
            total = sum(_round_exceptions.values())
            print("// round %d PyRule exceptions: %d total across %d distinct"
                  % (t, total, len(_round_exceptions)))
            for (typ, msg), n in _round_exceptions.most_common():
                print("//   %dx %s: %s" % (n, typ, msg))
        t += 1
        bar.update(1)
        _emit_progress(t, config.N_boost)
        if stop_reason is not None:
            if trace.tracing(trace.SUMM):
                print("// %s after round %d, stopping" % (stop_reason, t - 1))
            if tot_errs is not None and test_data:
                final = soft_error_rate(hyp, test_data)
                for j in range(t, config.N_boost):
                    tot_errs[j] += final
            break
    bar.close()


# ---------------------------------------------------------------------------
# growing a single rule
# ---------------------------------------------------------------------------

def build_rule(form, deriv, cl, dataset):
    """Greedily refine, then (optionally) simplify a rule.  Returns (p, n)."""
    if not config.Simplify:
        soft_refine(form, deriv, cl, dataset)
        return counting.count_examples(form, cl, dataset)
    grow_data = data.new_data()
    prune_data = data.new_data()
    weighted_partition(dataset, grow_data, prune_data)
    soft_refine(form, deriv, cl, grow_data)
    return soft_simplify(form, deriv, cl, grow_data, prune_data)


def soft_refine(ref, deriv, cl, dataset):
    """Greedily add the best-scoring grammar refinement until none improves."""
    data1 = data.copy_data(dataset)
    p, n = counting.count_examples(ref, cl, data1)
    last_val = config.Value_function(None, 0, 0, 0, p, n)

    while True:
        best_val = -_MAXREAL
        best_i = -1
        refine.compute_designated_refinements(ref, deriv, 0)
        for i in range(refine.n_designated_refinements()):
            vali = value.value(refine.refinement(i), cl, data1, ref, 0, p, n)
            if vali > best_val:
                best_val = vali
                best_i = i

        if best_val <= last_val:
            if trace.tracing(trace.LONG):
                print("// no improvement: best %g<=%g" % (best_val, last_val))
            return
        # adopt the refinement
        ref[:] = [g.copy() for g in refine.refinement(best_i)]
        deriv[:] = [s.copy() for s in refine.derivation(best_i)]
        last_val = best_val
        data.remove_uncovered_examples(ref, data1)
        if trace.tracing(trace.LONG):
            p1, n1 = counting.count_examples(ref, cl, data1)
            print("// refined to (v:%g, %g/%g):%s"
                  % (last_val, p1, n1, refine.form_str(ref)))


def _total_weight(dataset):
    return sum(ex.wt for ex in dataset)


def _simplify_fct(form, cl, totprune, grow_data, prune_data, baseprune=0.0,
                  flip=False):
    """Return (val, outp, outn) for sizing a rule against a holdout split.

    ``val`` is MINIMIZED by soft_simplify.  ``baseprune`` is the positive weight
    in prune_data, needed only by sf=5 (the FOIL objective) for its base rate.
    ``flip`` (coverlist only, sf=5 only) scores the rule as a *negative* rule --
    the same gain measured about the other class.  ``outp``/``outn`` are always
    reported against ``cl``, whichever side is being scored.
    """
    pprune, nprune = counting.count_examples(form, cl, prune_data)
    pgrow, ngrow = counting.count_examples(form, cl, grow_data)
    outp = pprune + pgrow
    outn = nprune + ngrow
    eps = config.Epsilon
    sf = config.Simplify_function
    if sf == 1:
        uncov = totprune - pprune - nprune
        val = (uncov
               + pprune * math.sqrt((ngrow + eps) / (pgrow + eps))
               + nprune * math.sqrt((pgrow + eps) / (ngrow + eps)))
    elif sf == 2:
        pprune /= totprune
        nprune /= totprune
        uncov = 1.0 - pprune - nprune
        val = (uncov + pprune * math.exp(_alphah(nprune, pprune))
               + nprune * math.exp(_alphah(pprune, nprune)))
    elif sf == 3:
        p = pgrow + pprune
        nn = ngrow + nprune
        uncov = 1.0 - p - nn
        val = uncov + p * math.exp(_alphah(nn, p)) + nn * math.exp(_alphah(p, nn))
    elif sf == 4:
        val = (ngrow + nprune) - (pgrow + pprune)
    elif sf == 5:
        # FOIL gain of the candidate rule against the base rate, measured on the
        # holdout (negated, since soft_simplify minimizes).  The counterpart of
        # value.foil_gain for generalization: growth asks "what does this
        # condition buy over the rule so far", sizing asks "what does the rule
        # that's left buy over predicting the base rate", on data it didn't see.
        # 0.0 (= no gain, ties the universal rule) whenever it can't be scored.
        # A coverlist rule grown for the negative class is the same measurement
        # with the classes exchanged, base rate included.
        gp, gn, gbase = ((nprune, pprune, totprune - baseprune) if flip
                         else (pprune, nprune, baseprune))
        if gp <= 0 or gbase <= 0 or totprune <= 0:
            val = 0.0
        else:
            val = -gp * (math.log2(gp / (gp + gn))
                         - math.log2(gbase / totprune))
    else:
        fatal("unimplemented simplification function")
    return val, outp, outn


def soft_simplify(form, deriv, cl, grow_data, prune_data):
    """Greedily generalize a rule to minimize the holdout-sizing objective.
    Returns (p, n)."""
    totprune = _total_weight(prune_data)
    baseprune = sum(ex.wt for ex in prune_data if ex.lab.nom is cl)
    best_val, outp, outn = _simplify_fct(form, cl, totprune, grow_data, prune_data,
                                         baseprune)
    # Which class this rule is for is a property of the rule, not of each
    # candidate generalization: decide it once from the grown form and size
    # every candidate on that side, so the sizing objective and the consequent
    # soft_fit will assign agree.
    flip = _coverlist() and outn > outp
    if flip:
        best_val, outp, outn = _simplify_fct(form, cl, totprune, grow_data,
                                             prune_data, baseprune, flip=True)
    best_j = -1
    gate_aware = _covering() and config.Cover_sizing == "gate"
    refine.compute_designated_generalizations(cl, form, deriv)
    for j in range(refine.n_designated_generalizations()):
        val, pj, nj = _simplify_fct(refine.generalization(j), cl, totprune,
                                    grow_data, prune_data, baseprune, flip=flip)
        # Every sizing objective rewards coverage, so left alone it will happily
        # generalize a grown rule back across soft_fit's p > n acceptance gate --
        # and under cover mode a rejected rule doesn't just cost a round, it can
        # end the run.  --cover-sizing gate opts out of that: only take a
        # generalization the gate would still accept ((pj, nj) counts grow+prune,
        # i.e. the whole sample the gate will measure), and give ties to the more
        # specific rule instead of to the last, most general candidate.  The
        # un-generalized form is never skipped either way -- it is the fallback.
        # Under coverlist "the gate would accept" reads on the rule's own side,
        # which also rules out a generalization that flips it to the other class
        # (a different rule, not a more general one); a tie still fails.
        passes = (nj > pj) if flip else (pj > nj)
        take = (val < best_val and passes) if gate_aware else val <= best_val
        if take:
            best_val = val
            best_j = j
            outp = pj
            outn = nj
    if best_j != -1:
        form[:] = [g.copy() for g in refine.generalization(best_j)]
        deriv[:] = [s.copy() for s in refine.derivation(best_j)]
    return outp, outn


def weighted_partition(dataset, grow_data, prune_data):
    """Split ``dataset`` so prune_data gets ~1/3 of the weight, grow_data the
    rest (after a stratified shuffle into 3 parts)."""
    grow_data.clear()
    prune_data.clear()
    data.stratify_and_shuffle_data(dataset, 3)
    n = len(dataset)
    w = 0.0
    for j in range(3):
        for i in range(j, n, 3):
            ex = dataset[i]
            w += ex.wt
            if w <= 1.0 / 3.0:
                prune_data.append(ex)
            else:
                grow_data.append(ex)


# ---------------------------------------------------------------------------
# distributions / weights
# ---------------------------------------------------------------------------

def make_dist(dataset, d, save):
    totw = 0.0
    for i, exi in enumerate(dataset):
        exi.index = i
        save[i] = d[i] = exi.wt
        totw += exi.wt
    normalize_dist(dataset, d, totw)


def normalize_dist(dataset, d, z):
    for exi in dataset:
        d[exi.index] /= z


def count_weight(posc, dataset, d):
    posw = negw = 0.0
    for exi in dataset:
        if exi.lab.nom is posc:
            posw += d[exi.index]
        else:
            negw += d[exi.index]
    return posw, negw


def reweight_default(posc, dataset, d, posw, negw):
    a = _alphah(posw, negw)
    new_totw = 0.0
    for exi in dataset:
        yi = 1 if exi.lab.nom is posc else -1
        d[exi.index] *= math.exp(-1.0 * yi * a)
        new_totw += d[exi.index]
    normalize_dist(dataset, d, new_totw)


def reweight_rule(conseq, antec, dataset, d, posw, negw):
    factor = -1.0 * _alphah(posw, negw)
    new_totw = 0.0
    for exi in dataset:
        yi = 1 if exi.lab.nom is conseq else -1
        if counting.form_covers(antec, exi.inst):
            d[exi.index] *= math.exp(yi * factor)
        new_totw += d[exi.index]
    normalize_dist(dataset, d, new_totw)


def reweight_with_rule(rule, conseq, dataset, d, posw, negw):
    """Polymorphic reweighting: PyRules use their stored fn; grammar rules
    delegate to reweight_rule (so the existing path is bit-identical)."""
    if _extracting():
        # Same multiplicative update; only y changes: +1 where the extracted
        # value is right, -1 where it is wrong, and no update where the rule
        # abstained (extract.rule_y returns 0).
        factor = -1.0 * _alphah(posw, negw)
        new_totw = 0.0
        for exi in dataset:
            yi = extract.rule_y(rule, exi)
            if yi:
                d[exi.index] *= math.exp(yi * factor)
            new_totw += d[exi.index]
        normalize_dist(dataset, d, new_totw)
        return
    if isinstance(rule, PyRule):
        factor = -1.0 * _alphah(posw, negw)
        new_totw = 0.0
        for exi in dataset:
            yi = 1 if exi.lab.nom is conseq else -1
            if _rule_covers(rule, exi.inst):
                d[exi.index] *= math.exp(yi * factor)
            new_totw += d[exi.index]
        normalize_dist(dataset, d, new_totw)
        return
    reweight_rule(conseq, rule.antec, dataset, d, posw, negw)


def recover_dist(rule, conseq, dataset, d):
    """Greedy set-cover reweighting (--weighting cover).

    Every example ``rule`` covers drops out (weight 0); the survivors share the
    weight uniformly, so the next round's rule builder — and its -M sample —
    sees only not-yet-covered examples.  Returns False when nothing worth
    covering is left (no uncovered example carrying ``conseq``), which is the
    natural stopping point for a covering learner.

    ``conseq`` is the positive class under coverlist too, even for a rule that
    concluded the negative one: once every positive is covered by some rule the
    default majority class already handles everything that falls through, so
    "no uncovered positives left" stays the exhaustion test.
    """
    for exi in dataset:
        if _rule_covers(rule, exi.inst):
            d[exi.index] = 0.0
    live = [exi for exi in dataset if d[exi.index] > 0.0]
    if not any(exi.lab.nom is conseq for exi in live):
        return False
    for exi in live:
        d[exi.index] = 1.0 / len(live)
    return True


def load_weights(dataset, d):
    for exi in dataset:
        exi.wt = d[exi.index]


# ---------------------------------------------------------------------------
# removing duplicate rules from a ruleset
# ---------------------------------------------------------------------------

def _rule_weight_cmp(r1, r2):
    d = _alphah(r1.nposx, r1.nnegx) - _alphah(r2.nposx, r2.nnegx)
    if d < 0:
        return 1
    if d > 0:
        return -1
    return 0


def _conseq_index(r):
    # Extraction rules have no consequent (the value is the prediction).
    return -1 if r.conseq is None else r.conseq.index


def _rule_cmp(r1, r2):
    # Sort PyRules (and extraction rules) after grammar rules; within them dedup
    # by source string -- for a RegexRule that's its canonical spec -- so only
    # identical rules collapse under compress_soft_rules.
    py1 = not isinstance(r1, Rule)
    py2 = not isinstance(r2, Rule)
    if py1 and not py2:
        return 1
    if py2 and not py1:
        return -1
    if py1 and py2:
        c = _conseq_index(r1) - _conseq_index(r2)
        if c != 0:
            return c
        if r1.source < r2.source:
            return -1
        if r1.source > r2.source:
            return 1
        return 0
    c = r1.conseq.index - r2.conseq.index
    if c != 0:
        return c
    c = len(r1.antec) - len(r2.antec)
    if c != 0:
        return c
    for i in range(len(r1.antec)):
        g1 = r1.antec[i]
        g2 = r2.antec[i]
        if g1.nonterm and not g2.nonterm:
            return 1
        if g2.nonterm and not g1.nonterm:
            return -1
        if not g1.nonterm:
            c = g1.attr_index - g2.attr_index
            if c != 0:
                return c
            c = int(g1.op) - int(g2.op)
            if c != 0:
                return c
            if g1.value.nom and not g2.value.nom:
                return -1
            if g2.value.nom and not g1.value.nom:
                return -1
            if g1.value.nom:
                c = g1.value.nom.index - g2.value.nom.index
                if c != 0:
                    return c
            else:
                if g1.value.num > g2.value.num:
                    return 1
                if g1.value.num < g2.value.num:
                    return -1
    return 0


def compress_soft_rules(hyp):
    if _covering():
        # A cover hypothesis is an ordered disjunction: dedup identical rules but
        # keep learning order (and don't merge confidences — they aren't summed).
        seen, out = [], []
        for r in hyp.rules:
            if not any(_rule_cmp(r, s) == 0 for s in seen):
                seen.append(r)
                out.append(r)
        if len(out) < len(hyp.rules) and trace.tracing(trace.SUMM):
            print("// %d duplicate rules removed" % (len(hyp.rules) - len(out)))
        hyp.rules = out
        return
    hyp.rules.sort(key=cmp_to_key(_rule_cmp))
    rules = hyp.rules
    out = []
    lo = 0
    while lo < len(rules):
        r = rules[lo]
        hi = lo + 1
        while hi < len(rules) and _rule_cmp(r, rules[hi]) == 0:
            r2 = rules[hi]
            if trace.tracing(trace.SUMM):
                print("// adding weight of rule %d to weight of rule %d" % (hi, lo))
            r.nposx = (math.exp(2 * _alphah(r.nposx, r.nnegx) + 2 * _alphah(r2.nposx, r2.nnegx))
                       * (r.nnegx + config.Epsilon) - config.Epsilon)
            if r.nposx < 0:
                r.nposx = 0
            hi += 1
        out.append(r)
        lo = hi
    if len(out) < len(rules):
        if trace.tracing(trace.SUMM):
            print("// %d rules removed in compression" % (len(rules) - len(out)))
    hyp.rules = out
    hyp.rules.sort(key=cmp_to_key(_rule_weight_cmp))


# ---------------------------------------------------------------------------
# print / classify / score a soft ruleset
# ---------------------------------------------------------------------------

def _rule_str(r):
    stats = "(%+g : %g/%g)." % (_alphah(r.nposx, r.nnegx), r.nposx, r.nnegx)
    if not isinstance(r, Rule):
        # PyRule, or an extraction rule (no consequent: it returns the value).
        body = textwrap.indent((r.source or "").rstrip(), "    ")
        head = "extract" if r.conseq is None else sym_str(r.conseq)
        return "%s :- py\n%s\n  %s" % (head, body, stats)
    parts = [sym_str(r.conseq)]
    sep = " :- "
    for g in r.antec:
        if g.nonterm is None:
            parts.append(sep + fprint_gsym(g))
            sep = ", "
    parts.append(" " + stats)
    return "".join(parts)


def fprint_soft_concept(fp, c):
    for r in c.rules:
        fp.write(_rule_str(r) + "\n")
    fp.write("default %s (%+g : %g/%g).\n"
             % (sym_str(c.def_.nom), _alphah(c.nposx, c.nnegx), c.nposx, c.nnegx))
    if c.res != Strategy.FIRST:
        fp.write("resolution=best.\n")


def fshow_soft_concept(fp, c):
    fshow_concept(fp, c)
    fp.write("epsilon %g\n" % config.Epsilon)


def soft_classify(sc, inst):
    """Return (predicted_class_symbol, confidence)."""
    Classes = names_mod.Classes
    if len(Classes) != 2:
        fatal("only two-class version is implemented!")
    negc = sc.def_.nom
    posc = Classes[0 if negc.index else 1].nom
    if _covering():
        # Ordered list: the first rule that fires decides, and answers with its
        # own consequent.  Under cover every rule carries posc, so this is the
        # disjunction's "covered by any rule"; under coverlist a rule can also
        # conclude negc.  No confidence to sum -- report +-1, signed by the
        # predicted class, so callers (.pred) keep a sign.
        for ri in sc.rules:
            if _rule_covers(ri, inst):
                return (ri.conseq, -1.0 if ri.conseq is negc else 1.0)
        return negc, -1.0
    sumcon = _alphah(sc.nposx, sc.nnegx)
    for ri in sc.rules:
        if _rule_covers(ri, inst):
            if ri.conseq is negc:
                sumcon += _alphah(ri.nnegx, ri.nposx)
            else:
                sumcon += _alphah(ri.nposx, ri.nnegx)
    return (negc if sumcon <= 0 else posc), sumcon


def soft_error_rate(c, dataset):
    if _extracting():
        # Same two-argument signature, so crossval and cli._test_slipper are
        # untouched; 1 - exact match instead of 0/1 class error.
        return extract.extract_error_rate(c, dataset)
    err = tot = 0.0
    for ex in dataset:
        tot += ex.wt
        cls, _ = soft_classify(c, ex.inst)
        if cls is not ex.lab.nom:
            err += ex.wt
    if tot == 0:
        return 0.0
    return err / tot
