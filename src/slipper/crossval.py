"""crossval.py - general k-fold cross-validation harness."""

import math

from . import data, trace, util


def _std_err(v, tot, n):
    if n == 1:
        return 0.0
    mean = tot / n
    variance_sum = sum((x - mean) ** 2 for x in v)
    sd = math.sqrt(variance_sum / n)
    correction = float(n) / (n - 1)
    return (sd * correction) / math.sqrt(n)


def cross_validate(folds, train_data, train, test):
    """Run k-fold CV: ``train(data1)`` learns, ``test(data2)`` returns an error
    rate.  ``folds == 0`` means leave-one-out."""
    if folds == 0:
        folds = len(train_data)
    m = len(train_data) // folds
    if trace.tracing(trace.SUMM):
        print("// will use %d of %d examples for testing" % (m, len(train_data)))

    errors = []
    times = []
    total_err = total_tm = 0.0
    data1 = data.new_data()
    data2 = data.new_data()
    data.stratify_and_shuffle_data(train_data, folds)
    for i in range(folds):
        data.ith_stratified_partition(train_data, i, folds, data1, data2)
        print("------------------------- run %2d -------------------------" % (i + 1))
        util.start_clock()
        if trace.tracing(trace.SUMM):
            print("// timing training %d..." % (i + 1))
        train(data1)
        tm = util.elapsed_time()
        if trace.tracing(trace.SUMM):
            print("// training took %.2f sec" % tm)
        err = 100 * test(data2)
        print("Error rate on holdout data is %g%%" % err)
        print("Running average of error rate is %g%%" % ((total_err + err) / (i + 1)))
        times.append(tm)
        total_tm += tm
        errors.append(err)
        total_err += err

    print("============================ statistical summary ============================")
    print("Average error: %.2f%% +/- %.2f%%     <<"
          % (total_err / folds, _std_err(errors, total_err, folds)))
    print("Average time:  %.2f  +/- %.2f"
          % (total_tm / folds, _std_err(times, total_tm, folds)))
