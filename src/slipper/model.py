"""model.py - class counting and ordering.

``count_classes`` tallies the weighted size of each class; ``reorder_classes``
sorts the class list (by default, least-frequent first) and reassigns each
class symbol's index to its new position.  For the two-class slipper2 this
puts the positive (rarer) class at index 0.
"""

from . import config, trace
from . import names as names_mod
from .counting import count_class_freq

Class_counts = None


def reset():
    global Class_counts
    Class_counts = None


def count_classes(data):
    global Class_counts
    Classes = names_mod.Classes
    nc = len(Classes)
    if Class_counts is None or len(Class_counts) != nc:
        Class_counts = [0.0] * nc
    for k in range(nc):
        p, _ = count_class_freq(Classes[k].nom, data)
        Class_counts[k] = p


def reorder_classes(data):
    Classes = names_mod.Classes
    nc = len(Classes)
    class_val = [0.0] * nc

    if config.Class_ordering in (config.UNORDERED, config.GIVEN):
        for i in range(nc):
            class_val[i] = i + 1
        how_ordered = "given"
    else:  # INCFREQ or DECFREQ
        for i in range(nc):
            class_val[i] = Class_counts[i]
        Classes.sort(key=lambda a: class_val[a.nom.index],
                     reverse=(config.Class_ordering == config.DECFREQ))
        how_ordered = "increasing frequency"

    if trace.tracing(trace.SUMM):
        print("// classes in %s order:" % how_ordered)
        for cli in Classes:
            print("// class %s:\t%.3f" % (cli.nom.name, class_val[cli.nom.index]))

    # re-assign indices and re-order class counts to match the new ordering
    for i in range(nc):
        class_val[i] = Class_counts[i]
    for i in range(nc):
        j = Classes[i].nom.index
        Classes[i].nom.index = i
        Class_counts[j] = class_val[i]

    if (config.FN_cost != 1.0 or config.FP_cost != 1.0) and nc != 2:
        from . import errors
        errors.warning("loss ratio only supported for 2-class problems!")
