"""cli.py - command-line driver for the ``slipper2`` learner.

The input is a dataset JSON file (stem.json) rather than separate
.names/.data/.test text files, and the grammar is always the
auto-generated default (there are no grammar/names/stdin-format options).
"""

import getopt
import math
import sys

from . import (boosting, config, crossval, dataset, extract, grammar, record, rng,
               trace, util)
from . import names as names_mod
from .errors import ACCOUNT_ERROR_EXIT, LLMAccountError, fatal, warning

PROGRAM = "slipper2"

HELP = [
    "syntax: slipper2 [options] stem        (reads stem.json)",
    "  learn a soft ruleset from examples (2-class version)",
    "",
    "options are (short / long form):",
    " -v# / --trace-level=#        trace level (0, 1, 2, or 3)",
    " -kN / --cv-folds=N           estimate error rate by k-fold cross-validation",
    " -l  / --leave-one-out        estimate error rate via leave-one-out method",
    " -a  / --class-ordering=STR   + (incfreq), - (decfreq), g (given), u (unordered)",
    " -2  / --two-sided            learn rules pos or neg correlated with the class",
    " -n# / --max-rounds=#         max # rounds of boosting (default 100)",
    " -i# / --internal-folds=#     #folds for xval to pick #rounds (0=max, -1=1/3 holdout)",
    " -p  / --predict              predict on test data (write stem.pred)",
    " -U# / --upward-trend-max=#   in CV, stop after hold-out error rises for # rules",
    " -G  / --print-grammar        print the completed grammar and exit",
    " -N  / --print-names          print the names info and exit",
    " -S# / --simplify=#           holdout rule-sizing objective (0 = don't size rules; 5 = FOIL gain, the --weighting cover default)",
    " -F# / --min-coverage=#       rules must cover at least # examples",
    " -!ns / --negations=ns        allow != tests on nominal / !~ tests on set attributes",
    " -R  / --llm-model=STR        litellm model string ('' = discrete search, default)",
    " -P  / --prompt-file=FILE     prompt template file for the LLM rule builder",
    " -M# / --sample-size=#        max examples in the rule-builder sample (0 = full data; default: 100 if -R else 0)",
    " -L  / --log-stem=STEM        log each round's prompt examples to STEM-NNN.txt, the full assembled prompt to STEM-NNN.prompt.txt (+ tool-call sequence to STEM-NNN-STAGE.tools.json when tools are on)",
    " -T# / --tools=#              tool-call budget for the LLM agent (0 = disable, default 6)",
    "       --claimed-tool-budget=#  budget the prompt CLAIMS (enforcement stays -T#); ablation for whether the model self-limits",
    "       --llm-extra-body JSON  merge JSON into every LLM request's extra_body (provider escape hatch), e.g. '{\"chat_template_kwargs\": {\"enable_thinking\": false}}' to switch off a thinking mode",
    "       --require-tool-output  add a prompt line telling the model to answer by calling the final_result tool, not in prose (some models return the rule in a markdown code block and never call the tool)",
    "       --llm-max-tokens N     cap completion tokens per LLM request (default: unset = provider default).  Local servers may default far too low -- mlx_lm.server stops at 512, which truncates the structured output and reads as 'Exceeded maximum output retries'.",
    "       --reasoning-effort E   send extra_body {\"reasoning_effort\": E} on every LLM request (default: unset = provider default).  OpenAI gpt-5.6-* needs 'none': it rejects function tools -- which pydantic-ai's structured output uses -- alongside its default effort.",
    " -x  / --maximize-criterion   add the maximization-criterion block to the LLM prompt",
    " -b  / --lambda-body-tool     offer the count_coverage_of_lambda_body tool (default ON)",
    "       --no-lambda-body-tool  drop it",
    "       --config=FILE          load a saved YAML config first (CLI flags still override it)",
    "       --save-config=FILE     write the fully-resolved config to FILE (for reproducible reruns)",
    "       --value-tools          also offer best_threshold / best_symbol full-data pickers (needs -T)",
    "       --candidate-conditions  prepend a best-condition-per-attribute table (accurate counts) to the prompt",
    "       --candidate-conditions-tool  also offer that table as a filter-taking tool (needs -T)",
    "       --retrieval=MODE        additional training rows: off (default), targeted, or random (needs -T)",
    "       --require-retrieval     require an example query before finalizing each proposal attempt",
    "       --retrieval-max-rows=N  total additional rows per proposal, across retries (default 20)",
    "       --retrieval-per-call=N  maximum additional rows in one call (default 10)",
    "       --retrieval-max-chars=N  cumulative JSON row characters per proposal (default 20000)",
    "       --count-coverage       add the base count_coverage tool (default off)",
    "       --no-count-coverage    drop it (the default)",
    "       --count-coverage-max-capmean=X  per-round gate: offer count_coverage only when captured wt <= X",
    "       --tools-prompt=NAME    LLM tools-hint wording: 'variant1' (default) or 'baseline'",
    "       --example-format=FMT   LLM sample rendering: 'table' (default, markdown) or 'dict'",
    "       --schema-format=FMT    attribute schema in prompt: 'full' (default) or 'compressed'",
    "       --cell-format=FMT      table-format cell values: 'repr' (default, quoted) or 'str' (bare)",
    "       --table-minibatch-size=R  render the table-format sample as subtables of <= R rows, each with its own header (0 = off, default)",
    "       --prompt-variant=NAME  LLM task variant: 'task_variant_2' (default, single conjunction), 'baseline', 'task_variant_1' (cluster/None)",
    "       --include-advice=EXT   if STEM<EXT> exists (e.g. .info), add its text to the LLM prompt as task advice",
    "       --include-info         same as --include-advice .info (the name used by older recorded configs)",
    "       --stratify             draw the first round's -M sample class-balanced (k/2 pos + k/2 neg)",
    "       --weighting=NAME       between-round example weighting: 'boost' (default, SLIPPER's confidence-rated update), 'cover' (greedy set cover: covered examples drop out, hypothesis is a disjunction of positive rules) or 'coverlist' (same cover step, but each rule carries its own consequent: a CN2/RIPPER ordered decision list, gate p != n)",
    "       --cover-objective=NAME rule-growth objective under --weighting cover/coverlist: 'foil' (default, FOIL gain) or 'linear' (p - n + 1, the objective the acceptance gate measures).  Under coverlist each becomes its two-sided form",
    "       --cover-sizing=NAME    rule sizing under --weighting cover/coverlist: 'legacy' (default, the original -S# generalization search) or 'gate' (skip generalizations that would fail the acceptance gate, and break ties toward the more specific rule)",
    "       --cover-patience=K     consecutive rounds failing the acceptance gate tolerated before --weighting cover/coverlist stops (default 1 = stop on the first failure, the legacy behaviour).  K > 1 skips the failed round and retries with a fresh sample, so it only helps with -M or -R",
    "       --oneshot              LLM builds the COMPLETE rule set in one call (implies --weighting cover + a single round; needs -R).  With an explicit -nK the prompt suggests about K rules; without one the model decides.",
    "       --pruning-agent        two-stage LLM: tool-less grow agent, then a prune agent simplifies via tools (needs -R, tools)",
    "       --rule-pool=FILE       extraction only: JSON list of regex rule specs "
    "({pattern, flags, template}); each round keeps the pool's best-scoring rule "
    "(no LLM).  Mutually exclusive with -R",
    "       --rule-body=NAME       extraction + -R only: what the LLM writes each "
    "round -- 'python' (default, a sandboxed extract(x)) or 'regex' (a "
    "pattern/flags/template spec)",
    "       --hide-text            keep 'text'-kind attributes out of the LLM's view (word sets only)",
    "       --use-text             tell the LLM it may use regex on 'text'-kind attributes (needs -R)",
    "       --seed=N               reseed both Python and numpy RNGs (default 1)",
    " -h  / --help                 show this help and exit",
    "",
    "default: slipper2 -v0 -i5 -n100 -R'' <stem>",
    "",
]

# Suffix for the --weighting echo; 'boost' is the default and gets none.
_WEIGHTING_NOTE = {
    "cover": " (greedy set cover; hypothesis is a disjunction)",
    "coverlist": " (greedy separate-and-conquer; hypothesis is an ordered "
                 "decision list, each rule with its own consequent)",
}

_gave_help = False


def give_help():
    global _gave_help
    from . import VERSION
    print("Program: %s --- %s\n" % (PROGRAM, VERSION))
    if not _gave_help:
        for line in HELP:
            print(line)
        _gave_help = True


def _binomial_std_err(err_rate, n):
    return math.sqrt(err_rate * (1 - err_rate) / (n - 1.0))


def _add_ext(stem, ext):
    return util.add_ext(stem, ext)


# cross-validation interface (holds the current hypothesis being fit)
class _CV:
    hyp = None


def _train_slipper(data):
    if _CV.hyp is not None:
        _CV.hyp = None
    _CV.hyp = boosting.soft_model(data, None)
    print("Hypothesis:")
    boosting.fprint_soft_concept(sys.stdout, _CV.hyp)


def _test_slipper(data):
    return boosting.soft_error_rate(_CV.hyp, data)


def _print_extract_counts(which, hyp, data):
    """Extraction detail under the error-rate line.  Added lines only: the
    harness regexes anchor on ^Train/Test error rate:, which are untouched."""
    exact, abstain, n = extract.extract_counts(hyp, data)
    # %-19s puts the value in the same column as the error-rate lines above.
    print("%-19s%d of %d" % (which + " exact match:", exact, n))
    print("%-19s%.2f%% (%d of %d)" % (which + " abstain:",
                                      100.0 * abstain / n if n else 0.0, abstain, n))


def main(argv):
    # On Windows, stdout/stderr to a pipe default to the cp1252 locale encoding,
    # which cannot print learned rules quoting raw document text (emoji etc.
    # under --use-text).  Force utf-8 with replacement so printing a hypothesis
    # can never crash a run.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    trace.set_trace_level(trace.NONE)
    folds = 5
    crossval_flag = False
    predict = False
    print_grammar_flag = False
    print_names_flag = False
    config.Optimizations = 0

    longopts = [
        "trace-level=", "cv-folds=", "leave-one-out",
        "class-ordering=", "two-sided",
        "max-rounds=", "internal-folds=", "predict", "upward-trend-max=",
        "print-grammar", "print-names", "simplify=", "min-coverage=",
        "negations=",
        "llm-model=", "prompt-file=", "sample-size=", "log-stem=",
        "tools=", "maximize-criterion", "lambda-body-tool", "value-tools",
        "candidate-conditions", "candidate-conditions-tool",
        "retrieval=", "retrieval-max-rows=", "retrieval-per-call=", "retrieval-max-chars=",
        "require-retrieval",
        "no-count-coverage", "count-coverage", "no-lambda-body-tool",
        "count-coverage-max-capmean=", "seed=",
        "tools-prompt=", "reasoning-effort=", "llm-max-tokens=",
        "require-tool-output", "llm-extra-body=",
        "example-format=", "schema-format=", "cell-format=", "prompt-variant=",
        "table-minibatch-size=",
        "include-advice=", "include-info", "stratify", "pruning-agent", "weighting=", "oneshot",
        "config=", "save-config=", "use-text", "hide-text",
        "rule-pool=", "rule-body=", "claimed-tool-budget=", "cover-objective=", "cover-sizing=", "cover-patience=",
        "help",
    ]
    # Long-form alias for each short flag (one-to-one).
    long_for_short = {
        "-i": "--internal-folds", "-U": "--upward-trend-max", "-p": "--predict",
        "-n": "--max-rounds", "-v": "--trace-level", "-k": "--cv-folds",
        "-l": "--leave-one-out", "-2": "--two-sided", "-a": "--class-ordering",
        "-!": "--negations", "-G": "--print-grammar", "-N": "--print-names",
        "-M": "--sample-size", "-R": "--llm-model", "-P": "--prompt-file",
        "-L": "--log-stem", "-F": "--min-coverage", "-S": "--simplify",
        "-T": "--tools", "-x": "--maximize-criterion",
        "-b": "--lambda-body-tool", "-h": "--help",
    }

    try:
        opts, args = getopt.getopt(
            argv[1:], "i:hv:k:lg:f:sGN!:M:F:n:pa:2S:U:R:P:L:T:xb", longopts)
    except getopt.GetoptError as e:
        give_help()
        fatal("bad option: %s", e)

    # Option echoes are deferred: each branch only mutates state and queues a
    # thunk; the thunks run after parsing (and after the -M default is resolved),
    # so every message reports the FINAL value regardless of option order. (E.g.
    # "-i0 -n4" must say "boost 4 times", not the pre-"-n" default of 100.)
    # A saved config (--config) is applied FIRST, so any explicit CLI flag
    # below overrides it (precedence: defaults < config file < CLI flags).
    for o, a in opts:
        if o == "--config":
            config.load_yaml(a)
            print("option: loaded config from %s" % a)

    echoes = []
    _save_config_path = None       # --save-config FILE: dump final config after resolution
    _simplify_given = False        # did the user pass -S/--simplify explicitly?
    _maxrounds_given = False       # ... or -n/--max-rounds? (--oneshot reads it as K)
    _variant_given = False         # ... or --prompt-variant? (--oneshot presets it)

    for o, a in opts:
        # Normalize short → long so each option has one branch below.
        o = long_for_short.get(o, o)
        if o == "--internal-folds":
            config.Slipper_folds = int(float(a))
            if config.Slipper_folds == 0:
                echoes.append(lambda: print("option: boost %d times" % config.N_boost))
            else:
                echoes.append(lambda: print(
                    "option: use %d-fold internal xval to pick #times to boost"
                    % config.Slipper_folds))
        elif o == "--upward-trend-max":
            config.Upward_trend_max = int(float(a))
            echoes.append(lambda: print(
                "option: stop CV when error rates trend upward for %d rules"
                % config.Upward_trend_max))
        elif o == "--predict":
            predict = True
            echoes.append(lambda: print("option: predict performance on test data"))
        elif o == "--max-rounds":
            _maxrounds_given = True          # --oneshot reads this as its K
            config.N_boost = int(a)
            echoes.append(lambda: print("option: learn ruleset of size %d" % config.N_boost))
        elif o == "--trace-level":
            trace.set_trace_level(int(a))
            echoes.append(lambda: print("option: trace level set to %d" % trace.trace_level()))
        elif o == "--cv-folds":
            crossval_flag = True
            folds = int(a)
            echoes.append(lambda folds=folds: print("option: %d-fold cross-validation" % folds))
        elif o == "--leave-one-out":
            crossval_flag = True
            folds = 0
            echoes.append(lambda: print("option: leave-one-out cross-validation"))
        elif o == "--two-sided":
            config.Allow_negative_rules = True
            echoes.append(lambda: print(
                "option: learn rules negatively correlated with asserted class too"))
        elif o == "--class-ordering":
            c = a[0] if a else ""
            if c == "+":
                config.Class_ordering = config.INCFREQ
                echoes.append(lambda: print("option: find rules for least frequent classes first"))
            elif c == "-":
                config.Class_ordering = config.DECFREQ
                echoes.append(lambda: print("option: find rules for most frequent classes first"))
            elif c in ("g", "G"):
                config.Class_ordering = config.GIVEN
                echoes.append(lambda: print("option: use class ordering given in names file"))
            elif c in ("m", "M"):
                config.Class_ordering = config.MDL
                echoes.append(lambda: print("option: choose ordering of classes using MDL"))
            elif c in ("u", "U"):
                config.Class_ordering = config.UNORDERED
                echoes.append(lambda: print("option: unordered classes"))
            else:
                warning("bad argument to -a/--class-ordering option")
        elif o == "--negations":
            for ch in a:
                if ch in ("n", "N"):
                    config.Eq_negations = True
                if ch in ("s", "S"):
                    config.Set_negations = True
            echoes.append(lambda: print(
                "option: %s allow inequality tests for nominal attributes"
                % ("will" if config.Eq_negations else "will not")))
            echoes.append(lambda: print(
                "option: %s allow inequality tests for set-valued attributes"
                % ("will" if config.Set_negations else "will not")))
        elif o == "--print-grammar":
            print_grammar_flag = True
        elif o == "--print-names":
            print_names_flag = True
        elif o == "--sample-size":
            config.Sample_size = max(0, int(a))
            if config.Sample_size == 0:
                echoes.append(lambda: print(
                    "option: rule-builder sees full training set (no sampling)"))
            else:
                echoes.append(lambda: print(
                    "option: rule-builder sample size %d" % config.Sample_size))
        elif o == "--llm-model":
            config.Rule_llm_model = a
            if a:
                echoes.append(lambda: print(
                    "option: rule builder = LLM (%s)" % config.Rule_llm_model))
            else:
                echoes.append(lambda: print("option: rule builder = discrete grammar search"))
        elif o == "--prompt-file":
            config.Prompt_file = a if a else None
            if a:
                echoes.append(lambda: print(
                    "option: LLM prompt template = %s" % config.Prompt_file))
        elif o == "--log-stem":
            config.Log_stem = a if a else None
            if a:
                echoes.append(lambda: print(
                    "option: log per-round LLM prompt examples to %s-NNN.txt" % config.Log_stem))
        elif o == "--tools":
            config.Tool_budget = max(0, int(a))
            if config.Tool_budget > 0:
                echoes.append(lambda: print(
                    "option: LLM rule-builder tool budget = %d calls/rule" % config.Tool_budget))
            else:
                echoes.append(lambda: print("option: LLM rule-builder tools disabled"))
        elif o == "--llm-extra-body":
            config.Llm_extra_body = a
            echoes.append(lambda: print("option: extra_body = %s" % config.Llm_extra_body))
        elif o == "--require-tool-output":
            config.Require_tool_output = True
            echoes.append(lambda: print(
                "option: prompt tells the model to answer via the final_result tool"))
        elif o == "--llm-max-tokens":
            config.Llm_max_tokens = max(0, int(a))
            if config.Llm_max_tokens:
                echoes.append(lambda: print(
                    "option: max completion tokens per LLM request = %d"
                    % config.Llm_max_tokens))
        elif o == "--reasoning-effort":
            config.Reasoning_effort = a.strip()
            if config.Reasoning_effort:
                echoes.append(lambda: print(
                    "option: reasoning_effort = %s (sent as extra_body)"
                    % config.Reasoning_effort))
        elif o == "--maximize-criterion":
            config.Maximize_criterion = True
            echoes.append(lambda: print(
                "option: add maximization-criterion block to the LLM prompt"))
        elif o == "--lambda-body-tool":
            config.Lambda_body_tool = True
            echoes.append(lambda: print(
                "option: offer the count_coverage_of_lambda_body tool"))
        elif o == "--value-tools":
            config.Value_tools = True
            echoes.append(lambda: print(
                "option: offer the best_threshold / best_symbol full-data tools"))
        elif o == "--candidate-conditions":
            config.Candidate_conditions = True
            echoes.append(lambda: print(
                "option: prepend the candidate-conditions table to the LLM prompt"))
        elif o == "--candidate-conditions-tool":
            config.Candidate_conditions_tool = True
            echoes.append(lambda: print(
                "option: offer the candidate_conditions tool"))
        elif o == "--retrieval":
            if a not in ("off", "targeted", "random"):
                fatal("--retrieval must be off, targeted, or random")
            config.Retrieval_mode = a
            echoes.append(lambda: print("option: training-row retrieval = %s" % config.Retrieval_mode))
        elif o == "--require-retrieval":
            config.Retrieval_required = True
        elif o in ("--retrieval-max-rows", "--retrieval-per-call", "--retrieval-max-chars"):
            if int(a) < 1:
                fatal("%s must be positive", o)
            key = {"--retrieval-max-rows": "Retrieval_max_rows",
                   "--retrieval-per-call": "Retrieval_per_call",
                   "--retrieval-max-chars": "Retrieval_max_chars"}[o]
            setattr(config, key, int(a))
        elif o == "--no-count-coverage":
            config.Count_coverage_tool = False
            echoes.append(lambda: print(
                "option: drop the base count_coverage tool"))
        elif o == "--count-coverage":
            config.Count_coverage_tool = True
            echoes.append(lambda: print(
                "option: add the base count_coverage tool"))
        elif o == "--no-lambda-body-tool":
            config.Lambda_body_tool = False
            echoes.append(lambda: print(
                "option: drop the count_coverage_of_lambda_body tool"))
        elif o == "--count-coverage-max-capmean":
            config.Count_coverage_max_capmean = float(a)
            echoes.append(lambda a=a: print(
                "option: gate count_coverage to rounds with captured wt <= %s" % a))
        elif o == "--tools-prompt":
            v = a.strip().lower()
            if v not in ("baseline", "variant1"):
                fatal("--tools-prompt must be 'baseline' or 'variant1', got %r" % a)
            config.Tools_prompt = v
            echoes.append(lambda v=v: print("option: LLM tools-hint wording = %s" % v))
        elif o == "--example-format":
            fmt = a.strip().lower()
            if fmt not in ("dict", "table"):
                fatal("--example-format must be 'dict' or 'table', got %r" % a)
            config.Example_format = fmt
            echoes.append(lambda fmt=fmt: print("option: LLM sample format = %s" % fmt))
        elif o == "--schema-format":
            fmt = a.strip().lower()
            if fmt not in ("full", "compressed"):
                fatal("--schema-format must be 'full' or 'compressed', got %r" % a)
            config.Schema_format = fmt
            echoes.append(lambda fmt=fmt: print("option: LLM schema format = %s" % fmt))
        elif o == "--cell-format":
            fmt = a.strip().lower()
            if fmt not in ("str", "repr"):
                fatal("--cell-format must be 'str' or 'repr', got %r" % a)
            config.Cell_format = fmt
            echoes.append(lambda fmt=fmt: print("option: LLM table cell format = %s" % fmt))
        elif o == "--table-minibatch-size":
            r = int(a)
            if r < 0:
                fatal("--table-minibatch-size must be >= 0, got %r" % a)
            config.Table_minibatch_size = r
            echoes.append(lambda r=r: print(
                "option: LLM table minibatch size = %s" % (r or "off (one table)")))
        elif o == "--prompt-variant":
            _variant_given = True
            v = a.strip()
            from .llm_rule_builder import _PROMPT_VARIANTS
            if v not in _PROMPT_VARIANTS:
                fatal("--prompt-variant must be one of %s, got %r"
                      % (", ".join(sorted(_PROMPT_VARIANTS)), a))
            config.Prompt_variant = v
            echoes.append(lambda v=v: print("option: LLM prompt variant = %s" % v))
        elif o in ("--include-advice", "--include-info"):
            ext = ".info" if o == "--include-info" else a.strip()
            if not ext:
                fatal("--include-advice needs a file extension, e.g. "
                      "--include-advice .info")
            if not ext.startswith("."):
                ext = "." + ext
            config.Advice_ext = ext
            echoes.append(lambda ext=ext: print(
                "option: load task advice from STEM%s into the LLM prompt" % ext))
        elif o == "--stratify":
            config.Stratify = True
            echoes.append(lambda: print(
                "option: stratify the first round's sample (k/2 pos + k/2 neg)"))
        elif o == "--weighting":
            w = a.strip().lower()
            if w not in ("boost", "cover", "coverlist"):
                fatal("--weighting must be 'boost', 'cover' or 'coverlist', got %r" % a)
            config.Weighting = w
            echoes.append(lambda w=w: print("option: example weighting = %s%s"
                                            % (w, _WEIGHTING_NOTE.get(w, ""))))
        elif o == "--cover-objective":
            co = a.strip().lower()
            if co not in ("foil", "linear"):
                fatal("--cover-objective must be 'foil' or 'linear', got %r" % a)
            config.Cover_objective = co
            # Echoed after parsing, so --weighting has settled and the message
            # can name the objective actually installed (coverlist takes the
            # two-sided form of whichever one is chosen).
            echoes.append(lambda co=co: print(
                "option: cover growth objective = %s" % (
                    ("|p - n| + 1 (value.two_sided_cover_value; matches the p != n gate)"
                     if co == "linear" else "two-sided FOIL gain (value.two_sided_foil_gain)")
                    if config.Weighting == "coverlist" else
                    ("p - n + 1 (value.cover_value; matches the p > n gate)"
                     if co == "linear" else "FOIL gain (value.foil_gain)"))))
        elif o == "--cover-sizing":
            cs = a.strip().lower()
            if cs not in ("legacy", "gate"):
                fatal("--cover-sizing must be 'legacy' or 'gate', got %r" % a)
            config.Cover_sizing = cs
            echoes.append(lambda cs=cs: print(
                "option: cover rule sizing = %s" % (
                    "gate-aware (only generalizations that keep p > n; ties to the specific rule)"
                    if cs == "gate" else "legacy (the original generalization search)")))
        elif o == "--cover-patience":
            k = int(a)
            if k < 1:
                fatal("--cover-patience must be >= 1, got %r" % a)
            config.Cover_patience = k
            echoes.append(lambda k=k: print(
                "option: cover gate patience = %d consecutive failed round(s)%s"
                % (k, " (stop on the first failure)" if k == 1
                   else " (skip and retry until then)")))
        elif o == "--oneshot":
            config.Oneshot = True
        elif o == "--pruning-agent":
            config.Pruning_agent = True
            echoes.append(lambda: print(
                "option: two-stage LLM rule builder (grow agent + prune agent)"))
        elif o == "--hide-text":
            config.Hide_text = True
            echoes.append(lambda: print(
                "option: hide text attributes from the LLM (word-set view only)"))
        elif o == "--use-text":
            config.Use_text = True
            echoes.append(lambda: print(
                "option: expose text attributes to the LLM (regex rules allowed)"))
        elif o == "--min-coverage":
            config.Min_coverage = max(int(a), 1)
            echoes.append(lambda: print("option: rules must cover %d example(s)" % config.Min_coverage))
        elif o == "--simplify":
            _simplify_given = True
            config.Simplify_function = int(float(a))
            config.Simplify = (config.Simplify_function != 0)
            echoes.append(lambda: print(
                "option: size rules with a holdout set: %s" % "ny"[config.Simplify]))
            echoes.append(lambda: print(
                "option: use resizing function %d" % config.Simplify_function))
        elif o == "--rule-pool":
            config.Rule_pool = a
            echoes.append(lambda a=a: print(
                "option: extraction rule pool = %s (deterministic rule builder)" % a))
        elif o == "--rule-body":
            b = a.strip().lower()
            from .llm_rule_builder import _RULE_BODIES
            if b not in _RULE_BODIES:
                fatal("--rule-body must be one of %s, got %r"
                      % (", ".join(sorted(_RULE_BODIES)), a))
            config.Rule_body = b
            echoes.append(lambda b=b: print(
                "option: extraction rule body = %s" % b))
        elif o == "--seed":
            rng.seed(int(a))
            echoes.append(lambda s=int(a): print(
                "option: reseeded Python+numpy RNGs with seed %d" % s))
        elif o in ("-g", "-f", "-s"):
            warning("option %s is not supported (input is JSON); ignored", o)
        elif o == "--config":
            pass          # already applied before the loop (see above)
        elif o == "--save-config":
            _save_config_path = a
        elif o == "--help":
            give_help()
            return 0
        else:
            give_help()
            fatal("option not implemented")

    # Resolve the -M / --sample-size default now that we know whether -R is set:
    # LLM path needs a finite prompt sample; discrete path defaults to no sampling.
    if config.Sample_size is None:
        config.Sample_size = 100 if config.Rule_llm_model else 0
        if config.Sample_size:
            echoes.append(lambda: print(
                "option: rule-builder sample size %d (default for -R)" % config.Sample_size))

    if config.Use_text and not config.Rule_llm_model:
        fatal("--use-text requires an LLM model (-R); text attributes are "
              "invisible to the discrete grammar regardless")

    # Rule sizing has to use the same objective the hypothesis does.  The -S
    # defaults are boosting losses; under cover weighting they generalize a good
    # rule right back to the universal one (which then fires on everything), so
    # default to -S5, the FOIL objective matching value.foil_gain used to grow.
    # An explicit -S/--simplify still wins.
    if (config.Weighting in ("cover", "coverlist") and config.Simplify
            and not _simplify_given):
        config.Simplify_function = 5
        echoes.append(lambda w=config.Weighting: print(
            "option: rule sizing = -S5 (FOIL gain on the holdout; default under "
            "--weighting %s)" % w))

    # --oneshot is a config preset, not a code path: one round of the cover
    # learner whose rule builder is handed the "write the whole rule set"
    # prompt.  Set after parsing so it wins over the defaults but reports the
    # final values; an explicit --prompt-variant/--weighting still applies.
    if config.Oneshot:
        if not config.Rule_llm_model:
            fatal("--oneshot requires an LLM model (-R); the one-pass rule set "
                  "is built by the LLM rule builder")
        if not _variant_given:
            config.Prompt_variant = "oneshot"
        # -n has no other job once the loop is one round, so it carries the
        # suggested rule count; unset means "the model decides".
        config.Oneshot_rules = config.N_boost if _maxrounds_given else 0
        if config.Weighting == "boost":
            config.Weighting = "cover"
        config.Slipper_folds = 0        # nothing to cross-validate: one round
        config.N_boost = 1
        echoes.append(lambda: print(
            "option: one-pass rule set (prompt variant %s, weighting %s, 1 round, "
            "%s)" % (config.Prompt_variant, config.Weighting,
                     "suggest ~%d rules" % config.Oneshot_rules
                     if config.Oneshot_rules else "rule count up to the model")))

    if config.Pruning_agent:
        if not config.Rule_llm_model:
            fatal("--pruning-agent requires an LLM model (-R); the grow/prune "
                  "agents are LLM-only")
        _prune_tools = (config.Tool_budget > 0 and
                        (config.Count_coverage_tool or config.Lambda_body_tool
                         or config.Value_tools))
        if not _prune_tools:
            fatal("--pruning-agent needs at least one tool for the prune agent; "
                  "pass -T N (>0) with count_coverage on and/or --value-tools / -b")

    # Now that every value is final, emit the queued option echoes.
    for echo in echoes:
        echo()

    # Dump the fully-resolved config for reproducibility (--save-config FILE).
    if _save_config_path:
        config.save_yaml(_save_config_path)
        print("option: saved config to %s" % _save_config_path)

    if trace.tracing(trace.SUMM):
        print("// parameter settings:")
        print("//   %s != tests in rules" % ("allow" if config.Eq_negations else "disallow"))
        print("//   %s !~ tests in rules" % ("allow" if config.Set_negations else "disallow"))
        print("//   rules cover %d examples" % config.Min_coverage)

    if not args:
        give_help()
        fatal("no file stem specified")
    stem = args[0]
    if len(args) > 1:
        warning("not all arguments were used: %s ...", args[1])

    json_path = stem if stem.endswith(".json") else stem + ".json"
    base = json_path[:-5]

    if config.Advice_ext:
        advice_path = _add_ext(base, config.Advice_ext)
        try:
            with open(advice_path) as fp:
                config.Advice_text = fp.read()
            print("option: loaded task advice from %s (%d chars)"
                  % (advice_path, len(config.Advice_text)))
        except OSError:
            warning("--include-advice: no %s file; prompt will have no advice",
                    advice_path)

    if trace.tracing(trace.SUMM):
        print("// loading data from %s..." % json_path)
    train_data, test_data = dataset.load_dataset(json_path)
    if not train_data:
        fatal("no training data!")

    # The task comes from the file ("task": "extract"), not from a flag.
    extracting = config.Task_mode == "extract"
    if extracting:
        print("task: value extraction (rules return values; default = abstain)")
        if config.Rule_llm_model and config.Rule_pool:
            fatal("--rule-pool and -R are two different rule builders for the "
                  "same slot; pass one, not both")
        if not config.Rule_llm_model and not config.Rule_pool:
            fatal("extraction mode needs --rule-pool FILE (the deterministic "
                  "rule builder) or -R MODEL (the LLM rule builder)")
        if config.Weighting == "cover":
            fatal("--weighting cover has no meaning in extraction mode: rules "
                  "vote on a value, they don't OR into a disjunction")
    elif config.Rule_pool:
        warning("--rule-pool is ignored: %s is not an extraction dataset", json_path)

    if crossval_flag:
        test_data = None
    if trace.tracing(trace.SUMM):
        print("// loaded %d train / %d test examples, %d features, %d values"
              % (len(train_data), len(test_data) if test_data else 0,
                 names_mod.n_fields(), names_mod.n_symbolic_values()))

    # No grammar in extraction mode: nothing consumes it (rules are regex or
    # Python) and building conditions over note text would enumerate one
    # symbolic value per note.  The LLM classification path only needs it for
    # the optional discrete-search tools, so a dataset with no grammar-usable
    # attributes (e.g. a single text column) is fatal only for discrete search.
    if not extracting and not grammar.ld_grammar(train_data) \
            and not config.Rule_llm_model:
        fatal("error in loading grammar")
    if print_grammar_flag:
        grammar.print_grammar()
        return 0
    if print_names_flag:
        names_mod.print_names()
        return 0

    if crossval_flag:
        crossval.cross_validate(folds, train_data, _train_slipper, _test_slipper)
        return 0

    with open(_add_ext(base, ".shyp"), "w", encoding="utf-8") as out_fp:
        util.start_clock()
        with record.recorder() as records:
            hyp = boosting.soft_model(train_data, test_data)
        tm = util.elapsed_time()
        boosting.fshow_soft_concept(out_fp, hyp)

    from . import python_emit
    pyhyp_path = python_emit.write_python(_add_ext(base, ".pyhyp.py"), hyp)
    if pyhyp_path:
        print("Wrote standalone Python classifier to %s" % pyhyp_path)

    print("Final hypothesis is:")
    boosting.fprint_soft_concept(sys.stdout, hyp)
    print("=============================== summary ===============================")
    err = boosting.soft_error_rate(hyp, train_data)
    print("Train error rate:  %.2f%% +/- %.2f%% (%d datapoints)    <<"
          % (err * 100, 100 * _binomial_std_err(err, len(train_data)), len(train_data)))
    if extracting:
        _print_extract_counts("Train", hyp, train_data)
    if records["n_llm_calls"]:
        cost = ("cost n/a (unpriced model)" if records["cost_unavailable"]
                else "$%.4f" % records["cost_usd"])
        print("LLM side info:     %d calls, %.1fs latency, %d in / %d out tokens, %s"
              % (records["n_llm_calls"], records["latency_s"],
                 records["input_tokens"], records["output_tokens"], cost))
    # Every provider request, successful attempt or not, plus the failures
    # themselves.  "LLM side info" above counts only attempts that returned a
    # rule; a run whose attempts all failed printed nothing at all, which is
    # how failed attempts went uncounted.  Kept as a separate line so parsers
    # of the old one are unaffected.
    if records["api_requests"] or records["failed_attempts"]:
        api_cost = ("cost n/a (unpriced model)" if records["api_cost_unavailable"]
                    else "$%.4f" % records["api_cost_usd"])
        print("LLM accounting:    %d requests (%d failed), %d in / %d out tokens, "
              "%s, %d failed attempts, %d lost rounds"
              % (records["api_requests"], records["api_failures"],
                 records["api_input_tokens"], records["api_output_tokens"],
                 api_cost, records["failed_attempts"], records["lost_rounds"]))
    if test_data:
        err = boosting.soft_error_rate(hyp, test_data)
        print("Test error rate:   %.2f%% +/- %.2f%% (%d datapoints)   <<"
              % (100 * err, 100 * _binomial_std_err(err, len(test_data)), len(test_data)))
        if extracting:
            _print_extract_counts("Test", hyp, test_data)
        if predict:
            with open(_add_ext(base, ".pred"), "w", encoding="utf-8") as predfp:
                for exi in test_data:
                    if extracting:
                        # TAB-separated: extracted values contain spaces.
                        v, con = extract.extract_classify(
                            hyp, extract.inst_dict(exi.inst))
                        predfp.write("%r\t%g\t0\t%s\n" % (v, con, exi.lab.nom.name))
                        continue
                    p, con = boosting.soft_classify(hyp, exi.inst)
                    if con < 0:
                        con = -con
                    predfp.write("%s %g 0 %s\n" % (p.name, con, exi.lab.nom.name))
    from .concept import concept_size
    print("Hypothesis size:   %d rules, %d conditions"
          % (len(hyp.rules), concept_size(hyp)))
    print("Learning time:     %.2f sec" % tm)
    return 0


def console_main():
    """No-arg entry point for the ``slipper`` console script.  An LLM account
    error (no credit, bad key) ends the run with ACCOUNT_ERROR_EXIT and no test
    error line, so the suite runner records no row and stops launching cells."""
    try:
        status = main(sys.argv)
    except LLMAccountError as e:
        print("FATAL: LLM account error, stopping: %s" % e, file=sys.stderr)
        raise SystemExit(ACCOUNT_ERROR_EXIT)
    raise SystemExit(status)
