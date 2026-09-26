"""config.py - global tunable parameters.

These are gathered as module-level globals so the rest of the package can
read/write them the same way (``import config; config.Min_coverage``).
"""

# --- class ordering codes (as in RIPPER, Cohen 1995) ---
GIVEN = 0
MDL = 1
INCFREQ = 2
DECFREQ = 3
UNORDERED = 4

# --- rule-fitting parameters ---
Simplify = True            # use incremental reduced-error pruning when growing rules
FP_cost = 1.0
FN_cost = 1.0
Max_decompression = 64.0
Min_coverage = 1           # (unused in the slipper2 path)
Optimizations = 0          # always disabled in this implementation
Class_ordering = INCFREQ

# --- grammar options ---
Eq_negations = False       # allow "!=" tests on nominal attributes
Set_negations = False      # allow "!~" tests on set-valued attributes
N_intervals = 0

# --- rule-value options ---
Allow_negative_rules = False   # learn rules negatively correlated with the class

# --- boosting parameters (Cohen & Singer, 1999) ---
Epsilon = 0.0              # smoothing; set to 0.5/len(train) in soft_model
Simplify_function = 1      # which holdout-sizing objective (-S)
Slipper_folds = 5          # internal xval folds to pick #boosting rounds (-i)
Upward_trend_max = 0       # stop CV after this many upward error steps (-U)
N_boost = 100              # max rounds of boosting (-n)

# --- value function pointer ---
# Set at runtime (boosting installs value.newboost_value).  Signature:
#   f(form, oldinfo, oldp, oldn, p, n) -> float
Value_function = None

# --- rule-builder options (-R / -P / -M / -L) ---
Rule_llm_model = ""        # empty = discrete grammar search; else litellm model string
Prompt_file = None         # None = built-in template in llm_rule_builder.py
# Per-round weighted sample passed to the rule builder (both paths).
# 0 means "no sampling, use the full training set".
# None means "unset on the command line"; cli.py resolves to 100 if -R else 0.
Sample_size = None
Log_stem = None            # if set, write each round's rendered prompt examples to
                           # "{stem}-NNN.txt"; when tools are enabled also write the
                           # round's ordered tool-call sequence (args + result/error)
                           # to "{stem}-NNN-STAGE.tools.json" (STAGE = build|prune).
                           # See llm_rule_builder._ToolLog.
Tool_budget = 6            # -T N: per-rule tool-call budget for LLM agent
                           # (0 = disable tools; N > 0 enables them with hard cap N).
                           # Default 6 with the lambda tool only -- the tuned
                           # `tv2_lambda_toolsv1` development config.
Claimed_tool_budget = None  # --claimed-tool-budget N: the number the prompt's
                           # budget line CLAIMS.  Changes only what the model is
                           # told, never the enforced cap (still Tool_budget via
                           # _tool_limits) -- an ablation knob for whether the
                           # model self-limits.  None = claim the real budget.
Llm_extra_body = ""        # --llm-extra-body JSON: merged into every request's
                           # extra_body.  Provider-specific escape hatch; the
                           # case it was added for is turning off a thinking
                           # mode, e.g. '{"chat_template_kwargs":
                           # {"enable_thinking": false}}' for Qwen served by
                           # mlx_lm.server, which otherwise spends thousands of
                           # tokens reasoning and never calls the
                           # structured-output tool.
Require_tool_output = False  # --require-tool-output: append a line telling the
                           # model to answer by calling the structured-output
                           # tool rather than in prose.  Some models (both Qwen
                           # variants tried) return a correct rule inside a
                           # markdown code block and never call the tool, which
                           # pydantic-ai reports as "Exceeded maximum output
                           # retries".  Off by default: it is prompt text, so it
                           # would perturb every tuned condition.
Llm_max_tokens = 0         # --llm-max-tokens N: cap on completion tokens per
                           # LLM request (0 = send nothing, the provider's own
                           # default).  Needed for servers whose default is too
                           # small for a rule: mlx_lm.server stops at 512, which
                           # truncates the structured output mid-JSON and shows
                           # up as "Exceeded maximum output retries".
Reasoning_effort = ""      # --reasoning-effort E: sent as extra_body
                           # {"reasoning_effort": E} on every LLM request.
                           # Empty = send nothing (the provider's own default).
                           # Needed for OpenAI gpt-5.6-*, which rejects function
                           # tools together with its default effort on
                           # /v1/chat/completions -- and pydantic-ai's structured
                           # output IS a function tool, so -T0 doesn't avoid it.
Maximize_criterion = False  # -x: add the maximization-criterion block to the prompt
Count_coverage_tool = False  # --count-coverage turns this back on (and
                           # --no-count-coverage off again).  Off by default: the
                           # tuned config runs the lambda tool alone, which beat
                           # count_coverage + lambda on the suite.
Count_coverage_max_capmean = None  # --count-coverage-max-capmean X: per-round
                           # gate -- offer count_coverage only on rounds whose
                           # captured weight fraction is <= X (None = no gate).
                           # (per-round vs per-dataset).
Lambda_body_tool = True    # -b / --no-lambda-body-tool: the
                           # count_coverage_of_lambda_body tool.  On by default:
                           # with count_coverage off it is the whole tool set of
                           # the tuned config.
Value_tools = False        # --value-tools: also offer best_threshold / best_symbol,
                           # which return the full-data optimal cutpoint / ranked
                           # symbol values (needs -T > 0, like the other tools)
Candidate_conditions = False  # --candidate-conditions: prepend a table of the
                           # best single condition per attribute (accurate
                           # weighted sample + full-data coverage) to the LLM
                           # prompt, for the vacuous always-pass filter.  A
                           # standing counting/optimization aid for models that
                           # miscount large samples.  See
                           # llm_rule_builder._candidate_conditions_table.
Candidate_conditions_tool = False  # --candidate-conditions-tool: also expose that
                           # table as a tool taking a filter (a
                           # count_coverage_of_lambda_body-style body in x), so
                           # the model can re-run it on the subset passing its
                           # partial rule (needs -T > 0, like the other tools).
Retrieval_mode = "off"     # --retrieval off|targeted|random: additional training rows
Retrieval_required = False  # require an example query before finalizing each attempt
Retrieval_max_rows = 20    # new rows per rule proposal, shared across retries
Retrieval_per_call = 10    # maximum rows returned by one retrieval call
Retrieval_max_chars = 20000  # cumulative JSON row characters per rule proposal
Tools_prompt = "variant1"  # --tools-prompt: which wording the LLM tools-hint uses
                           # -- "variant1" (default), the reworked
                           # intro/lambda/value tool descriptions that stress the
                           # sample is a subset (~1/captured x smaller) and lean on
                           # the full-data tools. Only affects prompts when tools
                           # are enabled (-T > 0).
Example_format = "table"   # --example-format: LLM sample rendering, "dict" or "table"
Schema_format = "full"     # --schema-format: attribute schema in the prompt,
                           # "full" (explicit attribute|kind|domain menu) or
                           # "compressed" (omit it; rely on inline type hints).
                           # Restoring the full menu recovers the table-format
                           # penalty on wide datasets.
Cell_format = "repr"       # --cell-format: table-format cell rendering, "repr"
                           # (default; repr()-wrapped, so strings are quoted and
                           # distinguishable from numbers) or "str" (bare values).
                           # Only affects --example-format table; dict already
                           # uses %r. repr vs str is a wash on accuracy (10-seed
                           # test, macro -0.18, no sig per-dataset effect -- see
                           # development suite); repr is the harmless default.
Table_minibatch_size = 0   # --table-minibatch-size: split the table-format sample
                           # into consecutive subtables of at most this many rows,
                           # each repeating the header + kind rows (0 = off, one
                           # table).  Tests the conjecture that the -M penalty is
                           # the model losing column binding down a long table --
                           # Only affects --example-format table.
Prompt_variant = "task_variant_2"  # --prompt-variant: which task-definition block the
                           # LLM prompt uses -- "baseline" (original bool
                           # covers()), "task_variant_1" (cluster framing,
                           # bool|None return), or "task_variant_2" (cluster
                           # framing but a single conjunctive rule per round).
                           # Registry: llm_rule_builder._PROMPT_VARIANTS.
Stratify = False           # --stratify: draw the first round's -M sample
                           # class-balanced (k/2 positive + k/2 negative) instead
                           # of from the pooled distribution; later rounds
                           # self-stratify as boosting up-weights hard examples.
                           # No-op without subsampling (0 < -M < N). See
                           # data.stratified_weighted_sample.
Weighting = "boost"        # --weighting: how examples are reweighted between
                           # rounds. "boost" is SLIPPER's confidence-rated
                           # multiplicative update. "cover" is greedy set cover:
                           # every example a rule covers drops to weight 0 and
                           # the rest stay uniform, so each round sees only the
                           # not-yet-covered examples.  Under "cover" the
                           # hypothesis is a plain disjunction (first rule that
                           # fires wins; all rules share the positive
                           # consequent) with the default class as the
                           # fallback -- confidences are printed but unused.
                           # "coverlist" shares all of that machinery but lets
                           # each accepted rule carry the consequent its own
                           # (p, n) chose, making the hypothesis a CN2/RIPPER
                           # ordered decision list rather than a disjunction:
                           # the gate becomes p != n, so a minority-positive
                           # region is learnable as a negative rule instead of
                           # ending the run.  See boosting._covering /
                           # _coverlist.
Cover_objective = "foil"   # --cover-objective: which growth objective the rule
                           # search maximizes under --weighting cover /
                           # coverlist (no effect under "boost").  "foil"
                           # (default, the status quo) is value.foil_gain;
                           # "linear" is value.cover_value, the p - n + 1
                           # objective aligned with soft_fit's p > n acceptance
                           # gate.  Under coverlist each resolves to its
                           # two-sided sibling (value.two_sided_*), since a rule
                           # there may be grown for either class.
Cover_sizing = "legacy"    # --cover-sizing: how soft_simplify picks among the
                           # generalizations of a grown rule under --weighting
                           # cover / coverlist (no effect under "boost").
                           # "legacy" (default, the status quo) is the original
                           # sizing: the `<=` tie-break, which hands ties to the
                           # last and most general candidate, and no awareness
                           # of the gate.  "gate" takes only generalizations
                           # that still pass soft_fit's acceptance gate, and
                           # breaks ties toward the more specific rule; under
                           # coverlist it also refuses a generalization that
                           # would flip the rule to the other class.
Cover_patience = 1         # --cover-patience: how many consecutive rounds whose
                           # rule fails the acceptance gate are tolerated before
                           # the run stops.  1 (the default) is the legacy behaviour:
                           # stop on the first failure.  K > 1 opts into
                           # skip-and-retry -- a failed round adds nothing and the
                           # next one draws a fresh sample.  Only worth raising on
                           # the stochastic paths (a per-round -M sample, or the
                           # LLM builder), where one unlucky sample used to end
                           # the run.
Oneshot = False            # --oneshot: build the whole rule set in ONE LLM call
                           # instead of one rule per round.  A PyRule is just
                           # Python, so the "rule set" is a single covers() whose
                           # body is a disjunction of rules; no new plumbing is
                           # needed beyond the prompt.  The flag configures
                           # Prompt_variant="oneshot" + cover weighting + a
                           # single round (see cli.py); everything else -- -M,
                           # tools, --tools-prompt, logging -- is unchanged.
Oneshot_rules = 0          # suggested number of rules for --oneshot, taken from
                           # an explicit -n (which has no other job once the
                           # loop is pinned to one round).  0 = no suggestion,
                           # the model decides how many to write.  This is the
                           # one-pass analogue of the bestj round budget the
                           # iterative path gets from cross-validation --
                           # run_learning_curve already passes -n <bestj> per
                           # cell, so the curve supplies it with no extra
                           # plumbing.  See llm_rule_builder._rules_hint.
Advice_ext = None          # --include-advice EXT: if STEM<EXT> exists (e.g.
                           # ".info"), insert its text into the LLM prompt as
                           # task advice/background (orthogonal to
                           # --prompt-variant). None = disabled; ".info"
                           # reproduces the old --include-info behavior.
Advice_text = None         # contents of STEM<Advice_ext> (loaded by cli.py once
                           # the stem is known; None = no advice)
Rule_pool = None           # --rule-pool FILE: a JSON list of extract.RegexRule
                           # specs.  In extraction mode this replaces the rule
                           # builder: each round scores the whole pool under the
                           # live weights and keeps the best (the deterministic,
                           # no-LLM path).  Ignored in classification mode.
Rule_body = "python"       # --rule-body: which arm the extraction LLM rule
                           # builder uses -- "python" (design A: the model
                           # writes `def extract(x: dict)`, run in the same
                           # smolagents sandbox as a PyRule) or "regex" (design
                           # A': the model returns a RegexRule spec --
                           # pattern/flags/template -- as structured output, no
                           # code).  Read only in extraction mode with -R set;
                           # the --rule-pool path is regex-only by construction
                           # and classification ignores it.  Registry:
                           # llm_rule_builder._RULE_BODIES.
Task_mode = "classify"     # what the loaded dataset asks for: "classify" (the
                           # SLIPPER two-class learner) or "extract" (value
                           # extraction, see extract.py).  Set from the JSON
                           # file's "task" key by dataset.load_dataset and reset
                           # by dataset.reset_all -- NOT a command-line knob,
                           # hence its place in _NONCONFIG below.
Pruning_agent = False      # --pruning-agent: two-stage LLM path -- a tool-less
                           # grow agent proposes a rule from the -M sample, then
                           # a second prune agent simplifies it using tools only
                           # (coverage tools see the grow sample, best_* tools
                           # see the full training set). Requires -R and >=1 tool.
                           # See llm_rule_builder.prune_pyrule.
Hide_text = False          # --hide-text: keep "text"-kind attributes out of the
                           # LLM's view (x, schema, sample), so a dataset with both
                           # WORDS and TEXT gives a word-set-only CLIPPER.
Use_text = False           # --use-text: add a prompt note that `re` may be used
                           # on "text"-kind attributes (raw document strings).
                           # Text attributes are visible to the LLM either way
                           # (the MedCalc runs rely on this); the discrete
                           # grammar never sees them.


# ---------------------------------------------------------------------------
# YAML config I/O -- save/load a complete tunable config so an experiment is
# reproducible (`--save-config FILE` / `--config FILE` in cli.py).
# ---------------------------------------------------------------------------

# Runtime-derived / non-serializable globals excluded from the saved config:
# Value_function is installed at runtime by boosting.soft_model; Epsilon is
# recomputed from the training-set size; Advice_text is the loaded contents of
# the Advice_ext file (Advice_ext, the knob, IS saved); Task_mode comes from the
# dataset file, not the command line.
_NONCONFIG = {"Value_function", "Epsilon", "Advice_text", "Task_mode"}


def _is_config_name(name, value) -> bool:
    """A tunable config global: a Mixed-case name (so the ALL-CAPS enum
    constants GIVEN/MDL/... are excluded), not runtime-derived, holding a
    YAML-friendly scalar or None."""
    if name.startswith("_") or name in _NONCONFIG:
        return False
    if not name[0].isupper() or name.isupper():
        return False
    return isinstance(value, (str, int, float, bool, type(None)))


def config_dict() -> dict:
    """The current tunable config as a plain ``{name: value}`` dict."""
    g = globals()
    return {k: g[k] for k in sorted(g) if _is_config_name(k, g[k])}


def save_yaml(path) -> dict:
    """Write the current tunable config to ``path`` as YAML; return the dict."""
    import yaml
    d = config_dict()
    with open(path, "w") as fp:
        yaml.safe_dump(d, fp, default_flow_style=False, sort_keys=True)
    return d


def load_yaml(path) -> dict:
    """Read a saved config from ``path`` and APPLY it to the live config
    globals (so a run reproduces the saved one).  Unknown keys warn and are
    skipped; the runtime-derived keys in ``_NONCONFIG`` are ignored.  Returns
    the applied ``{name: value}`` dict."""
    import yaml
    from .errors import warning
    with open(path) as fp:
        data = yaml.safe_load(fp) or {}
    valid = set(config_dict())
    g = globals()
    applied = {}
    for k, v in data.items():
        if k not in valid:
            warning("ignoring unknown/non-config key %r in %s", k, path)
            continue
        g[k] = v
        applied[k] = v
    return applied
