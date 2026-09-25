"""slipper - a SLIPPER-style boosted rule learner, plus CLIPPER, an LLM rule proposer.

The two-class boosting loop follows Cohen & Singer, "A simple, fast, and
effective rule learner" (AAAI 1999), which in turn builds on the RIPPER
rule-growing procedure of Cohen, "Fast effective rule induction" (ICML 1995).
Rules are grown greedily over a small grammar of attribute tests, scored by
SLIPPER's confidence-rated boosting objective, and combined into a weighted
vote.

Design choices in this implementation:
  * Input is JSON (see ``tools/to_json.py``); there is no ``.names``/``.data``
    text-format reader.
  * Only the auto-generated *default* grammar is supported; ``.gram`` files
    are not read.
  * Randomness uses a clean seedable RNG.

CLIPPER extends the same boosting loop with an LLM-based rule proposer
(``llm_rule_builder.py``) that can also read free-text attributes and return
value-extraction rules, not just yes/no classification rules.

Module map:

    symbols.py           -- interned symbols and atoms used by the grammar
    names.py             -- attribute metadata and the class list
    data.py              -- examples, partitions, and shuffling
    dataset.py           -- load a dataset from JSON
    grammar.py           -- grammar data types and default-grammar construction
    refine.py            -- designated refinements/generalizations of a rule
    counting.py          -- coverage and weighted counting over examples
    optimize.py          -- choose optimal thresholds/symbols for wildcard tests
    value.py             -- scoring functions used when refining a rule
    concept.py           -- rule and concept (rule set) types
    model.py             -- class counting and ordering
    boosting.py          -- the confidence-rated boosting core
    crossval.py          -- k-fold cross-validation harness
    cli.py               -- command-line driver
    llm_rule_builder.py  -- LLM-based rule proposer (CLIPPER)
    extract.py           -- value extraction: value-returning rules, (p, n)
                             by right/wrong-where-fired, alpha-vote over values
    normalize.py         -- scoring-time value canonicalizer used by extract.py
    record.py            -- recorder() context manager: LLM latency/cost/tokens
    config.py            -- tunable parameters, plus save_yaml/load_yaml for
                             reproducible configs
    retrieval.py         -- bounded retrieval from a rule proposal's training
                             population
    errors.py            -- error/warning message helpers
    rng.py               -- seedable random number source
    trace.py             -- trace-level logging helpers
    util.py              -- small timing/CLI helpers
    python_emit.py       -- emit a standalone Python classifier from a
                             learned concept
"""

VERSION = "2.6"
