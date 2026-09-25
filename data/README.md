# Paper results and the script that rebuilds them

This folder holds the per-seed results behind every number in the paper and
one script that prints all tables, redraws all figures, and prints the
in-text numbers that are not shown verbatim in a table. It trains no models
and calls no LLMs. It contains aggregate results only (test errors,
accuracies, counts), not dataset rows.

## Run

```sh
python make_paper_outputs.py                 # tables + in-text numbers + figures
python make_paper_outputs.py --only tables   # or: figures, numbers
```

Requires Python 3.10+ with numpy, pandas, matplotlib, and scipy. The run takes
a few seconds. Outputs:

- `output/tables/<name>.tex`: LaTeX table bodies, named as in the paper source.
  All tables are also printed as plain text.
- `output/figures/<name>.pdf` and `.png`: the paper's figures.

## Method names

The CSVs keep the original experiment identifiers. `slopper` is **CLIPPER**,
`candcond` is CLIPPER with condition hints, and `slipper` is **SLIPPER**.
All rendered labels use the paper's names.

| Identifier | Paper name |
| --- | --- |
| `slipper` | SLIPPER, full data |
| `slipper_M100` | SLIPPER, M=100 sample per proposal |
| `slopper_notools` | CLIPPER sample (no tools) |
| `slopper` | CLIPPER coverage (coverage tools) |
| `candcond_notools` | CLIPPER hints ("suggestions") |
| `candcond_tool_T2` | CLIPPER hints+tools |
| `slopper_cover`, `slopper_cover_p3` | CLIPPER inside set covering, patience 1 / 3 |
| `candcond_cover` | CLIPPER hints inside set covering |
| `oneshot_slopper`, `oneshot_candcond` | one-shot CLIPPER, one-shot CLIPPER hints |
| `discrete_cover`, `discrete_cover_repaired` | legacy / repaired discrete set covering (SLIPPER's search) |
| `rulefit`, `boosted_rules`, `figs`, `corels`, `tao_tree`, `hstree`, `greedy_rule_list`, `fpskope`, `skope_rules`, `ripper`, `imodels_slipper` | discrete baselines (RuleFit, boosted stumps, FIGS, CORELS, TAO tree, HS tree, greedy rule list, FPSkope, SkopeRules, RIPPER, imodels SLIPPER) |
| `agent_gpt56_sol_norun` | Sol (Codex), with the explicit no-learners instruction |
| `agent_gpt56_sol` | Sol (spec gaming): the original prompt, under which Sol sometimes fitted scikit-learn models. Appendix only |
| `agent_gpt56_terra`, `agent_gpt56_luna` | Terra, Luna (Codex) |
| `agent_opus5`, `agent_sonnet5`, `agent_haiku45` | Opus 5, Sonnet 5, Haiku 4.5 (Claude Code) |
| `agent_gemini35_flash`, `agent_gemini31_flashlite` | Gemini 3.5 Flash, Gemini 3.1 Flash-Lite (Gemini CLI) |
| `slopper_<model>_T0`, `slopper_notools` | CLIPPER with another proposal model, no tools (`sonnet`, `g38` = Gemini 3.8 Flash, `g25fl` = Gemini 2.5 Flash-Lite, `qwen9b` = Qwen 3.5 9B, `haiku`) |
| `slopper_<model>`, `slopper` | the same models with coverage tools (`luna` = GPT-5.6 Luna). The default model is Gemini 3.1 Flash-Lite |
| `slopper_raw_M20`, `slopper_raw_M100` | CLIPPER on raw MedCalc notes, M=20 / 100 |
| `slipper_full`, `slipper_M20`, `slipper_M100` (MedCalc) | SLIPPER on note word sets, full data / M=20 / M=100 |
| `extract_pool`, `extract_{python,regex}_M{20,55,110}` | hand-written regex pool; CLIPPER Python / regex extraction rules |
| `slopper_text` (text suite) | CLIPPER on raw text, regular expressions allowed, no coverage tools, M=100 |
| `slopper` (text suite) | CLIPPER reading each document's set of words (SLIPPER's input), no coverage tools, M=100 |
| `rulechef` | RuleChef 0.2.2 with default settings, same model (Gemini 3.1 Flash-Lite); test documents no rule matches are assigned the training subset's majority class |
| `slipper`, `hstree`, `figs`, `tao_tree`, `boosted_rules`, `corels` (text suite) | full-data SLIPPER on word sets; imodels baselines on TF-IDF features |
| `clipper_raw_M100`, `clipper_raw_M100_tools` (Banking77) | CLIPPER on raw query text, M=100, without / with coverage tools (at most six calls per proposal); one-versus-rest over the five intents |
| `rulechef` (Banking77) | RuleChef 0.2.2 run through its authors' BANKING77 script with their published settings, same model; unmatched queries get the training subset's majority intent |

## Files

All error, accuracy, precision, recall, and F1 values are percentages.

### `results/tabular/`

- **`tabular-runs.csv`**: one row per tabular fit (label, dataset, training
  size, seed). Columns: `label`, `dataset`, `size`, `seed`, `err` (test
  error %), `rules` (retained rules, excluding the default). Ten seeds per
  cell; sizes 100 to 2,000; eight datasets (College only up to n=600).
  Used by Figure 3 (all panels), Tables `outer-loop-main`,
  `harness-main`, `fixed-seven-summary`, `model-complexity` (a), and all
  dataset-level appendix tables (`hero-*`).
- **`symbolic-baselines.csv`**: `label`, `dataset`, `size`, `seed`, `err` for
  SLIPPER and the discrete baselines, in the row order used for Figure 2's
  seeded jitter. Used by Figure 2 (`all-baselines`) and for RIPPER in Figure 3(c).
- **`set-cover-development.csv`**: `label`, `dataset`, `size`, `seed`, `err`,
  `rules` for six discrete covering variants on the development suite (440
  cells each). Used by the discrete-covering table in the development appendix
  (written as `setcover-development.tex`) and the numbers in its text.

- **`development-configurations.csv`**: `label`, `dataset`, `seed`, `err` for
  six configurations on the ten-dataset development suite (full training data,
  ten seeds). Macro error averages the ten per-dataset means. Labels:
  `discrete` (discrete SLIPPER), `table_repr` (CLIPPER, table sample, no tools,
  earlier prompt), `table_repr_valtools` (per-attribute SLIPPER condition
  tools), `tv2_lambda` (one-conjunction framing with coverage tool),
  `tv2_lambda_toolsv1` (revised coverage-tool prompt), `candcond_tool_T2`
  (condition-hints table with iterative queries). Used by the first table in
  the development appendix (written as `development-configurations.tex`).

### `results/medcalc/`

- **`calculator-prediction-seeds.csv`**: 55-way calculator prediction on the
  1,100-row official test split, one row per condition, training cap, and seed.
  Columns: `condition`, `cap` (training notes per calculator), `seed`, `split`,
  `n_eval`, `accuracy` (exact accuracy from centered scores), `macro_f1`,
  `accuracy_uncentered`. Used by Table `calculator-final-repeats` and Figure
  4(a) (plotted as 100 minus accuracy).
- **`value-extraction-seeds.csv`**: value extraction on the official test split
  (500 rows, 1,120 requested values, ten keys). Columns: `condition`, `M`,
  `seed`, `split`, `keys`, `rows`, `tp`, `fp`, `fn`, `micro_p`, `micro_r`,
  `micro_f1`, `exact_set` (rows with every requested value correct). The pool
  has one deterministic run. Used by Tables `extraction-final-repeats` and
  `extraction-f1-main`.

### `results/agents/`

- **`n250-runs.csv`**: n=250 runs for the coding-agent, backend, and reference
  conditions. Columns: `label`, `dataset`, `size`, `seed`, `bestj` (boosting
  rounds selected by SLIPPER's cross-validation), `err`, `rules`, `llm_calls`,
  `latency_s`, `in_tok`, `out_tok`, `cost_usd` (per fit). Comparisons use seeds
  1 to 3; CLIPPER with coverage tools (`slopper`) has ten seeds for the
  anonymization pairing. Used by Tables `agents-main`, `backends-main`,
  `new-harness-full`, `new-backend-full`, `new-anonymization-full`, Figure 4(c),
  the appendix Sol figure (`new-sol-behavior`), and the cost figures in the text.
- **`n250-anonymized-runs.csv`**: the 80 CLIPPER fits on anonymized datasets
  (same columns; `anonymized_dataset` is the arbitrary task name and
  `original_dataset` the dataset it hides). Paired with `slopper` in
  `n250-runs.csv` on dataset and seed. Used by Figure 4(c) and Table
  `new-anonymization-full`.

### `results/text/`

- **`text-runs.csv`**: text classification on five datasets (`sms_spam`,
  `sentiment_amazon`, `sentiment_imdb`, `sentiment_yelp`, `youtube_spam`),
  sizes 50 to 750, ten seeds, for CLIPPER without tools on raw text
  (`slopper_text`) and on word sets (`slopper`),
  full-data SLIPPER, and the imodels baselines. Columns: `label`, `dataset`,
  `size`, `seed`, `err` (test error %).
- **`rulechef-runs.csv`**: every RuleChef fit (sizes 50 to 1,000; the paper
  uses 50 to 750). Columns: `label`, `dataset`, `size`, `seed`, `bestj`
  (unused, 0), `err`, `rules` (rules in the final rule set), `coverage` (% of
  test documents matched by at least one rule), `llm_calls`, `latency_s`,
  `in_tok`, `out_tok`, `cost_usd`.
- **`rulechef-settings.json`**: RuleChef version and settings (0.2.2,
  gemini-3.1-flash-lite, at most 10 rules and 5 per class, 50 examples shown,
  3 refinement iterations; unmatched documents get the training-majority class).

- **`banking77-runs.csv`**: CLIPPER and RuleChef on the five BANKING77
  intents of RuleChef's own benchmark, sizes 50 to 600, ten seeds, scored on
  the 200 official test queries. Columns: `label`, `size`, `seed`, `err` (test
  error %), `err_strict` (RuleChef only: error counting queries no rule labels
  as wrong), `coverage` (RuleChef only: % of test queries labeled by a rule),
  `rules` (rules in the final model, summed over the five one-versus-rest
  models for CLIPPER), `llm_calls`, `cost_usd`. Used by the `banking77` figure
  and table.

- **`recent-runs.csv`**: `label`, `dataset`, `size`, `seed`, `err` on
  `inflation_abstracts`, a text dataset whose labels were published after the
  model's training cutoff; sizes 50 to 750, ten seeds, for CLIPPER on raw text
  and on word sets, full-data SLIPPER, and the imodels baselines. Used for the
  n=750 comparison in the background-knowledge paragraph.

These feed Figure 4(b), Tables `text-suite-error` and `text-suite-rulechef`,
and the numbers in the RuleChef comparison and the text-suite appendix.

### `results/model-complexity/`

- **`python-extraction-rules.csv`**: one row per retained rule in the 300
  Python extraction ensembles (3 sample caps, 10 seeds, 10 keys). Columns:
  `condition`, `sample_cap`, `seed`, `key`, `rule_index`, `positive_coverage`,
  `negative_coverage` (training coverage stored with the rule), `code_lines`
  (physical lines with code, excluding comments and docstrings), `if_tests`
  (`if`/`elif`/conditional expressions), `loops`, `comprehensions`,
  `boolean_join_operators`, `exception_handlers`, `source_bytes`. Used by
  Table `model-complexity` (b).

### `results/figure1/`

- **`learned-rules.json`**: the three learned rules drawn in Figure 1(b)
  (tabular classification on Adult, calculator prediction on MedCalc, and
  bicarbonate extraction), with their learned weights and the short input
  excerpts the figure prints.

## Code

- `make_paper_outputs.py`: entry point.
- `lib/common.py`: loaders, per-dataset and macro summaries (macro SEM is
  sqrt(sum SEM_d^2)/D), rounding (half up), and plain-text rendering.
- `lib/tables.py`: every table. Bold marks the lowest displayed value in a
  column or block, as in the paper.
- `lib/figures.py`, `lib/figure_baselines.py`, `lib/style.py`: every figure.
- `lib/numbers.py`: in-text numbers.

## Paper outputs

| Paper item | Output |
| --- | --- |
| Figure 1 | `tabular-main` |
| Figure 2 | `all-baselines` |
| Figure 3 (tabular: equal examples, condition hints, rule learners) | `figure3-tabular` |
| Figure 4 (MedCalc calculators, text classification, anonymized feature names) | `figure4-text` |
| Main tables | `outer-loop-main`, `agents-main`, `backends-main`, `extraction-f1-main` |
| MedCalc appendix | `calculator-final-repeats`, `extraction-final-repeats` |
| Development appendix | `development-configurations` (inline table), `harness-main`, `setcover-development` (inline table) |
| Dataset-level results appendix | `fixed-seven-summary`, `hero-250-symbolic`, `hero-1200-symbolic-corrected`, `hero-250-slopslip`, `hero-1200-slopslip`, `hero-250-harness`, `hero-1200-harness`, `hero-punchline-curve` |
| Coding-agent appendix | `new-harness-full`, `new-sol-behavior` (figure), `new-anonymization-full`, `new-backend-full` |
| Text-suite appendix | `text-suite-error`, `text-suite-rulechef` |
| Model-complexity appendix | `model-complexity` |

Two development-appendix statements come from development runs that are not
included here: the sample-size sentence (macro errors 20.5, 20.8, and 20.9% at
M=100, 200, 400) and the one-conjunction framing's 1.0-point improvement over
its prompt control.
