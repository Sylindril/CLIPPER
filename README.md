# CLIPPER

Code and results for the CLIPPER paper. CLIPPER keeps SLIPPER's boosting loop
and full-training-set rule scoring, but asks a coding LLM to propose each rule as
a Python predicate from a weighted sample of at most `M` training examples,
optionally with coverage tools and condition hints computed on all training rows.

The package contains:

| Path | Contents |
|---|---|
| `src/slipper/` | The learner: SLIPPER (Cohen & Singer, 1999; building on RIPPER, Cohen, 1995) plus the LLM rule proposer (CLIPPER), text attributes, and value extraction. Command: `slipper`. |
| `src/tools/` | Dataset converters, the anonymizer, and the MedCalc builders and extraction scorer. |
| `datasets/` | `download.py`, which fetches and verifies every public dataset; see `datasets/README.md`. |
| `scripts/` | Experiment runners and summaries (below). |
| `experiments/` | Recorded run configurations, the MedCalc reference regex pools, and the expected MedCalc input hashes. |
| `data/` | The numbers behind every table and figure in the paper, and `make_paper_outputs.py`, which prints the tables and redraws the figures. See `data/README.md`. |
| `tests/` | Unit and end-to-end tests (`uv run pytest`). |

**Names.** The code keeps our internal identifiers. `slopper` is **CLIPPER** and
`slipper` is **SLIPPER** (the discrete grammar search, selected by `-R ''`). The
package and command are called `slipper` because CLIPPER runs inside that learner
with `-R MODEL`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. The vendored CORELS
wheel in `vendor/` is built for CPython 3.11 on macOS and Windows; see
`vendor/README.md` to rebuild it. CORELS needs `gmp` at run time.

```bash
uv sync --all-groups          # learner + baselines (imodels, wittgenstein, CORELS) + MedCalc (datasets)
uv run pytest -q              # tests that need datasets or API keys are skipped
uv run python datasets/download.py            # tabular and text suites
uv run python src/tools/anonymize_suite.py
uv run python src/tools/build_medcalc_extraction_suite.py
```

Model access goes through [LiteLLM](https://docs.litellm.ai/) model strings.
Set the key for each provider you use: `GEMINI_API_KEY` (the default model,
`gemini/gemini-3.1-flash-lite`), `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
`OPENROUTER_API_KEY`.

## Reproducing the paper's tables and figures from saved results

```bash
uv run python data/make_paper_outputs.py      # prints every table, writes data/output/{tables,figures}
```

This reads only `data/results/` and makes no model calls.

## Running the learner

```bash
uv run slipper -R '' test_suite/telco_churn                              # SLIPPER, internal CV for the number of rounds
uv run slipper -R gemini/gemini-3.1-flash-lite -i0 -n10 test_suite/telco_churn   # CLIPPER, 10 rounds, default configuration
uv run slipper -h                                                         # all options
```

`-R MODEL` alone selects the CLIPPER configuration used in the paper: table
sample, `M=100`, the one-conjunction prompt, and the coverage tool (`-T6`).
`-T0` removes the tools. Condition hints are
`--candidate-conditions` (initial table) and `--candidate-conditions-tool`
(iterative queries). `--weighting cover` replaces boosting with set covering, and
`--oneshot` asks for a whole rule set in one call. `--save-config FILE` and
`--config FILE` write and read the complete resolved configuration.

## Running the experiments

The harnesses are resumable. They append one row per (dataset, size, seed) to
`scripts/<suite>-lc-out/results.csv` (the development suite uses
`scripts/lc-out/` and `scripts/bestj-out/`) and write a `<label>_config.yaml`
describing the run. A cell draws its training subset deterministically from
(dataset, size, seed), so every condition sees the same rows. SLIPPER's
five-fold internal cross-validation on that subset chooses the number of
boosting rounds, with a minimum of two, and each condition reuses that budget.
`experiments/configs/` holds the configuration files recorded for the paper's
runs. The size and seed lists below are the grids reported in the paper.

```bash
LC="uv run python scripts/run_learning_curve.py --suite test --min-bestj 2"
NINE=100,175,250,400,500,600,800,1200,2000
FIVE=100,250,600,1200,2000
FL=gemini/gemini-3.1-flash-lite
HINTS="--example-format table --schema-format full --cell-format repr --prompt-variant task_variant_2"
```

### CLIPPER against SLIPPER and condition hints (Figure 3a–b, fixed-seven summary)

```bash
$LC --label slipper          --sizes $NINE -- -R ''
$LC --label slipper_M100     --sizes $NINE -- -R '' -M100
$LC --label slopper          --sizes $NINE -- -R $FL
$LC --label slopper_notools  --sizes $NINE -- -R $FL -T0
$LC --label candcond_notools --sizes $NINE -- -R $FL $HINTS -T0 --candidate-conditions
$LC --label candcond_tool_T2 --sizes $NINE -- -R $FL $HINTS --tools-prompt variant1 -T2 \
    --no-lambda-body-tool --candidate-conditions --candidate-conditions-tool
$LC --label slopper_notools_M200 --sizes 250,1200 -- -R $FL -T0 -M200
$LC --label slopper_notools_M400 --sizes 250,1200 -- -R $FL -T0 -M400
```

### Established rule learners (Figures 2 and 3c)

```bash
uv run python scripts/run_baseline_curve.py --suite test --sizes $FIVE --seeds 10 \
    --models ripper,corels,greedy_rule_list,skope_rules,fpskope,rulefit,figs,hstree,tao_tree,boosted_rules,imodels_slipper
```

`uv run python scripts/run_baseline_curve.py --list` shows the estimator
settings. The baselines see the same training subsets as the learners above.

### Anonymized names (Figure 4c)

```bash
uv run python scripts/run_learning_curve.py --suite anon --label slopper --sizes 250 --seeds 10 \
    --min-bestj 2 --bestj-from scripts/test-lc-out/results.csv:slopper -- -R $FL
uv run python scripts/paired_anon.py --label slopper --size 250
```

The anonymized suite draws the same subsets as the original datasets, and
`--bestj-from` gives each pair the same round budget.

### Outer loop: boosting, set cover, one-shot (Table 1 and the covering appendix)

```bash
$LC --label slopper_cover    --sizes $FIVE -- -R $FL --weighting cover
$LC --label slopper_cover_p3 --sizes $FIVE -- -R $FL --weighting cover --cover-patience 3
$LC --label oneshot_slopper  --sizes $FIVE -- -R $FL --oneshot
$LC --label discrete_cover   --sizes $FIVE -- -R '' --weighting cover
$LC --label candcond_cover   --sizes $FIVE -- -R $FL $HINTS --tools-prompt variant1 -T2 \
    --no-lambda-body-tool --candidate-conditions --candidate-conditions-tool --weighting cover
$LC --label oneshot_candcond --sizes $FIVE -- -R $FL --example-format table --schema-format full \
    --cell-format repr --tools-prompt variant1 -T2 --no-lambda-body-tool \
    --candidate-conditions --candidate-conditions-tool --oneshot
# Repaired discrete covering (gate-aware sizing + linear growth)
uv run python scripts/run_repaired_cover_control.py --out runs/repaired_cover --sizes 100,250,600,1200,2000
```

The repaired-covering runner reuses each cell's round budget. It first checks
that the `slipper`, `discrete_cover`, `slopper`, and `slopper_cover` rows exist
and agree on the budget, so run those conditions first.

### Changing CLIPPER's model (Table 3; n=250, three seeds)

```bash
B="$LC --sizes 250 --seeds 3"
$B --label slopper_g38_T0    -- -R gemini/gemini-3.8-flash -T0
$B --label slopper_g38       -- -R gemini/gemini-3.8-flash
$B --label slopper_g25fl_T0  -- -R gemini/gemini-2.5-flash-lite -T0
$B --label slopper_g25fl     -- -R gemini/gemini-2.5-flash-lite
$B --label slopper_sonnet_T0 -- -R anthropic/claude-sonnet-5 -T0
$B --label slopper_haiku_T0  -- -R anthropic/claude-haiku-4-5-20251001 -T0
$B --label slopper_haiku     -- -R anthropic/claude-haiku-4-5-20251001
$B --label slopper_luna      -- -R openai/gpt-5.6-luna --reasoning-effort none
$B --label slopper_qwen9b_T0 -- -R openrouter/qwen/qwen3.5-9b -T0
```

The Flash-Lite rows reuse seeds 1–3 of `slopper` and `slopper_notools`.

### General-purpose coding agents (Table 2 and appendix)

Each agent runs headless in a [sandy](https://github.com/rappdw/sandy) Docker
container. The container holds `train.csv`, `schema.json`, and `TASK.md` and has
no test data, web tools, or GitHub token. Install Docker and sandy, and put the
provider keys in `~/.sandy/.secrets`. See the docstring of
`scripts/run_agent_curve.py` for setup and scoring.

```bash
A="uv run python scripts/run_agent_curve.py --suite test --sizes 250 --seeds 3"
$A --label agent_opus5    --agent claude --model claude-opus-5
$A --label agent_sonnet5  --agent claude --model claude-sonnet-5
$A --label agent_haiku45  --agent claude --model claude-haiku-4-5-20251001
$A --label agent_gpt56_sol   --agent codex --model gpt-5.6-sol --timeout 3000 --workers 1
$A --label agent_gpt56_terra --agent codex --model gpt-5.6-terra
$A --label agent_gpt56_luna  --agent codex --model gpt-5.6-luna
$A --label agent_gemini35_flash      --agent gemini --model gemini-3.5-flash --max-turns 80
$A --label agent_gemini31_flashlite  --agent gemini --model gemini-3.1-flash-lite
$A --label agent_gpt56_sol_norun --agent codex --model gpt-5.6-sol --forbid-learners
```

`--forbid-learners` adds the explicit instruction not to fit an existing
learning algorithm. The condition without it is "Sol (spec gaming)" in the
appendix.

### Text classification, RuleChef, and background knowledge (Figure 4b)

```bash
T="uv run python scripts/run_learning_curve.py --suite text --min-bestj 0"
$T --label slipper -- -R ''
OLD="--schema-format compressed --example-format dict --prompt-variant baseline -T0 --no-lambda-body-tool --tools-prompt baseline"
$T --label slopper_text -- -R $FL $OLD --use-text     # CLIPPER on the raw text
$T --label slopper      -- -R $FL $OLD --hide-text    # CLIPPER on the word sets only
uv run python scripts/run_baseline_curve.py --suite text --models hstree,figs,tao_tree,boosted_rules,corels

# The same comparison on a dataset whose labels were published after the model's training cutoff
R="uv run python scripts/run_learning_curve.py --suite recent --min-bestj 0"
$R --label slipper -- -R ''
$R --label slopper_text -- -R $FL $OLD --use-text
$R --label slopper      -- -R $FL $OLD --hide-text
uv run python scripts/run_baseline_curve.py --suite recent --models hstree,figs,tao_tree,boosted_rules,corels
```

Each document has its raw text (`TEXT`) and its set of words (`WORDS`).
`--use-text` gives CLIPPER the raw text and permits regular expressions;
`--hide-text` restricts it to the word sets that SLIPPER searches. The imodels
baselines use TF-IDF features. These CLIPPER runs predate the tuned `-R`
default, so their flags are spelled out (no coverage tools, `M=100`); they
reproduce the original prompts except for one blank line. These runs used no
minimum round budget (`--min-bestj 0`).

RuleChef 0.2.2 runs in its own environment, with the same model through
OpenRouter and the same training subsets:

```bash
uv venv .rulechef-venv --python 3.12
VIRTUAL_ENV=.rulechef-venv uv pip install rulechef==0.2.2 openai
PYTHONIOENCODING=utf-8 OPENROUTER_API_KEY=... .rulechef-venv/bin/python scripts/run_rulechef_curve.py
```

It uses RuleChef's default settings. Test documents that no rule matches get
the training subset's majority class, and the `coverage` column records how
often a rule fired.

### Banking77: CLIPPER and RuleChef

```bash
uv run python src/tools/build_banking77.py        # once; BANKING77 at a pinned revision
uv venv .rulechef-venv --python 3.12
VIRTUAL_ENV=.rulechef-venv uv pip install "rulechef[agentic]==0.2.2" datasets grex
uv run python scripts/run_banking77.py prepare --seeds 1 2 3 4 5 6 7 8 9 10
uv run python scripts/run_banking77.py run --rulechef-python .rulechef-venv/bin/python \
    --only 'clipper_raw_M100/*' 'clipper_raw_M100_tools/*' 'rulechef/*'
uv run python scripts/run_banking77.py summarize  # writes runs/banking77/results.csv
```

The data are the five BANKING77 intents that RuleChef's own benchmark uses:
the 685 official training queries for those intents, and all 200 official test
queries. `prepare` draws one training subset per (size, seed), and every method
trains on it. CLIPPER reads the raw query, with `M=100` and with or without
coverage tools, and fits one two-class model per intent for six rounds, as for
MedCalc. RuleChef runs its authors' benchmark script
(`vendor/rulechef/benchmark_banking77.py`, unmodified) with the settings of
their published run: regular-expression rules, at most 100 rules, 200 examples
in the prompt, up to 15 rounds of revision, and their agentic coordinator. It
writes rules from five examples per intent and revises them against the other
training queries. Both use Gemini 3.1 Flash-Lite (`GEMINI_API_KEY`). Test
queries that no RuleChef rule matches get the training subset's majority intent;
the `err_strict` column counts them as errors instead. `run` can be resumed and
runs cells in parallel; the grid also defines full-data SLIPPER cells, which the
paper does not report.

### MedCalc: calculator prediction and value extraction (Figure 4a, Table 4, and the MedCalc appendix)

```bash
uv run python src/tools/build_medcalc_extraction_suite.py        # once
uv run python scripts/run_medcalc_repeats.py prepare             # freezes inputs; checks them against our hashes
uv run python scripts/run_medcalc_repeats.py run --cpu-jobs 4 --llm-jobs 8
uv run python scripts/run_medcalc_repeats.py summarize
```

`prepare` downloads MedCalc-Bench-Verified at a pinned revision. It builds the
55-way calculator-prediction inputs at training caps 1, 4, 16, and 64, as word
sets for SLIPPER and raw notes for CLIPPER, and the ten extraction keys. It
then checks every input against `experiments/medcalc/expected_hashes.json`. The
grid has ten seeds for each condition:

- **Calculator prediction:** SLIPPER full data, `M=20`, and `M=100`; CLIPPER
  (raw notes) `M=20` and `M=100`; 55 one-versus-rest ensembles of six rounds.
- **Extraction:** CLIPPER Python and regex rules at `M=20, 55, 110`; ten rounds
  per key; one-second execution limit per rule call.
- **Reference:** the hand-written regex pool in `experiments/medcalc/pools/`,
  selected by the same scorer.

`run` works through the cells in separate processes and can be resumed.
`status` shows progress.

### Development suite (appendix)

```bash
uv run python scripts/run_bestj.py --label discrete -- -R ''
uv run python scripts/run_bestj.py --label tv2_lambda_toolsv1 -- -R $FL --example-format table \
    --schema-format full --cell-format repr --prompt-variant task_variant_2 -T6 -b \
    --no-count-coverage --tools-prompt variant1
uv run python scripts/run_bestj.py --label candcond_tool_T2 -- -R $FL $HINTS --tools-prompt variant1 \
    -T2 --no-lambda-body-tool --candidate-conditions --candidate-conditions-tool
uv run python scripts/summarize_results.py 'scripts/bestj-out/*_config.yaml'
uv run python scripts/run_format_grid.py                 # dict/table x full/compressed schema
uv run python scripts/retrieval_probe.py                 # can the model read counts off the sample table?
```

`run_bestj.py` runs one configuration on the ten development datasets. Each
dataset's round budget is fixed in `scripts/suite.py`. The flags for every
development configuration are in `experiments/configs/tuning_bestj/`. Set-cover
repairs on the development suite (appendix table) use `--suite tuning` with the
flags in `experiments/configs/setcover/`, for example:

```bash
uv run python scripts/run_learning_curve.py --suite tuning --label discrete_cover_fix12_linear \
    --sizes 100,200,300,400,800 --min-bestj 2 -- -R '' --weighting cover --cover-sizing gate --cover-objective linear
```

### Summaries

```bash
uv run python scripts/summarize_learning_curve.py 'scripts/test-lc-out/*_config.yaml'   # per-size tables
uv run python scripts/paired_lc.py --new slopper --base slipper_M100 --csv scripts/test-lc-out/results.csv
```

`data/make_paper_outputs.py` produces the paper's exact tables and figures from
the saved per-seed results in `data/results/`.

## Notes

- **Uncertainty.** Tabular results report mean test error and SEM over seeds.
  Macro error weights datasets equally, and its SEM is
  `sqrt(sum_d SEM_d^2) / D`. Main-text tabular comparisons use seven datasets
  (all but College).
- **Nondeterminism.** Seeds fix training subsets and the learner's own sampling.
  Provider-side generation is not seeded, so LLM conditions do not repeat
  exactly.
- **Imodels baselines.** The imodels baselines depend on those libraries'
  versions (pinned in `uv.lock`).
