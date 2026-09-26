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
uv run pytest -q              # the Gemini test is skipped without GEMINI_API_KEY
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

The commands below use these shell definitions (bash or zsh; run them in the
same session):

```bash
lc() { uv run python scripts/run_learning_curve.py --suite test --min-bestj 2 "$@"; }
NINE=100,175,250,400,500,600,800,1200,2000
FIVE=100,250,600,1200,2000
FL=gemini/gemini-3.1-flash-lite
HINTS=(--example-format table --schema-format full --cell-format repr --prompt-variant task_variant_2)
```

### CLIPPER against SLIPPER and condition hints (Figure 3a–b, fixed-seven summary)

```bash
lc --label slipper          --sizes $NINE -- -R ''
lc --label slipper_M100     --sizes $NINE -- -R '' -M100
lc --label slopper          --sizes $NINE -- -R $FL
lc --label slopper_notools  --sizes $NINE -- -R $FL -T0
lc --label candcond_notools --sizes $NINE -- -R $FL "${HINTS[@]}" -T0 --candidate-conditions
lc --label candcond_tool_T2 --sizes $NINE -- -R $FL "${HINTS[@]}" --tools-prompt variant1 -T2 \
    --no-lambda-body-tool --candidate-conditions --candidate-conditions-tool
```

### Established rule learners (Figures 2 and 3c)

```bash
uv run python scripts/run_baseline_curve.py --suite test --sizes $FIVE --seeds 10 \
    --models ripper,corels,greedy_rule_list,skope_rules,fpskope,rulefit,figs,hstree,tao_tree,boosted_rules,imodels_slipper
```

`uv run python scripts/imodels_baselines.py --list` shows the estimator
settings. The baselines see the same training subsets as the learners above.

### Anonymized names (Figure 4d and the anonymization appendix)

```bash
uv run python scripts/run_learning_curve.py --suite anon --label slopper --sizes 250 --seeds 10 \
    --min-bestj 2 --bestj-from scripts/test-lc-out/results.csv:slopper -- -R $FL
uv run python scripts/paired_anon.py --label slopper --size 250
```

The anonymized suite draws the same subsets as the original datasets, and
`--bestj-from` gives each pair the same round budget.

### Outer loop: boosting, set cover, one-shot (Table 1 and the covering appendix)

```bash
lc --label slopper_cover    --sizes $FIVE -- -R $FL --weighting cover
lc --label slopper_cover_p3 --sizes $FIVE -- -R $FL --weighting cover --cover-patience 3
lc --label oneshot_slopper  --sizes $FIVE -- -R $FL --oneshot
lc --label discrete_cover   --sizes $FIVE -- -R '' --weighting cover
lc --label candcond_cover   --sizes $FIVE -- -R $FL "${HINTS[@]}" --tools-prompt variant1 -T2 \
    --no-lambda-body-tool --candidate-conditions --candidate-conditions-tool --weighting cover
lc --label oneshot_candcond --sizes $FIVE -- -R $FL --example-format table --schema-format full \
    --cell-format repr --tools-prompt variant1 -T2 --no-lambda-body-tool \
    --candidate-conditions --candidate-conditions-tool --oneshot
# Repaired discrete covering (gate-aware sizing + linear growth)
uv run python scripts/run_repaired_cover_control.py --out runs/repaired_cover --sizes 100,250,600,1200,2000
```

The repaired-covering runner reuses each cell's round budget. It first checks
that the `slipper`, `discrete_cover`, `slopper`, and `slopper_cover` rows exist
and agree on the budget, so run those conditions first.

### Changing CLIPPER's model (Table 3 and appendix; n=250, three seeds)

```bash
b() { lc --sizes 250 --seeds 3 "$@"; }
b --label slopper_g38_T0    -- -R gemini/gemini-3.8-flash -T0
b --label slopper_g38       -- -R gemini/gemini-3.8-flash
b --label slopper_g25fl_T0  -- -R gemini/gemini-2.5-flash-lite -T0
b --label slopper_g25fl     -- -R gemini/gemini-2.5-flash-lite
b --label slopper_sonnet_T0 -- -R anthropic/claude-sonnet-5 -T0
b --label slopper_haiku_T0  -- -R anthropic/claude-haiku-4-5-20251001 -T0
b --label slopper_haiku     -- -R anthropic/claude-haiku-4-5-20251001
b --label slopper_luna      -- -R openai/gpt-5.6-luna --reasoning-effort none
b --label slopper_qwen9b_T0 -- -R openrouter/qwen/qwen3.5-9b -T0
```

The Flash-Lite rows reuse seeds 1–3 of `slopper` and `slopper_notools`.

### General-purpose coding agents (Table 2 and appendix)

Each agent runs headless in a [sandy](https://github.com/rappdw/sandy) Docker
container. The container holds `train.csv`, `schema.json`, and `TASK.md` and has
no test data, web tools, or GitHub token. Install Docker and sandy, and put the
provider keys in `~/.sandy/.secrets`. See the docstring of
`scripts/run_agent_curve.py` for setup and scoring.

```bash
agent() { uv run python scripts/run_agent_curve.py --suite test --sizes 250 --seeds 3 "$@"; }
agent --label agent_opus5    --agent claude --model claude-opus-5
agent --label agent_sonnet5  --agent claude --model claude-sonnet-5
agent --label agent_haiku45  --agent claude --model claude-haiku-4-5-20251001
agent --label agent_gpt56_sol   --agent codex --model gpt-5.6-sol --timeout 3000 --workers 1
agent --label agent_gpt56_terra --agent codex --model gpt-5.6-terra
agent --label agent_gpt56_luna  --agent codex --model gpt-5.6-luna
agent --label agent_gemini35_flash      --agent gemini --model gemini-3.5-flash --max-turns 80
agent --label agent_gemini31_flashlite  --agent gemini --model gemini-3.1-flash-lite
agent --label agent_gpt56_sol_norun --agent codex --model gpt-5.6-sol --forbid-learners
```

`--forbid-learners` adds the explicit instruction not to fit an existing
learning algorithm. The condition without it is "Sol (spec gaming)" in the
appendix.

### Text classification and background knowledge (Figure 4b and the text-datasets appendix)

```bash
TSIZES=50,100,200,300,400,500,750
OLD=(--schema-format compressed --example-format dict --prompt-variant baseline -T0
     --no-lambda-body-tool --tools-prompt baseline)
txt() { uv run python scripts/run_learning_curve.py --min-bestj 0 --sizes $TSIZES "$@"; }
# `text`: the five text datasets; `recent`: the dataset whose labels were
# published after the model's training cutoff
for S in text recent; do
  txt --suite $S --label slipper      -- -R ''
  txt --suite $S --label slopper_text -- -R $FL "${OLD[@]}" --use-text    # CLIPPER on the raw text
  txt --suite $S --label slopper      -- -R $FL "${OLD[@]}" --hide-text   # CLIPPER on the word sets
  uv run python scripts/run_baseline_curve.py --suite $S --sizes $TSIZES \
      --models hstree,figs,tao_tree,boosted_rules,corels
done
```

Each document has its raw text (`TEXT`) and its set of words (`WORDS`).
`--use-text` gives CLIPPER the raw text and permits regular expressions;
`--hide-text` restricts it to the word sets that SLIPPER searches. The imodels
baselines use TF-IDF features (CORELS uses word-presence indicators). These
CLIPPER runs predate the tuned `-R` default, so their flags are spelled out (no
coverage tools, `M=100`); they reproduce the original prompts except for one
blank line. These runs used no minimum round budget (`--min-bestj 0`).

### BANKING77: CLIPPER and RuleChef (Figure 4c and the BANKING77 appendix)

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

### MedCalc: value extraction and calculator prediction (Figure 4a and the MedCalc appendix)

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
uv run python scripts/run_bestj.py --label candcond_tool_T2 -- -R $FL "${HINTS[@]}" --tools-prompt variant1 \
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
