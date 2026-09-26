# Datasets

We do not redistribute the datasets. `download.py` fetches each raw file from a
public source, checks its sha256 against the file used in our experiments, runs
the converters in `src/tools/`, and checks the sha256 of every resulting JSON
file. A clean run therefore reproduces our inputs byte for byte.

```bash
uv run python datasets/download.py                 # all suites
uv run python datasets/download.py --suite test    # evaluation suite only
uv run python datasets/download.py --suite tuning  # development suite only
uv run python datasets/download.py --suite text    # text suite only
uv run python datasets/download.py --suite recent  # post-cutoff text dataset only
uv run python datasets/download.py --no-verify     # continue past a sha256 mismatch
```

`download.py` uses only the standard library.

## Tabular evaluation suite (`test_suite/`, 8 datasets)

The suite studied by Kruschel et al. Each dataset becomes a binary task with the
event class positive and a fixed, stratified 25% test split (seed 1); see
`src/tools/build_test_suite.py` for target columns, dropped ID columns, and
feature choices.

| Dataset | Source | Citation |
|---|---|---|
| Adult | UCI `adult/adult.data` | Kohavi (1996) |
| Bank marketing | UCI `00222/bank.zip` (`bank-full.csv`) | Moro, Cortez & Rita (2014) |
| College | GitHub mirror `cdiegor/MineracaoDeDados` of Kaggle `saddamazyazy/go-to-college-dataset` | Mukti (2022) |
| Water potability | GitHub mirror `MainakRepositor/Datasets` of Kaggle `adityakadiwal/water-potability` | Kadiwal (2021) |
| Stroke | GitHub mirror `JulioArapa/mypackage` of Kaggle `fedesoriano/stroke-prediction-dataset` | fedesoriano (2021) |
| Telco churn | GitHub `IBM/telco-customer-churn-on-icp4d` | IBM (2019) |
| COMPAS | GitHub `propublica/compas-analysis` (`compas-scores-two-years.csv`) | Larson et al. (2016) |
| HELOC | GitHub mirror `benoitparis/explainable-challenge` | FICO (2018) |

## Development suite (`tuning_suite/`, 10 datasets)

Used only to select the prompt and tool configuration (paper appendix,
"Development experiments").

| Dataset | Source | Notes |
|---|---|---|
| Hypothyroid | UCI `thyroid-disease/hypothyroid.data` | |
| Breast cancer Wisconsin | UCI `breast-cancer-wisconsin/breast-cancer-wisconsin.data` | original 1991 version |
| Cylinder bands | UCI `cylinder-bands/bands.data` | two ragged rows dropped |
| Abalone | UCI `abalone/abalone.data` | target: at least ten rings |
| German credit | UCI `statlog/german/german.data` | codes decoded to readable values |
| Obesity | UCI dataset 544 | obesity categories merged; height and weight removed |
| Student performance | UCI dataset 320 (`student-por.csv`) | target: final grade below ten; earlier grades removed |
| Pima diabetes | GitHub mirror `jbrownlee/Datasets` | zero-valued measurements kept as given |
| Titanic | GitHub mirror `datasciencedojo/datasets` | Kaggle training split |
| Move1 | original RIPPER/SLIPPER distribution (included here) | see below |

**Move1** has no standalone public download; it ships with the original
RIPPER/SLIPPER distribution. Its three raw files are therefore included in
`tuning_suite/` and are checked against the same sha256 table.

`pdata/vote1.*` is the congressional-voting sample from the same distribution,
with its JSON conversion; only the tests use it.

`tuning_suite/*.info` are dataset descriptions (from the UCI/Kaggle data cards)
read with `--include-advice .info` (older configs call this `--include-info`); `tuning_suite/*.advice` are the model-written notes
read by `--include-advice .claude-opus-4-8.advice`. Both were used only in
development-suite comparisons.

## Text-classification suite (`text_suite/`, 5 datasets)

The paper does not report results on this suite; `src/tools/build_banking77.py`
reuses its encoding. Each document keeps its raw text (`TEXT`)
and its set of words (`WORDS`); each dataset has a stratified 25% test split
(seed 1). See `src/tools/build_text_suite.py` for tokenization.

| Dataset | Source | Citation |
|---|---|---|
| SMS spam | UCI dataset 228 (`SMSSpamCollection`) | Almeida, Gómez Hidalgo & Yamakami (2011) |
| Amazon, IMDb, Yelp sentiment | UCI dataset 331 (Sentiment Labelled Sentences) | Kotzias et al. (2015) |
| YouTube spam | UCI dataset 380 (five videos merged) | Alberto, Lochter & Almeida (2015) |

## Post-cutoff text dataset (`recent_suite/`)

`inflation_abstracts`: economics-inflation research abstracts labeled by
whether they use machine-learning methods (UCI dataset 1125, 2025). The labels
were published after the model's training cutoff. It is encoded like the text
suite (`TEXT` and `WORDS`) with a stratified 25% test split. The paper does not
report results on it.

## BANKING77 (five intents)

The five intents that RuleChef's own BANKING77 benchmark uses
(`beneficiary_not_allowed`, `card_arrival`, `disposable_card_limits`,
`exchange_rate`, `pending_cash_withdrawal`), downloaded from Hugging Face
(`legacy-datasets/banking77`, revision
`f54121560de48f2852f90be299010d1d6dc612ec`; Casanueva et al., 2020; license
CC BY 4.0). The official split is kept: the 685 official training queries of
those intents form the training pool, and all 200 official test queries are
the test set. Queries are encoded like the text suite (`TEXT` and `WORDS`).

```bash
uv run python src/tools/build_banking77.py
```

## Anonymized evaluation suite (`anon_suite/`)

Derived from `test_suite/*.json`; dataset, attribute, class, and categorical
value names are replaced with meaningless identifiers, while values, rows, and
training subsets are unchanged.

```bash
uv run python src/tools/anonymize_suite.py        # test_suite -> anon_suite
```

## MedCalc-Bench-Verified

Downloaded from Hugging Face (`nsk7153/MedCalc-Bench-Verified`, revision
`591157b3343b4dda247294f9d929da4c75026fa8`) by the MedCalc builders; license
CC BY-SA 4.0. Calculator prediction is built directly by
`scripts/run_medcalc_repeats.py prepare`; value extraction needs the extraction
suite first:

```bash
uv run python src/tools/build_medcalc_extraction_suite.py   # -> datasets/medcalc_extract/
uv run python scripts/run_medcalc_repeats.py prepare         # checks every input against
                                                              # experiments/medcalc/expected_hashes.json
```

`src/tools/build_medcalc_suite.py` builds the same calculator-prediction data as
standalone one-versus-rest JSON files (`datasets/medcalc_suite*/`) for use with
the `slipper` command line.
