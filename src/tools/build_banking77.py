"""build_banking77.py - a 5-intent BANKING77 subset in the text-suite format.

The five intents are the ones RuleChef pinned for its published BANKING77
benchmark.  Rows keep BANKING77's official split: every official train row of
those intents forms the training pool (685 rows) and every official test row
(40 per intent, 200 rows) is the test set, the same test set RuleChef reports on.

Each row has the text suite's two attributes: ``WORDS`` (a token set built with
``build_text_suite.tokenize_words``) and ``TEXT`` (the raw query string).

Run from the repo root::

    uv run --group medcalc python src/tools/build_banking77.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_text_suite import SUITE, tokenize_words

HF_REPO = "legacy-datasets/banking77"
HF_REVISION = "f54121560de48f2852f90be299010d1d6dc612ec"
CLASSES = ["beneficiary_not_allowed", "card_arrival", "disposable_card_limits",
           "exchange_rate", "pending_cash_withdrawal"]
NAME = "banking77_5"


def rows(split, names):
    return [{"values": [sorted(tokenize_words(text)), text], "label": names[label]}
            for text, label in zip(split["text"], split["label"]) if names[label] in CLASSES]


def main():
    import datasets
    # The dataset is public; token=False avoids failing on a stale cached login.
    ds = datasets.load_dataset(HF_REPO, revision=HF_REVISION, token=False)
    names = ds["train"].features["label"].names
    out = {"classes": CLASSES,
           "attributes": [{"name": "WORDS", "kind": "set"}, {"name": "TEXT", "kind": "text"}],
           "source": {"hf_repo": HF_REPO, "hf_revision": HF_REVISION},
           "train": rows(ds["train"], names), "test": rows(ds["test"], names)}
    path = os.path.join(SUITE, NAME + ".json")
    os.makedirs(SUITE, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(out, fp)
    print(f"{path}: {len(out['train'])} train / {len(out['test'])} test")


if __name__ == "__main__":
    main()
