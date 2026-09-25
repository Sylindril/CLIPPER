"""build_text_suite.py - convert the text_suite/ raw downloads to slipper2 JSON.

A *third* benchmark suite (alongside ``test_suite/`` and ``tuning_suite/``),
this one for **text classification** -- the regime where the LLM rule-builder
(``slopper``, the ``-R`` path) is expected to have the largest edge over the
discrete grammar search, because a rule can lean on what words *mean*.

Every dataset is a two-class problem with **two attributes** per example:

  - ``WORDS`` (**set**) -- the bag of tokens, like the in-repo ``tdata/talks``
    precedent, which slipper's grammar tests with ``WORDS ~ token``.  This is
    what the discrete learner (and the current slopper) uses.  No vocabulary is
    declared in advance -- the learner discovers useful tokens from the data.
  - ``TEXT`` (**text**) -- the raw document string.  The ``text`` kind (a
    port-only extension) is skipped by the discrete grammar exactly like
    ``ignore`` (slipper results are identical whether or not it is present),
    but its value is preserved on load and exposed to the LLM rule-builder
    under ``--use-text``, which also authorizes ``re`` in the sandbox so
    ``covers()`` can use regex on the string.

Sources (raw files under ``text_suite/raw/``, fetched by
``datasets/download.py --suite text``):

  sms_spam          UCI SMS Spam Collection   (spam vs ham, 13% spam)
  sentiment_amazon  \
  sentiment_imdb     >  UCI Sentiment Labelled Sentences (Kotzias KDD-15),
  sentiment_yelp    /   1000 balanced sentences each, positive vs negative
  youtube_spam      UCI YouTube Spam Collection (5 videos merged; spam vs ham)

Tokenisation
------------
``tokenize_words`` turns one document into a **set** of tokens (presence, not
counts).  **No filtering / no data loss**: every row is kept, so the instance
counts match the raw sources exactly, and content the plain ``[a-z0-9']``
tokenizer would silently drop is instead captured as explicit tokens so it can
be a learnable feature.  Processed per whitespace-run:

  - plain words        lower-cased; accents & fullwidth Latin folded to ASCII
                       (NFKD + ``_LATIN_EXTRA``); smart/1252 apostrophes -> '
                       so contractions survive.  Then ``[a-z0-9']`` extraction.
  - currency           £/€/¥ -> ``[gbp]``/``[euro]``/``[yen]``
  - emoticons          ``:) :-D ;) :( :P`` ... -> ``[smile] [grin] [wink]
                       [frown] [tongue_out]`` ... (see ``_emoticon_token``)
  - emoji              hearts & heart-eyes -> ``[heart]``; face emoji collapse
                       onto the emoticon buckets plus ``[crying]``/``[sleeping]``
                       (see ``_face_bucket``); every other symbol keeps its
                       slugified Unicode name (``[slice_of_pizza]``, ``[siren]``)
  - ASCII art          a run containing box-drawing/block/geometric glyphs
                       collapses to a single ``[ascii_art]`` (absorbing anything
                       embedded in it)
  - non-Latin scripts  a per-script marker ``[hangul]``/``[arabic]``/...

The token maps are enumerated by hand from the raw data (no third-party deps);
extend them as new datasets introduce new characters.  A document that still
reduces to zero tokens is kept as an empty set (classified by the default rule).

Run from the repo root::

    python src/tools/build_text_suite.py            # writes text_suite/*.json
"""

import csv
import json
import os
import random
import re
import sys
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(_HERE, "..", "..", "text_suite")
RAW = os.path.join(SUITE, "raw")

TEST_FRAC = 0.25   # stratified held-out fraction (no dataset ships a native split)
SEED = 1

_WORD = re.compile(r"[a-z0-9']+")

# --- special-token maps (dependency-free; enumerated from the raw data) ----- #
# Everything below turns non-word
# content that the plain [a-z0-9'] tokenizer would drop into explicit tokens, so
# emoji/emoticons/currency become learnable features instead of silent gaps.

_CURRENCY = {"£": "[gbp]", "€": "[euro]", "¥": "[yen]"}
_APOS = {"’", "‘", "", ""}       # smart / cp1252 apostrophes -> '
_DROP = {"﻿", "️", "︎", " ", "­", "​"}  # BOM/VS/nbsp/shy

# Latin letters NFKD can't fold to ASCII (no canonical decomposition); mapped by
# hand so they stay part of their word instead of becoming a [latin] marker.
# Expand as new datasets introduce more.
_LATIN_EXTRA = {"ı": "i", "İ": "i", "ł": "l", "Ł": "l", "ø": "o", "Ø": "o",
                "đ": "d", "Đ": "d", "ð": "d", "þ": "th", "ß": "ss",
                "æ": "ae", "œ": "oe", "ħ": "h"}

# Emoticons: eyes [:;=], optional nose, mouth -> token.  (No '8' eye: it made
# "1948)" and "8pm" false-match.)  Mapped by (eye, mouth) in _emoticon_token.
_EMOTICON = re.compile(r"[:;=]-?[)(dpo|]|<3", re.I)


def _is_art(c):
    """Box-drawing, block-element and geometric-shape glyphs (ASCII art)."""
    return 0x2500 <= ord(c) <= 0x25FF


def _emoticon_token(s):
    s = s.lower()
    if s == "<3":
        return "[heart]"
    eye, mouth = s[0], s[-1]
    if mouth == "(":
        return "[frown]"
    if mouth == ")":
        return "[wink]" if eye == ";" else "[smile]"
    return {"d": "[grin]", "p": "[tongue_out]", "o": "[surprise]", "|": "[neutral]"}.get(mouth)


def _slug(name):
    return "[%s]" % re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# Negative-affect keywords used by _face_bucket.
_FACE_NEG = ("ANGRY", "POUTING", "WEARY", "TIRED", "DISAPPOINTED",
             "FROWNING", "ANGUISHED", "PERSEVERING", "CONFOUNDED", "UNAMUSED",
             "WORRIED", "CONFUSED", "PENSIVE", "DIZZY", "GRIMACING",
             "MASK", "FEARFUL", "SCREAMING", "TRIUMPH", "SICK", "SWEAT")

# Emoji whose slugified Unicode name is unwieldy, or that should join another
# token: a friendlier/redirected token. Checked before _face_bucket.
_EMOJI_ALIAS = {"POLICE CARS REVOLVING LIGHT": "[siren]",
                "SMILING FACE WITH HORNS": "[imp]"}


def _face_bucket(name):
    """Collapse a face emoji onto the emoticon buckets, or None if not a face.
    Strong-positive (SMILING/GRINNING) dominates mixed names like 'smiling face
    with cold sweat'; otherwise negative affect is checked before soft-positive
    so 'disappointed but relieved face' reads as [frown], not [smile]."""
    if "KISS" in name:
        return "[kiss]"
    if "TONGUE" in name:
        return "[tongue_out]"
    if "SMILING" in name or "GRINNING" in name:
        if "WINKING" in name:
            return "[wink]"
        return "[grin]" if ("GRINNING" in name or "OPEN MOUTH" in name) else "[smile]"
    if "WINKING" in name:
        return "[wink]"
    if "TEARS OF JOY" in name:
        return "[grin]"
    if "CRYING" in name:
        return "[crying]"
    if "SLEEP" in name:
        return "[sleeping]"
    if any(k in name for k in _FACE_NEG):
        return "[frown]"
    if any(k in name for k in ("ASTONISHED", "HUSHED", "FLUSHED", "OPEN MOUTH")):
        return "[surprise]"
    if any(k in name for k in ("NEUTRAL", "EXPRESSIONLESS", "WITHOUT MOUTH")):
        return "[neutral]"
    if any(k in name for k in ("SMIRK", "RELIEVED", "SAVOURING", "SMILE")):
        return "[smile]"
    return None


def _emoji_token(c):
    """Named token for a symbol/emoji char: hearts -> [heart]; face emoji ->
    the emoticon buckets ([smile]/[frown]/...); every other symbol keeps its
    (slugified) Unicode name."""
    try:
        name = unicodedata.name(c)
    except ValueError:
        return None
    # heart *symbols* and heart-eyed *faces* both -> [heart]
    if ("HEART" in name and "FACE" not in name) or "HEART-SHAPED EYES" in name:
        return "[heart]"
    return _EMOJI_ALIAS.get(name) or _face_bucket(name) or _slug(name)


def _reduces_to_ascii(c):
    """True if c folds onto ASCII letter(s) (accented / fullwidth / _LATIN_EXTRA
    Latin) -- handled by word normalisation, not a script marker."""
    if c in _LATIN_EXTRA:
        return True
    d = unicodedata.normalize("NFKD", c)
    return any(ch.isascii() and ch.isalpha() for ch in d)


def _script_token(c):
    """Per-script marker for a non-Latin letter: [hangul], [arabic], ..."""
    name = unicodedata.name(c, "")
    return "[%s]" % name.split()[0].lower() if name else None


# --------------------------------------------------------------------------- #
# Tokeniser
# --------------------------------------------------------------------------- #
def tokenize_words(text):
    """Bag-of-words *set* (unique tokens) for one document.

    Processed per whitespace-run so an ASCII-art run collapses to a single
    ``[ascii_art]`` token (absorbing any glyphs embedded in it).  Other runs
    contribute: currency (``[gbp]``/``[euro]``/``[yen]``), emoticons
    (``[smile]``/``[frown]``/...), emoji (``[heart]`` or the Unicode name), and
    non-Latin scripts (``[hangul]``/...), plus the plain words, which are
    lower-cased and accent-/fullwidth-normalised (NFKD) before ``[a-z0-9']``
    extraction.  Nothing is dropped silently: a run that yields no tokens simply
    contributes nothing, and its document may end up an empty set.
    """
    toks = set()
    for run in text.split():
        if any(_is_art(c) for c in run):
            toks.add("[ascii_art]")
            continue
        for sym, tk in _CURRENCY.items():
            if sym in run:
                toks.add(tk)
        for m in _EMOTICON.finditer(run):
            t = _emoticon_token(m.group(0))
            if t:
                toks.add(t)
        for c in run:
            if ord(c) < 128 or c in _DROP or c in _CURRENCY:
                continue
            if unicodedata.category(c) == "So":            # symbol / emoji
                t = _emoji_token(c)
                if t:
                    toks.add(t)
            elif c.isalpha() and not _reduces_to_ascii(c):
                t = _script_token(c)
                if t:
                    toks.add(t)
        norm = "".join(_LATIN_EXTRA.get(c) or ("'" if c in _APOS else c) for c in run)
        norm = unicodedata.normalize("NFKD", norm)
        norm = "".join(c for c in norm if not unicodedata.combining(c)).lower()
        toks.update(_WORD.findall(norm))
    return sorted(toks)


# --------------------------------------------------------------------------- #
# Raw readers -> (list of (raw_text, label), classes) with the positive class
# first.  Tokenisation happens later in build(), so each example can carry both
# the WORDS set and the raw TEXT string.
# --------------------------------------------------------------------------- #
def read_sms():
    rows = []
    with open(os.path.join(RAW, "sms", "SMSSpamCollection"), encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            lab, _, text = line.partition("\t")
            rows.append((text, "spam" if lab == "spam" else "ham"))
    return rows, ["spam", "ham"]


def _read_sentiment(fname):
    rows = []
    with open(os.path.join(RAW, "sentiment", fname), encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            text, _, score = line.rpartition("\t")
            score = score.strip()
            if score not in ("0", "1"):
                continue
            rows.append((text, "positive" if score == "1" else "negative"))
    return rows, ["positive", "negative"]


def read_youtube():
    rows = []
    files = ["Youtube01-Psy.csv", "Youtube02-KatyPerry.csv", "Youtube03-LMFAO.csv",
             "Youtube04-Eminem.csv", "Youtube05-Shakira.csv"]
    for f in files:
        with open(os.path.join(RAW, "youtube", f), encoding="utf-8", newline="") as fp:
            for r in csv.DictReader(fp):
                cls = (r.get("CLASS") or "").strip()
                if cls not in ("0", "1"):
                    continue
                rows.append((r.get("CONTENT") or "", "spam" if cls == "1" else "ham"))
    return rows, ["spam", "ham"]


DATASETS = {
    "sms_spam":         read_sms,
    "sentiment_amazon": lambda: _read_sentiment("amazon_cells_labelled.txt"),
    "sentiment_imdb":   lambda: _read_sentiment("imdb_labelled.txt"),
    "sentiment_yelp":   lambda: _read_sentiment("yelp_labelled.txt"),
    "youtube_spam":     read_youtube,
}
# (inflation_abstracts moved to build_recent_suite.py -- the post-cutoff
#  "recent datasets" grouping; it imports this module's helpers.)


# --------------------------------------------------------------------------- #
# Assembly: min-df vocab (train-only), stratified split, JSON emit
# --------------------------------------------------------------------------- #
def stratified_split(rows, classes):
    rng = random.Random(SEED)
    train, test = [], []
    for cls in classes:
        group = [r for r in rows if r[1] == cls]
        rng.shuffle(group)
        n_test = int(round(len(group) * TEST_FRAC))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build(name, loader):
    rows, classes = loader()                          # rows: (raw_text, label)
    train, test = stratified_split(rows, classes)     # keeps every row (empty ok)

    def pack(rows):
        # Two attributes per example: WORDS (the bag-of-words set) and TEXT (the
        # raw document string).  TEXT is a "text"-kind attribute -> skipped by
        # the discrete grammar (so slipper results are unchanged) but preserved
        # on load and exposed to the LLM under --use-text (regex rules).
        return [{"values": [tokenize_words(text), text], "label": lab}
                for text, lab in rows]

    tr, te = pack(train), pack(test)
    out = {"classes": classes,
           "attributes": [{"name": "WORDS", "kind": "set"},
                          {"name": "TEXT", "kind": "text"}],
           "train": tr, "test": te}
    path = os.path.join(SUITE, name + ".json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp)

    pos = sum(1 for ex in tr if ex["label"] == classes[0])
    empty = sum(1 for ex in tr + te if not ex["values"][0])
    vocab = len({t for ex in tr for t in ex["values"][0]})
    print("%-17s -> %-24s  %5d train / %4d test  pos=%s %4.1f%%  vocab=%d  empty=%d"
          % (name, name + ".json", len(tr), len(te), classes[0],
             100.0 * pos / max(len(tr), 1), vocab, empty))


def main(argv):
    which = argv[1:] or list(DATASETS)
    for name in which:
        if name not in DATASETS:
            sys.exit("unknown dataset %r (choices: %s)"
                     % (name, ", ".join(DATASETS)))
        build(name, DATASETS[name])


if __name__ == "__main__":
    main(sys.argv)
