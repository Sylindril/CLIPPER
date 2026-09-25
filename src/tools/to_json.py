#!/usr/bin/env python3
"""to_json.py - convert a RIPPER/SLIPPER dataset (stem.names + stem.data
[+ stem.test]) into the JSON format read by the `slipper` package.

This is a one-off conversion utility, deliberately separate from the learner
(which only reads JSON).  It is schema-driven: the .names file determines how
each field of the .data/.test files is tokenised, so set-valued fields keep
words like "2d416" whole while continuous fields parse numbers.

Usage:
    python3 to_json.py STEM [STEM ...]      # writes STEM.json next to inputs
    python3 to_json.py --out OUT STEM       # write to OUT instead
"""

import argparse
import json
import os
import sys


class Reader:
    """A small character reader over a text blob, with on-demand token
    methods that tokenize the .names/.data format closely enough for these
    datasets."""

    def __init__(self, text):
        self.s = text
        self.i = 0
        self.n = len(text)

    def _skip(self):
        while self.i < self.n:
            c = self.s[self.i]
            if c == "%":
                while self.i < self.n and self.s[self.i] != "\n":
                    self.i += 1
            elif c.isspace():
                self.i += 1
            else:
                break

    def eof(self):
        self._skip()
        return self.i >= self.n

    def peek(self):
        self._skip()
        return self.s[self.i] if self.i < self.n else None

    def word(self):
        """Read one token: a single punctuation char (. , :), a quoted
        string, or a maximal run of non-space, non-delimiter characters."""
        self._skip()
        if self.i >= self.n:
            return None
        c = self.s[self.i]
        if c in ".,:":
            self.i += 1
            return c
        if c == "'":
            self.i += 1
            start = self.i
            while self.i < self.n and self.s[self.i] not in "'\n":
                self.i += 1
            w = self.s[start:self.i]
            if self.i < self.n and self.s[self.i] == "'":
                self.i += 1
            return w
        start = self.i
        while (self.i < self.n and not self.s[self.i].isspace()
               and self.s[self.i] not in ".,:'%"):
            self.i += 1
        return self.s[start:self.i]

    def number(self):
        self._skip()
        start = self.i
        if self.i < self.n and self.s[self.i] in "+-":
            self.i += 1
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if (self.i + 1 < self.n and self.s[self.i] == "."
                and self.s[self.i + 1].isdigit()):
            self.i += 1
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
        if self.i < self.n and self.s[self.i] in "eE":
            j = self.i + 1
            if j < self.n and self.s[j] in "+-":
                j += 1
            if j < self.n and self.s[j].isdigit():
                self.i = j
                while self.i < self.n and self.s[self.i].isdigit():
                    self.i += 1
        return float(self.s[start:self.i])


KEYWORDS = {"continuous", "symbolic", "set", "bag", "ignore"}


def parse_names(text):
    r = Reader(text)
    # classes: words separated by commas, ended by '.'
    classes = []
    while True:
        w = r.word()
        if w is None or w == ".":
            break
        if w == ",":
            continue
        classes.append(w)

    attributes = []
    while not r.eof():
        name = r.word()
        if name is None:
            break
        colon = r.word()
        if colon != ":":
            raise ValueError("expected ':' after attribute %r, got %r" % (name, colon))
        spec = r.word()
        suppressed = False
        if spec == "suppressed":
            suppressed = True
            spec = r.word()
        attr = {"name": name}
        if suppressed:
            attr["suppressed"] = True
        if spec == "continuous":
            attr["kind"] = "continuous"
            _expect_stop(r)
        elif spec == "symbolic":
            attr["kind"] = "symbolic"
            _expect_stop(r)
        elif spec in ("set", "bag"):
            attr["kind"] = spec
            _expect_stop(r)
        elif spec == "ignore":
            attr["kind"] = "ignore"
            _expect_stop(r)
        else:
            # explicit value list: spec is the first value
            attr["kind"] = "symbolic"
            values = [spec]
            while True:
                t = r.word()
                if t is None or t == ".":
                    break
                if t == ",":
                    continue
                values.append(t)
            attr["values"] = values
        attributes.append(attr)
    return classes, attributes


def _expect_stop(r):
    t = r.word()
    if t != ".":
        raise ValueError("expected '.', got %r" % (t,))


def parse_data(text, attributes):
    r = Reader(text)
    rows = []
    while not r.eof():
        rows.append(_parse_example(r, attributes))
    return rows


def _parse_example(r, attributes):
    values = []
    for adef in attributes:
        kind = adef["kind"]
        if kind == "set" or kind == "bag":
            words = []
            while True:
                t = r.word()
                if t in (",", ".", None):
                    break
                words.append(t)
            if len(words) == 1 and words[0] == "?":
                values.append(None)
            else:
                values.append(words)
        elif kind == "ignore":
            while True:
                t = r.word()
                if t in (",", ".", None):
                    break
            values.append(None)
        elif kind == "continuous":
            if r.peek() == "?":
                r.word()
                values.append(None)
            else:
                values.append(r.number())
            r.word()  # consume separator ','
        else:  # symbolic
            w = r.word()
            values.append(None if w == "?" else w)
            r.word()  # consume separator ','

    label = r.word()
    weight = 1.0
    t = r.word()  # ':' or '.'
    if t == ":":
        weight = r.number()
        t = r.word()  # '.'
    row = {"values": values, "label": label}
    if weight != 1.0:
        row["weight"] = weight
    return row


def convert(stem):
    with open(stem + ".names") as fp:
        classes, attributes = parse_names(fp.read())
    out = {"classes": classes, "attributes": attributes}
    with open(stem + ".data") as fp:
        out["train"] = parse_data(fp.read(), attributes)
    if os.path.exists(stem + ".test"):
        with open(stem + ".test") as fp:
            out["test"] = parse_data(fp.read(), attributes)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stems", nargs="+", help="dataset stem(s) (without extension)")
    ap.add_argument("--out", help="output path (only with a single stem)")
    ns = ap.parse_args(argv[1:])
    if ns.out and len(ns.stems) != 1:
        ap.error("--out requires exactly one stem")
    for stem in ns.stems:
        data = convert(stem)
        out_path = ns.out or (stem + ".json")
        with open(out_path, "w") as fp:
            json.dump(data, fp)
        ntest = len(data.get("test", []))
        print("%s -> %s : %d classes, %d attributes, %d train, %d test"
              % (stem, out_path, len(data["classes"]), len(data["attributes"]),
                 len(data["train"]), ntest))


if __name__ == "__main__":
    main(sys.argv)
