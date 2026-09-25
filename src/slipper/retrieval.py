"""Bounded retrieval from one rule proposal's current training population.

All state is owned by the proposal, including its independent RNG. No weights,
examples, or learner sampling streams are mutated. Retries share the exposure
budget and seen-row set, while the existing tool-call budget stays per attempt.
"""
from dataclasses import dataclass, field
import json

import numpy as np


@dataclass
class RetrievalSession:
    dataset: list
    positive_class: object
    sample: list
    generator: object
    max_rows: int = 20
    per_call: int = 10
    max_chars: int = 20000
    seen: set = field(init=False)
    rows_used: int = field(default=0, init=False)
    chars_used: int = field(default=0, init=False)
    calls_made: int = field(default=0, init=False)

    def __post_init__(self):
        if min(self.max_rows, self.per_call, self.max_chars) <= 0:
            raise ValueError("retrieval limits must be positive")
        self.seen = {ex.index for ex in self.sample}
        if len({ex.index for ex in self.dataset}) != len(self.dataset):
            raise ValueError("retrieval requires unique training row indices")

    def fetch(self, *, where, label, k, compile_filter, to_dict):
        if label not in ("positive", "negative", "either"):
            raise ValueError("label must be positive, negative, or either")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        predicate = compile_filter(where) if where.strip() else lambda x: True
        self.calls_made += 1
        remaining = self.max_rows - self.rows_used
        rows, reason = [], "ok"
        if remaining <= 0 or self.chars_used >= self.max_chars:
            reason = "retrieval_budget_exhausted"
        else:
            eligible = []
            for ex in self.dataset:
                if ex.index in self.seen or ex.wt <= 0:
                    continue
                positive = ex.lab.nom is self.positive_class
                if label != "either" and positive != (label == "positive"):
                    continue
                values = to_dict(ex.inst)
                if predicate(values):
                    eligible.append((ex, values))
            count = min(k, self.per_call, remaining, len(eligible))
            if not count:
                reason = "no_unseen_matching_rows"
            else:
                weights = np.asarray([ex.wt for ex, _ in eligible], dtype=float)
                weights /= weights.sum()
                chosen = self.generator.choice(len(eligible), count, replace=False, p=weights)
                for i in chosen:
                    ex, values = eligible[i]
                    row = {"row_id": ex.index, "class": ex.lab.nom.name,
                           "boosting_weight": float(ex.wt), "x": values}
                    size = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    if self.chars_used + size > self.max_chars:
                        # Keep notes whole; silently clipping them could remove
                        # the very evidence that makes them counterexamples.
                        reason = "row_exceeds_remaining_character_budget"
                        break
                    rows.append(row)
                    self.seen.add(ex.index)
                    self.rows_used += 1
                    self.chars_used += size
        return {"rows": rows, "returned": len(rows), "status": reason,
                "remaining_rows": self.max_rows - self.rows_used,
                "remaining_characters": self.max_chars - self.chars_used}
