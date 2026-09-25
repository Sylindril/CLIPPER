"""Prompt assembly: the template fields that vary per prompt variant.

The header and sample-note lines became ``{header}`` / ``{sample_note}`` fields
when --oneshot arrived, because their hardcoded wording ("one rule", "the
current boosting distribution") is false in one-pass mode.  Every condition in
the development suite was run against the *previous* wording, so the default
rendering has to stay byte-identical -- a reworded prompt is a different
condition, and silently rewording one would invalidate the comparisons rather
than fail a test.  Hence the golden strings below.
"""
import pytest

from slipper import llm_rule_builder as L

# Exactly as the templates read before parameterization (git b914b0e and earlier).
GOLDEN_HEADER = "You are building one rule for a SLIPPER-style boosted rule learner."
GOLDEN_NOTE = """Below is a weighted sample of {n_sampled} training examples drawn from the
current boosting distribution; higher-weight rows are the ones the previous
rules misclassify and that this new rule most needs to handle.
"""
GOLDEN_NOTE_TABLE_HEAD = ("Below is a weighted sample of {n_sampled} training "
                          "examples drawn from the\ncurrent boosting "
                          "distribution, as a table.")


def test_default_framing_is_unchanged():
    assert L._HEADER_DEFAULT == GOLDEN_HEADER
    assert L._SAMPLE_NOTE_DEFAULT == GOLDEN_NOTE
    assert L._SAMPLE_NOTE_DEFAULT_TABLE.startswith(GOLDEN_NOTE_TABLE_HEAD)


@pytest.mark.parametrize("template", ["_DEFAULT_PROMPT", "_DEFAULT_PROMPT_TABLE"])
def test_templates_still_carry_every_field(template):
    """A missing field would raise KeyError at .format() time -- i.e. mid-run,
    against a live model, which is how the one cell in the M200 curve died."""
    fields = dict(header="H", task_definition="TD", grow_note="GN", task_info="TI",
                  maximization_criterion="MC", schema="SCH", examples="EX",
                  sample_note="SN", tools_hint="TH", noncover="False",
                  candidate_conditions="CC", text_note="",
                  posclass="P", negclass="N", n_sampled=42)
    out = getattr(L, template).format(**fields)
    for v in ("H", "TD", "SCH", "EX", "SN", "TH"):
        assert v in out


def test_oneshot_variant_is_registered_with_its_own_framing():
    assert "oneshot" in L._PROMPT_VARIANTS
    assert L._HEADERS["oneshot"] != L._HEADER_DEFAULT
    plain, table = L._SAMPLE_NOTES["oneshot"]
    for note in (plain, table):
        # the boosting framing is a lie in one-pass mode: no previous rules,
        # uniform weights
        assert "boosting distribution" not in note
        assert "previous rules to correct" in note
    assert "COMPLETE rule set" in L._PROMPT_VARIANTS["oneshot"][0]


def test_unlisted_variants_fall_back_to_the_default_framing():
    for variant in ("baseline", "task_variant_1", "task_variant_2"):
        assert L._HEADERS.get(variant, L._HEADER_DEFAULT) == L._HEADER_DEFAULT
        assert L._SAMPLE_NOTES.get(
            variant, (L._SAMPLE_NOTE_DEFAULT, L._SAMPLE_NOTE_DEFAULT_TABLE))[0] \
            == L._SAMPLE_NOTE_DEFAULT


# ---------------------------------------------------------------------------
# --oneshot's suggested rule count (K, carried by -n)
# ---------------------------------------------------------------------------

@pytest.fixture
def _restore_oneshot_rules():
    from slipper import config
    saved = config.Oneshot_rules
    yield config
    config.Oneshot_rules = saved


def test_rules_hint_free_when_no_suggestion(_restore_oneshot_rules):
    _restore_oneshot_rules.Oneshot_rules = 0
    hint = L._rules_hint("republican")
    assert "as many rules as the data calls for" in hint
    assert "budget" not in hint


def test_rules_hint_states_k_as_a_budget(_restore_oneshot_rules):
    _restore_oneshot_rules.Oneshot_rules = 7
    hint = L._rules_hint("republican")
    assert "about 7 rules" in hint
    # a quota would push the model to pad a short rule set
    assert "not pad" in hint


def test_every_task_definition_tolerates_the_rules_hint_field():
    """The hint is passed to whichever variant is active; only oneshot uses it,
    and the others must not blow up on the extra kwarg."""
    for name, (template, _noncover) in L._PROMPT_VARIANTS.items():
        out = template.format(posclass="P", negclass="N", rules_hint="RH")
        assert ("RH" in out) == (name == "oneshot"), name


# ---------------------------------------------------------------------------
# kind: "text" -- the raw-note attribute (LLM path only)
#
# One dataset carries a long, newline-and-pipe-bearing note so the whole path
# is exercised at once: loader -> inst_to_dict -> schema -> kind row -> cell.
# The point of the kind is that the string arrives *intact*, so the assertions
# are about the note surviving, not about formatting.
# ---------------------------------------------------------------------------

NOTE = ("A 47-year-old man | presented with chest pain.\n" * 80) + "TAILMARKER"

TEXT_SPEC = {
    "classes": ["pos", "neg"],
    "attributes": [{"name": "WORDS", "kind": "set"},
                   {"name": "NOTE", "kind": "text"}],
    "train": [{"values": [["chest", "pain"], NOTE], "label": "pos"},
              {"values": [["fever"], None], "label": "neg"}],
}


@pytest.fixture
def text_dataset(tmp_path):
    import json

    from slipper import dataset

    p = tmp_path / "text.json"
    p.write_text(json.dumps(TEXT_SPEC))
    train, _ = dataset.load_dataset(str(p))
    return train


def test_text_attribute_round_trips_verbatim(text_dataset):
    from slipper.dataset import inst_to_dict

    x = inst_to_dict(text_dataset[0].inst)
    assert x["NOTE"] == NOTE                      # not tokenized, not truncated
    assert x["WORDS"] == ["chest", "pain"]        # the set column is untouched
    assert inst_to_dict(text_dataset[1].inst)["NOTE"] is None   # null -> missing


def test_text_attribute_is_described_in_the_schema(text_dataset):
    assert "| NOTE | text | free text |" in L._render_schema()
    assert L._attr_kind_map()["NOTE"] == "text"


@pytest.mark.parametrize("cell_format", ["repr", "str"])
def test_text_cell_keeps_the_whole_note_on_one_row(text_dataset, cell_format, monkeypatch):
    from slipper import config

    monkeypatch.setattr(config, "Cell_format", cell_format)
    monkeypatch.setattr(config, "Table_minibatch_size", 0)
    table = L._render_examples_table(text_dataset)
    rows = [ln for ln in table.splitlines() if ln.startswith("| pos |")]
    assert len(rows) == 1                          # newlines did not split the row
    assert "TAILMARKER" in rows[0]                 # the tail survived: no elision
    assert "| text |" in table                     # the kind row names the column


def test_text_attribute_gets_no_grammar_conditions(text_dataset):
    from slipper import grammar

    assert grammar.ld_grammar(text_dataset)
    lhs = {r.lhs.name for r in grammar.Grammar}
    assert "cond_WORDS" in lhs and "cond_NOTE" not in lhs
    # NOTE is attribute index 1: no terminal condition anywhere mentions it.
    assert not any(g.attr_index == 1 for r in grammar.Grammar for g in r.rhs)
