"""normalize: one canonicalizer for gold and predicted extraction values."""
from slipper.normalize import canon, equiv, parse_value


def test_number_int_float_collapse():
    assert canon(48) == canon(48.0)
    assert canon(1 / 3) == canon(0.333333)


def test_numeric_string_coerces_to_float():
    # a rule returning "48" must match gold 48
    assert equiv("48", 48)
    assert equiv(" 1.4 ", 1.4)


def test_string_casefold_and_strip():
    assert equiv("Male", "male")
    assert canon("Years ") == "years"
    assert not equiv("Male", "Female")


def test_bool_unchanged_and_never_a_number():
    assert equiv(True, True)
    assert not equiv(True, False)
    # the trap: True == 1.0 in Python, so bools must not land in the number space
    assert canon(True) != canon(1)
    assert canon(False) != canon(0)


def test_none_and_unknown_pass_through():
    assert canon(None) is None
    assert equiv(None, None)
    obj = {"a": 1}
    assert canon(obj) is obj


def test_value_unit_two_lists():
    assert equiv([87, "years"], [87.0, "Years "])
    assert equiv([120, "mm Hg"], [120.0, "mmHg"])
    assert equiv([80, "beats per minute"], [80.0, "bpm"])
    assert equiv([6.0, "uL"], [6.0, "µL"])
    assert equiv([106, "umol/L"], [106.0, "µmol/L"])
    assert equiv([48.0, "kg"], (48, "KG"))  # list/tuple canonicalize alike


def test_units_are_spelling_only_never_conversion():
    assert not equiv([37.0, "degrees celsius"], [37.0, "degrees fahrenheit"])
    assert not equiv([140, "mEq/L"], [140, "mmol/L"])
    assert not equiv([163.0, "cm"], [163.0, "m"])


def test_three_list_steroid_conversion():
    assert equiv(["Dexamethasone PO", 7.96, "mg"], ["dexamethasone po", 7.96, "mg"])
    assert not equiv(["Dexamethasone PO", 7.96, "mg"],
                     ["Dexamethasone IV", 7.96, "mg"])


def test_parse_value_round_trips_a_dict_value():
    assert parse_value("[87, 'years']") == [87, "years"]
    assert parse_value("True") is True
    assert parse_value("1.4") == 1.4
    assert parse_value("'Male'") == "Male"


def test_parse_value_keeps_non_literal_strings():
    assert parse_value("Male") == "Male"
    assert parse_value("  mm Hg ") == "mm Hg"
    assert parse_value("[87, years]") == "[87, years]"


def test_parse_then_canon_matches_a_gold_repr():
    gold = "{'sex': 'Male', 'age': [87, 'years']}"
    entities = parse_value(gold)
    assert equiv("male", entities["sex"])
    assert equiv([87.0, "Years"], entities["age"])
