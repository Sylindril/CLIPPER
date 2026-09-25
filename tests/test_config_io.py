"""config.config_dict / save_yaml / load_yaml (reproducible-config I/O)."""
import pytest

from slipper import config


@pytest.fixture(autouse=True)
def restore_config():
    """Snapshot and restore the config globals around each test (they are
    module-level mutable state)."""
    saved = config.config_dict()
    yield
    g = config.__dict__
    for k, v in saved.items():
        g[k] = v


def test_config_dict_scope():
    d = config.config_dict()
    # ALL-CAPS enum constants are excluded
    assert "GIVEN" not in d and "MDL" not in d and "INCFREQ" not in d
    # runtime-derived fields are excluded
    assert "Value_function" not in d and "Epsilon" not in d and "Advice_text" not in d
    # real knobs are present
    for k in ("Rule_llm_model", "Tool_budget", "Prompt_variant", "Sample_size",
              "Simplify", "Class_ordering"):
        assert k in d


def test_config_dict_values_are_yaml_scalars():
    for k, v in config.config_dict().items():
        assert isinstance(v, (str, int, float, bool, type(None))), (k, v)


def test_save_load_roundtrip(tmp_path):
    d0 = config.config_dict()
    p = tmp_path / "c.yaml"
    config.save_yaml(str(p))
    # perturb several knobs, then reload
    config.Tool_budget = 12345
    config.Prompt_variant = "oneshot"
    config.Sample_size = 7
    config.Rule_llm_model = "x/y"
    applied = config.load_yaml(str(p))
    assert config.config_dict() == d0
    assert applied == d0                       # every knob was applied


def test_save_returns_and_writes(tmp_path):
    p = tmp_path / "c.yaml"
    returned = config.save_yaml(str(p))
    assert returned == config.config_dict()
    import yaml
    on_disk = yaml.safe_load(p.read_text())
    assert on_disk == config.config_dict()


def test_load_ignores_unknown_and_nonconfig(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("Tool_budget: 3\nNot_a_key: 9\nValue_function: 0\n")
    applied = config.load_yaml(str(p))
    assert applied == {"Tool_budget": 3}       # unknown + runtime keys skipped
    assert config.Tool_budget == 3


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        config.load_yaml(str(tmp_path / "nope.yaml"))
