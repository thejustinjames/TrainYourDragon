import pytest
import yaml

from conftest import EXAMPLE_CONFIG, ROOT
from dragon.config import Config
from dragon.errors import DragonError
from dragon.example import EXAMPLE_CONFIG as EMBEDDED


def test_the_root_example_matches_the_one_init_writes():
    assert (ROOT / "config.example.yaml").read_text() == EMBEDDED


def test_the_embedded_example_is_valid_yaml_and_has_every_source_kind():
    parsed = yaml.safe_load(EMBEDDED)
    kinds = {s["kind"] for s in parsed["corpus"]["sources"]}
    assert kinds == {"markdown", "pages", "terms", "text", "documents"}


def test_the_example_project_loads():
    config = Config.load(EXAMPLE_CONFIG)
    assert config.base_model.startswith("mlx-community/")
    assert config.corpus_root.is_dir()
    assert "the-quiet-part" in config.holdout


def test_a_missing_system_prompt_is_refused(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"base": "x"}, "corpus": {"root": "."}}))
    with pytest.raises(DragonError, match="system_prompt"):
        Config.load(path)


def test_an_unknown_source_kind_is_refused(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"base": "x"},
                "voice": {"system_prompt": "Write."},
                "corpus": {"root": ".", "sources": [{"name": "s", "kind": "pdf", "path": "."}]},
            }
        )
    )
    with pytest.raises(DragonError, match="kind"):
        Config.load(path)


def test_lora_config_carries_the_training_block():
    config = Config.load(EXAMPLE_CONFIG)
    lora = config.lora_config()
    assert lora["model"] == config.base_model
    assert lora["iters"] == 40
    assert lora["lora_parameters"]["rank"] == 8  # the default survives a partial override
