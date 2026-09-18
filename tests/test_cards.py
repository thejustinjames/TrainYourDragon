import yaml

from conftest import EXAMPLE_CONFIG
from dragon.cards import model_card, modelfile
from dragon.config import Config


def config():
    return Config.load(EXAMPLE_CONFIG)


def test_the_model_card_has_valid_frontmatter():
    card = model_card(config(), fused=False, private=True)
    assert card.startswith("---\n")
    meta = yaml.safe_load(card.split("---")[1])
    assert meta["base_model"] == config().base_model
    assert "private" in meta["tags"]


def test_the_card_carries_the_system_prompt_and_the_caveat():
    card = model_card(config(), fused=True, private=False)
    assert "British English" in card
    assert "impersonation kit" in card
    assert "private" not in yaml.safe_load(card.split("---")[1])["tags"]


def test_the_modelfile_names_the_source_and_the_prompt(tmp_path):
    text = modelfile(config(), tmp_path / "fused-fp16")
    assert f"FROM {tmp_path / 'fused-fp16'}" in text
    assert "SYSTEM" in text and "British English" in text
    assert "PARAMETER stop <|im_end|>" in text
