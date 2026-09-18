import yaml

from conftest import EXAMPLE_CONFIG
from dragon.cards import model_card, modelfile
from dragon.config import Config


def config():
    return Config.load(EXAMPLE_CONFIG)


def test_the_model_card_has_valid_frontmatter():
    card = model_card(config(), flavour="adapter", private=True)
    assert card.startswith("---\n")
    meta = yaml.safe_load(card.split("---")[1])
    assert meta["base_model"] == config().base_model
    assert "private" in meta["tags"]


def test_the_card_carries_the_system_prompt_and_the_caveat():
    card = model_card(config(), flavour="fused", private=False)
    assert "British English" in card
    assert "impersonation kit" in card
    assert "private" not in yaml.safe_load(card.split("---")[1])["tags"]


def test_the_modelfile_names_the_source_and_the_prompt(tmp_path):
    text = modelfile(config(), tmp_path / "fused-fp16")
    assert f"FROM {tmp_path / 'fused-fp16'}" in text
    assert "SYSTEM" in text and "British English" in text
    assert "PARAMETER stop <|im_end|>" in text


def test_the_fused_card_says_it_is_eight_bit_and_why():
    card = model_card(config(), flavour="fused", private=True)
    assert "8-bit MLX model" in card
    assert "Why eight bits" in card and "rounds most of it away" in card
    assert "Why eight bits" not in model_card(config(), flavour="adapter", private=True)


def test_the_gguf_card_tells_ollama_users_how_to_pull():
    card = model_card(config(), flavour="gguf", private=True)
    meta = yaml.safe_load(card.split("---")[1])
    assert meta["library_name"] == "gguf"
    assert "ollama run hf.co/" in card


def test_an_unknown_flavour_is_refused():
    import pytest

    with pytest.raises(ValueError, match="flavour"):
        model_card(config(), flavour="onnx", private=True)
