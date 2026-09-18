import json

import pytest

from conftest import EXAMPLE_CONFIG
from dragon import gguf
from dragon.config import Config
from dragon.errors import DragonError


def config():
    return Config.load(EXAMPLE_CONFIG)


def test_the_converter_is_found_through_the_environment(tmp_path, monkeypatch):
    (tmp_path / gguf.CONVERTER).write_text("# converter")
    monkeypatch.setenv("LLAMA_CPP", str(tmp_path))
    assert gguf.llama_cpp_dir(config()) == tmp_path
    assert gguf.converter(config()) == tmp_path / gguf.CONVERTER


def test_config_wins_over_the_environment(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / gguf.CONVERTER).write_text("")
    monkeypatch.setenv("LLAMA_CPP", str(b))
    cfg = config()
    cfg.raw["export"] = {"gguf": {"llama_cpp": str(a)}}
    assert gguf.llama_cpp_dir(cfg) == a


def test_a_missing_converter_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP", raising=False)
    monkeypatch.setattr(gguf, "CANDIDATE_CHECKOUTS", [])
    cfg = config()
    cfg.raw["export"] = {}
    with pytest.raises(DragonError, match="git clone"):
        gguf.converter(cfg)


def test_the_quantiser_is_found_in_the_checkout_build_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    (tmp_path / gguf.CONVERTER).write_text("")
    (tmp_path / "build" / "bin").mkdir(parents=True)
    (tmp_path / "build" / "bin" / gguf.QUANTISER).write_text("")
    monkeypatch.setenv("LLAMA_CPP", str(tmp_path))
    assert gguf.quantiser(config()) == tmp_path / "build" / "bin" / gguf.QUANTISER


def test_sidecars_give_ollama_the_voice(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr(gguf, "gguf_dir", lambda c: tmp_path)
    (tmp_path / "example-7b-f16.gguf").write_bytes(b"x" * 20)
    (tmp_path / "example-7b-Q4_K_M.gguf").write_bytes(b"x" * 5)
    written = gguf.write_sidecars(cfg, tmp_path)
    names = {p.name for p in written}
    assert names == {"template", "system", "params", "Modelfile"}
    assert "British English" in (tmp_path / "system").read_text()
    params = json.loads((tmp_path / "params").read_text())
    assert params["stop"] == ["<|im_end|>", "<|im_start|>"]
    assert "<|im_start|>" in (tmp_path / "template").read_text()
    # the Modelfile points at the quantised file, relatively, so it works in a Hub repo too
    assert "FROM ./example-7b-Q4_K_M.gguf" in (tmp_path / "Modelfile").read_text()


def test_the_quantised_file_is_the_smallest_non_f16(tmp_path, monkeypatch):
    monkeypatch.setattr(gguf, "gguf_dir", lambda c: tmp_path)
    (tmp_path / "m-f16.gguf").write_bytes(b"x" * 3)
    (tmp_path / "m-Q8_0.gguf").write_bytes(b"x" * 9)
    (tmp_path / "m-Q4_K_M.gguf").write_bytes(b"x" * 5)
    assert gguf.quantised_file(config()).name == "m-Q4_K_M.gguf"
