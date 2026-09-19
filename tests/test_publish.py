import pytest

from conftest import EXAMPLE_CONFIG
from dragon import publish as pub
from dragon.config import Config
from dragon.errors import DragonError


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo, **kw):
        self.calls.append(("create_repo", repo, kw.get("private")))

    def upload_folder(self, **kw):
        self.calls.append(("upload", kw["repo_id"], tuple(kw.get("allow_patterns") or ())))

    def create_tag(self, repo, **kw):
        self.calls.append(("tag", repo, kw["tag"]))


def project(tmp_path, monkeypatch):
    config = Config.load(EXAMPLE_CONFIG)
    config.root = tmp_path
    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters" / "adapters.safetensors").write_bytes(b"")
    api = FakeApi()
    monkeypatch.setattr(pub, "_api", lambda: (api, "someone"))
    return config, api


def test_the_adapter_goes_to_a_private_repo_named_for_the_model(tmp_path, monkeypatch):
    config, api = project(tmp_path, monkeypatch)
    urls = pub.publish(config)
    assert urls == ["https://huggingface.co/someone/example-7b-lora"]
    assert ("create_repo", "someone/example-7b-lora", True) in api.calls
    assert ("upload", "someone/example-7b-lora", tuple(pub.ADAPTER_FILES)) in api.calls
    assert not [c for c in api.calls if c[0] == "tag"]
    assert "British English" in (tmp_path / "adapters" / "README.md").read_text()


def test_a_tag_is_put_on_every_repository_published(tmp_path, monkeypatch):
    config, api = project(tmp_path, monkeypatch)
    (tmp_path / "fused").mkdir()
    (tmp_path / "fused" / "config.json").write_text("{}")
    urls = pub.publish(config, fused=True, tag="v2")
    assert urls[1].endswith("/example-7b-mlx-8bit")
    assert ("tag", "someone/example-7b-lora", "v2") in api.calls
    assert ("tag", "someone/example-7b-mlx-8bit", "v2") in api.calls


def test_a_missing_fused_model_is_a_clear_error(tmp_path, monkeypatch):
    config, api = project(tmp_path, monkeypatch)
    with pytest.raises(DragonError, match="dragon fuse"):
        pub.publish(config, fused=True)
