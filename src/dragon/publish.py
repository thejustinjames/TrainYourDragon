"""Putting the adapter, and optionally the fused model, on the Hugging Face Hub.

Repositories are created private. `--public` is possible and deliberately
awkward: a model that writes in a named person's voice does not belong on a
public index unless that person has said so.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from dragon.cards import model_card
from dragon.config import Config
from dragon.errors import DragonError

ADAPTER_FILES = ["adapters.safetensors", "adapter_config.json", "README.md"]


def _api():
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError as exc:
        raise DragonError("huggingface_hub is not installed. `pip install -e '.[hub]'`") from exc
    try:
        user = whoami()["name"]
    except Exception as exc:  # any auth failure, however the hub reports it
        raise DragonError(
            'not signed in to Hugging Face. Run: python -c "from huggingface_hub '
            'import login; login()" with a write token.'
        ) from exc
    return HfApi(), user


def publish(
    config: Config,
    *,
    fused: bool = False,
    gguf: bool = False,
    private: bool = True,
    adapter_repo: str | None = None,
    fused_repo: str | None = None,
    gguf_repo: str | None = None,
    owner: str | None = None,
    tag: str | None = None,
) -> list[str]:
    """Upload, and return the repository URLs. With `tag`, each repository's new
    head is tagged, so a version can be pulled by name later rather than by
    commit sha: `snapshot_download(repo, revision="v2")`."""
    api, user = _api()
    owner = owner or user
    today = datetime.date.today().strftime("%d %B %Y")
    urls = []

    def tagged(repo: str) -> None:
        if tag:
            api.create_tag(repo, tag=tag, tag_message=f"{tag}, {today}", exist_ok=True)
            print(f"  tagged {tag}")

    adapter_dir = config.adapter_dir
    if not (adapter_dir / "adapters.safetensors").exists():
        raise DragonError(f"no adapter at {adapter_dir}. Train one first.")

    repo = adapter_repo or f"{owner}/{config.model_name}-lora"
    api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
    (adapter_dir / "README.md").write_text(
        model_card(config, flavour="adapter", private=private), encoding="utf-8"
    )
    api.upload_folder(
        repo_id=repo,
        folder_path=str(adapter_dir),
        repo_type="model",
        allow_patterns=ADAPTER_FILES,
        commit_message=f"Adapter, {today}",
    )
    url = f"https://huggingface.co/{repo}"
    print(f"adapter → {url} ({'private' if private else 'PUBLIC'})")
    tagged(repo)
    urls.append(url)

    if fused:
        fused_dir = config.fused_dir
        if not (fused_dir / "config.json").exists():
            raise DragonError(f"no fused model at {fused_dir}. Run `dragon fuse` first.")
        repo = fused_repo or f"{owner}/{config.model_name}-mlx-{config.fuse_bits}bit"
        api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
        (fused_dir / "README.md").write_text(
            model_card(config, flavour="fused", private=private), encoding="utf-8"
        )
        api.upload_folder(
            repo_id=repo,
            folder_path=str(fused_dir),
            repo_type="model",
            commit_message=f"Fused model, {today}",
        )
        url = f"https://huggingface.co/{repo}"
        print(f"fused → {url} ({'private' if private else 'PUBLIC'})")
        tagged(repo)
        urls.append(url)

    if gguf:
        from dragon.gguf import existing, gguf_dir, write_sidecars

        if not existing(config):
            raise DragonError(f"no GGUF files in {gguf_dir(config)}. Run `dragon gguf` first.")
        write_sidecars(config, gguf_dir(config))
        repo = gguf_repo or f"{owner}/{config.model_name}-gguf"
        api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
        (gguf_dir(config) / "README.md").write_text(
            model_card(config, flavour="gguf", private=private), encoding="utf-8"
        )
        api.upload_folder(
            repo_id=repo,
            folder_path=str(gguf_dir(config)),
            repo_type="model",
            allow_patterns=["*.gguf", "README.md", "template", "system", "params"],
            commit_message=f"GGUF, {today}",
        )
        url = f"https://huggingface.co/{repo}"
        print(f"gguf \u2192 {url} ({'private' if private else 'PUBLIC'})")
        print(f"  ollama run hf.co/{repo}")
        tagged(repo)
        urls.append(url)

    return urls


def local_card(config: Config, path: Path, *, flavour: str = "adapter") -> Path:
    path.write_text(model_card(config, flavour=flavour, private=True), encoding="utf-8")
    return path
