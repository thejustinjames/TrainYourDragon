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
    private: bool = True,
    adapter_repo: str | None = None,
    fused_repo: str | None = None,
    owner: str | None = None,
) -> list[str]:
    api, user = _api()
    owner = owner or user
    today = datetime.date.today().strftime("%d %B %Y")
    urls = []

    adapter_dir = config.adapter_dir
    if not (adapter_dir / "adapters.safetensors").exists():
        raise DragonError(f"no adapter at {adapter_dir}. Train one first.")

    repo = adapter_repo or f"{owner}/{config.model_name}-lora"
    api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
    (adapter_dir / "README.md").write_text(
        model_card(config, fused=False, private=private), encoding="utf-8"
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
    urls.append(url)

    if fused:
        fused_dir = config.fused_dir
        if not (fused_dir / "config.json").exists():
            raise DragonError(f"no fused model at {fused_dir}. Run `dragon fuse` first.")
        repo = fused_repo or f"{owner}/{config.model_name}-mlx"
        api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
        (fused_dir / "README.md").write_text(
            model_card(config, fused=True, private=private), encoding="utf-8"
        )
        api.upload_folder(
            repo_id=repo,
            folder_path=str(fused_dir),
            repo_type="model",
            commit_message=f"Fused model, {today}",
        )
        url = f"https://huggingface.co/{repo}"
        print(f"fused → {url} ({'private' if private else 'PUBLIC'})")
        urls.append(url)

    return urls


def local_card(config: Config, path: Path, *, fused: bool = False) -> Path:
    path.write_text(model_card(config, fused=fused, private=True), encoding="utf-8")
    return path
