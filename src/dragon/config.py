"""Read and check `config.yaml`.

One file describes the whole project: which base model, which voice, which
corpus, and how to train. Everything else in the package reads it from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dragon.errors import DragonError

CONFIG_NAME = "config.yaml"

SOURCE_KINDS = {
    "markdown",  # a directory of .md files with frontmatter: essays, posts, notes
    "pages",  # a directory of .md files whose frontmatter is most of the content
    "terms",  # JSON files of [{"term": ..., "definition": ...}]
    "text",  # one plain-text or markdown file, split on its headings
    "documents",  # an explicit list of files, each with names it answers to
}

DEFAULT_TRAINING: dict[str, Any] = {
    "iters": 1200,
    "batch_size": 2,
    "learning_rate": 1.0e-5,
    "num_layers": 16,
    "max_seq_length": 2560,
    "grad_checkpoint": True,
    "steps_per_report": 20,
    "steps_per_eval": 100,
    "val_batches": 25,
    "save_every": 200,
    "lora_parameters": {
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.05,
        "keys": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    },
}


def expand(value: str | Path) -> Path:
    """`~` and `$VARS` in a configured path, resolved."""
    return Path(os.path.expandvars(str(value))).expanduser()


@dataclass
class Source:
    """One place writing comes from."""

    name: str
    kind: str
    path: str | None = None
    label: str = "Document"
    fields: dict[str, Any] = field(default_factory=dict)
    documents: list[dict[str, Any]] = field(default_factory=list)
    prompt: str | None = None
    language: str | None = None
    min_words: int = 30

    def field_name(self, key: str, default: str | None = None) -> str | None:
        """The frontmatter key this corpus uses for a given idea (title, summary...)."""
        return self.fields.get(key, default)


@dataclass
class Config:
    root: Path
    path: Path
    raw: dict[str, Any]

    base_model: str
    model_name: str
    license: str
    system_prompt: str

    corpus_root: Path
    sources: list[Source]
    exclude: set[str]
    holdout: set[str]

    valid_fraction: float
    max_words: int
    seed: int
    recall: bool | int

    training: dict[str, Any]
    stock_prompts: list[str]

    # -------------------------------------------------------------- paths
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def adapter_dir(self) -> Path:
        return self.root / "adapters"

    @property
    def fused_dir(self) -> Path:
        return self.root / "fused"

    @property
    def fused_fp16_dir(self) -> Path:
        return self.root / "fused-fp16"

    @property
    def work_dir(self) -> Path:
        d = self.root / ".dragon"
        d.mkdir(exist_ok=True)
        return d

    # --------------------------------------------------------------- load
    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg_path = Path(path) if path else Path.cwd() / CONFIG_NAME
        if not cfg_path.exists():
            raise DragonError(
                f"no {cfg_path}. Copy config.example.yaml to {CONFIG_NAME} and edit it."
            )
        try:
            raw = yaml.safe_load(cfg_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise DragonError(f"{cfg_path} is not valid YAML: {exc}") from exc
        return cls.from_dict(raw, cfg_path)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], cfg_path: Path) -> Config:
        root = cfg_path.resolve().parent
        model = raw.get("model") or {}
        voice = raw.get("voice") or {}
        corpus = raw.get("corpus") or {}
        dataset = raw.get("dataset") or {}

        base_model = model.get("base")
        if not base_model:
            raise DragonError("config: model.base is required (a Hugging Face model id)")

        system_prompt = (voice.get("system_prompt") or "").strip()
        if not system_prompt:
            raise DragonError(
                "config: voice.system_prompt is required. It is the instruction every "
                "training pair carries, and the one the model expects at inference."
            )

        corpus_root = corpus.get("root")
        if not corpus_root:
            raise DragonError("config: corpus.root is required (where your writing lives)")
        corpus_root = expand(corpus_root)
        if not corpus_root.is_absolute():
            corpus_root = (root / corpus_root).resolve()

        sources = [_source(entry, i) for i, entry in enumerate(corpus.get("sources") or [])]
        if not sources:
            raise DragonError("config: corpus.sources is empty; there is nothing to learn from")

        training = _merge(DEFAULT_TRAINING, raw.get("training") or {})

        return cls(
            root=root,
            path=cfg_path,
            raw=raw,
            base_model=base_model,
            model_name=model.get("name") or "voice-model",
            license=str(model.get("license") or "apache-2.0"),
            system_prompt=system_prompt,
            corpus_root=corpus_root,
            sources=sources,
            exclude=set(corpus.get("exclude") or []),
            holdout=set(corpus.get("holdout") or []),
            valid_fraction=float(dataset.get("valid_fraction", 0.06)),
            max_words=int(dataset.get("max_words_per_example", 1300)),
            seed=int(dataset.get("seed", 7)),
            recall=dataset.get("recall_pairs", True),
            training=training,
            stock_prompts=list(voice.get("replaces_default_prompts") or []),
        )

    # ------------------------------------------------------------ helpers
    def resolve(self, relative: str | Path) -> Path:
        """A corpus-relative path, made absolute."""
        p = expand(relative)
        return p if p.is_absolute() else (self.corpus_root / p)

    def lora_config(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """The config mlx_lm.lora wants, assembled from ours."""
        cfg = {
            "model": self.base_model,
            "train": True,
            "data": str(self.data_dir),
            "adapter_path": str(self.adapter_dir),
            "seed": self.seed,
            **self.training,
        }
        cfg.update(extra or {})
        return cfg


def _source(entry: Any, index: int) -> Source:
    if not isinstance(entry, dict):
        raise DragonError(f"config: corpus.sources[{index}] should be a mapping")
    kind = entry.get("kind")
    if kind not in SOURCE_KINDS:
        raise DragonError(
            f"config: corpus.sources[{index}].kind is {kind!r}; "
            f"expected one of {', '.join(sorted(SOURCE_KINDS))}"
        )
    name = entry.get("name") or f"{kind}-{index}"
    if kind == "documents":
        if not entry.get("documents"):
            raise DragonError(f"config: source {name!r} is kind 'documents' but lists none")
    elif not entry.get("path"):
        raise DragonError(f"config: source {name!r} needs a path")
    return Source(
        name=name,
        kind=kind,
        path=entry.get("path"),
        label=entry.get("label") or "Document",
        fields=entry.get("fields") or {},
        documents=entry.get("documents") or [],
        prompt=entry.get("prompt"),
        language=entry.get("language"),
        min_words=int(entry.get("min_words", 30)),
    )


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
