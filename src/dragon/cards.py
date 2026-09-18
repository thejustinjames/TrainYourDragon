"""The two documents this project generates for other tools: a Hugging Face
model card and an Ollama Modelfile.

Both are written from `config.yaml` and the adapter's own config, so they
cannot drift from what was actually trained.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from dragon.config import Config

CHATML_TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

CHATML_STOPS = ["<|im_end|>", "<|im_start|>"]


def adapter_facts(config: Config) -> dict:
    path = config.adapter_dir / "adapter_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return config.lora_config()


FLAVOURS = {
    "adapter": "a LoRA adapter for",
    "fused": "the fused {bits}-bit MLX model of",
    "gguf": "the GGUF export of",
}


def model_card(config: Config, *, flavour: str = "adapter", private: bool) -> str:
    facts = adapter_facts(config)
    lora = facts.get("lora_parameters", {})
    today = datetime.date.today().strftime("%d %B %Y")
    if flavour not in FLAVOURS:
        raise ValueError(f"unknown card flavour {flavour!r}")
    what = FLAVOURS[flavour].format(bits=config.fuse_bits)
    joined = "on" if flavour == "adapter" else "fused into"
    library = "gguf" if flavour == "gguf" else "mlx"
    tags = ["lora", library, "writing-style", "fine-tune"]
    if private:
        tags.append("private")

    settings = "\n".join(
        [
            f"- iterations {facts.get('iters')}, batch {facts.get('batch_size')}, "
            f"learning rate {facts.get('learning_rate')}",
            f"- layers {facts.get('num_layers')}, max sequence {facts.get('max_seq_length')}",
            f"- LoRA rank {lora.get('rank')}, scale {lora.get('scale')}, "
            f"dropout {lora.get('dropout')}",
            "- keys: " + ", ".join(lora.get("keys", [])),
        ]
    )

    why_bits = ""
    if flavour != "adapter":
        why_bits = (
            "## Why eight bits\n\n"
            f"Fused at {config.fuse_bits} bits on purpose. A light adapter moves each weight by "
            "less than a 4-bit quantisation step, so fusing into a 4-bit base and re-quantising "
            "rounds most of it away: the voice survives, the recall of anything specific does "
            "not. Load this directory directly; the voice prompt is the chat template's default.\n\n"
        )

    stats = config.root / "data" / "stats.json"
    corpus_line = ""
    if stats.exists():
        try:
            s = json.loads(stats.read_text())
            corpus_line = (
                f"Trained on {s['train']:,} brief-and-prose pairs, "
                f"{s['target_words']:,} words of target text.\n\n"
            )
        except (json.JSONDecodeError, KeyError):
            corpus_line = ""

    return f"""---
base_model: {config.base_model}
library_name: {library}
license: {config.license}
language: [en]
tags: [{", ".join(tags)}]
---

# {config.model_name} — {what} one writer's voice

A rank-{lora.get("rank", "?")} LoRA {joined} `{config.base_model}`,
trained on one person's published writing. It learns the register — sentence
length, rhythm, vocabulary, the shape of an argument — and very little else.

{corpus_line}It does not reliably know facts, not even facts that were in its training
corpus, and it will invent them fluently. Every output is a first draft for the
author to read, correct and sign.

Built {today} with [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple
silicon, using [TrainYourDragon](https://github.com/thejustinjames/TrainYourDragon).

## The system prompt

The adapter was trained with this prompt on every example, and expects it:

```
{" ".join(config.system_prompt.split())}
```

## Load it

{_load_snippet(config, flavour)}

## Training configuration

{settings}

{why_bits}## What it is not

Not a writer, not a source of facts, and not a way round any disclosure
obligation. A model trained to write in a named person's voice is an
impersonation kit in the wrong hands; keep it private unless every person whose
writing is in the corpus has agreed otherwise.
"""


def _load_snippet(config: Config, flavour: str) -> str:
    if flavour == "gguf":
        return (
            "With Ollama, on any machine, straight from this repository:\n\n"
            "```bash\n"
            "ollama run hf.co/<this repo>:Q8_0\n"
            "```\n\n"
            "The `template`, `system` and `params` files here give Ollama the chat\n"
            "template, the voice prompt and the sampling settings, so no Modelfile is\n"
            "needed. Any llama.cpp build loads the `.gguf` files directly."
        )
    if flavour == "fused":
        return '```python\nfrom mlx_lm import load, generate\nmodel, tok = load("<this repo>")\n```'
    return (
        "```python\n"
        "from mlx_lm import load, generate\n"
        "from huggingface_hub import snapshot_download\n"
        'adapter = snapshot_download("<this repo>")\n'
        f'model, tok = load("{config.base_model}", adapter_path=adapter)\n'
        "```"
    )


def modelfile(config: Config, source: Path | str, *, params: dict | None = None) -> str:
    export = config.raw.get("export") or {}
    ollama = export.get("ollama") or {}
    template = ollama.get("template", CHATML_TEMPLATE)
    stops = ollama.get("stop", CHATML_STOPS)
    settings = {"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192}
    settings.update(ollama.get("parameters") or {})
    settings.update(params or {})

    lines = [
        f"# {config.model_name} for Ollama.",
        "# Generated by dragon; edit config.yaml, not this file.",
        f"FROM {source}",
        "",
        f'TEMPLATE """{template}"""',
        "",
        f'SYSTEM """{" ".join(config.system_prompt.split())}"""',
        "",
    ]
    lines += [f"PARAMETER {k} {v}" for k, v in settings.items()]
    lines += [f"PARAMETER stop {s}" for s in stops]
    return "\n".join(lines) + "\n"
