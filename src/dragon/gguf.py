"""GGUF export, so the model can be pulled by Ollama on any machine.

MLX weights only run on Apple silicon and only in MLX. GGUF runs in llama.cpp
and everything built on it, Ollama included, and Ollama can pull a GGUF
repository straight from the Hub: `ollama run hf.co/<user>/<repo>:Q8_0`.

mlx-lm's own GGUF writer only knows the Llama architecture, so this uses
llama.cpp's converter on the dequantised fused model, then llama-quantize.
Both come from a llama.cpp checkout (or `brew install llama.cpp` for the
quantiser); neither is bundled here.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from dragon.cards import CHATML_STOPS, CHATML_TEMPLATE
from dragon.config import Config
from dragon.errors import DragonError
from dragon.mlxops import fuse, run

CONVERTER = "convert_hf_to_gguf.py"
QUANTISER = "llama-quantize"
DEFAULT_QUANT = "Q8_0"  # not Q4: a 4-bit re-quantisation rounds a light adapter away

CANDIDATE_CHECKOUTS = ["~/llama.cpp", "~/src/llama.cpp", "~/Documents/GitHub/llama.cpp"]

HOW_TO_GET_CONVERTER = (
    "llama.cpp's converter is needed and was not found. Either:\n"
    "  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp\n"
    "  pip install -e '.[gguf]'          # gguf, torch, transformers for the converter\n"
    "or point export.gguf.llama_cpp in config.yaml (or $LLAMA_CPP) at an existing checkout."
)
HOW_TO_GET_QUANTISER = (
    "llama-quantize was not found, so only the f16 file was written. "
    "`brew install llama.cpp` provides it, or build it in the checkout."
)


def llama_cpp_dir(config: Config) -> Path | None:
    """The llama.cpp checkout, from config, the environment, or the usual places."""
    export = (config.raw.get("export") or {}).get("gguf") or {}
    candidates = [export.get("llama_cpp"), os.environ.get("LLAMA_CPP"), *CANDIDATE_CHECKOUTS]
    for c in candidates:
        if not c:
            continue
        d = Path(os.path.expandvars(str(c))).expanduser()
        if (d / CONVERTER).exists():
            return d
    return None


def converter(config: Config) -> Path:
    d = llama_cpp_dir(config)
    if d is None:
        raise DragonError(HOW_TO_GET_CONVERTER)
    return d / CONVERTER


def quantiser(config: Config) -> Path | None:
    found = shutil.which(QUANTISER)
    if found:
        return Path(found)
    d = llama_cpp_dir(config)
    if d:
        for rel in ("build/bin", "."):
            p = d / rel / QUANTISER
            if p.exists():
                return p
    return None


def gguf_dir(config: Config) -> Path:
    return config.root / "gguf"


def existing(config: Config) -> list[Path]:
    d = gguf_dir(config)
    return sorted(d.glob("*.gguf")) if d.is_dir() else []


def quantised_file(config: Config) -> Path | None:
    """The smallest GGUF present, which is the one to run locally."""
    files = [p for p in existing(config) if "f16" not in p.stem.lower()]
    return min(files, key=lambda p: p.stat().st_size) if files else None


# ------------------------------------------------------------------ export
def export(config: Config, *, quant: str = DEFAULT_QUANT, force: bool = False) -> list[Path]:
    """fused-fp16/ → gguf/<name>-f16.gguf → gguf/<name>-<quant>.gguf, plus the
    three sidecar files Ollama reads from a Hub repository."""
    out = gguf_dir(config)
    out.mkdir(exist_ok=True)
    f16 = out / f"{config.model_name}-f16.gguf"
    quantised = out / f"{config.model_name}-{quant}.gguf"
    written: list[Path] = []

    if f16.exists() and not force:
        print(f"{f16} exists; --force to rebuild")
    else:
        script = converter(config)
        source = fuse(config, dequantize=True)
        args = [sys.executable, str(script), str(source), "--outfile", str(f16), "--outtype", "f16"]
        if run(args, cwd=config.root) != 0:
            raise DragonError(
                "the converter failed. Its own output above says why; the usual cause is "
                "a missing Python package: pip install -e '.[gguf]'"
            )
    written.append(f16)

    tool = quantiser(config)
    if tool is None:
        print(HOW_TO_GET_QUANTISER)
    elif quantised.exists() and not force:
        print(f"{quantised} exists; --force to rebuild")
        written.append(quantised)
    else:
        if run([str(tool), str(f16), str(quantised), quant], cwd=config.root) != 0:
            raise DragonError("llama-quantize failed")
        written.append(quantised)

    written += write_sidecars(config, out)
    return written


def write_sidecars(config: Config, out: Path) -> list[Path]:
    """`template`, `system` and `params`: what Ollama reads alongside a GGUF on
    the Hub so that `ollama run hf.co/<repo>` gets the voice, not the base
    persona. Plus a Modelfile for a local `ollama create`."""
    export = (config.raw.get("export") or {}).get("ollama") or {}
    template = export.get("template", CHATML_TEMPLATE)
    stops = export.get("stop", CHATML_STOPS)
    params = {"temperature": 0.7, "top_p": 0.9, "num_ctx": 8192}
    params.update(export.get("parameters") or {})
    params["stop"] = stops

    files = {
        out / "template": template,
        out / "system": " ".join(config.system_prompt.split()) + "\n",
        out / "params": json.dumps(params, indent=2) + "\n",
    }
    local = quantised_file(config) or next(iter(existing(config)), None)
    if local:
        from dragon.cards import modelfile

        files[out / "Modelfile"] = modelfile(config, f"./{local.name}")
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    return list(files)
