"""Everything that shells out to mlx-lm: train, test, chat, fuse, serve.

mlx-lm is only imported inside the functions that need it, so the dataset
builder and the tests run anywhere, including on Linux CI where MLX will not
install at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from dragon.config import Config
from dragon.errors import DragonError

STOCK_PROMPTS = [
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "You are a helpful assistant.",
    "You are a helpful AI assistant.",
]


def require_mlx() -> None:
    if sys.platform != "darwin":
        raise DragonError(
            "mlx-lm runs on Apple silicon. Building a dataset works anywhere; training does not."
        )
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise DragonError(
            "mlx-lm is not installed in this environment. `pip install -e '.[mlx]'`"
        ) from exc


def run(args: list[str], *, cwd: Path | None = None, tee: Path | None = None) -> int:
    """Run a command, echoing it first. Output is streamed, optionally to a log too."""
    print("+ " + " ".join(args), flush=True)
    if tee is None:
        return subprocess.run(args, cwd=cwd, check=False).returncode
    with (
        subprocess.Popen(
            args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as proc,
        open(tee, "w", encoding="utf-8") as log,
    ):
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return proc.wait()


def module(name: str) -> list[str]:
    return [sys.executable, "-m", name]


# ---------------------------------------------------------------- training
def write_lora_config(config: Config, extra: dict | None = None) -> Path:
    path = config.work_dir / "lora.yaml"
    path.write_text(yaml.safe_dump(config.lora_config(extra), sort_keys=False))
    return path


def train(config: Config, *, resume: bool = False, extra_args: list[str] | None = None) -> int:
    require_mlx()
    if not (config.data_dir / "train.jsonl").exists():
        raise DragonError("no data/train.jsonl. Run `dragon build` first.")
    extra = None
    if resume:
        extra = {"resume_adapter_file": str(config.adapter_dir / "adapters.safetensors")}
    lora = write_lora_config(config, extra)
    args = module("mlx_lm.lora") + ["-c", str(lora), *(extra_args or [])]
    return run(args, cwd=config.root, tee=config.root / "train.log")


def test(config: Config) -> int:
    require_mlx()
    if not (config.data_dir / "test.jsonl").exists():
        raise DragonError("no data/test.jsonl. Hold at least one piece back in corpus.holdout.")
    args = module("mlx_lm.lora") + [
        "--model",
        config.base_model,
        "--adapter-path",
        str(config.adapter_dir),
        "--data",
        str(config.data_dir),
        "--test",
        "--test-batches",
        "-1",
    ]
    return run(args, cwd=config.root)


def chat(config: Config, extra_args: list[str] | None = None) -> int:
    require_mlx()
    args = module("mlx_lm.chat") + [
        "--model",
        config.base_model,
        "--adapter-path",
        str(config.adapter_dir),
        *(extra_args or []),
    ]
    return run(args, cwd=config.root)


# ------------------------------------------------------------------ fusing
def fuse(config: Config, *, dequantize: bool = False, force: bool = False) -> Path:
    """Bake the adapter into the base weights, giving a standalone model folder."""
    require_mlx()
    target = config.fused_fp16_dir if dequantize else config.fused_dir
    if (target / "config.json").exists() and not force:
        print(f"{target} exists; --force to rebuild")
        return target
    if not (config.adapter_dir / "adapters.safetensors").exists():
        raise DragonError(f"no adapter at {config.adapter_dir}. Train one first.")
    args = module("mlx_lm.fuse") + [
        "--model",
        config.base_model,
        "--adapter-path",
        str(config.adapter_dir),
        "--save-path",
        str(target),
    ]
    if dequantize:
        args.append("--dequantize")
    if run(args, cwd=config.root) != 0:
        raise DragonError("mlx_lm.fuse failed")
    if not dequantize:
        apply_default_prompt(config, target)
    return target


def apply_default_prompt(config: Config, model_dir: Path) -> int:
    """Put the voice prompt into the chat template's default.

    Every training pair carried the system prompt, so the adapter's effect is
    conditioned on it. A client that sends no system message otherwise gets the
    base model's stock persona and none of the voice.
    """
    template = model_dir / "chat_template.jinja"
    if not template.exists():
        return 0
    text = template.read_text(encoding="utf-8")
    prompt = " ".join(config.system_prompt.split())
    replaced = 0
    for stock in [*config.stock_prompts, *STOCK_PROMPTS]:
        if stock and stock in text:
            replaced += text.count(stock)
            text = text.replace(stock, prompt)
    if replaced:
        template.write_text(text, encoding="utf-8")
    print(f"{template}: replaced {replaced} default prompt(s)")
    return replaced


# ----------------------------------------------------------------- serving
def serve(config: Config, *, host: str = "127.0.0.1", port: int = 8787) -> int:
    """An OpenAI-compatible endpoint on the loopback interface.

    The fused model is served rather than base-plus-adapter so that whichever
    id a client reads back from /v1/models resolves to the same weights.
    """
    require_mlx()
    fused = fuse(config)
    apply_default_prompt(config, fused)
    print(f"\nhttp://{host}:{port}/v1 — model id: default_model\n")
    args = module("mlx_lm.server") + [
        "--model",
        str(fused),
        "--host",
        host,
        "--port",
        str(port),
    ]
    return run(args, cwd=config.root)


# ------------------------------------------------------------------ ollama
def ollama_import(config: Config, modelfile: Path, tag: str, quantise: str = "q4_K_M") -> int:
    if not shutil.which("ollama"):
        print("ollama is not installed; skipped")
        return 0
    args = ["ollama", "create", tag, "--quantize", quantise, "-f", str(modelfile)]
    code = run(args, cwd=config.root)
    if code == 0:
        print(f"ollama run {tag}")
    return code


def lm_studio_link(fused: Path, name: str) -> Path | None:
    """Link the fused model into LM Studio's models folder, if there is one."""
    base = Path(os.path.expanduser("~/.cache/lm-studio/models"))
    if not base.exists():
        print("LM Studio's models folder not found; skipped")
        return None
    target = base / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(fused)
    print(f"LM Studio: {target}")
    return target
