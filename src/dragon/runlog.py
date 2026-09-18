"""Reading a run after the fact: the loss curve from train.log, the
checkpoints in adapters/, and how many passes a configuration makes over the
corpus. None of it needs mlx-lm, so it works on any machine that has the files.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from dragon.config import Config
from dragon.errors import DragonError

_VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
_TRAIN = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
_CHECKPOINT = re.compile(r"^(\d+)_adapters\.safetensors$")

# Rough, and good enough for a pass count: English prose runs about 1.3
# tokens a word on the tokenisers these models use.
TOKENS_PER_WORD = 1.3


@dataclass
class Point:
    iteration: int
    loss: float


def curve(log: Path) -> tuple[list[Point], list[Point]]:
    """(validation points, training points) from an mlx_lm.lora log."""
    if not log.exists():
        raise DragonError(f"no {log}. Nothing has been trained here yet.")
    text = log.read_text(encoding="utf-8", errors="replace")
    val = [Point(int(i), float(x)) for i, x in _VAL.findall(text)]
    train = [Point(int(i), float(x)) for i, x in _TRAIN.findall(text)]
    return val, train


def describe(val: list[Point]) -> list[str]:
    """What the validation curve is saying, in plain words."""
    if len(val) < 3:
        return ["Too few validation points to say anything yet."]
    low = min(val, key=lambda p: p.loss)
    last = val[-1]
    lines = [f"Lowest validation loss {low.loss:.3f} at iteration {low.iteration}."]
    if last is not low and last.loss > low.loss + 0.02:
        lines.append(
            f"The last point ({last.loss:.3f} at {last.iteration}) is above it: "
            f"memorisation starting. Keep the checkpoint nearest {low.iteration}."
        )
    tail = val[-3:]
    spread = max(p.loss for p in tail) - min(p.loss for p in tail)
    if spread < 0.02:
        lines.append(f"Flat since iteration {tail[0].iteration}: the useful part is done.")
    return lines


def checkpoints(adapter_dir: Path) -> list[tuple[int, Path]]:
    if not adapter_dir.is_dir():
        return []
    found = []
    for path in adapter_dir.iterdir():
        m = _CHECKPOINT.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def promote(config: Config, iteration: int) -> Path:
    """Make one saved checkpoint the live adapter."""
    available = dict(checkpoints(config.adapter_dir))
    if iteration not in available:
        have = ", ".join(str(i) for i in available) or "none"
        raise DragonError(f"no checkpoint at iteration {iteration}. Saved: {have}")
    live = config.adapter_dir / "adapters.safetensors"
    shutil.copyfile(available[iteration], live)
    return live


def passes(config: Config, target_words: int, brief_words: int = 0) -> float:
    """How many times a run at these settings sees the corpus. Approximate."""
    tokens = (target_words + brief_words) * TOKENS_PER_WORD
    if tokens <= 0:
        return 0.0
    t = config.training
    seen = t["iters"] * t["batch_size"] * t["max_seq_length"]
    # Sequences are padded to the longest in the batch, not to max_seq_length,
    # so this overstates a little. Direction is what matters.
    return seen / tokens
