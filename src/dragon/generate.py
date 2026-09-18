"""Writing with the model: one brief in, one draft out."""

from __future__ import annotations

from pathlib import Path

from dragon.config import Config
from dragon.errors import DragonError
from dragon.mlxops import require_mlx


def brief_from_notes(notes: str, title: str, section: str | None, label: str = "piece") -> str:
    where = (
        f"the section '{section}' of the {label} '{title}'"
        if section
        else f"the opening of the {label} '{title}'"
    )
    return f"Turn these dictated notes into {where}.\n\n{notes.strip()}"


def write(
    config: Config,
    brief: str,
    *,
    base: bool = False,
    adapter: Path | None = None,
    max_tokens: int = 900,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system: str | None = None,
) -> str:
    require_mlx()
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    adapter_path = None if base else str(adapter or config.adapter_dir)
    if adapter_path and not Path(adapter_path).exists():
        raise DragonError(f"no adapter at {adapter_path}. Train one, or pass --base.")

    model, tokeniser = load(config.base_model, adapter_path=adapter_path)
    messages = [
        {"role": "system", "content": system or config.system_prompt},
        {"role": "user", "content": brief},
    ]
    prompt = tokeniser.apply_chat_template(messages, add_generation_prompt=True)
    out = generate(
        model,
        tokeniser,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p),
        verbose=False,
    )
    return out.strip()
