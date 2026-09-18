"""Turning written prose into something a trainer can eat.

Nothing here is model-specific. It strips the markup a static site adds, finds
the headings an author already wrote, and cuts long pieces at paragraph
boundaries so no example ends mid-sentence.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)
_HEADING = re.compile(r"^(#{2,3})\s+(.+?)\s*$")


def frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a `---` YAML header from the body. No header is not an error."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    return (data if isinstance(data, dict) else {}), text[m.end() :]


def clean(md: str) -> str:
    """Strip the scaffolding: components, images, footnotes, links, entities."""
    md = re.sub(r"<(style|script)\b.*?</\1>", "", md, flags=re.S | re.I)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)  # images
    # Bodies before references: a body starts with a reference, so stripping
    # references first would leave the definition text behind.
    md = re.sub(r"^\[\^[^\]]+\]:.*$", "", md, flags=re.M)  # footnote bodies
    md = re.sub(r"\[\^[^\]]+\]", "", md)  # footnote references
    md = "\n".join(ln for ln in md.split("\n") if not re.match(r"^\s*</?[a-zA-Z]", ln))
    md = re.sub(r"<[^>]+>", "", md)  # stray inline tags
    md = re.sub(r"\[([^\]]+)\]\(<?[^)]*>?\)", r"\1", md)  # links become their text
    for entity, char in (
        ("&rarr;", "→"),
        ("&mdash;", "—"),
        ("&ndash;", "–"),
        ("&amp;", "&"),
        ("&nbsp;", " "),
        ("&hellip;", "…"),
    ):
        md = md.replace(entity, char)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def sections(body: str) -> list[tuple[str | None, str]]:
    """Split on headings into `(heading, text)`. Text before the first is `(None, ...)`."""
    out: list[tuple[str | None, str]] = []
    head: str | None = None
    buf: list[str] = []
    for line in body.split("\n"):
        m = _HEADING.match(line)
        if m:
            if "".join(buf).strip():
                out.append((head, "\n".join(buf).strip()))
            head, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((head, "\n".join(buf).strip()))
    return out


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def words(text: str) -> int:
    return len(text.split())


def chunk(text: str, max_words: int) -> list[str]:
    """Cut long text into pieces of at most `max_words`, never mid-paragraph."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs(text):
        n = words(para)
        if current and count + n > max_words:
            pieces.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += n
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def notes_from(text: str, max_words_per_note: int = 14) -> str:
    """A rough dictation of a passage: the opening clause of each paragraph.

    This is the one synthetic thing in the dataset. It gives the model the task
    it will actually be asked to do — sparse notes in, finished prose out —
    without anyone having to write notes for every essay by hand.
    """
    notes = []
    for para in paragraphs(text):
        if para.startswith((">", "-", "*", "|", "```", "#")):
            continue
        first = re.split(r"(?<=[.!?])\s", para, maxsplit=1)[0]
        parts = first.split()
        if len(parts) < 4:
            continue
        notes.append("- " + " ".join(parts[:max_words_per_note]).rstrip(".,;:"))
    return "\n".join(notes)


def dashed(left: str, right: str) -> str:
    """`left — right`, or whichever of the two is there."""
    left, right = (left or "").strip(), (right or "").strip()
    if left and right:
        return f"{left} — {right}"
    return left or right


def bullets(values: Any) -> str:
    if not isinstance(values, (list, tuple)):
        return ""
    out = []
    for v in values:
        if isinstance(v, dict):
            title = v.get("title") or v.get("name") or ""
            body = v.get("description") or v.get("summary") or ""
            out.append(dashed(f"**{title}**" if title else "", body))
        elif v:
            out.append(f"- {v}")
    return "\n".join(out)
