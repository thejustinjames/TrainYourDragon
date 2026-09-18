"""Reading a corpus off disk.

Five kinds of source cover most people's writing: a folder of articles, a
folder of pages whose frontmatter *is* the content, JSON term lists, one long
text file, and a handful of named documents. Each becomes a `Document`, and
the dataset builder only ever sees `Document`s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dragon.config import Config, Source
from dragon.errors import DragonError
from dragon.textutil import clean, frontmatter


@dataclass
class Document:
    key: str  # the slug: how exclude/holdout lists name it
    title: str
    body: str = ""  # cleaned, heading-structured prose
    summary: str = ""
    description: str = ""
    tagline: str = ""
    meta_line: str = ""  # "Category: X · Status: Y", from configured fields
    extras: str = ""  # bulleted frontmatter lists, for pages
    aliases: list[str] = field(default_factory=list)
    terms: list[tuple[str, str]] = field(default_factory=list)
    source: Source | None = None
    path: Path | None = None
    holdout: bool = False

    @property
    def label(self) -> str:
        return self.source.label if self.source else "Document"

    @property
    def names(self) -> list[str]:
        """Every name this document answers to, longest first, de-duplicated."""
        seen, out = set(), []
        for n in [self.title, *self.aliases]:
            n = (n or "").strip()
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out


def load(config: Config) -> list[Document]:
    """Every document the config points at, in a stable order."""
    docs: list[Document] = []
    for source in config.sources:
        loader = _LOADERS[source.kind]
        docs.extend(loader(config, source))
    for doc in docs:
        doc.holdout = doc.key in config.holdout
    return docs


# --------------------------------------------------------------- loaders
def _markdown_files(config: Config, source: Source) -> list[Path]:
    directory = config.resolve(source.path or "")
    if not directory.is_dir():
        raise DragonError(
            f"source {source.name!r}: {directory} is not a directory. "
            "Check corpus.root and the source path."
        )
    return sorted(directory.glob("*.md")) + sorted(directory.glob("*.markdown"))


def _from_markdown(config: Config, source: Source) -> list[Document]:
    docs = []
    for path in _markdown_files(config, source):
        if path.stem in config.exclude:
            continue
        meta, body = frontmatter(path.read_text(encoding="utf-8"))
        if not _language_ok(source, meta):
            continue
        docs.append(_document(source, path, meta, clean(body)))
    return docs


def _from_pages(config: Config, source: Source) -> list[Document]:
    # Pages differ from articles only in where the weight sits: most of a page
    # is frontmatter (a tagline, a description, a list of features), and the
    # body is a short remainder.
    return _from_markdown(config, source)


def _from_terms(config: Config, source: Source) -> list[Document]:
    directory = config.resolve(source.path or "")
    if not directory.is_dir():
        raise DragonError(f"source {source.name!r}: {directory} is not a directory")
    docs = []
    for path in sorted(directory.glob("*.json")):
        if path.stem in config.exclude:
            continue
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DragonError(f"{path} is not readable JSON: {exc}") from exc
        terms = [
            (str(i["term"]).strip(), str(i["definition"]).strip())
            for i in items
            if isinstance(i, dict) and i.get("term") and i.get("definition")
        ]
        if terms:
            docs.append(
                Document(key=path.stem, title=path.stem, terms=terms, source=source, path=path)
            )
    return docs


def _from_text(config: Config, source: Source) -> list[Document]:
    path = config.resolve(source.path or "")
    if not path.exists():
        raise DragonError(
            f"source {source.name!r}: {path} does not exist. "
            "If a build step produces it, run that first."
        )
    text = path.read_text(encoding="utf-8")
    stop = source.fields.get("stop_at")
    if stop and stop in text:
        text = text.split(stop)[0]
    return [Document(key=path.stem, title=source.label, body=clean(text), source=source, path=path)]


def _from_documents(config: Config, source: Source) -> list[Document]:
    docs = []
    for entry in source.documents:
        path = config.resolve(entry.get("path", ""))
        if not path.exists():
            print(f"  ({source.name}: missing, skipped — {path})")
            continue
        meta, body = frontmatter(path.read_text(encoding="utf-8"))
        body = clean(body)
        doc = _document(source, path, meta, body)
        doc.title = entry.get("title") or doc.title
        doc.aliases = list(entry.get("aliases") or [])
        docs.append(doc)
    return docs


_LOADERS = {
    "markdown": _from_markdown,
    "pages": _from_pages,
    "terms": _from_terms,
    "text": _from_text,
    "documents": _from_documents,
}


# --------------------------------------------------------------- helpers
def _language_ok(source: Source, meta: dict) -> bool:
    if not source.language:
        return True
    key = source.field_name("language", "lang")
    return str(meta.get(key, source.language)) == source.language


def _document(source: Source, path: Path, meta: dict, body: str) -> Document:
    from dragon.textutil import bullets

    def get(key: str, default: str = "") -> str:
        name = source.field_name(key, key)
        value = meta.get(name, default) if name else default
        return str(value).strip() if value else ""

    meta_bits = []
    for key in source.fields.get("meta", []) or []:
        value = meta.get(key)
        if value:
            meta_bits.append(f"{str(key).replace('_', ' ').capitalize()}: {value}")

    extras = []
    for key in source.fields.get("lists", []) or []:
        rendered = bullets(meta.get(key))
        if rendered:
            extras.append(rendered)

    title = get("title") or path.stem
    return Document(
        key=path.stem,
        title=title,
        body=body,
        summary=get("summary"),
        description=get("description"),
        tagline=get("tagline"),
        meta_line=" · ".join(meta_bits),
        extras="\n\n".join(extras),
        source=source,
        path=path,
    )
