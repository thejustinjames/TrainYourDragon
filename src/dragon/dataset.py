"""Turning documents into training pairs.

Every example is a brief and a piece of finished prose. That is deliberate: it
is how the model will be used, and it is the only shape that teaches writing
rather than reciting. The kinds of pair are:

  summary   the one-line summary of a piece, from its title
  opening   the text before the first heading
  section   a section, from its heading and the one before it
  continue  the rest of a long section, given how it ends so far
  notes     sparse dictated notes turned into the finished section
  page      a page's introduction, assembled from its frontmatter
  term      a glossary definition in the author's tone
  recall    "what is X?" answered from what the corpus says X is

The last one matters more than it looks. Without it, every brief hands the
model the thing's name *and* its description, so it never has to remember what
anything is. See docs/voice-and-knowledge.md.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dragon.config import Config
from dragon.corpus import Document
from dragon.errors import DragonError
from dragon.textutil import chunk, dashed, notes_from, paragraphs, sections, words

RECALL_ASKS = [
    "What is {name}?",
    "Tell me about {name}.",
    "What does {name} do?",
    "Describe {name}.",
    "What is {name}, in a paragraph?",
    "Who is {name} for?",
    "Why did you build {name}?",
    "What problem does {name} solve?",
    "Give me the one-line pitch for {name}.",
    "{name} — what is it?",
    "Summarise {name}.",
]

TERM_PROMPT = (
    "Define '{term}' for the glossary of '{title}'. One or two sentences, in your own tone."
)

ARTICLE_RECALL_ASKS = [
    "What is '{name}' about?",
    "Summarise your piece '{name}'.",
    "What did you argue in '{name}'?",
]


@dataclass
class Example:
    messages: list[dict[str, str]]
    tag: str

    def json(self) -> str:
        return json.dumps({"messages": self.messages}, ensure_ascii=False)


@dataclass
class Dataset:
    train: list[Example] = field(default_factory=list)
    valid: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)

    @property
    def target_words(self) -> int:
        return sum(words(e.messages[-1]["content"]) for e in self.train)

    @property
    def brief_words(self) -> int:
        return sum(
            words(e.messages[0]["content"]) + words(e.messages[1]["content"]) for e in self.train
        )

    @property
    def by_tag(self) -> dict[str, int]:
        return dict(sorted(Counter(e.tag for e in self.train).items()))

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", self.train), ("valid", self.valid), ("test", self.test)):
            with open(directory / f"{name}.jsonl", "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(row.json() + "\n")


def build(config: Config, documents: list[Document]) -> Dataset:
    """All the pairs, shuffled, split into train/valid/test."""
    rng = random.Random(config.seed)
    maker = _Maker(config)

    kept: list[Example] = []
    held: list[Example] = []
    for doc in documents:
        examples = maker.for_document(doc)
        (held if doc.holdout else kept).extend(examples)

    if not kept:
        raise DragonError(
            "the corpus produced no training pairs. Check corpus.sources, and that "
            "your files carry the frontmatter fields named in `fields`."
        )

    rng.shuffle(kept)
    n_valid = min(max(20, int(len(kept) * config.valid_fraction)), len(kept) // 4)
    return Dataset(train=kept[n_valid:], valid=kept[:n_valid], test=held)


class _Maker:
    def __init__(self, config: Config):
        self.config = config
        self.system = config.system_prompt
        self.max_words = config.max_words
        self.recall = config.recall

    # ------------------------------------------------------------ one doc
    def for_document(self, doc: Document) -> list[Example]:
        if doc.terms:
            return self._terms(doc)
        out: list[Example] = []
        kind = doc.source.kind if doc.source else "markdown"
        if kind == "pages":
            out += self._page_intro(doc)
        elif doc.summary:
            out.append(
                self._ex(
                    f"Write the summary line for {_indefinite(doc.label)} titled '{doc.title}'.",
                    doc.summary,
                    "summary",
                )
            )
        out += self._body(doc)
        if self.recall and not doc.holdout:
            out += self._recall(doc, kind)
        return out

    # -------------------------------------------------------------- parts
    def _body(self, doc: Document) -> list[Example]:
        out: list[Example] = []
        previous: str | None = None
        min_words = doc.source.min_words if doc.source else 30
        for heading, text in sections(doc.body):
            if words(text) < min_words:
                previous = heading or previous
                continue
            brief = self._brief(doc, heading, previous)
            pieces = chunk(text, self.max_words)
            for i, piece in enumerate(pieces):
                if i == 0:
                    out.append(self._ex(brief, piece, "opening" if heading is None else "section"))
                else:
                    tail = paragraphs(pieces[i - 1])[-1]
                    out.append(
                        self._ex(
                            f"{self._head(doc, heading, previous)}\n\n"
                            f"Continue this section. It so far ends:\n\n{tail}",
                            piece,
                            "continue",
                        )
                    )
            if heading and len(paragraphs(text)) >= 3 and words(text) <= self.max_words:
                notes = notes_from(text)
                if notes.count("\n") >= 2:
                    out.append(
                        self._ex(
                            f"Turn these dictated notes into the section '{heading}' of "
                            f"{_indefinite(doc.label.lower())} '{doc.title}'.\n\n{notes}",
                            text,
                            "notes",
                        )
                    )
            previous = heading or previous
        return out

    def _page_intro(self, doc: Document) -> list[Example]:
        out = []
        intro = "\n\n".join(p for p in (doc.description, doc.extras) if p)
        head = self._head(doc, None, None, section=False)
        if intro:
            out.append(
                self._ex(
                    f"{head}\n\nWrite the introduction: the description and the headline facts.",
                    intro,
                    "page",
                )
            )
        if doc.summary:
            out.append(
                self._ex(f"{head}\n\nWrite the one-paragraph summary.", doc.summary, "page-summary")
            )
        return out

    def _terms(self, doc: Document) -> list[Example]:
        template = (doc.source.prompt if doc.source else None) or TERM_PROMPT
        return [
            self._ex(template.format(term=term, title=doc.title), definition, "term")
            for term, definition in doc.terms
        ]

    def _recall(self, doc: Document, kind: str) -> list[Example]:
        """Ask by name only, and answer from what the corpus says."""
        out: list[Example] = []
        if kind in ("pages", "documents"):
            answer = self._what_it_is(doc)
            if not answer:
                return []
            for name in doc.names:
                for ask in self._asks(RECALL_ASKS):
                    out.append(self._ex(ask.format(name=name), answer, "recall"))
        elif doc.summary:
            for ask in self._asks(ARTICLE_RECALL_ASKS):
                out.append(self._ex(ask.format(name=doc.title), doc.summary, "recall"))
        return out

    def _asks(self, asks: list[str]) -> list[str]:
        """`recall_pairs` may be true, or a number of phrasings per document."""
        if self.recall is True:
            return asks
        return asks[: int(self.recall)]

    def _what_it_is(self, doc: Document) -> str:
        body = doc.description or doc.summary
        if not body:
            intro = next((t for h, t in sections(doc.body) if h is None), "")
            body = "\n\n".join(paragraphs(intro)[:3])
        if not body:
            return ""
        head = dashed(doc.title, doc.tagline)
        parts = [f"{head}\n\n{body}" if head != body else body]
        if doc.extras:
            parts.append(doc.extras)
        if doc.meta_line:
            parts.append(doc.meta_line)
        return "\n\n".join(parts)

    # ------------------------------------------------------------ framing
    def _head(
        self, doc: Document, heading: str | None, previous: str | None, *, section: bool = True
    ) -> str:
        title = dashed(doc.title, doc.tagline)
        lines = [f"{doc.label}: {title}"]
        if doc.meta_line:
            lines.append(doc.meta_line)
        if doc.summary:
            lines.append(f"Summary: {doc.summary}")
        if section:
            lines.append(f"Section: {heading or 'the opening'}")
        if previous:
            lines.append(f"Follows the section: {previous}")
        return "\n".join(lines)

    def _brief(self, doc: Document, heading: str | None, previous: str | None) -> str:
        instruction = (
            "Write the opening, before the first heading."
            if heading is None
            else "Write this section."
        )
        return f"{self._head(doc, heading, previous)}\n\n{instruction}"

    def _ex(self, user: str, assistant: str, tag: str) -> Example:
        return Example(
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": assistant.strip()},
            ],
            tag=tag,
        )


def _indefinite(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun
