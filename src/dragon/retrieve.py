"""Retrieval: the facts come from your writing at the moment of asking.

Fine-tuning teaches a model how you write. It is a poor way to teach it what
you have written: a light adapter recalls a product page by name and invents
the rest, fluently and in your voice. So the corpus is also an index. Every
document is cut into passages of a couple of hundred words, each passage is
embedded by any OpenAI-compatible `/embeddings` endpoint (Ollama's bge-m3 is
the default), and a question is answered with the passages that match it in
front of the model, with an instruction to take its facts from them and from
nowhere else.

Three rankings are fused: the embedding's, a BM25 keyword score, and an exact
match on the names a document answers to. Embeddings are good at what a
question is about and bad at proper nouns; "SILO.RED" and "AOS" are found by
the other two.

The index lives in `.dragon/index/`. Re-indexing only embeds passages whose
text has changed, so a new essay costs a few seconds.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from array import array
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dragon.corpus import Document
from dragon.errors import DragonError
from dragon.pick import Endpoint, post_json
from dragon.textutil import paragraphs, sections, words

DEFAULT_EMBED_ENDPOINT = "http://127.0.0.1:11434/v1"  # Ollama on this machine
DEFAULT_EMBED_MODEL = "bge-m3"
PASSAGE_WORDS = 220  # long enough to carry a fact with its context, short enough to rank
TERMS_PER_PASSAGE = 8
BATCH = 16

# Appended to the voice prompt whenever passages go with a request.
INSTRUCTIONS = """\
Each request comes with notes from your own published writing. Take every fact \
from them: names, numbers, dates, prices, what a product does, what an essay \
argues. If the notes do not say it, do not say it; where they do not cover \
something, leave it out or say plainly that you have not written about it.

Write as yourself, in plain prose. Do not mention the notes, number them or \
quote them at length. Avoid the accent of machine prose: no "not X, but Y" \
contrasts, no lists of three where the third is padding, no stacked adverbs, \
no headings or bullets unless asked, and no tidy line at the end that sums up \
what was just said."""

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'’\-][a-z0-9]+)*")
_STOP = set(
    "a an and are as at be but by for from has have how i if in into is it its of on or "
    "that the their them there these this to was what when where which who why will with "
    "you your about does do did can my me we our tell".split()
)


@dataclass
class Passage:
    id: str  # "<slug>#<n>", or "<slug>/glossary#<n>"
    key: str
    title: str
    label: str
    heading: str | None
    text: str
    names: list[str] = field(default_factory=list)

    @property
    def where(self) -> str:
        return f"{self.label}: {self.title}" + (f" · {self.heading}" if self.heading else "")

    @property
    def embedded(self) -> str:
        """What is embedded: the passage under its document's name, so it ranks in context."""
        return f"{self.where}\n\n{self.text}"

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.embedded.encode("utf-8")).hexdigest()


@dataclass
class Hit:
    passage: Passage
    score: float
    why: list[str]  # which rankings found it: meaning, words, name


# ---------------------------------------------------------------- passages
def passages(docs: list[Document]) -> list[Passage]:
    """Every document cut into passages, in a stable order."""
    out: list[Passage] = []
    # A glossary file is named after its essay; its passages carry the essay's title.
    titles = {d.key: d.title for d in docs if not d.terms}
    for doc in docs:
        texts: list[tuple[str | None, str]] = []
        head = _overview(doc)
        if head:
            texts.append((None, head))
        if doc.extras:
            lines = [ln for ln in doc.extras.split("\n") if ln.strip()]
            texts.extend(("Highlights and features", t) for t in _pack(lines, joiner="\n"))
        for heading, text in sections(doc.body):
            texts.extend((heading, t) for t in _pack(paragraphs(text)))
        for i in range(0, len(doc.terms), TERMS_PER_PASSAGE):
            group = doc.terms[i : i + TERMS_PER_PASSAGE]
            texts.append((None, "\n".join(f"{t} — {d}" for t, d in group)))
        glossary = bool(doc.terms)
        # A glossary shares its essay's slug, and so its names, but not its ids.
        prefix = f"{doc.key}/glossary" if glossary else doc.key
        for n, (heading, text) in enumerate(texts):
            out.append(
                Passage(
                    id=f"{prefix}#{n}",
                    key=doc.key,
                    title=titles.get(doc.key, doc.title) if glossary else doc.title,
                    label="Glossary" if glossary else doc.label,
                    heading=heading,
                    text=text,
                    names=doc.names,
                )
            )
    return out


def _overview(doc: Document) -> str:
    """What a page says about itself before its body: tagline, status, summary."""
    parts = [doc.tagline, doc.meta_line, doc.summary, doc.description]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _pack(paras: list[str], joiner: str = "\n\n") -> list[str]:
    """Paragraphs gathered into passages of about PASSAGE_WORDS, never split mid-paragraph
    unless one paragraph is itself far too long, and then at sentence ends."""
    out, buf, n = [], [], 0
    for p in paras:
        for piece in _sentences(p) if words(p) > PASSAGE_WORDS * 1.5 else [p]:
            w = words(piece)
            if buf and n + w > PASSAGE_WORDS:
                out.append(joiner.join(buf))
                buf, n = [], 0
            buf.append(piece)
            n += w
    if buf:
        out.append(joiner.join(buf))
    return out


def _sentences(text: str) -> list[str]:
    out, buf = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if buf and words(buf) + words(s) > PASSAGE_WORDS:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    return out + ([buf] if buf else [])


# --------------------------------------------------------------- embedding
def embedder(endpoint: Endpoint) -> Callable[[list[str]], list[list[float]]]:
    """A function from texts to vectors, through an OpenAI-compatible /embeddings."""

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            reply = post_json(endpoint, "/embeddings", {"model": endpoint.model, "input": batch})
            try:
                rows = sorted(reply["data"], key=lambda d: d.get("index", 0))
                out.extend(r["embedding"] for r in rows)
            except (KeyError, TypeError) as exc:
                raise DragonError(f"{endpoint.url} did not send embeddings") from exc
            if len(out) != i + len(batch):
                raise DragonError(f"{endpoint.url} sent {len(out) - i} embeddings for {len(batch)}")
        return out

    return embed


def _unit(v: list[float]) -> array:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return array("f", (x / norm for x in v))


def _dot(a: array, b: array) -> float:
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------- index
class Index:
    """Passages, their vectors, and a keyword index built on load."""

    def __init__(self, items: list[Passage], vectors: list[array], meta: dict):
        self.passages = items
        self.vectors = vectors
        self.meta = meta
        self._terms = [Counter(_tokens(p.embedded)) for p in items]
        self._length = [sum(c.values()) for c in self._terms]
        self._avg = (sum(self._length) / len(items)) if items else 1.0
        df: Counter = Counter()
        for c in self._terms:
            df.update(c.keys())
        n = len(items)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._names = _name_patterns(items)

    # ------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        items: list[Passage],
        embed: Callable[[list[str]], list[list[float]]],
        *,
        model: str,
        previous: Index | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[Index, int]:
        """Embed what is new; reuse vectors whose passage text is unchanged.
        Returns the index and how many passages were embedded."""
        known: dict[str, array] = {}
        if previous and previous.meta.get("model") == model:
            known = {p.digest: v for p, v in zip(previous.passages, previous.vectors)}
        todo = [p for p in items if p.digest not in known]
        for i in range(0, len(todo), BATCH * 4):
            batch = todo[i : i + BATCH * 4]
            for p, v in zip(batch, embed([p.embedded for p in batch])):
                known[p.digest] = _unit(v)
            if progress:
                progress(min(i + len(batch), len(todo)), len(todo))
        vectors = [known[p.digest] for p in items]
        meta = {
            "model": model,
            "dimensions": len(vectors[0]) if vectors else 0,
            "passages": len(items),
            "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return cls(items, vectors, meta), len(todo)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "passages.jsonl").open("w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        with (directory / "vectors.f32").open("wb") as f:
            for v in self.vectors:
                v.tofile(f)
        (directory / "index.json").write_text(json.dumps(self.meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> Index:
        meta_path = directory / "index.json"
        if not meta_path.exists():
            raise DragonError(f"no index at {directory}. Run `dragon index` first.")
        meta = json.loads(meta_path.read_text())
        items = [
            Passage(**json.loads(line))
            for line in (directory / "passages.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dims = int(meta["dimensions"])
        flat = array("f")
        with (directory / "vectors.f32").open("rb") as f:
            flat.fromfile(f, dims * len(items))
        vectors = [flat[i * dims : (i + 1) * dims] for i in range(len(items))]
        return cls(items, vectors, meta)

    # ------------------------------------------------------------ search
    def search(
        self, query: str, vector: list[float], *, k: int = 6, per_document: int = 2
    ) -> list[Hit]:
        """The passages for a question, by reciprocal rank fusion of three rankings."""
        if not self.passages:
            return []
        q = _unit(vector)
        sims = [_dot(q, v) for v in self.vectors]
        meaning = sorted(range(len(sims)), key=sims.__getitem__, reverse=True)[:50]
        keyword = self._bm25(query)[:50]
        named_docs = {key for key, pattern in self._names if pattern.search(query)}
        # A document asked for by name: its overview first, its features next, then
        # the passages the question is closest to.
        named = sorted(
            (i for i, p in enumerate(self.passages) if p.key in named_docs),
            key=lambda i: (
                self.passages[i].id == f"{self.passages[i].key}#0",
                self.passages[i].heading == "Highlights and features",
                sims[i],
            ),
            reverse=True,
        )[:20]

        score: Counter = Counter()
        why: dict[int, list[str]] = {}
        for label, ranking in (("meaning", meaning), ("words", keyword), ("name", named)):
            for rank, i in enumerate(ranking):
                score[i] += 1 / (60 + rank)
                why.setdefault(i, []).append(label)

        # A named page's overview and features always go, whatever else ranks above them.
        pinned = [
            i
            for i in named
            if self.passages[i].id == f"{self.passages[i].key}#0"
            or self.passages[i].heading == "Highlights and features"
        ][:4]
        order = pinned + [i for i, _ in score.most_common() if i not in pinned]
        hits, per = [], Counter()
        for i in order:
            p, s = self.passages[i], score[i]
            if per[p.key] >= per_document + (p.key in named_docs):
                continue
            per[p.key] += 1
            hits.append(Hit(p, round(s, 5), why[i]))
            if len(hits) == k:
                break
        return hits

    def _bm25(self, query: str, k1: float = 1.4, b: float = 0.75) -> list[int]:
        terms = [t for t in set(_tokens(query)) if t in self._idf]
        if not terms:
            return []
        scores = []
        for i, tf in enumerate(self._terms):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += (
                        self._idf[t]
                        * f
                        * (k1 + 1)
                        / (f + k1 * (1 - b + b * self._length[i] / self._avg))
                    )
            if s:
                scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)]


def _tokens(text: str) -> list[str]:
    out = []
    for m in _TOKEN.findall(text.lower()):
        if m in _STOP:
            continue
        out.append(m)
        if re.search(r"[.'’\-]", m):  # silo.red is also silo and red
            out.extend(p for p in re.split(r"[.'’\-]", m) if p and p not in _STOP)
    return out


def _name_patterns(items: list[Passage]) -> list[tuple[str, re.Pattern]]:
    """One pattern per document name, matched whole and without regard to case.
    Names under three letters are skipped: they match too much."""
    seen, out = set(), []
    for p in items:
        for name in p.names:
            if len(name) < 3 or (p.key, name.lower()) in seen:
                continue
            seen.add((p.key, name.lower()))
            out.append((p.key, re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.I)))
    return out


# -------------------------------------------------------------- the brief
def brief(question: str, hits: list[Hit]) -> str:
    """The user message: the passages, then the question."""
    if not hits:
        return question
    notes = "\n\n".join(f"— {h.passage.where}\n{h.passage.text}" for h in hits)
    return f"Notes from your own writing:\n\n{notes}\n\n---\n\n{question}"


class Retriever:
    """An index and the endpoint that embeds questions for it."""

    def __init__(
        self,
        index: Index,
        embed: Callable[[list[str]], list[list[float]]],
        *,
        k: int = 6,
        per_document: int = 2,
        instructions: str = INSTRUCTIONS,
    ):
        self.index, self.embed = index, embed
        self.k, self.per_document, self.instructions = k, per_document, instructions

    def find(self, question: str) -> list[Hit]:
        [vector] = self.embed([question])
        return self.index.search(question, vector, k=self.k, per_document=self.per_document)

    def system(self, voice: str | None) -> str:
        return "\n\n".join(s for s in (voice, self.instructions) if s)

    def describe(self) -> str:
        return f"{self.index.meta.get('passages', 0)} passages, {self.k} a question"
