# Answers from your own writing

```bash
dragon index                         # cut the corpus into passages and embed them
dragon ask "What is Lantern?"        # an answer, written from the passages it found
dragon ask --sources "What is Lantern?"   # just the passages, to see what it would use
```

A fine-tuned adapter writes like you and remembers very little of what you
wrote. Asked about a product it was trained on, it will usually name it
correctly and invent the rest. [Voice and knowledge](voice-and-knowledge.md)
explains why; this is what to do about it. The facts are not trained in. They
are looked up at the moment of asking, from the same corpus the adapter was
trained on, and handed to the model with the question.

## What `dragon index` does

Every document the config points at is cut into passages of about 220 words,
at paragraph boundaries and under its own headings. A page's tagline, status
and summary are one passage, and its highlights and features another. A
glossary's terms go eight to a passage, under the title of the essay they
belong to.

Each passage is embedded by an OpenAI-compatible `/embeddings` endpoint. The
default is Ollama on this machine with `bge-m3`:

```bash
ollama pull bge-m3
```

The index is written to `.dragon/index/`. Running it again only embeds the
passages whose text has changed, so a new essay takes seconds. `--rebuild`
embeds everything, and changing the embedding model does the same on its own.
A corpus of 430,000 words is about 1,700 passages and takes under a minute on
an M3 Max.

## How a question finds its passages

Three rankings, fused:

- the embedding: what the question is about, in other words than its own;
- keywords, scored by BM25: the exact terms, which embeddings are bad at;
- names: a document whose title or alias appears in the question.

Embeddings alone miss proper nouns. "What is AOS?" has no meaning an embedding
can place; the page it refers to is called ATLAS, and only an alias connects
them. So names count, and aliases are worth adding for anything known by an
acronym or an older name:

```yaml
corpus:
  aliases:
    atlas: ["AOS", "the agentic operating system"]
```

A page asked for by name always brings its overview and its features. No
document takes more than `per_document` places (two by default, three for one
asked for by name), so a long essay cannot crowd out everything else.

## What the model is told

The passages go in front of the question, headed by where each came from, and
the voice prompt gains an instruction: take every fact from the notes; if they
do not say it, do not say it; where they do not cover something, leave it out
or say so. It also names the habits that make generated prose sound generated,
the false contrast, the padded list of three, the tidy summing-up line, and
asks for none of them. `retrieval.instructions` replaces that text if you want
your own.

## Which model writes

Any OpenAI-compatible endpoint, under `retrieval.writer`:

```yaml
retrieval:
  writer:
    endpoint: http://127.0.0.1:11434/v1
    model: gpt-oss:20b
    max_tokens: 3000
```

The choice is between two different faults.

The adapter, served by `dragon serve` or as a GGUF, has the voice and little
judgement. Given passages, it tends to stitch them together nearly verbatim:
the facts are right because they are your sentences, and the answer reads like
a page of extracts.

A larger general model writes from the passages properly, and brings its own
accent with it, the "in short" and the joke you did not make. It also
sometimes generalises a fact slightly past what the passage says.

[`dragon pick`](pick.md) can put the two side by side, blind, with the same
passages, which is the quickest way to find out which fault you mind less.

A reasoning model (gpt-oss, Qwen3) thinks inside the same token budget as it
answers. Give it 2,500 tokens or more; if it runs out while thinking,
`dragon ask` says so rather than printing nothing.

## What it does not do

It does not search the web or anything outside the corpus. It does not cite in
the answer, because the instruction asks it not to mention the notes, but
`dragon ask` prints the passages it used, and `dragon pick` records them with
each verdict. Nothing leaves the machine except the passages and the question,
to the endpoints you name.
