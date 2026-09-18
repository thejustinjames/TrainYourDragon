# Preparing a corpus

The dataset builder reads your writing where it already is. You should not have
to reorganise a site or rename frontmatter to train on it.

## How much, and of what

A few hundred thousand words of your own prose is comfortable. A hundred
thousand is enough to see the voice transfer. Below about fifty thousand you
are mostly teaching the model to repeat your favourite sentences.

What counts: anything you wrote, in the register you want back. Essays, posts,
long-form documentation, product pages, newsletters, talks you wrote out.

What does not: anything you edited rather than wrote, anything co-written,
anything in a different language, and anything in a register you do not want
the model to reach for. A corpus that is 20% conference abstracts will give you
a model that writes conference abstracts one time in five.

## The five kinds of source

Each entry under `corpus.sources` names a kind, a path relative to
`corpus.root`, and a `fields` map saying which frontmatter keys you use.

### `markdown` — a folder of articles

The common case. Frontmatter for the title and summary, prose below, `##`
headings dividing it.

```yaml
- name: essays
  kind: markdown
  path: src/content/posts
  label: Essay          # how a brief names one: "Essay: <title>"
  language: en          # skip anything whose `lang` field says otherwise
  min_words: 40         # sections shorter than this are not worth a pair
  fields:
    title: title
    summary: summary
    language: lang
```

Produces: the summary line from the title; the opening; each section from its
heading and the one before it; long sections continued; dictated notes turned
into the finished section; and, unless the piece is held out, a few pairs
asking what the piece was about.

### `pages` — where the frontmatter is the content

Product pages, services, case studies: a tagline, a description, a list of
highlights, and a short body.

```yaml
- name: products
  kind: pages
  path: src/content/products
  label: Product
  fields:
    title: name                   # your key is `name`, not `title`
    tagline: tagline
    summary: summary
    description: description
    lists: [highlights, features] # rendered as bullets
    meta: [category, status]      # rendered as "Category: X · Status: Y"
```

`lists` handles both plain lists of strings and lists of
`{title, description}` objects.

Produces: an introduction assembled from the frontmatter, a one-paragraph
summary, the body sections, and the recall pairs that ask what the thing is by
name alone.

### `terms` — glossaries

JSON files of `[{"term": ..., "definition": ...}]`. One pair per term, which
teaches short, tonally consistent definitions — a surprisingly good thing to
have in the mix.

```yaml
- name: glossary
  kind: terms
  path: pdf/glossaries
  prompt: "Define '{term}' for the glossary of '{title}'. Two sentences."
```

The filename stem is `{title}`, so a glossary named after the piece it belongs
to gets the context for free.

### `text` — one long file

A static site's `llms-full.txt`, an exported document, anything you can produce
as one file with headings.

```yaml
- name: site
  kind: text
  path: dist/llms-full.txt
  label: Site
  fields:
    stop_at: "\n## Products\n"    # ignore everything from here down
```

`stop_at` is there because these exports usually end with a dump of content you
are already reading from source, and you do not want it twice.

### `documents` — named files the model should know about

An explicit list, each with the names a question might use. This is how you let
the model describe things that are not on the site: internal notes, a README,
the write-up of the model itself.

```yaml
- name: notes
  kind: documents
  documents:
    - path: ~/Documents/GitHub/my-model/docs/PROGRESS.md
      title: The voice model
      aliases: [myvoice, "the voice model", "your private model"]
```

## Leaving things out

```yaml
corpus:
  exclude:
    - the-lights-of-manila-tl     # a translation, not your voice
    - the-adversary-replies       # four voices, three of them not yours
  holdout:
    - the-feed-got-crowded        # never trained on, used to measure
```

Both lists name files by their stem — the filename without the extension. A
piece in `holdout` is read, turned into pairs, and put in `test.jsonl` and
nowhere else. It contributes no recall pairs either, or the model would learn
what it is about by another route.

Hold something back. One piece is thin evidence and three is better, but one is
enormously better than none, and adding it afterwards means retraining.

## Checking your work

```bash
dragon doctor                      # every source, with a count
dragon build --dry-run --sample 5  # count the pairs, read a few
```

`doctor` fails if a source finds nothing, which is almost always a path or a
field name. Read the sample output before a long run: if the briefs look wrong
to you, they will teach the wrong thing.

## What the builder does to your prose

Before anything becomes a training pair it is stripped of markup: components,
images, footnotes, HTML, and links reduced to their text. Entities become
characters. Blank lines are normalised.

Long sections are cut into pieces at paragraph boundaries, never mid-sentence,
at `dataset.max_words_per_example`. Each piece after the first becomes a
"continue this, it so far ends: ..." pair.

One thing is synthetic. The `notes` pairs are made by reducing each paragraph
of a section to a gist of its content words, lower case, in order, with the
paragraph's first few words skipped, and calling the result dictated notes. It
is a crude imitation of how you would actually brief the model, and it works
because the task it teaches — sparse input, finished prose out — is the one
you will use it for.

The gist is deliberately not the paragraph's opening clause. The first version
of this used exactly that, and the trained model began every paragraph with the
note verbatim, lower case and all, because in 323 of 345 training pairs that
was the right answer. Synthetic pairs teach whatever pattern is in them; check
a few before a long run.
