# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-09-18

First public release: the tooling behind a private voice model, generalised so
it reads someone else's corpus.

### Added

- `dragon` command line: `init`, `doctor`, `build`, `train`, `test`, `write`,
  `chat`, `fuse`, `serve`, `export`, `publish`, `card`.
- One configuration file describing the base model, the voice, the corpus and
  the training run, with a commented example that `dragon init` writes.
- Five kinds of corpus source — `markdown`, `pages`, `terms`, `text` and
  `documents` — each with a `fields` map, so existing frontmatter does not have
  to be renamed.
- Training pairs for summaries, openings, sections, continuations, dictated
  notes, glossary terms and recall by name.
- Recall pairs, which ask what something is by name alone, and can be capped
  with `dataset.recall_pairs`.
- Held-out pieces, excluded from training and from recall, for `dragon test`.
- Fusing, with the system prompt written into the fused model's chat template
  as its default.
- An OpenAI-compatible local endpoint, an LM Studio link and a generated Ollama
  Modelfile.
- Publishing to the Hugging Face Hub: private by default, `--public` behind a
  typed confirmation, with a model card generated from the config and the
  adapter's own configuration.
- `scripts/train-awake.sh`, which holds the Mac awake for the length of a run.
- Documentation: how to use it, preparing a corpus, voice and knowledge,
  training, serving, Hugging Face, responsible use, troubleshooting.
- A small example corpus, which the test suite runs against.

[Unreleased]: https://github.com/thejustinjames/TrainYourDragon/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/thejustinjames/TrainYourDragon/releases/tag/v0.1.0
