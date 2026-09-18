# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Dictated-notes pairs are now a gist of each paragraph's content words rather
  than its opening clause. The old recipe made the answer begin with the note's
  own words in nearly every pair, and the trained model duly began every
  paragraph with the note verbatim.
- Fused models are 8-bit, made by an exact fp16 fuse re-quantised with
  `mlx_lm.convert` (`export.fuse_bits`, default 8). Fusing straight into the
  4-bit base rounded a light adapter away: the voice survived and recall did
  not. The Ollama import is `q8_0` and the GGUF default is `Q8_0` for the same
  reason, and the fused Hub repository is named `<name>-mlx-8bit`. The model
  card and `docs/serving.md` explain it.
- `dragon serve` documents why it serves the fused directory: `mlx_lm.server`
  resolves `default_model` before looking an adapter up, so base-plus-adapter
  never applies the adapter.

### Added

- Documentation for keeping fused models on an external drive behind symlinks,
  and for verifying a copy before removing the original.
- `dragon studio`: the observability studio. A read-only page on localhost that
  draws the loss curve live, marks checkpoints, estimates time left, explains
  what the numbers mean, and works on any `mlx_lm.lora` log, config or not.
  Every `train*.log` in the project is a tab, and the other runs' validation
  curves can be overlaid on the current one.
- The studio carries the Agencie.io Labs mark, header and footer, and its
  typefaces. `scripts/studio.sh` starts it; a taken port is reported by
  process and the studio offers to end it, move to the next free port, or take
  one you type (`--if-busy ask|next|kill|fail`).
- `dragon gguf`: GGUF export through llama.cpp's converter and `llama-quantize`,
  with the `template`, `system` and `params` files Ollama reads from a Hub
  repository. `dragon publish --gguf` uploads it, so `ollama run hf.co/<repo>`
  works on any machine.
- `model.license` on the model card; a `gguf` card flavour; `dragon export`
  imports an existing GGUF into Ollama rather than re-quantising.

- `dragon curve`: the validation loss by iteration from `train.log`, with a
  reading of where the low point is and whether the curve has flattened.
- `dragon promote`: list saved checkpoints, or make one the live adapter.
- `dragon compare`: the same brief through the base model and the adapter,
  one after the other.
- `dragon build` estimates how many passes the configured run makes over the
  corpus, and says whether that is in the range a voice model wants.
- `dragon doctor` flags an `exclude` or `holdout` slug that matches no file.
- `model.license` in `config.yaml`, carried onto the model card. It was
  hard-coded to Apache 2.0, which is wrong for several base model families.
- `dragon serve` warns when bound to anything other than the loopback address.

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
