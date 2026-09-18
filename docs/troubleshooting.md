# Troubleshooting

## Setting up

**`mlx-lm is not installed`** — `pip install -e '.[mlx]'`, and check you are in
the virtual environment. On anything that is not a Mac with Apple silicon it
will not install at all; dataset building still works.

**`no config.yaml`** — `dragon init`, or pass `-c path/to/config.yaml`.

**A source finds nothing.** Almost always the path or a field name. `corpus`
paths are relative to `corpus.root`; `fields` names the frontmatter keys you
actually use, not the ones the example uses. `dragon doctor` prints what it
found for each source.

**`the corpus produced no training pairs`** — usually `min_words` set too high
for short pieces, or frontmatter that is not being read. Check with
`dragon build --dry-run --sample 3`.

## Building

**Recall pairs swamp the set.** Set `dataset.recall_pairs` to a number rather
than `true`. Six per document is reasonable; on a small corpus, fewer.

**A piece you excluded is still there.** `exclude` and `holdout` name files by
stem — the filename without its extension, not the title.

**Odd fragments in the output.** Something in the source markup is not being
stripped. Read the pairs; most of it comes from custom components, which the
cleaner removes only when they are on their own line.

## Training

**Out of memory.** Halve `max_seq_length` first, then `batch_size`. A 7B at
4-bit with the defaults peaks around 13 GB.

**It takes twice as long as it should.** The Mac is dozing between wake-ups.
`bash scripts/train-awake.sh`, which holds it awake for exactly as long as the
run.

**Validation loss rises from the start.** Learning rate too high, or malformed
data. Look at a few pairs.

**Validation loss flattens early.** Nothing wrong. Stop the run and promote the
checkpoint where it flattened; the rest is time for very little.

**Interrupted run.** Checkpoints are in `adapters/` every `save_every`
iterations. `dragon train --resume` continues from the current adapter.

## Using it

**The voice is not there.** Check the system prompt is being sent. Every
training pair carried the one in `config.yaml`, and the adapter is conditioned
on it. `dragon write` and `dragon chat` send it; a raw `mlx_lm.load` does not.

**The endpoint gives generic output.** The client has probably picked the base
model off `/v1/models`, which lists it from your Hugging Face cache alongside
the fused one. Use `default_model`.

**LM Studio ignores the template default.** It sends its own empty system
prompt. Paste yours into the model's settings, once.

**Ollama cannot pull the repository.** It runs GGUF, not MLX. Fuse with
`--dequantize` and import locally; `dragon export` does both.

**It writes beautifully and says nothing.** Working as designed, unfortunately.
It has learnt that your sections run six paragraphs and has material for four.
[Voice and knowledge](voice-and-knowledge.md) explains why, and what to do
instead.

**It is confidently wrong about your own products.** Same document. Recall
pairs help; retrieval is the honest fix.

## Publishing

**`not signed in to Hugging Face`** —
`python -c "from huggingface_hub import login; login()"` with a write token.

**`no adapter at adapters/`** — train one, or point at another directory.

**The upload is enormous.** `--fused` uploads several gigabytes. The adapter
alone is around 50 MB and is what you want to version.
