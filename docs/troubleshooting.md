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

**It recalls things fine with `dragon write` and forgets them all through the
endpoint, LM Studio or Ollama.** The fused model was quantised to 4 bits,
which rounds the adapter away. Check `export.fuse_bits` is 8, `dragon fuse
--force`, and re-export; for Ollama, `q8_0`, not `q4_K_M`. [Serving](serving.md)
has the mechanism.

## The studio

**Port 8790 is in use.** At a terminal the studio says by what and asks
whether to end it, move to the next port, or quit; type a number to pick one.
Elsewhere it moves on its own. `--if-busy` and `--port` set it explicitly.

**The page loads but says "studio not reachable".** The server it was opened
from has gone. `bash scripts/studio.sh` again.

**No previous runs in the tabs.** Only `train*.log` files in the project are
found. Keep an old log as `train-run1.log` (and its adapters as
`adapters-run1/`) before starting the next run.

## Publishing

**`not signed in to Hugging Face`** —
`python -c "from huggingface_hub import login; login()"` with a write token.

**`no adapter at adapters/`** — train one, or point at another directory.

**`[METAL] Command buffer execution failed: Caused GPU Timeout Error` during
`fuse`, `export` or `gguf`.** The weights being read sit on a slow external
volume. `dragon fuse` stages locally; if you ran `mlx_lm.convert` by hand,
point it at a local copy. If it happens during training with the corpus on
the internal disk, look for another process holding the GPU or most of the
memory; a container runtime's VM is the usual one.

**`Discarded (victim of GPU error/recovery)` during training.** Something else
faulted the GPU and the trainer was collateral, almost always memory pressure.
Quit whatever else is large (a Docker VM, a loaded model in LM Studio) and
restart the run; nothing is lost beyond the time since the last checkpoint.

**The disk is full.** The fused models are 8 GB and 15 GB each. Move them to
an external drive and leave symlinks at `fused/` and `fused-fp16/`; [serving](serving.md)
has the sequence. Do not move `adapters/`.

**The upload is enormous.** `--fused` uploads several gigabytes. The adapter
alone is around 50 MB and is what you want to version.
