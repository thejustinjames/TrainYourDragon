# Training

What the settings do, what the loss curve is telling you, and which checkpoint
to keep.

## The run

```bash
dragon train                  # rebuilds the dataset first, then fine-tunes
dragon train --no-build       # use data/ as it stands
dragon train --resume         # carry on from the current adapter
dragon train -- --iters 400   # anything after -- goes to mlx_lm.lora
```

Output goes to the screen and to `train.log`. Checkpoints land in `adapters/`
every `save_every` iterations, and the live adapter is
`adapters/adapters.safetensors`.

Keep the Mac awake. A laptop left to itself will doze between wake-ups and
double the wall-clock time for no benefit:

```bash
bash scripts/train-awake.sh
```

## The settings that matter

```yaml
training:
  iters: 1200
  batch_size: 2
  learning_rate: 1.0e-5
  num_layers: 16
  max_seq_length: 2560
  lora_parameters:
    rank: 8
    scale: 16.0
    dropout: 0.05
```

**`iters`** is the one to think about. What matters is how many times the model
sees your corpus: roughly `iters × batch_size × max_seq_length` tokens against
the total in `data/train.jsonl`. One to two passes is right for a voice model.
Three or four and it starts reproducing favourite sentences verbatim, which
reads as self-plagiarism rather than as voice. Scale `iters` with the corpus,
not with your patience.

**`num_layers`** is how many layers from the top get an adapter. Sixteen of a
28-layer model is a good default: style lives near the top, and adapting the
lower layers mostly costs memory.

**`rank`** and **`scale`** set the adapter's capacity. Rank 8 with scale 16 is
plenty for register. Raising the rank gives it more room to memorise, which is
the opposite of what you want here.

**`learning_rate`** at 1e-5 is deliberately gentle. If the loss is still falling
steeply at the end of the run, add iterations before you add learning rate.

**`max_seq_length`** and **`batch_size`** set the memory ceiling. A 7B at 4-bit
with these values peaks around 13 GB. If you run out, halve the sequence length
before you touch anything else; most examples are shorter than the limit.

## Reading the loss curve

Validation loss every hundred iterations, from a real run:

| 1 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 1100 | 1200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.84 | 1.98 | 1.96 | 1.84 | 1.79 | 1.68 | 1.81 | 1.67 | 1.63 | 1.64 | 1.63 | 1.63 | 1.66 |

Three things to read off it.

**The fall in the first hundred iterations** is the model learning the shape of
the task — the system prompt, the brief format, that it should produce prose.
It is not yet learning your voice.

**The flattening from 800** is the useful part being done. Beyond it you are
paying time for very little.

**The bump at 600** is noise, not a signal. A single point going the wrong way
on 25 validation batches means nothing; two consecutive points do.

**The tick up at 1200** is the beginning of memorisation. The checkpoint at
1000 is the one to keep.

## Promoting a checkpoint

`adapters/` holds every checkpoint. To make an earlier one live:

```bash
cp adapters/0001000_adapters.safetensors adapters/adapters.safetensors
```

Then re-fuse (`dragon fuse --force`) and re-export if you have already done
either, because those copy the weights rather than referring to them.

## Measuring

```bash
dragon test
```

runs the held-out pieces through the adapter and reports loss and perplexity —
around 2.2 and 9.3 on the run above. The absolute figure means very little. The
same figure next month, on the same held-out pieces, means something.

The measurement that actually decides whether a run was worth keeping is the
side-by-side:

```bash
dragon write --notes notes.txt --title "..." > tuned.txt
dragon write --notes notes.txt --title "..." --base > base.txt
```

Read both. You are looking for your own paragraph length, your punctuation, the
way you close a section. You are also looking for the failure described in
[voice and knowledge](voice-and-knowledge.md) — a paragraph with your rhythm
and no content — because a model that has learnt your voice will pad in it.

## When a run goes wrong

**Loss falls and the output is still generic.** Not enough passes, or the
corpus is too small or too mixed. Check the word count from `dragon build`.

**Output is recognisably you but quotes you verbatim.** Too many passes.
Promote an earlier checkpoint, and lower `iters` next time.

**Output has your voice but the wrong register** — chatty when you are dry, or
the reverse. Look at what is in the corpus: a folder of talk transcripts or
marketing copy will pull the whole model towards it. Exclude it.

**Validation loss rises from the start.** The learning rate is too high, or the
data is malformed. Read a few pairs with `dragon build --sample 5`.

**It ignores the voice entirely at inference.** The system prompt. The adapter
learnt to respond to the exact prompt in `config.yaml`; a client sending a
different one, or none, gets much less. See [serving](serving.md).
