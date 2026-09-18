# How to use it

Start to finish, on one machine, assuming nothing. Allow an evening of setting
up and a night of training.

## 1. Install

```bash
git clone https://github.com/thejustinjames/TrainYourDragon.git
cd TrainYourDragon
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mlx,hub]'
dragon --version
```

If you use `uv`, `uv venv --python 3.12 .venv && uv pip install -e '.[mlx,hub]'`
does the same in less time.

`mlx-lm` only installs on Apple silicon. Everything up to and including
`dragon build` works anywhere; training does not.

## 2. Choose a base model

The default is `mlx-community/Qwen2.5-7B-Instruct-4bit`: 7B parameters,
quantised to four bits, about 4 GB on disk and 13 GB of memory while training.
It is a good trade at the time of writing.

| Memory | Sensible choice |
|---|---|
| 16 GB | a 3B instruct model at 4-bit |
| 24 GB | 7B at 4-bit, shorter sequences |
| 36 GB and up | 7B at 4-bit comfortably, or 14B if you have patience |

Two things to check before you commit. It must be an *instruct* model, because
every training pair is a chat exchange. And its licence must allow what you
intend to do with the result — that constraint follows the weights, not the
adapter, and differs between families.

## 3. Point it at your writing

```bash
mkdir ~/my-voice-model && cd ~/my-voice-model
dragon init
```

Open `config.yaml` and work through it. The four things that matter:

**`corpus.root`** — the folder your writing lives in. Everything else is
relative to it.

**`corpus.sources`** — one entry per kind of material. The `fields` map tells
the builder which frontmatter keys you actually use, so you do not have to
rename anything. [docs/corpus.md](corpus.md) goes through each kind.

**`corpus.holdout`** — at least one piece, named by its filename without the
extension, held back from training entirely. It is the only honest measurement
you will get. Three is better than one.

**`voice.system_prompt`** — the instruction every training pair carries, and the
one you must send at inference. Describe the register, not the subject:

```yaml
system_prompt: >-
  You are <name>, writing for <where>. British English. First person. Short
  paragraphs, often a single sentence. Dry, understated, rigorous, with a wry
  aside where it earns its place. Never corporate, never breathless.
```

This matters more than it looks. The adapter learns to respond *to this
prompt*; change it later and some of the effect goes with it.

Then:

```bash
dragon doctor
```

which checks the machine, reads every source, and tells you what it found. Fix
anything it flags before going further.

## 4. Build the dataset

```bash
dragon build
```

This writes `data/train.jsonl`, `data/valid.jsonl` and `data/test.jsonl`, and
prints what it made:

```
train 1,842 examples / 328,972 words of target text · valid 116 · held out 14
by kind: continue 187, notes 96, opening 51, page 24, recall 463, section 802, ...
at iters 1200 × batch 2 × seq 2560: roughly 1.6 passes over the corpus — about right for a voice model
```

Look at the numbers. The last line is the one to act on: one to two passes is
where a voice model should sit, and `training.iters` is the knob. A few hundred thousand words of target text is a healthy
corpus. If `recall` is more than about a quarter of the total, set
`dataset.recall_pairs` to a number instead of `true` — see
[voice and knowledge](voice-and-knowledge.md).

Read a few pairs before you train on them:

```bash
dragon build --dry-run --sample 5
```

The dataset is yours and it stays local. It is in `.gitignore` for a reason.

## 5. Train

```bash
bash scripts/train-awake.sh        # or: dragon train
```

Several hours. A 7B at 4-bit on around 1,800 pairs with the default 1,200
iterations takes roughly three hours on an M4 with the machine kept awake, and
closer to six if it is allowed to doze between wake-ups — which is what
`train-awake.sh` is for. Everything is written to `train.log` as well as the
screen.

In another terminal:

```bash
dragon studio
```

opens a page that draws the loss curve as it happens, marks each checkpoint,
estimates the time left and explains what the numbers mean. Keep it open; it
is easier to read than the log. [docs/studio.md](studio.md).

Validation loss is printed every hundred iterations. What you want to see is a
steep fall, then a flattening. What you are watching for is the point where
validation loss starts rising while training loss keeps falling: that is the
model beginning to memorise, and the checkpoint before it is the one to keep.
[docs/training.md](training.md) has the detail, including how to promote an
earlier checkpoint.

## 6. Find out whether it worked

```bash
dragon test
```

gives the loss on the pieces you held back. Useful as a number to compare
against next week's number, and almost meaningless on its own.

First, the curve:

```bash
dragon curve          # validation loss by iteration, read from train.log
dragon promote        # list the saved checkpoints
dragon promote 1000   # make one of them the live adapter
```

Then the test that tells you something, which is the side-by-side. Take some
real notes and run them both ways in one go:

```bash
dragon compare --notes notes.txt --title "The Cost of a Number"
```

The base model will restate your notes in a paragraph of competent, anonymous
prose. If the fine-tune reads like you — your paragraph length, your habits,
your closing move — it worked. Look also for the padding described in
[voice and knowledge](voice-and-knowledge.md): a paragraph with your cadence
and nothing in it is the characteristic tell, and knowing it in advance is
most of the battle.

## 7. Use it

```bash
# From dictated notes, one section at a time
dragon write --notes notes.txt --title "..." --section "..."

# From a bare brief
dragon write "Essay: The Cost of a Number\nSection: Why boards love a number\n\nWrite this section."

# Interactive
dragon chat

# Endpoint for other tools
dragon serve

# Into LM Studio and Ollama
dragon export
```

[docs/serving.md](serving.md) covers wiring it into anything that speaks the
OpenAI API.

## 8. Keep it somewhere

```bash
dragon publish            # the adapter, private, about 50 MB
dragon publish --fused    # the whole model, several GB
dragon gguf && dragon publish --gguf    # a GGUF, so Ollama anywhere can pull it
```

Private by default, and `--public` makes you confirm. See
[docs/hugging-face.md](hugging-face.md) and, before you publish anything,
[docs/responsible-use.md](responsible-use.md).

## 9. Retrain when there is more to learn

`dragon train` rebuilds the dataset from your corpus first, so retraining after
a few new pieces is one command. Keep `iters` roughly proportional to the size
of the corpus: more passes over the same material is how a voice model turns
into a quotation machine.

Each publish is a new commit on the Hub, so earlier versions stay reachable.
