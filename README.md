# TrainYourDragon

Train a small language model on your own writing, on your own machine, in a
night. What comes out sounds like you and knows nothing. That turns out to be
the interesting result.

This is the tooling behind a private voice model: a LoRA fine-tune of a
small open model on one person's published work, trained overnight on a
MacBook with Apple's MLX. Nothing leaves the machine unless you send it there
on purpose.

```bash
pip install -e '.[mlx,hub]'
dragon init            # a starting config.yaml
dragon doctor          # check the machine, the config and the corpus
dragon build           # your writing becomes a training set
dragon train           # a few hours
dragon studio          # watch it in the browser while it runs
dragon test            # loss on the pieces it never saw
dragon write --notes notes.txt --title "The Cost of a Number"
```

## What it does, and what it does not

It learns register: sentence length, rhythm, punctuation habits, the shape of
an argument, the words you reach for and the ones you never use. On a corpus of
a few hundred thousand words that transfers well, and it transfers quickly.

It does not learn facts. It will describe a product you built for years and get
it wrong, fluently, in your voice. Voice and knowledge are separate problems and
fine-tuning only solves one of them. The distinction is the whole point of
[docs/voice-and-knowledge.md](docs/voice-and-knowledge.md), and if you read one
document here, read that one.

So the honest description of the output is: a first draft in the right register,
from rough notes, for you to read, correct and sign.

## What you need

- A Mac with Apple silicon. A 7B model at 4-bit trains in about 13 GB, so 16 GB
  works and 36 GB is comfortable. Building a dataset works anywhere; training
  does not.
- Python 3.10 or later.
- Your own writing, as Markdown files, JSON term lists, or one long text file.
  A few hundred thousand words is plenty. A hundred thousand is enough to see
  the effect.
- Six hours, most of them while you are asleep.

## Install

```bash
git clone https://github.com/thejustinjames/TrainYourDragon.git
cd TrainYourDragon
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mlx,hub]'
```

`mlx` pulls in mlx-lm, which is Apple silicon only. `hub` pulls in
`huggingface_hub`, needed only to publish. Neither is required to build a
dataset, which is why the tests run on any platform.

## The shape of it

One file, `config.yaml`, describes the whole project: which base model, which
voice, where the writing is, what to hold back, and how hard to train. Run
`dragon init` to get a commented starting point, and see
[docs/corpus.md](docs/corpus.md) for how to point it at your own material.

Five kinds of source cover most people's writing:

| Kind | What it reads | Example |
|---|---|---|
| `markdown` | a folder of articles with frontmatter | essays, posts, newsletters |
| `pages` | a folder of pages whose frontmatter is most of the content | products, services, case studies |
| `terms` | JSON lists of `{term, definition}` | glossaries |
| `text` | one long text file, split on its headings | a site's `llms-full.txt` |
| `documents` | named files the model should be able to describe | internal notes |

Every training example is a brief and a piece of finished prose, because that
is how the model will be used: notes in, draft out. The builder makes several
kinds of pair from each piece — the opening, each section given its heading and
the one before it, a long section continued, dictated notes turned into the
finished thing — and, importantly, pairs that ask what something *is* by name
alone. Without those last ones the model never has to remember anything.

There is a tiny worked example in `examples/`. It is far too small to train on
and exists so you can see the data:

```bash
dragon -c examples/config.yaml build --dry-run --sample 2
```

## Try it

```bash
# Read your corpus, write data/{train,valid,test}.jsonl
dragon build

# Fine-tune. Several hours; keep the Mac awake with scripts/train-awake.sh
dragon train

# Loss on the pieces held back from training
dragon test

# A draft from dictated notes
dragon write --notes notes.txt --title "The Cost of a Number" \
             --section "Why boards love a number"

# The same notes through the base model and the adapter, side by side
dragon compare --notes notes.txt --title "The Cost of a Number"

# The validation curve from train.log, and which checkpoint to keep
dragon curve
dragon promote 1000

# Interactive
dragon chat
```

The comparison matters more than any loss number. If you cannot tell the two
halves of `dragon compare` apart, the run did nothing; if the fine-tune is
recognisably you, it worked, whatever the figures say.
[docs/training.md](docs/training.md) covers reading the loss curve and
choosing which checkpoint to keep.

## Watch it train

```bash
bash scripts/studio.sh        # or: dragon studio
```

A page on localhost that draws the loss curve live, marks each checkpoint,
estimates the time left and says in plain words what the numbers mean — where
the low point is, whether memorisation has started, how many passes over the
corpus so far. Read-only, no dependencies, and it works on a run started by
hand. If the port is taken it tells you by what and offers to end it or move.
[docs/studio.md](docs/studio.md).

## Use it from other tools

```bash
dragon serve      # http://127.0.0.1:8787/v1, OpenAI-compatible, model id default_model
dragon export     # fused model into LM Studio, and imported into Ollama
dragon gguf       # a GGUF, so Ollama on any machine can pull it from the Hub
```

Anything that speaks the OpenAI API can point at the endpoint. See
[docs/serving.md](docs/serving.md), including two traps: a client picking the
untouched base model off `/v1/models`, and the bigger one, that fusing into
a 4-bit base rounds a light adapter away. Exports here are 8-bit for that
reason, and the document says why.

## Put it on Hugging Face

```bash
dragon publish            # the adapter, private, with a generated model card
dragon publish --fused    # the whole fused model too, 8-bit, about 8 GB
dragon publish --gguf     # the GGUF export: ollama run hf.co/<you>/<name>-gguf
```

Repositories are created private, and `--public` asks you to confirm in words.
[docs/hugging-face.md](docs/hugging-face.md) covers tokens, access, pulling the
model onto another machine, and versioning a retrain.

## Responsible use

A model trained on a named person's writing reproduces that person's voice well
enough to be mistaken for them. Trained on your own work and kept to yourself,
it is a writing tool. Published openly, or trained on someone else's writing
without asking, it is an impersonation kit, and it cannot be recalled once
downloaded.

[docs/responsible-use.md](docs/responsible-use.md) is short and worth the two
minutes. The defaults here follow it: private repositories, a confirmation
before anything goes public, and a model card that says plainly what the thing
is not.

## Documentation

- [How to use it](docs/how-to-use.md) — the walkthrough, start to finish
- [Preparing a corpus](docs/corpus.md) — sources, fields, what to leave out
- [Voice and knowledge](docs/voice-and-knowledge.md) — what fine-tuning does and does not teach
- [Training](docs/training.md) — hyperparameters, the loss curve, which checkpoint
- [Observability studio](docs/studio.md) — watching a run in the browser
- [Serving and exporting](docs/serving.md) — endpoint, LM Studio, Ollama
- [Hugging Face](docs/hugging-face.md) — publishing, access, another machine
- [Responsible use](docs/responsible-use.md)
- [Troubleshooting](docs/troubleshooting.md)

## Licence

MIT, in [LICENSE](LICENSE). The base models you train on have their own
licences, and those govern what you may do with the result: check before you
publish anything. Qwen2.5 and Llama, to take two common choices, differ.
