# Picking the better draft

```bash
dragon pick                       # against `dragon serve` on this machine
```

opens a page at <http://127.0.0.1:8791/> that asks the model for two drafts
from the same prompt, with different seeds, and puts them side by side. You say
which is better. Each verdict is a line in `preferences.jsonl`, and when there
are enough of them they are the data for the next kind of training.

## Why bother

Fine-tuning on your writing teaches the model what your prose looks like. It
does not teach it what you would rather it did not do. The faults a trained
voice model actually has — the sixth paragraph with nothing in it, the figure
it made up, the closing line that has your cadence and no meaning — are all
things a good example cannot show, because a good example simply does not
contain them.

A preference can. Two drafts of the same brief, one preferred: repeated a few
hundred times, that is what preference tuning (DPO, ORPO) learns from, and the
judge is the one person whose taste the model is supposed to have.

## The four verdicts

| Verdict | What is saved | Training pairs |
|---|---|---|
| A is better / B is better | both drafts, the choice | one: the winner over the loser |
| …and you edit the winner | both drafts, the choice, your version | two: your version over each draft |
| Both poor, and you write it as it should be | both drafts, your version | two |
| Both poor, nothing written | both drafts | none |
| About the same | both drafts | none |

A tie and an unrescued pair of bad drafts are saved, so you can see how often
it happens, but they make no training pairs: forcing a choice between two
equal drafts teaches the model noise.

The edit is the strongest signal there is. Correcting the chosen draft takes a
minute and turns one weak preference into two strong ones. If you only have
time for some, edit the ones where the fault is specific: the invented number,
the paragraph that should not be there.

Keys: `⌘↵` asks for two drafts and saves an edit; `1`–`4` give the verdict.

## Pointing it at a model

Any OpenAI-compatible endpoint, or two of them. With a `config.yaml`, the voice prompt goes
with every request and `pick:` supplies the rest:

```yaml
pick:
  endpoint: https://192.168.1.156:8443/v1   # Ollama behind a gateway
  model: jjvoice:7b-run4
  api_key_file: ~/.config/ollama-client/api-key
  cacert: ~/.config/ollama-client/ollama.crt
  temperature: 0.7
```

Or on the command line, which also works without a config:

```bash
dragon pick --endpoint http://127.0.0.1:8099/v1 --model jjvoice \
            --system @system-prompt.txt --out ~/voice/preferences.jsonl
```

The key can also come from `DRAGON_PICK_API_KEY`. It is sent as a bearer token
and never printed or written to the verdicts.

Sample at the temperature you would actually use. Two drafts at 0.2 are nearly
the same draft, and the verdict says little; at 0.7 they differ enough to
choose between, and the faults you are trying to train out are the ones that
appear at that temperature.

## Two models, blind

```yaml
pick:
  endpoint: http://127.0.0.1:8099/v1     # the adapter, as a GGUF
  model: jjvoice
  against:
    endpoint: http://127.0.0.1:11434/v1  # a general model
    model: gpt-oss:20b
```

or `--against MODEL` for another model at the same endpoint, and
`--against-endpoint URL` for one elsewhere. Draft A and draft B then come from
the two models. Which one is A is decided by a coin each round, and you are told
after you have chosen. Each verdict records which model wrote which draft, so
a tally of wins is a line of `jq` away.

## With your own writing in front of it

```bash
dragon index
dragon pick --retrieve
```

Each prompt goes with the passages of your writing it matches, the same
passages for both drafts, and the page lists where they came from. The verdict
saves the passages' ids and the full brief the model saw, and the training
pairs use that brief as their prompt: a model tuned on them learns to write
well from passages, which is the job it will be doing.
[Answers from your own writing](retrieval.md) covers the index.

## Turning verdicts into training data

```bash
dragon pick --pairs dpo.jsonl
```

writes one line per pair, in the shape preference trainers read:

```json
{"system": "You are…", "prompt": "Turn these notes…", "chosen": "…", "rejected": "…"}
```

How many is enough: a few hundred pairs to see an effect, and more of them
edits than not. Each verdict records the model and endpoint it came from, so
pairs from an older model can be kept or dropped when you retrain.

## What it does not do

It does not train. Preference tuning needs its own run, on its own settings,
and is not yet a `dragon` command. `preferences.jsonl` matches `*.jsonl` in
`.gitignore`, because it holds your prompts and your writing.
