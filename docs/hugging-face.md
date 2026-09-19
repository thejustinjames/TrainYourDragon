# Hugging Face

Where to put the model, who can reach it, and how to get it onto another
machine.

## What to publish

Two artefacts, and they serve different purposes.

| | What it is | Size | Publish it when |
|---|---|---|---|
| The adapter | `adapters.safetensors` and its config | around 50 MB | Always. It is the thing to version |
| The fused model | the adapter baked into the base at 8 bits: weights, config, tokeniser, chat template | about 8 GB for a 7B | You want to load and go, or hand LM Studio a folder |
| The GGUF | the fused model converted for llama.cpp, quantised, with Ollama's `template`, `system` and `params` files | 4–5 GB | You want `ollama run` to work on a machine that is not a Mac, or without any of this installed |

The adapter is the interesting object: small enough to keep every version of,
and useless without the base model, which is a mild security property in its
own right. The fused model is 8-bit rather than 4 on purpose; a 4-bit fuse
keeps the voice and loses the recall, for reasons in [serving](serving.md).

```bash
dragon publish            # the adapter and a generated model card
dragon publish --fused    # the fused model too
dragon publish --gguf     # the GGUF export, after `dragon gguf`
```

## Signing in

A one-off, per machine. Create a token at
<https://huggingface.co/settings/tokens> — read is enough to download, write is
needed to publish — then:

```bash
pip install -U huggingface_hub
python -c "from huggingface_hub import login; login()"   # paste the token
```

It is cached at `~/.cache/huggingface/token` and every tool here picks it up.
Revoke it from the same page if the machine is lost. `dragon doctor` tells you
who you are signed in as.

## Private by default

Repositories are created private. They do not appear in search, and a request
without a valid token gets a 401.

`dragon publish --public` exists and asks you to type `yes` first. Before you
do, read [responsible use](responsible-use.md). A model trained to write in a
named person's voice cannot be recalled once it has been downloaded.

If you need to let one other person in: open the repository page, Settings,
Collaborators, add their Hugging Face username. They authenticate with their own
token. There is no reason to share yours.

For something between private and public, use Settings → Gated model, where
each request is approved by hand.

## Naming

By default the repositories are `<you>/<model.name>-lora`,
`<you>/<model.name>-mlx-8bit` and `<you>/<model.name>-gguf`, from the `name` in
`config.yaml`. Override per run:

```bash
dragon publish --repo myorg/house-voice-lora
dragon publish --fused --fused-repo myorg/house-voice-mlx-8bit
dragon publish --gguf --gguf-repo myorg/house-voice-gguf
dragon publish --owner myorg          # publish under an organisation
```

## The model card

Each repository carries a README that the Hub renders as its model card. It is
generated from `config.yaml` and the adapter's own `adapter_config.json`, so it
cannot drift from what was actually trained: base model, licence (from
`model.license` in `config.yaml`, which you should set to match the base
model's), the system
prompt the adapter expects, how to load it, the training configuration, and a
plain statement of what the thing is not.

```bash
dragon card                    # print it without publishing
dragon card --flavour fused
dragon card --flavour gguf
```

If you want to see what a publish would upload, this is how.

## Getting it onto another machine

Sign in as above, then:

```python
from mlx_lm import load, generate

# The fused model, straight from the Hub
model, tok = load("you/myvoice-7b-mlx-8bit")

# Or the base model with the adapter on top
from huggingface_hub import snapshot_download
adapter = snapshot_download("you/myvoice-7b-lora")
model, tok = load("mlx-community/Qwen2.5-7B-Instruct-4bit", adapter_path=adapter)
```

Just the files:

```bash
python -c "from huggingface_hub import snapshot_download as d; print(d('you/myvoice-7b-mlx-8bit'))"
```

prints the local folder, which is what you link into LM Studio:

```bash
ln -s "$(python -c "from huggingface_hub import snapshot_download as d; print(d('you/myvoice-7b-mlx-8bit'))")" \
  ~/.cache/lm-studio/models/you/myvoice-7b-mlx-8bit
```

Serve it directly:

```bash
mlx_lm.server --model you/myvoice-7b-mlx-8bit --port 8787
```

Remember the system prompt. It is in the fused model's chat template as the
default, but a client that sends its own empty one overrides that. The model
card records the prompt for exactly this reason.

**Ollama** cannot pull the adapter or the fused repository, because it runs
GGUF and those are MLX. It can pull the GGUF one, on any machine, with nothing
else installed:

```bash
ollama run hf.co/you/myvoice-7b-gguf:Q8_0
```

For a private repository, put a read token in Ollama's settings first. The
`template`, `system` and `params` files in the repository give it the voice.
See [serving](serving.md) for making the GGUF.

## Versioning a retrain

Each publish is a new commit, so earlier versions stay reachable by sha. Give
the ones worth keeping a name:

```bash
dragon publish --fused --tag v2
```

tags the new head of each repository it touches, and from then on

```python
snapshot_download("you/myvoice-7b-lora", revision="v2")
model, tok = load("you/myvoice-7b-mlx-8bit", revision="v2")
```

fetch that version whatever has been pushed since. Tags are cheap; a run that
went into use deserves one.

Publish the adapter after every run worth keeping — it is 50 MB — and the fused
model only when you actually need it somewhere. Note which checkpoint you
promoted in the commit; the model card records the configuration but not which
of the twelve checkpoints you chose.
