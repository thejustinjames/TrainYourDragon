# Serving and exporting

Four ways to use the model once it exists: the `dragon write` command, a local
HTTP endpoint, LM Studio, and Ollama.

## Fusing

Most of what follows needs the adapter baked into the base weights:

```bash
dragon fuse                 # fused/        4-bit MLX, about 4 GB
dragon fuse --dequantize    # fused-fp16/   about 14 GB, only for the Ollama import
dragon fuse --force         # rebuild after promoting a different checkpoint
```

Fusing also writes the system prompt into the model's chat template as its
default, which matters more than it sounds like it should. See below.

## The endpoint

```bash
dragon serve                      # http://127.0.0.1:8787/v1
dragon serve --port 9000
```

An OpenAI-compatible server on the loopback interface. Anything that can be
pointed at a base URL and a model id can use it:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "default_model",
       "messages": [{"role": "user", "content": "Write the opening of an essay about measurement."}]}'
```

It binds to `127.0.0.1` on purpose. It has no authentication, so do not move it
to `0.0.0.0` on a shared network and assume nobody will find it.

**The trap.** `/v1/models` also lists the untouched base model, because it is
sitting in your Hugging Face cache. A client that offers a dropdown will happily
show it, and picking it gets you the base model with none of the voice and no
error to tell you so. Use `default_model`, or the explicit path to `fused/`.

## The system prompt

Every training pair carried the prompt from `config.yaml`, so the adapter's
effect is conditioned on it. A request that sends no system message gets
whatever the chat template puts there by default — for Qwen, "You are Qwen,
created by Alibaba Cloud" — and the voice largely disappears.

`dragon fuse` rewrites that default to your prompt, which fixes the API and
anything calling it. Two things it does not fix:

- **LM Studio's chat window** sends its own empty system prompt, overriding the
  template. Paste your prompt into the model's settings there, once.
- **Base-plus-adapter loading** (`dragon chat`, `dragon write`) sends the prompt
  explicitly, so it is fine, but anything else you write against `mlx_lm.load`
  needs to send it too.

If your base model ships a different stock persona, add it to
`voice.replaces_default_prompts` in `config.yaml` and re-fuse.

## LM Studio and Ollama

```bash
dragon export                    # both
dragon export --no-ollama
dragon export --tag myvoice:7b
```

**LM Studio** reads MLX models straight from its models folder, so the export
links `fused/` into it under the name from `export.lm_studio_name`. It then
appears in the model list like anything downloaded from the Hub. Remember the
system prompt note above.

**Ollama** runs GGUF rather than MLX, so it cannot use `fused/` directly. The
export fuses a dequantised copy, generates a Modelfile from your config — chat
template, system prompt, sampling parameters, stop tokens — and imports it,
quantising on the way in:

```bash
ollama run myvoice-7b
```

The generated Modelfile is at `.dragon/Modelfile`. Change `config.yaml` and
re-export rather than editing it; it is overwritten each time.

The default template is ChatML, which suits Qwen and most instruct models of
that lineage. For a base model that wants something else, put the template and
stop tokens in `config.yaml`:

```yaml
export:
  ollama:
    template: |
      ...
    stop: ["</s>"]
    parameters:
      temperature: 0.7
      num_ctx: 8192
```

## Wiring it into your own tools

The pattern that works is to make the local model an option rather than a
default: a named provider in whatever settings your tool already has, with the
base URL and model id filled in. It is slower and less capable than a hosted
model at everything except the one thing you trained it for, and you want to be
able to switch in one click when it is the wrong tool.

Send your system prompt explicitly rather than relying on the template default.
It costs nothing and it survives re-fusing, re-exporting, and someone else
loading the model on another machine.
