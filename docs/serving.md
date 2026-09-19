# Serving and exporting

Four ways to use the model once it exists: the `dragon write` command, a local
HTTP endpoint, LM Studio, and Ollama.

## Fusing

Most of what follows needs the adapter baked into the base weights:

```bash
dragon fuse                 # fused-fp16/ (exact, ~15 GB) then fused/ (8-bit MLX, ~8 GB)
dragon fuse --dequantize    # stop at the fp16 fuse
dragon fuse --force         # rebuild after promoting a different checkpoint
```

Fusing also writes the system prompt into the model's chat template as its
default, which matters more than it sounds like it should. See below.

### Eight bits, not four

The obvious thing to do is fuse the adapter into the 4-bit base you trained
on and get a 4-bit model back. Do not. It was the default here until an
afternoon was lost to it.

A rank-8 adapter trained at a low learning rate moves each weight by less
than a 4-bit quantisation step. Re-quantise after the fuse and most of the
change rounds back to where it started. The voice survives, because it is
spread thinly across millions of weights and the rounding averages out. The
recall of a specific page does not, because it lives in a few weights and
each of those is rounded individually. The symptom is a model that sounds
exactly right and, asked about something it plainly learnt, describes a
generic thing instead. It is easy to mistake for a training failure.

So `dragon fuse` always goes by way of fp16, which is exact, and then
re-quantises with `mlx_lm.convert` at `export.fuse_bits`, 8 by default. At
eight bits the step is sixteen times smaller and the adapter's changes
survive. The cost is a 7B model of about 8 GB rather than 4. The same rule
applies to the Ollama import (`q8_0`, not `q4_K_M`) and the GGUF export
(`Q8_0`). If you set `fuse_bits: 4` the command warns and does it anyway.

Base-plus-adapter is not a way round this for the endpoint, because
`mlx_lm.server` resolves `default_model` to the real path before it looks the
adapter up and never applies it. `dragon serve` serves the fused directory for
that reason.

### Where to keep them

A 7B model fused this way is 8 GB at 8 bits and 15 GB at fp16, and a
retrain doubles it if you keep the old one. They do not belong on a laptop's
internal disk for long. Move the directories to an external drive and leave
symlinks at the old paths; everything here follows a symlink, including the
LM Studio link (a link to a link) and the Ollama Modelfile's relative `FROM`:

```bash
mv fused fused-fp16 "/Volumes/Models/myvoice/"
ln -s "/Volumes/Models/myvoice/fused" fused
ln -s "/Volumes/Models/myvoice/fused-fp16" fused-fp16
```

Copy, verify, then delete, rather than `mv` across volumes in one step:
`rsync -a` to the drive, `rsync -a -n -i` back to see that nothing differs,
then remove the originals. The adapters are a few hundred megabytes and are
the thing to version, so they stay in the project.

Unload the model from LM Studio and stop `dragon serve` first; a model that is
loaded has its files open.

One thing not to do: GPU work against the drive. Fusing *to* a USB volume is
only slow, but quantising *from* one died twice with a Metal GPU timeout, the
GPU stalling on page faults from a 30 MB/s device. `dragon fuse` therefore
writes to a staging directory under `.dragon/` and moves the finished model
through the symlink afterwards, and copies the fp16 locally before quantising
if it lives on another volume. `mlx_lm.convert` also refuses to write into a
directory that already exists, which the staging path sidesteps.

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

## GGUF, for Ollama anywhere

MLX weights run on Apple silicon only. A GGUF runs in llama.cpp and everything
built on it, and Ollama can pull one straight from the Hub on any machine:

```bash
dragon gguf                     # gguf/<name>-f16.gguf and gguf/<name>-Q8_0.gguf
dragon gguf --quant Q6_K        # anything llama-quantize knows; below Q8 you lose recall
dragon publish --gguf           # → ollama run hf.co/<you>/<name>-gguf:Q8_0
```

Two tools are needed and neither is bundled. The converter is
`convert_hf_to_gguf.py` from a [llama.cpp](https://github.com/ggml-org/llama.cpp)
checkout, found through `export.gguf.llama_cpp` in `config.yaml`, `$LLAMA_CPP`,
or `~/llama.cpp`; its Python dependencies come with `pip install -e '.[gguf]'`,
which is heavy because it includes PyTorch. The quantiser is `llama-quantize`,
which `brew install llama.cpp` provides. `dragon doctor` reports whether it can
see both. Without the quantiser you get the f16 file only, which works but is
fourteen gigabytes.

mlx-lm's own GGUF writer is not used because it only knows the Llama
architecture; a Qwen adapter would come out mislabelled.

Alongside the `.gguf` files the export writes `template`, `system` and
`params`, which are the three files Ollama reads from a Hub repository for the
chat template, the system prompt and the sampling settings. That is how a
machine that pulls the model gets the voice rather than the base persona,
without a Modelfile. A Modelfile is written too, for a local `ollama create`.

Once a GGUF exists, `dragon export` imports that into the local Ollama instead
of re-quantising from the fp16 safetensors, which is faster. Either way the
Ollama model is 8-bit; `export.ollama.quantise` changes it, and the section
above says why you should not.

## Wiring it into your own tools

The pattern that works is to make the local model an option rather than a
default: a named provider in whatever settings your tool already has, with the
base URL and model id filled in. It is slower and less capable than a hosted
model at everything except the one thing you trained it for, and you want to be
able to switch in one click when it is the wrong tool.

Send your system prompt explicitly rather than relying on the template default.
It costs nothing and it survives re-fusing, re-exporting, and someone else
loading the model on another machine.
