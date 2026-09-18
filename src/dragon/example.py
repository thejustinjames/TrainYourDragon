"""The starting config `dragon init` writes.

It lives here rather than in a data file so that an installed copy of the
package can always produce it. `config.example.yaml` at the root of the
repository is written from this string, and a test keeps the two identical.
"""

EXAMPLE_CONFIG = """# TrainYourDragon — one file describes the whole project.
# Copy to config.yaml, edit, then run `dragon doctor`.

model:
  # Any model mlx-lm can load. A 7B at 4-bit trains comfortably in 36 GB;
  # a 3B will do on 16 GB. Check the base model's licence before you publish.
  base: mlx-community/Qwen2.5-7B-Instruct-4bit
  # Used to name the adapter, the Hugging Face repositories and the Ollama tag.
  name: myvoice-7b
  # Goes on the model card. The adapter inherits the base model's licence, so
  # look it up: Qwen2.5 is apache-2.0, Llama is llama3.x, Gemma is gemma.
  license: apache-2.0

voice:
  # The instruction every training pair carries, and the one you must send at
  # inference. Describe the register, not the subject matter. Be specific:
  # this prompt is half of what the adapter learns to respond to.
  system_prompt: >-
    You are <your name>, writing for your journal at <your site>.
    British English. First person. Short paragraphs, often a single sentence.
    Dry, understated, rigorous, with a wry aside where it earns its place.
    Never corporate, never breathless.

  # Stock personas to overwrite in a fused model's chat template, so a client
  # that sends no system message still gets the voice. The common ones are
  # known already; add yours if your base model ships something else.
  replaces_default_prompts: []

corpus:
  # Everything below is relative to this.
  root: ~/Documents/GitHub/my-site

  sources:
    # A folder of articles: frontmatter for the title and summary, prose below.
    - name: essays
      kind: markdown
      path: src/content/posts
      label: Essay                # how a brief names one: "Essay: <title>"
      language: en                # skip anything whose `lang` says otherwise
      min_words: 40               # sections shorter than this are not worth a pair
      fields:
        title: title
        summary: summary
        language: lang

    # A folder of pages whose frontmatter is most of the content: products,
    # services, case studies. The body is a short remainder.
    - name: products
      kind: pages
      path: src/content/products
      label: Product
      fields:
        title: name
        tagline: tagline
        summary: summary
        description: description
        lists: [highlights, features]     # rendered as bullets
        meta: [category, status]          # rendered as "Category: X · Status: Y"

    # JSON files of [{"term": ..., "definition": ...}].
    - name: glossary
      kind: terms
      path: pdf/glossaries
      # prompt: "Define '{term}' for the glossary of '{title}'. Two sentences."

    # One long text file, split on its own headings. A static site's
    # llms-full.txt, or anything else you can export as one document.
    - name: site
      kind: text
      path: dist/llms-full.txt
      label: Site
      fields:
        stop_at: "\\n## Products\\n"       # ignore everything from here down

    # Named documents the model should be able to describe when asked.
    - name: notes
      kind: documents
      label: Document
      documents:
        - path: ~/Documents/GitHub/my-model/docs/PROGRESS.md
          title: The voice model
          aliases: [myvoice, "the voice model", "your private model"]

  # Slugs (the filename without its extension) to leave out entirely:
  # translations, co-written pieces, anything not in your own voice.
  exclude:
    - a-piece-someone-else-wrote

  # Slugs held back from training and used only to measure. At least one, or
  # `dragon test` has nothing to say. Three is better: one is thin evidence.
  holdout:
    - one-piece-the-model-never-sees

dataset:
  valid_fraction: 0.06
  max_words_per_example: 1300
  seed: 7
  # Pairs that ask "what is X?" by name alone. Without them the model never has
  # to remember what anything is. `true` uses every phrasing; a number caps how
  # many per document, which matters on a small corpus where they would
  # otherwise outnumber the prose. See docs/voice-and-knowledge.md.
  recall_pairs: 6

training:
  # A light touch. Past a couple of passes over the corpus the model starts
  # quoting you back rather than writing like you.
  iters: 1200
  batch_size: 2
  learning_rate: 1.0e-5
  num_layers: 16
  max_seq_length: 2560
  grad_checkpoint: true
  steps_per_report: 20
  steps_per_eval: 100
  val_batches: 25
  save_every: 200
  lora_parameters:
    rank: 8
    scale: 16.0
    dropout: 0.05
    keys:
      - self_attn.q_proj
      - self_attn.k_proj
      - self_attn.v_proj
      - self_attn.o_proj
      - mlp.gate_proj
      - mlp.up_proj
      - mlp.down_proj

export:
  lm_studio_name: myname/myvoice-7b-mlx-4bit
  gguf:
    # A llama.cpp checkout, for `dragon gguf`. Also found via $LLAMA_CPP or
    # ~/llama.cpp. The quantiser is `llama-quantize` on PATH (brew install llama.cpp).
    llama_cpp: ~/llama.cpp
  ollama:
    parameters:
      temperature: 0.7
      top_p: 0.9
      num_ctx: 8192
"""
