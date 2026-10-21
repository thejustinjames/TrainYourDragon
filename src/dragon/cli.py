"""`dragon` — the command line.

dragon init       start a project here from the example config
dragon doctor     check the machine, the config and the corpus
dragon build      read the corpus, write data/{train,valid,test}.jsonl
dragon train      fine-tune the adapter
dragon test       loss on the held-out pieces the model never saw
dragon write      a draft from a brief or from dictated notes
dragon chat       an interactive session with the adapter loaded
dragon fuse       bake the adapter into a standalone model
dragon serve      an OpenAI-compatible endpoint on localhost
dragon studio     watch a training run in the browser
dragon export     fused model into LM Studio and Ollama
dragon gguf       a GGUF export, so Ollama can pull it on any machine
dragon publish    adapter (and optionally the model) to Hugging Face
dragon curve      the validation loss curve, read from train.log
dragon promote    make a saved checkpoint the live adapter
dragon compare    the same brief through the base model and the adapter
dragon pick       two drafts side by side; your verdict is preference data
dragon index      cut the corpus into passages and embed them, for retrieval
dragon ask        an answer from your own writing, with the passages it used
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dragon import __version__
from dragon.config import CONFIG_NAME, Config
from dragon.errors import DragonError
from dragon.example import EXAMPLE_CONFIG


# ------------------------------------------------------------------ helpers
def _config(args: argparse.Namespace) -> Config:
    return Config.load(args.config)


def _corpus(config: Config):
    from dragon import corpus

    return corpus.load(config)


def _endpoint(
    settings: dict,
    *,
    url=None,
    model=None,
    key_file=None,
    cacert=None,
    default_url=None,
    default_model=None,
    env_key=None,
):
    """An Endpoint from flags, then a config section, then defaults. The key comes
    from a file or an environment variable, never from the command line."""
    import os

    from dragon.pick import Endpoint

    key_file = key_file or settings.get("api_key_file")
    api_key = os.environ.get(env_key) if env_key else None
    if key_file:
        path = Path(os.path.expanduser(key_file))
        if not path.exists():
            raise DragonError(f"no API key file at {path}")
        api_key = path.read_text().strip()
    cacert = cacert or settings.get("cacert")
    return Endpoint(
        url=url or settings.get("endpoint") or default_url,
        model=model or settings.get("model") or default_model,
        api_key=api_key,
        cacert=os.path.expanduser(cacert) if cacert else None,
    )


def _index_dir(config: Config) -> Path:
    where = (config.raw.get("retrieval") or {}).get("index")
    return (config.root / Path(where).expanduser()) if where else config.work_dir / "index"


def _embed_endpoint(config: Config):
    from dragon.retrieve import DEFAULT_EMBED_ENDPOINT, DEFAULT_EMBED_MODEL

    settings = (config.raw.get("retrieval") or {}).get("embed") or {}
    return _endpoint(
        settings, default_url=DEFAULT_EMBED_ENDPOINT, default_model=DEFAULT_EMBED_MODEL
    )


def _retriever(config: Config, k: int | None = None):
    from dragon.retrieve import INSTRUCTIONS, Index, Retriever, embedder

    settings = config.raw.get("retrieval") or {}
    index = Index.load(_index_dir(config))
    endpoint = _embed_endpoint(config)
    if index.meta.get("model") != endpoint.model:
        raise DragonError(
            f"the index was embedded with {index.meta.get('model')}, not {endpoint.model}; "
            "run `dragon index` again"
        )
    return Retriever(
        index,
        embedder(endpoint),
        k=k or int(settings.get("k", 6)),
        per_document=int(settings.get("per_document", 2)),
        instructions=(settings.get("instructions") or INSTRUCTIONS).strip(),
    )


# ----------------------------------------------------------------- commands
def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.config or CONFIG_NAME)
    if target.exists() and not args.force:
        raise DragonError(f"{target} already exists; --force to overwrite")
    target.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(f"wrote {target}")
    print("Edit corpus.root, the sources, and voice.system_prompt, then run `dragon doctor`.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    print(f"dragon {__version__} · python {sys.version.split()[0]} · {sys.platform}")

    try:
        config = _config(args)
        print(f"config     {config.path}")
    except DragonError as exc:
        print(f"config     ✗ {exc}")
        return 1

    if sys.platform == "darwin":
        try:
            import mlx_lm  # noqa: F401

            print("mlx-lm     installed")
        except ImportError:
            ok = False
            print("mlx-lm     ✗ not installed — pip install -e '.[mlx]'")
    else:
        print("mlx-lm     not applicable on this platform (training needs Apple silicon)")

    try:
        from huggingface_hub import whoami

        try:
            print(f"hugging face  signed in as {whoami()['name']}")
        except Exception:
            print("hugging face  not signed in (only needed to publish)")
    except ImportError:
        print("hugging face  huggingface_hub not installed (only needed to publish)")

    from dragon.gguf import llama_cpp_dir, quantiser

    d = llama_cpp_dir(config)
    print(f"llama.cpp  {d if d else 'not found (only needed for `dragon gguf`)'}")
    q = quantiser(config)
    print(f"quantiser  {q if q else 'llama-quantize not found (brew install llama.cpp)'}")

    print(f"corpus     {config.corpus_root}")
    if not config.corpus_root.exists():
        ok = False
        print("           ✗ that directory does not exist")

    try:
        documents = _corpus(config)
    except DragonError as exc:
        print(f"           ✗ {exc}")
        return 1

    for source in config.sources:
        found = [d for d in documents if d.source is source]
        note = f"{len(found)} document(s)"
        if source.kind == "terms":
            note += f", {sum(len(d.terms) for d in found)} terms"
        print(f"  {source.name:<14} {source.kind:<10} {note}")
        if not found:
            ok = False
            print("                 ✗ nothing found here")

    # An excluded slug must exist on disk; a held-out one must have been loaded.
    # Either way, a name that matches nothing is a typo waiting to waste a run.
    on_disk, loaded = _all_keys(config), {d.key for d in documents}
    for name, entries, known in (
        ("exclude", config.exclude, on_disk),
        ("holdout", config.holdout, loaded),
    ):
        for slug in sorted(entries):
            if slug not in known:
                ok = False
                print(
                    f"{name:<10} ✗ {slug!r} matches no file (a slug is the filename, no extension)"
                )

    holdout = [d.key for d in documents if d.holdout]
    if holdout:
        print("held out   " + ", ".join(holdout))
    else:
        ok = False
        print("held out   ✗ nothing, so `dragon test` will have nothing to measure")

    print("ok" if ok else "\nSomething above needs attention.")
    return 0 if ok else 1


def _all_keys(config: Config) -> set[str]:
    """Every slug on disk, including ones `exclude` removed before loading."""
    keys = set()
    for source in config.sources:
        if source.kind in ("markdown", "pages", "terms") and source.path:
            directory = config.resolve(source.path)
            if directory.is_dir():
                keys.update(p.stem for p in directory.iterdir() if p.is_file())
    return keys


def cmd_build(args: argparse.Namespace) -> int:
    from dragon import dataset

    config = _config(args)
    documents = _corpus(config)
    data = dataset.build(config, documents)
    if args.dry_run:
        print("(dry run, nothing written)")
    else:
        data.write(config.data_dir)
        stats = {
            "train": len(data.train),
            "valid": len(data.valid),
            "test": len(data.test),
            "target_words": data.target_words,
            "brief_words": data.brief_words,
            "by_tag": data.by_tag,
        }
        (config.data_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(
        f"train {len(data.train):,} examples / {data.target_words:,} words of target text"
        f" · valid {len(data.valid):,} · held out {len(data.test):,}"
    )
    print("by kind: " + ", ".join(f"{k} {v:,}" for k, v in data.by_tag.items()))
    from dragon.runlog import passes

    n = passes(config, data.target_words, data.brief_words)
    t = config.training
    note = (
        "about right for a voice model"
        if 1.0 <= n <= 2.5
        else (
            "on the low side; consider more iterations"
            if n < 1.0
            else "high: expect it to start quoting you. Fewer iterations"
        )
    )
    print(
        f"at iters {t['iters']} × batch {t['batch_size']} × seq {t['max_seq_length']}: "
        f"roughly {n:.1f} passes over the corpus — {note}"
    )
    if args.sample:
        for example in data.train[: args.sample]:
            print("\n" + "─" * 72)
            print(example.messages[1]["content"])
            print("─" * 72)
            print(example.messages[2]["content"][:600])
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from dragon import mlxops

    config = _config(args)
    if not args.no_build:
        cmd_build(argparse.Namespace(config=args.config, dry_run=False, sample=0))
    print("to watch it: dragon studio   (in another terminal)\n")
    return mlxops.train(config, resume=args.resume, extra_args=args.extra)


def cmd_test(args: argparse.Namespace) -> int:
    from dragon import mlxops

    return mlxops.test(_config(args))


def cmd_write(args: argparse.Namespace) -> int:
    from dragon.generate import brief_from_notes, write

    config = _config(args)
    if args.notes:
        notes = sys.stdin.read() if args.notes == "-" else Path(args.notes).read_text()
        brief = brief_from_notes(notes, args.title, args.section, args.label)
    elif args.brief:
        brief = args.brief.replace("\\n", "\n")
    else:
        raise DragonError("give a brief, or --notes FILE (or --notes - for stdin)")

    text = write(
        config,
        brief,
        base=args.base,
        max_tokens=args.max_tokens,
        temperature=args.temp,
    )
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from dragon import mlxops

    return mlxops.chat(_config(args), extra_args=args.extra)


def cmd_fuse(args: argparse.Namespace) -> int:
    from dragon import mlxops

    config = _config(args)
    path = mlxops.fuse(config, dequantize=args.dequantize, force=args.force)
    print(path)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from dragon import mlxops

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding to {args.host}. The endpoint has no authentication; "
            "anything that can reach this machine can use the model.",
            file=sys.stderr,
        )
    return mlxops.serve(_config(args), host=args.host, port=args.port)


def cmd_export(args: argparse.Namespace) -> int:
    from dragon import cards, mlxops

    config = _config(args)
    fused = mlxops.fuse(config)
    if not args.no_lm_studio:
        export = config.raw.get("export") or {}
        mlxops.lm_studio_link(fused, export.get("lm_studio_name") or config.model_name)
    if not args.no_ollama:
        from dragon.gguf import quantised_file

        # A quantised GGUF, if one has been made, imports in seconds; otherwise
        # Ollama reads the fp16 safetensors and quantises on the way in.
        source = quantised_file(config)
        if source is None:
            source = mlxops.fuse(config, dequantize=True)
        path = config.work_dir / "Modelfile"
        path.write_text(cards.modelfile(config, source), encoding="utf-8")
        tag = args.tag or f"{config.model_name}:latest"
        # q8_0, not q4_K_M: see mlxops.fuse for why four bits loses the adapter.
        ollama = (config.raw.get("export") or {}).get("ollama") or {}
        quantise = None if source.is_file() else str(ollama.get("quantise", "q8_0"))
        mlxops.ollama_import(config, path, tag, quantise=quantise)
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    from dragon.studio import discover, serve

    # Works with or without a config, so it can watch a run started by hand.
    try:
        config = _config(args)
        root, name = config.root, config.model_name
        stats, lora, iters = (
            config.data_dir / "stats.json",
            config.work_dir / "lora.yaml",
            config.training.get("iters"),
        )
    except DragonError:
        root, name, stats, lora, iters = (
            Path.cwd(),
            Path.cwd().name,
            Path.cwd() / "data" / "stats.json",
            None,
            None,
        )
    logs = [Path(p) for p in (args.log or [])]
    if logs and not args.config:
        root, name = logs[0].resolve().parent, logs[0].resolve().parent.name

    def runs():
        # Rediscovered on every poll, so a log that appears later is a new tab.
        return discover(
            root,
            extra_logs=logs,
            adapters=Path(args.adapters) if args.adapters else None,
            stats=stats,
            lora=lora,
            iters=args.iters or iters,
            name=name,
        )

    if_busy = args.if_busy or ("ask" if sys.stdin.isatty() else "next")
    serve(runs, host=args.host, port=args.port, open_browser=not args.no_open, if_busy=if_busy)
    return 0


def cmd_gguf(args: argparse.Namespace) -> int:
    from dragon.gguf import export

    for path in export(_config(args), quant=args.quant, force=args.force):
        print(path)
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from dragon.publish import publish

    config = _config(args)
    if not args.private and not args.yes:
        print(
            "\nA model trained on one person's writing reproduces that person's voice.\n"
            "Published openly it can be used to impersonate them, and it cannot be\n"
            "recalled once downloaded. Publish publicly only if everyone whose writing\n"
            "is in the corpus has agreed.\n"
        )
        if input("Publish publicly? type 'yes' to continue: ").strip().lower() != "yes":
            print("stopped")
            return 1
    publish(
        config,
        fused=args.fused,
        gguf=args.gguf,
        private=args.private,
        adapter_repo=args.repo,
        fused_repo=args.fused_repo,
        gguf_repo=args.gguf_repo,
        owner=args.owner,
        tag=args.tag,
    )
    return 0


def cmd_curve(args: argparse.Namespace) -> int:
    from dragon.runlog import curve, describe

    config = _config(args)
    val, train = curve(Path(args.log) if args.log else config.root / "train.log")
    if not val:
        print("no validation points yet")
        return 0
    print("iteration  val loss")
    for p in val:
        print(f"{p.iteration:>9}  {p.loss:.3f}")
    if train:
        print(f"\nlast training loss {train[-1].loss:.3f} at iteration {train[-1].iteration}")
    print()
    for line in describe(val):
        print(line)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from dragon.runlog import checkpoints, promote

    config = _config(args)
    if args.iteration is None:
        saved = checkpoints(config.adapter_dir)
        if not saved:
            print(f"no checkpoints in {config.adapter_dir}")
            return 1
        print("saved checkpoints: " + ", ".join(str(i) for i, _ in saved))
        return 0
    live = promote(config, args.iteration)
    print(f"{live} is now checkpoint {args.iteration}")
    print("Re-run `dragon fuse --force` and `dragon export` if you have made either.")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """The test that matters: the same brief, base model and adapter, side by side."""
    from dragon.generate import brief_from_notes, write

    config = _config(args)
    if args.notes:
        notes = sys.stdin.read() if args.notes == "-" else Path(args.notes).read_text()
        brief = brief_from_notes(notes, args.title, args.section, args.label)
    elif args.brief:
        brief = args.brief.replace("\\n", "\n")
    else:
        raise DragonError("give a brief, or --notes FILE")
    for label, base in (("BASE MODEL", True), ("WITH ADAPTER", False)):
        print("═" * 72)
        print(label)
        print("═" * 72)
        print(write(config, brief, base=base, max_tokens=args.max_tokens, temperature=args.temp))
        print()
    return 0


def cmd_pick(args: argparse.Namespace) -> int:
    from dragon.pick import DEFAULT_ENDPOINT, DEFAULT_MODEL, pairs, read, serve

    # Works without a config, so it can sit in front of any endpoint; with one,
    # it sends the voice prompt and reads `pick:` for the endpoint.
    try:
        config = _config(args)
        root, system, settings = config.root, config.system_prompt, config.raw.get("pick") or {}
    except DragonError:
        config, root, system, settings = None, Path.cwd(), None, {}
    if args.system:
        system = (
            Path(args.system[1:]).read_text().strip()
            if args.system.startswith("@")
            else args.system
        )

    out = Path(args.out or settings.get("out") or root / "preferences.jsonl").expanduser().resolve()
    if args.pairs:
        made = pairs(read(out))
        target = Path(args.pairs)
        target.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in made))
        print(f"{len(made)} pairs from {len(read(out))} verdicts → {target}")
        return 0

    endpoint = _endpoint(
        settings,
        url=args.endpoint,
        model=args.model,
        key_file=args.api_key_file,
        cacert=args.cacert,
        default_url=DEFAULT_ENDPOINT,
        default_model=DEFAULT_MODEL,
        env_key="DRAGON_PICK_API_KEY",
    )
    # The second model, for a blind comparison: its own `pick.against:` section,
    # or just another model at the same endpoint.
    against = settings.get("against") or {}
    if isinstance(against, str):
        against = {"model": against}
    endpoint_b = None
    if args.against or args.against_endpoint or against:
        endpoint_b = _endpoint(
            {**settings, **against} if not against.get("endpoint") else against,
            url=args.against_endpoint,
            model=args.against,
            default_url=endpoint.url,
            default_model=endpoint.model,
            env_key="DRAGON_PICK_API_KEY",
        )
        if not (against.get("endpoint") or args.against_endpoint):
            endpoint_b.api_key, endpoint_b.cacert = endpoint.api_key, endpoint.cacert
    retriever = None
    if args.retrieve or settings.get("retrieve"):
        if config is None:
            raise DragonError("--retrieve needs a config: the index is built from its corpus")
        retriever = _retriever(config)
    if_busy = args.if_busy or ("ask" if sys.stdin.isatty() else "next")
    serve(
        endpoint,
        system=system,
        out=out,
        temperature=args.temp if args.temp is not None else float(settings.get("temperature", 0.7)),
        max_tokens=args.max_tokens,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        if_busy=if_busy,
        endpoint_b=endpoint_b,
        retriever=retriever,
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    import time

    from dragon.retrieve import Index, embedder, passages

    config = _config(args)
    directory = _index_dir(config)
    endpoint = _embed_endpoint(config)
    items = passages(_corpus(config))
    if not items:
        raise DragonError("the corpus has no text to index")
    previous = None
    if not args.rebuild:
        try:
            previous = Index.load(directory)
        except (DragonError, OSError, ValueError, KeyError, TypeError):
            previous = None  # a missing or half-written index is rebuilt from scratch

    def progress(done: int, total: int) -> None:
        print(f"\r  embedded {done}/{total}", end="", flush=True)

    print(f"index: {len(items)} passages, embedding with {endpoint.describe()}")
    started = time.time()
    index, embedded = Index.build(
        items, embedder(endpoint), model=endpoint.model, previous=previous, progress=progress
    )
    if embedded:
        print()
    index.save(directory)
    reused = len(items) - embedded
    print(
        f"  {embedded} embedded, {reused} unchanged, in {time.time() - started:.0f}s → {directory}"
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    import random

    from dragon.pick import DEFAULT_ENDPOINT, DEFAULT_MODEL, complete
    from dragon.retrieve import brief

    config = _config(args)
    question = args.question
    if question in (None, "-"):
        question = sys.stdin.read()
    question = (question or "").strip()
    if not question:
        raise DragonError("nothing to ask")
    retriever = _retriever(config, k=args.k)
    hits = retriever.find(question)
    if args.sources or not args.quiet:
        for h in hits:
            print(f"  {h.score:.4f}  {h.passage.where}  ({', '.join(h.why)})", file=sys.stderr)
    if args.sources:
        return 0

    settings = config.raw.get("retrieval") or {}
    writer = settings.get("writer") or config.raw.get("pick") or {}
    endpoint = _endpoint(
        writer,
        url=args.endpoint,
        model=args.model,
        key_file=args.api_key_file,
        cacert=args.cacert,
        default_url=DEFAULT_ENDPOINT,
        default_model=DEFAULT_MODEL,
        env_key="DRAGON_PICK_API_KEY",
    )
    temperature = args.temp if args.temp is not None else float(writer.get("temperature", 0.5))
    max_tokens = args.max_tokens or int(writer.get("max_tokens", 2500))
    if not args.quiet:
        print(f"  → {endpoint.describe()}\n", file=sys.stderr)
    print(
        complete(
            endpoint,
            retriever.system(config.system_prompt),
            brief(question, hits),
            temperature=temperature,
            seed=random.randrange(1, 2**31),
            max_tokens=max_tokens,
        )
    )
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    from dragon.cards import model_card

    print(model_card(_config(args), flavour=args.flavour, private=True))
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dragon", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"dragon {__version__}")
    p.add_argument("-c", "--config", help=f"path to {CONFIG_NAME} (default: the current directory)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="write a starting config.yaml here")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("doctor", help="check the machine, the config and the corpus")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("build", help="read the corpus and write the training set")
    s.add_argument("--dry-run", action="store_true", help="count the pairs, write nothing")
    s.add_argument("--sample", type=int, default=0, metavar="N", help="print N example pairs")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("train", help="fine-tune the adapter")
    s.add_argument("--no-build", action="store_true", help="use data/ as it stands")
    s.add_argument("--resume", action="store_true", help="carry on from the current adapter")
    s.add_argument("extra", nargs="*", help="passed through to mlx_lm.lora")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("test", help="loss on the held-out pieces")
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("write", help="a draft from a brief or from notes")
    s.add_argument("brief", nargs="?")
    s.add_argument("--notes", help="file of dictated notes, or - for stdin")
    s.add_argument("--title", default="Untitled")
    s.add_argument("--section")
    s.add_argument("--label", default="piece", help="what to call the thing: essay, page, post")
    s.add_argument("--base", action="store_true", help="the base model, no adapter, for comparison")
    s.add_argument("--max-tokens", type=int, default=900)
    s.add_argument("--temp", type=float, default=0.7)
    s.add_argument("-o", "--out", help="write to a file instead of stdout")
    s.set_defaults(func=cmd_write)

    s = sub.add_parser("chat", help="interactive session with the adapter loaded")
    s.add_argument("extra", nargs="*", help="passed through to mlx_lm.chat")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("fuse", help="bake the adapter into a standalone model")
    s.add_argument("--dequantize", action="store_true", help="stop at the exact fp16 fuse")
    s.add_argument("--force", action="store_true", help="rebuild even if it exists")
    s.set_defaults(func=cmd_fuse)

    s = sub.add_parser("serve", help="OpenAI-compatible endpoint on localhost")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("export", help="into LM Studio and Ollama")
    s.add_argument("--tag", help="the Ollama tag (default: <model name>:latest)")
    s.add_argument("--no-ollama", action="store_true")
    s.add_argument("--no-lm-studio", action="store_true")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("studio", help="watch a training run in the browser")
    s.add_argument(
        "--log",
        action="append",
        help="a log to watch; repeatable. train*.log in the project are found anyway",
    )
    s.add_argument("--adapters", help="an adapters/ directory other than this project's")
    s.add_argument(
        "--iters", type=int, help="the run's target, if there is no config to read it from"
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8790)
    s.add_argument("--no-open", action="store_true", help="do not open a browser")
    s.add_argument(
        "--if-busy",
        choices=["ask", "next", "kill", "fail"],
        help="what to do if the port is taken (default: ask at a terminal, next otherwise)",
    )
    s.set_defaults(func=cmd_studio)

    s = sub.add_parser("gguf", help="a GGUF export, for Ollama on any machine")
    s.add_argument(
        "--quant", default="Q8_0", help="llama-quantize type (default Q8_0; Q4 loses the adapter)"
    )
    s.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    s.set_defaults(func=cmd_gguf)

    s = sub.add_parser("publish", help="to the Hugging Face Hub")
    s.add_argument("--fused", action="store_true", help="the fused model too (about 8 GB for a 7B)")
    s.add_argument("--gguf", action="store_true", help="the GGUF export too, for Ollama")
    s.add_argument("--public", dest="private", action="store_false", help="read the warning first")
    s.add_argument("--yes", action="store_true", help="skip the confirmation for --public")
    s.add_argument("--repo", help="adapter repository id (default: <user>/<model name>-lora)")
    s.add_argument("--fused-repo", help="fused repository id")
    s.add_argument("--gguf-repo", help="GGUF repository id (default: <user>/<model name>-gguf)")
    s.add_argument("--owner", help="publish under an organisation instead of your account")
    s.add_argument("--tag", help="tag the new head of each repository, e.g. v2")
    s.set_defaults(func=cmd_publish, private=True)

    s = sub.add_parser("curve", help="the validation loss curve from train.log")
    s.add_argument("--log", help="a log file other than train.log")
    s.set_defaults(func=cmd_curve)

    s = sub.add_parser("promote", help="make a saved checkpoint the live adapter")
    s.add_argument("iteration", nargs="?", type=int, help="omit to list what is saved")
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("compare", help="the same brief through the base model and the adapter")
    s.add_argument("brief", nargs="?")
    s.add_argument("--notes", help="file of dictated notes, or - for stdin")
    s.add_argument("--title", default="Untitled")
    s.add_argument("--section")
    s.add_argument("--label", default="piece")
    s.add_argument("--max-tokens", type=int, default=600)
    s.add_argument("--temp", type=float, default=0.7)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("pick", help="two drafts side by side; your verdict is preference data")
    s.add_argument("--endpoint", help="OpenAI-compatible base URL (default: dragon serve's)")
    s.add_argument("--model", help="model id at the endpoint (default: default_model)")
    s.add_argument("--api-key-file", help="a file holding a bearer token for the endpoint")
    s.add_argument("--cacert", help="a CA certificate for an endpoint with its own")
    s.add_argument("--system", help="the system prompt, or @FILE (default: the config's voice)")
    s.add_argument("--temp", type=float, help="sampling temperature (default 0.7)")
    s.add_argument("--max-tokens", type=int, default=700)
    s.add_argument("--out", help="where verdicts go (default: preferences.jsonl in the project)")
    s.add_argument("--pairs", metavar="FILE", help="write chosen/rejected pairs and exit")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8791)
    s.add_argument("--no-open", action="store_true", help="do not open a browser")
    s.add_argument(
        "--if-busy",
        choices=["ask", "next", "kill", "fail"],
        help="what to do if the port is taken (default: ask at a terminal, next otherwise)",
    )
    s.add_argument("--against", metavar="MODEL", help="draft B from this model, blind")
    s.add_argument("--against-endpoint", metavar="URL", help="…at this endpoint, if not A's")
    s.add_argument(
        "--retrieve", action="store_true", help="send the passages of your writing that match"
    )
    s.set_defaults(func=cmd_pick)

    s = sub.add_parser("index", help="embed the corpus in passages, for retrieval")
    s.add_argument("--rebuild", action="store_true", help="embed everything, not only what changed")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("ask", help="an answer from your own writing")
    s.add_argument("question", nargs="?", help="the question or brief, or - for stdin")
    s.add_argument("--sources", action="store_true", help="list the passages found, and stop")
    s.add_argument("--k", type=int, help="how many passages (default 6)")
    s.add_argument("--endpoint", help="the writing model's base URL (default: retrieval.writer)")
    s.add_argument("--model", help="the writing model")
    s.add_argument("--api-key-file", help="a file holding a bearer token for the endpoint")
    s.add_argument("--cacert", help="a CA certificate for an endpoint with its own")
    s.add_argument("--temp", type=float, help="sampling temperature (default 0.5)")
    s.add_argument("--max-tokens", type=int, help="default 2500: reasoning models think first")
    s.add_argument("-q", "--quiet", action="store_true", help="the answer only")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("card", help="print the model card that publish would upload")
    s.add_argument("--flavour", choices=["adapter", "fused", "gguf"], default="adapter")
    s.set_defaults(func=cmd_card)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DragonError as exc:
        print(f"dragon: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
