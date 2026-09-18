"""The observability studio: a page that watches a training run.

One small HTTP server from the standard library, one self-contained HTML page,
no build step. It reads `train.log`, `adapters/` and `data/stats.json` every
couple of seconds and shows what the trainer is doing, what the numbers mean,
and when to stop. It never writes anything, so it is safe to point at a run
that is in progress.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from dragon.runlog import TOKENS_PER_WORD, checkpoints, describe
from dragon.runlog import curve as parse_curve

_TRAIN_LINE = re.compile(
    r"Iter (?P<iter>\d+): Train loss (?P<loss>[\d.]+), Learning Rate (?P<lr>[\d.e+-]+), "
    r"It/sec (?P<its>[\d.]+), Tokens/sec (?P<tps>[\d.]+), Trained Tokens (?P<tokens>\d+)"
    r"(?:, Peak mem (?P<mem>[\d.]+) GB)?"
)
_SAVED = re.compile(r"Iter (\d+): Saved adapter weights")
_TRAINABLE = re.compile(r"Trainable parameters: ([\d.]+)% \(([\d.]+M)/([\d.]+M)\)")
STALE_AFTER_SECONDS = 180


STOPPED_AFTER_SECONDS = 3600


@dataclass
class Paths:
    log: Path
    adapters: Path
    stats: Path | None = None
    lora: Path | None = None  # the resolved mlx_lm.lora config, for iters
    iters: int | None = None  # an override, when there is no config at all
    name: str = "training run"

    @property
    def key(self) -> str:
        return self.log.stem


def discover(
    root: Path,
    *,
    extra_logs: list[Path] | None = None,
    adapters: Path | None = None,
    stats: Path | None = None,
    lora: Path | None = None,
    iters: int | None = None,
    name: str | None = None,
) -> list[Paths]:
    """Every run this project has logged, newest first.

    `train.log` is the current run. Earlier ones are whatever was kept as
    `train-<something>.log`, with `adapters-<something>/` beside it if it exists.
    """
    logs = {p.resolve() for p in root.glob("train*.log")}
    logs |= {p.resolve() for p in (extra_logs or [])}
    runs = []
    for log in sorted(logs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        suffix = log.stem[len("train") :].lstrip("-_")
        run_adapters = adapters or (root / "adapters")
        if suffix and (root / f"adapters-{suffix}").is_dir():
            run_adapters = root / f"adapters-{suffix}"
        label = name or root.name
        runs.append(
            Paths(
                log=log,
                adapters=run_adapters,
                stats=stats,
                lora=lora if not suffix else None,
                iters=iters if not suffix else None,
                name=f"{label} \u00b7 {suffix}" if suffix else f"{label} \u00b7 current",
            )
        )
    if not runs:
        runs.append(
            Paths(
                log=root / "train.log",
                adapters=adapters or root / "adapters",
                stats=stats,
                lora=lora,
                iters=iters,
                name=name or root.name,
            )
        )
    return runs


@dataclass
class Snapshot:
    name: str
    stage: str  # waiting | loading | training | finished | stalled
    live: bool
    iteration: int = 0
    iters: int | None = None
    progress: float = 0.0
    eta_seconds: float | None = None
    val: list[dict[str, float]] = field(default_factory=list)
    train: list[dict[str, float]] = field(default_factory=list)
    saved: list[int] = field(default_factory=list)
    lowest: dict[str, float] | None = None
    latest: dict[str, Any] = field(default_factory=dict)
    trainable: dict[str, Any] | None = None
    corpus: dict[str, Any] | None = None
    passes: float | None = None
    reading: list[str] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)
    log_age_seconds: float | None = None
    updated: float = 0.0


def snapshot(paths: Paths) -> Snapshot:
    now = time.time()
    iters = _iters(paths)
    corpus = _stats(paths.stats)

    if not paths.log.exists():
        return Snapshot(
            name=paths.name,
            stage="waiting",
            live=False,
            iters=iters,
            corpus=corpus,
            reading=["Waiting for train.log to appear. Start `dragon train`."],
            updated=now,
        )

    text = paths.log.read_text(encoding="utf-8", errors="replace")
    age = now - paths.log.stat().st_mtime
    val, _ = parse_curve(paths.log)
    train = [_train_point(m) for m in _TRAIN_LINE.finditer(text)]
    saved = sorted(
        {int(i) for i in _SAVED.findall(text)} | {i for i, _ in checkpoints(paths.adapters)}
    )

    iteration = max([p.iteration for p in val] + [int(t["iteration"]) for t in train] + [0])
    latest = train[-1] if train else {}
    finished = "Saved final weights" in text or (iters is not None and iteration >= iters)

    if finished:
        stage = "finished"
    elif not val and not train:
        stage = "loading"
    elif age > STOPPED_AFTER_SECONDS:
        stage = "stopped"
    elif age > STALE_AFTER_SECONDS:
        stage = "stalled"
    else:
        stage = "training"
    if finished and iters is None:
        iters = iteration
    live = stage in ("loading", "training")

    eta = None
    if stage == "training" and iters and latest.get("it_per_sec"):
        eta = (iters - iteration) / float(latest["it_per_sec"])

    lowest = None
    if val:
        low = min(val, key=lambda p: p.loss)
        lowest = {"iteration": low.iteration, "loss": low.loss}

    passes = None
    if corpus and latest.get("trained_tokens"):
        words = corpus.get("target_words", 0) + corpus.get("brief_words", 0)
        if words:
            passes = float(latest["trained_tokens"]) / (words * TOKENS_PER_WORD)

    m = _TRAINABLE.search(text)
    trainable = (
        {"percent": float(m.group(1)), "adapter": m.group(2), "model": m.group(3)} if m else None
    )

    return Snapshot(
        name=paths.name,
        stage=stage,
        live=live,
        iteration=iteration,
        iters=iters,
        progress=(iteration / iters) if iters else 0.0,
        eta_seconds=eta,
        val=[{"iteration": p.iteration, "loss": p.loss} for p in val],
        train=train,
        saved=saved,
        lowest=lowest,
        latest=latest,
        trainable=trainable,
        corpus=corpus,
        passes=passes,
        reading=_reading(stage, val, iteration, iters, passes, age),
        log_tail=[ln for ln in text.splitlines() if _worth_showing(ln)][-12:],
        log_age_seconds=age,
        updated=now,
    )


# --------------------------------------------------------------- helpers
_NOISE = re.compile(r"^(Calculating loss|Fetching \d+ files)|\d+%\|")


def _worth_showing(line: str) -> bool:
    """Drop tqdm progress bars, which dominate a raw tail and say nothing."""
    return bool(line.strip()) and not _NOISE.search(line)


def _train_point(m: re.Match) -> dict[str, float]:
    d = m.groupdict()
    out = {
        "iteration": int(d["iter"]),
        "loss": float(d["loss"]),
        "lr": float(d["lr"]),
        "it_per_sec": float(d["its"]),
        "tokens_per_sec": float(d["tps"]),
        "trained_tokens": int(d["tokens"]),
    }
    if d.get("mem"):
        out["peak_mem_gb"] = float(d["mem"])
    return out


def _iters(paths: Paths) -> int | None:
    if paths.iters:
        return paths.iters
    if paths.lora and paths.lora.exists():
        try:
            return int((yaml.safe_load(paths.lora.read_text()) or {}).get("iters") or 0) or None
        except (yaml.YAMLError, ValueError):
            return None
    return None


def _stats(path: Path | None) -> dict[str, Any] | None:
    if path and path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _reading(stage, val, iteration, iters, passes, age) -> list[str]:
    """What is going on, in words a first-time trainer can use."""
    if stage == "loading":
        return [
            "The trainer is loading the base model and the dataset. The first "
            "validation figure appears at iteration 1, and it will look terrible: "
            "that is the untouched model being asked to write in a voice it has never seen."
        ]
    if stage == "stopped":
        return [
            f"This run ended at iteration {iteration} without saving final weights, so it "
            "was stopped or it crashed. The checkpoints it saved are still usable."
        ]
    if stage == "stalled":
        return [
            f"train.log has not changed for {int(age // 60)} minutes and the run is not "
            "finished. Either the trainer was stopped, or the Mac is dozing: "
            "`caffeinate -ims -w <pid>` or scripts/train-awake.sh keeps it up."
        ]
    lines: list[str] = []
    if stage == "finished":
        lines.append(
            "The run is finished. The live adapter is whatever was saved last, which is "
            "not necessarily the best checkpoint."
        )
    lines += describe(val)
    if passes is not None:
        lines.append(
            f"About {passes:.1f} passes over the corpus so far. One to two is where a voice "
            "model should end; beyond that it starts quoting rather than writing."
        )
    if stage == "training" and len(val) >= 2:
        lines.append(
            "Validation loss is measured on pieces the model does not train on, so it is the "
            "honest curve. Training loss will keep falling whatever happens; the gap between "
            "them opening up is memorisation."
        )
    return lines


# ---------------------------------------------------------------- server
def page() -> str:
    return resources.files("dragon").joinpath("studio.html").read_text(encoding="utf-8")


def make_server(
    runs: Paths | list[Paths], *, host: str = "127.0.0.1", port: int = 8790
) -> ThreadingHTTPServer:
    """The studio's HTTP server, not yet running. `serve()` runs it; tests poke it."""
    runs = [runs] if isinstance(runs, Paths) else list(runs)
    by_key = {r.key: r for r in runs}
    html = page().encode("utf-8")
    lock = threading.Lock()
    cache: dict[str, tuple[float, bytes]] = {}

    def state(key: str) -> bytes:
        with lock:
            at, body = cache.get(key, (0.0, b""))
            if time.time() - at > 1.0:
                body = json.dumps(asdict(snapshot(by_key[key]))).encode("utf-8")
                cache[key] = (time.time(), body)
            return body

    def index() -> bytes:
        rows = []
        for r in runs:
            s = snapshot(r)
            rows.append(
                {
                    "key": r.key,
                    "name": r.name,
                    "stage": s.stage,
                    "iteration": s.iteration,
                    "iters": s.iters,
                    "lowest": s.lowest,
                    "live": s.live,
                }
            )
        return json.dumps(rows).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server's name)
            path, _, query = self.path.partition("?")
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/api/state":
                key = params.get("run", runs[0].key)
                if key not in by_key:
                    self._send(404, "text/plain", b"no such run")
                    return
                self._send(200, "application/json; charset=utf-8", state(key))
            elif path == "/api/runs":
                self._send(200, "application/json; charset=utf-8", index())
            elif path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", html)
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    runs: Paths | list[Paths],
    *,
    host: str = "127.0.0.1",
    port: int = 8790,
    open_browser: bool = True,
) -> None:
    runs = [runs] if isinstance(runs, Paths) else list(runs)
    server = make_server(runs, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"observability studio: {url}")
    for r in runs:
        print(f"  {r.key:<16} {r.log}")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
