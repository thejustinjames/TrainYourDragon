"""Pick the better of two drafts: preference data from your own taste.

The same prompt goes to the model twice with different seeds, the two drafts
sit side by side, and you say which is better, that they are about the same,
or that both are poor. You can correct the one you chose before saving. Each
verdict is a line in `preferences.jsonl`, and `pairs()` turns those lines into
the chosen/rejected pairs a preference-tuning run (DPO, ORPO) reads.

Supervised fine-tuning teaches the model what your writing looks like. This
teaches it what you would rather it did not do: the padded sixth paragraph,
the invented figure, the closing line that sounds right and says nothing.

The model is any OpenAI-compatible endpoint: `dragon serve` on this machine by
default, or Ollama elsewhere behind a gateway with a bearer token and its own
certificate. One small server from the standard library, one page, nothing
leaves the machine except the prompt to the endpoint you name.
"""

from __future__ import annotations

import json
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from dragon.errors import DragonError

DEFAULT_ENDPOINT = "http://127.0.0.1:8787/v1"  # dragon serve
DEFAULT_MODEL = "default_model"  # what mlx_lm.server calls the model it was started with
CHOICES = ("a", "b", "same", "both_poor")


@dataclass
class Endpoint:
    url: str  # the base, ending in /v1
    model: str
    api_key: str | None = None
    cacert: str | None = None
    timeout: float = 600.0

    def describe(self) -> str:
        return f"{self.model} at {self.url}"


def complete(
    endpoint: Endpoint,
    system: str | None,
    prompt: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
) -> str:
    """One chat completion. Errors come back as DragonError, without the key."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {
        "model": endpoint.model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    request = urllib.request.Request(
        endpoint.url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
    )
    context = ssl.create_default_context(cafile=endpoint.cacert) if endpoint.cacert else None
    try:
        with urllib.request.urlopen(request, timeout=endpoint.timeout, context=context) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as exc:
        raise DragonError(f"{endpoint.url} answered {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise DragonError(f"could not reach {endpoint.url}: {reason}") from exc
    try:
        return reply["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise DragonError(f"{endpoint.url} sent something that is not a chat completion") from exc


# ------------------------------------------------------------------ records
def record(path: Path, entry: dict[str, Any]) -> int:
    """Append one verdict and return how many the file now holds."""
    if entry.get("choice") not in CHOICES:
        raise DragonError(f"choice must be one of {', '.join(CHOICES)}")
    for key in ("prompt", "a", "b"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            raise DragonError(f"a verdict needs a {key!r}")
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(read(path))


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written last line is not worth failing over
    return out


def pairs(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Chosen/rejected pairs from verdicts, in the shape DPO trainers read.

    A beats B: one pair. Your edit of the winner beats both drafts: two pairs,
    and the stronger signal. Both poor with an edit: the edit beats both. Both
    poor without one, or about the same: nothing, because neither draft is the
    one you wanted and a forced choice between two bad drafts teaches noise.
    Two identical drafts (a memorised answer) make no pair from a pick, and
    one pair, not two, from an edit.
    """
    out: list[dict[str, str]] = []
    for r in records:
        a, b, choice = r.get("a", ""), r.get("b", ""), r.get("choice")
        edited = (r.get("edited") or "").strip()
        base = {"system": r.get("system") or "", "prompt": r.get("prompt", "")}
        if edited:
            if choice == "same":
                continue
            for rejected in dict.fromkeys((a.strip(), b.strip())):
                if rejected and rejected != edited:
                    out.append({**base, "chosen": edited, "rejected": rejected})
        elif a.strip() == b.strip():
            continue
        elif choice == "a":
            out.append({**base, "chosen": a, "rejected": b})
        elif choice == "b":
            out.append({**base, "chosen": b, "rejected": a})
    return out


# ------------------------------------------------------------------- server
def page() -> str:
    return resources.files("dragon").joinpath("pick.html").read_text(encoding="utf-8")


def make_server(
    endpoint: Endpoint,
    *,
    system: str | None,
    out: Path,
    temperature: float = 0.7,
    max_tokens: int = 700,
    host: str = "127.0.0.1",
    port: int = 8791,
    complete_fn=complete,
) -> ThreadingHTTPServer:
    """The picker's HTTP server, not yet running. `serve()` runs it; tests poke it."""
    html = page().encode("utf-8")
    lock = threading.Lock()

    def info() -> dict[str, Any]:
        records = read(out)
        return {
            "model": endpoint.describe(),
            "system": bool(system),
            "out": str(out),
            "count": len(records),
            "pairs": len(pairs(records)),
            "temperature": temperature,
        }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server's name)
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", html)
            elif self.path == "/api/info":
                self._json(200, info())
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "not JSON"})
                return
            try:
                if self.path == "/api/draft":
                    prompt = str(body.get("prompt") or "").strip()
                    if not prompt:
                        raise DragonError("nothing to write from")
                    seed = random.randrange(1, 2**31)
                    started = time.time()
                    text = complete_fn(
                        endpoint,
                        system,
                        prompt,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=max_tokens,
                    )
                    self._json(
                        200, {"text": text, "seed": seed, "seconds": round(time.time() - started)}
                    )
                elif self.path == "/api/choice":
                    entry = {
                        "prompt": body.get("prompt"),
                        "system": system or "",
                        "a": body.get("a"),
                        "b": body.get("b"),
                        "choice": body.get("choice"),
                        "edited": (body.get("edited") or "").strip() or None,
                        "seeds": body.get("seeds"),
                        "model": endpoint.model,
                        "endpoint": endpoint.url,
                        "temperature": temperature,
                    }
                    with lock:
                        record(out, entry)
                    self._json(200, info())
                else:
                    self._send(404, "text/plain", b"not found")
            except DragonError as exc:
                self._json(502 if self.path == "/api/draft" else 400, {"error": str(exc)})

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

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
    endpoint: Endpoint,
    *,
    system: str | None,
    out: Path,
    temperature: float,
    max_tokens: int,
    host: str = "127.0.0.1",
    port: int = 8791,
    open_browser: bool = True,
    if_busy: str = "ask",
) -> None:
    from dragon.studio import choose_port

    port = choose_port(host, port, if_busy=if_busy)
    server = make_server(
        endpoint,
        system=system,
        out=out,
        temperature=temperature,
        max_tokens=max_tokens,
        host=host,
        port=port,
    )
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"pick: {url}")
    print(f"  model    {endpoint.describe()}")
    print(f"  verdicts {out} ({len(read(out))} so far)")
    if not system:
        print("  no system prompt: without a config the voice prompt is not sent")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
