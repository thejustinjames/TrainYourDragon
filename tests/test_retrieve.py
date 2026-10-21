import hashlib
import json
import math
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from conftest import EXAMPLE_CONFIG
from dragon import corpus, pick
from dragon.cli import main
from dragon.config import Config
from dragon.errors import DragonError
from dragon.pick import Endpoint, read
from dragon.retrieve import INSTRUCTIONS, Index, Retriever, brief, embedder, passages


def bag(text, dims=64):
    """A stand-in embedding: words hashed into buckets. Shared words, nearby vectors."""
    v = [0.0] * dims
    for w in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dims] += 1.0
    return v


def fake_embed(texts):
    fake_embed.calls += len(texts)
    return [bag(t) for t in texts]


fake_embed.calls = 0


@pytest.fixture
def docs():
    return corpus.load(Config.load(EXAMPLE_CONFIG))


@pytest.fixture
def index(docs):
    built, _ = Index.build(passages(docs), fake_embed, model="bag")
    return built


def find(index, question, **kw):
    return index.search(question, bag(question), **kw)


# --------------------------------------------------------------- passages
def test_every_document_becomes_passages_of_bounded_length(docs):
    items = passages(docs)
    assert {p.key for p in items} == {d.key for d in docs}
    assert len({p.id for p in items}) == len(items)
    assert all(len(p.text.split()) <= 400 for p in items)


def test_a_page_has_an_overview_and_its_features_apart(docs):
    lantern = [p for p in passages(docs) if p.key == "lantern"]
    assert lantern[0].heading is None and lantern[0].text.startswith("A reading light")
    features = [p for p in lantern if p.heading == "Highlights and features"]
    assert features and "Provenance by default" in features[0].text


def test_a_glossary_carries_its_essays_title(docs):
    glossary = next(p for p in passages(docs) if p.label == "Glossary")
    assert glossary.title == "The Cost of a Number"
    assert "Goodhart's law —" in glossary.text


# ------------------------------------------------------------------ search
def test_a_question_finds_the_page_it_names(index):
    hits = find(index, "What is Lantern?")
    assert hits[0].passage.key == "lantern" and "name" in hits[0].why
    assert hits[0].passage.id == "lantern#0"  # the overview first


def test_keywords_find_what_the_embedding_might_not(index):
    hits = find(index, "Goodhart")
    assert any(h.passage.label == "Glossary" for h in hits)
    assert "words" in next(h for h in hits if h.passage.label == "Glossary").why


def test_no_document_takes_every_place(index):
    hits = find(index, "measurement numbers targets boards", k=6, per_document=1)
    keys = [h.passage.key for h in hits]
    assert len(keys) == len(set(keys))


def test_an_alias_finds_its_page(tmp_path):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["corpus"]["root"] = str((EXAMPLE_CONFIG.parent / raw["corpus"]["root"]).resolve())
    raw["corpus"]["aliases"] = {"lantern": ["the reading light", "LNTN"]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = Config.load(path)
    lantern = next(d for d in corpus.load(config) if d.key == "lantern")
    assert lantern.names == ["Lantern", "the reading light", "LNTN"]
    built, _ = Index.build(passages(corpus.load(config)), fake_embed, model="bag")
    assert find(built, "Is LNTN shipping?")[0].passage.key == "lantern"


def test_bad_aliases_are_refused(tmp_path):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["corpus"]["aliases"] = ["lantern"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(DragonError, match="aliases"):
        Config.load(path)


# ------------------------------------------------------------------- index
def test_an_index_saves_loads_and_searches_the_same(tmp_path, index):
    index.save(tmp_path / "index")
    again = Index.load(tmp_path / "index")
    assert [p.id for p in again.passages] == [p.id for p in index.passages]
    q = "What is Lantern?"
    assert [h.passage.id for h in find(again, q)] == [h.passage.id for h in find(index, q)]
    assert math.isclose(sum(x * x for x in again.vectors[0]), 1.0, rel_tol=1e-4)


def test_reindexing_only_embeds_what_changed(docs, index):
    items = passages(docs)
    items[0].text += " A new sentence."
    fake_embed.calls = 0
    _, embedded = Index.build(items, fake_embed, model="bag", previous=index)
    assert embedded == 1 and fake_embed.calls == 1
    _, embedded = Index.build(items, fake_embed, model="other", previous=index)
    assert embedded == len(items)  # another model's vectors are not reused


def test_no_index_says_what_to_run(tmp_path):
    with pytest.raises(DragonError, match="dragon index"):
        Index.load(tmp_path)


# ------------------------------------------------------------------- brief
def test_the_brief_puts_the_passages_before_the_question(index):
    hits = find(index, "What is Lantern?", k=2)
    text = brief("What is Lantern?", hits)
    assert text.startswith("Notes from your own writing:")
    assert text.endswith("---\n\nWhat is Lantern?")
    assert "— Product: Lantern" in text
    assert brief("Anything", []) == "Anything"


def test_the_system_prompt_gains_the_instructions(index):
    r = Retriever(index, fake_embed)
    assert r.system("You are me.") == f"You are me.\n\n{INSTRUCTIONS}"
    assert "If the notes do not say it, do not say it" in r.system(None)


# --------------------------------------------------------- the endpoints
class FakeOllama(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append((self.path, body))
        if self.path.endswith("/embeddings"):
            data = [{"index": i, "embedding": bag(t)} for i, t in enumerate(body["input"])]
            reply = {"data": list(reversed(data))}  # out of order, as a server may send
        else:
            user = body["messages"][-1]["content"]
            reply = {"choices": [{"message": {"content": f"{body['model']}: {len(user)}"}}]}
        out = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_):
        pass


@pytest.fixture
def ollama():
    FakeOllama.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_the_embedder_batches_and_keeps_order(ollama):
    embed = embedder(Endpoint(url=ollama, model="bge-m3"))
    texts = [f"text number {i}" for i in range(20)]
    assert embed(texts) == [bag(t) for t in texts]
    assert [len(b["input"]) for _, b in FakeOllama.seen] == [16, 4]


def project(tmp_path, endpoint):
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    raw["corpus"]["root"] = str((EXAMPLE_CONFIG.parent / raw["corpus"]["root"]).resolve())
    raw["retrieval"] = {
        "embed": {"endpoint": endpoint, "model": "bge-m3"},
        "writer": {"endpoint": endpoint, "model": "writer"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_index_and_ask_from_the_command_line(tmp_path, ollama, capsys):
    cfg = str(project(tmp_path, ollama))
    assert main(["-c", cfg, "index"]) == 0
    assert (tmp_path / ".dragon" / "index" / "index.json").exists()
    assert "0 unchanged" in capsys.readouterr().out
    assert main(["-c", cfg, "index"]) == 0
    assert " 0 embedded" in capsys.readouterr().out

    assert main(["-c", cfg, "ask", "--sources", "What is Lantern?"]) == 0
    assert "Product: Lantern" in capsys.readouterr().err

    FakeOllama.seen = []
    assert main(["-c", cfg, "ask", "-q", "What is Lantern?"]) == 0
    assert capsys.readouterr().out.startswith("writer: ")
    chat = next(b for path, b in FakeOllama.seen if path.endswith("/chat/completions"))
    assert INSTRUCTIONS in chat["messages"][0]["content"]
    assert chat["messages"][1]["content"].startswith("Notes from your own writing:")


def test_ask_refuses_an_index_from_another_model(tmp_path, ollama):
    cfg = project(tmp_path, ollama)
    assert main(["-c", str(cfg), "index"]) == 0
    raw = yaml.safe_load(cfg.read_text())
    raw["retrieval"]["embed"]["model"] = "nomic-embed-text"
    cfg.write_text(yaml.safe_dump(raw))
    assert main(["-c", str(cfg), "ask", "--sources", "Lantern"]) == 1


# -------------------------------------------------------------- the picker
def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def test_the_picker_sends_passages_and_hides_which_model_is_which(tmp_path, ollama, index):
    out = tmp_path / "preferences.jsonl"
    server = pick.make_server(
        Endpoint(url=ollama, model="jjvoice"),
        endpoint_b=Endpoint(url=ollama, model="gpt-oss"),
        retriever=Retriever(index, fake_embed),
        system="You are me.",
        out=out,
        port=0,
    )
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    try:
        port = server.server_address[1]
        seen_a = set()
        for _ in range(12):  # A is either model, at random
            r = post(port, "/api/round", {"prompt": "What is Lantern?"})
            assert r["blind"] and any(s.startswith("Product: Lantern") for s in r["sources"])
            a = post(port, "/api/draft", {"round": r["round"], "slot": "A"})
            b = post(port, "/api/draft", {"round": r["round"], "slot": "B"})
            seen_a.add(a["text"].split(":")[0])
            assert a["text"].split(":")[0] != b["text"].split(":")[0]
        assert seen_a == {"jjvoice", "gpt-oss"}

        info = post(
            port,
            "/api/choice",
            {"round": r["round"], "prompt": "What is Lantern?", "a": a["text"], "b": b["text"],
             "choice": "a"},
        )  # fmt: skip
        assert set(info["reveal"].values()) == {"jjvoice", "gpt-oss"}
        assert "blind" in info["model"] and info["retrieval"]
        saved = read(out)[0]
        assert saved["models"] == info["reveal"]
        assert saved["brief"].startswith("Notes from your own writing:")
        assert "lantern#0" in saved["sources"]
        assert saved["system"].startswith("You are me.") and INSTRUCTIONS in saved["system"]
        # The pair trains on what the model saw: passages and all.
        assert pick.pairs([saved])[0]["prompt"] == saved["brief"]
    finally:
        server.shutdown()
        server.server_close()
