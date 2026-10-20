import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dragon import pick
from dragon.cli import main
from dragon.errors import DragonError
from dragon.pick import Endpoint, complete, pairs, read, record


def verdict(choice, edited=None, a="Draft one.", b="Draft two."):
    return {
        "prompt": "Write about holds.",
        "system": "You are me.",
        "a": a,
        "b": b,
        "choice": choice,
        "edited": edited,
    }


# ------------------------------------------------------------------- pairs
def test_a_preference_is_one_pair():
    assert pairs([verdict("a")]) == [
        {
            "system": "You are me.",
            "prompt": "Write about holds.",
            "chosen": "Draft one.",
            "rejected": "Draft two.",
        }
    ]
    assert pairs([verdict("b")])[0]["chosen"] == "Draft two."


def test_an_edit_beats_both_drafts():
    made = pairs([verdict("a", edited="Better.")])
    assert [(p["chosen"], p["rejected"]) for p in made] == [
        ("Better.", "Draft one."),
        ("Better.", "Draft two."),
    ]


def test_ties_and_unrescued_poor_drafts_make_no_pairs():
    assert pairs([verdict("same"), verdict("both_poor")]) == []
    assert len(pairs([verdict("both_poor", edited="As it should be.")])) == 2


def test_an_edit_identical_to_a_draft_is_not_paired_against_itself():
    made = pairs([verdict("a", edited="Draft one.")])
    assert [(p["chosen"], p["rejected"]) for p in made] == [("Draft one.", "Draft two.")]


def test_identical_drafts_make_no_pair_from_a_pick_and_one_from_an_edit():
    assert pairs([verdict("a", a="Same.", b="Same.\n")]) == []
    made = pairs([verdict("a", edited="Better.", a="Same.", b="Same.")])
    assert [(p["chosen"], p["rejected"]) for p in made] == [("Better.", "Same.")]


# ----------------------------------------------------------------- records
def test_records_append_and_count(tmp_path):
    out = tmp_path / "preferences.jsonl"
    assert record(out, verdict("a")) == 1
    assert record(out, verdict("same")) == 2
    rows = read(out)
    assert rows[0]["choice"] == "a" and "time" in rows[0]


def test_a_bad_verdict_is_refused(tmp_path):
    with pytest.raises(DragonError):
        record(tmp_path / "p.jsonl", verdict("maybe"))
    with pytest.raises(DragonError):
        record(tmp_path / "p.jsonl", {**verdict("a"), "b": ""})


def test_a_half_written_last_line_is_skipped(tmp_path):
    out = tmp_path / "p.jsonl"
    record(out, verdict("a"))
    with out.open("a") as f:
        f.write('{"prompt": "cut off')
    assert len(read(out)) == 1


# --------------------------------------------------------- the endpoint
class FakeModel(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeModel.seen.append((self.path, dict(self.headers), body))
        if self.headers.get("Authorization") == "Bearer wrong":
            self.send_response(401)
            self.end_headers()
            return
        reply = {"choices": [{"message": {"content": f"  seed {body['seed']}  "}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


@pytest.fixture
def fake_model():
    FakeModel.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_complete_sends_the_voice_prompt_seed_and_token(fake_model):
    ep = Endpoint(url=fake_model, model="jjvoice:7b-run4", api_key="secret")
    text = complete(ep, "You are me.", "Hello", temperature=0.5, seed=42, max_tokens=50)
    assert text == "seed 42"
    path, headers, body = FakeModel.seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert body["model"] == "jjvoice:7b-run4" and body["seed"] == 42
    assert body["messages"][0] == {"role": "system", "content": "You are me."}


def test_a_refused_request_does_not_leak_the_key(fake_model):
    ep = Endpoint(url=fake_model, model="m", api_key="wrong")
    with pytest.raises(DragonError) as err:
        complete(ep, None, "Hello", temperature=0.5, seed=1, max_tokens=5)
    assert "401" in str(err.value) and "wrong" not in str(err.value)


def test_an_unreachable_endpoint_says_so():
    ep = Endpoint(url="http://127.0.0.1:9/v1", model="m", timeout=2)
    with pytest.raises(DragonError, match="could not reach"):
        complete(ep, None, "Hello", temperature=0.5, seed=1, max_tokens=5)


# ------------------------------------------------------------------ server
def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def test_the_page_drafts_and_saves_a_verdict(tmp_path, fake_model):
    out = tmp_path / "preferences.jsonl"
    server = pick.make_server(
        Endpoint(url=fake_model, model="m"), system="You are me.", out=out, port=0
    )
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    try:
        port = server.server_address[1]
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3).read().decode()
        assert "<title>Pick the better draft · Agencie.io Labs</title>" in page
        a = post(port, "/api/draft", {"prompt": "Write about holds."})
        b = post(port, "/api/draft", {"prompt": "Write about holds."})
        assert a["text"].startswith("seed ") and a["seed"] != b["seed"]
        info = post(
            port,
            "/api/choice",
            {
                "prompt": "Write about holds.",
                "a": a["text"],
                "b": b["text"],
                "choice": "b",
                "edited": "Mine.",
                "seeds": [a["seed"], b["seed"]],
            },
        )
        assert info["count"] == 1 and info["pairs"] == 2
        saved = read(out)[0]
        assert saved["system"] == "You are me." and saved["edited"] == "Mine."
        with pytest.raises(urllib.error.HTTPError) as err:
            post(port, "/api/choice", {"prompt": "x", "a": "y", "b": "z", "choice": "nope"})
        assert err.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as err:
            post(port, "/api/draft", {"prompt": "  "})
        assert err.value.code == 502
    finally:
        server.shutdown()
        server.server_close()


def test_pairs_from_the_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "preferences.jsonl"
    record(out, verdict("a"))
    record(out, verdict("same"))
    assert main(["pick", "--pairs", "dpo.jsonl"]) == 0
    lines = (tmp_path / "dpo.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["chosen"] == "Draft one."
    assert "1 pairs from 2 verdicts" in capsys.readouterr().out
