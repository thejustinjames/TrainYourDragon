import json

from conftest import EXAMPLE_CONFIG
from dragon import corpus, dataset
from dragon.config import Config


def built():
    config = Config.load(EXAMPLE_CONFIG)
    return config, dataset.build(config, corpus.load(config))


def test_every_example_is_a_three_message_exchange():
    config, data = built()
    for example in data.train + data.valid + data.test:
        roles = [m["role"] for m in example.messages]
        assert roles == ["system", "user", "assistant"]
        assert example.messages[0]["content"] == config.system_prompt
        assert example.messages[1]["content"] and example.messages[2]["content"]


def test_the_held_out_piece_is_only_in_the_test_split():
    _, data = built()
    quiet = "Held back from training on purpose"
    assert any(quiet in e.messages[2]["content"] for e in data.test)
    assert not any(
        "Holding something back" in e.messages[2]["content"] for e in data.train + data.valid
    )


def test_a_held_out_piece_contributes_no_recall_pairs():
    _, data = built()
    assert not any(e.tag == "recall" for e in data.test)


def test_recall_pairs_ask_by_name_alone():
    _, data = built()
    recalls = [e for e in data.train + data.valid if e.tag == "recall"]
    assert recalls
    ferry = next(e for e in recalls if "Ferry" in e.messages[1]["content"])
    assert len(ferry.messages[1]["content"]) < 80  # the brief is only the question
    assert "plumbing" in ferry.messages[2]["content"]  # the answer is what the corpus says


def test_recall_pairs_can_be_capped():
    config = Config.load(EXAMPLE_CONFIG)
    docs = corpus.load(config)
    config.recall = 1
    one = dataset.build(config, docs)
    config.recall = True
    many = dataset.build(config, docs)
    count = lambda d: sum(e.tag == "recall" for e in d.train + d.valid)  # noqa: E731
    assert count(one) < count(many)


def test_notes_pairs_hand_over_notes_and_expect_prose():
    _, data = built()
    notes = next(e for e in data.train + data.valid if e.tag == "notes")
    brief, answer = notes.messages[1]["content"], notes.messages[2]["content"]
    assert brief.startswith("Turn these dictated notes")
    assert brief.count("\n- ") >= 2
    assert len(answer.split()) > len(brief.split())


def test_no_example_is_longer_than_the_configured_maximum():
    config, data = built()
    for example in data.train:
        assert len(example.messages[2]["content"].split()) <= config.max_words * 1.1


def test_writing_produces_readable_jsonl(tmp_path):
    _, data = built()
    data.write(tmp_path)
    lines = (tmp_path / "train.jsonl").read_text().splitlines()
    assert len(lines) == len(data.train)
    assert json.loads(lines[0])["messages"][0]["role"] == "system"
    assert (tmp_path / "valid.jsonl").exists() and (tmp_path / "test.jsonl").exists()


def test_the_build_is_reproducible():
    first, second = built()[1], built()[1]
    assert [e.json() for e in first.train] == [e.json() for e in second.train]
