from conftest import EXAMPLE_CONFIG
from dragon import corpus
from dragon.config import Config


def documents():
    config = Config.load(EXAMPLE_CONFIG)
    return config, corpus.load(config)


def test_every_source_finds_something():
    config, docs = documents()
    for source in config.sources:
        assert [d for d in docs if d.source is source], source.name


def test_frontmatter_fields_are_mapped_by_name():
    _, docs = documents()
    lantern = next(d for d in docs if d.key == "lantern")
    assert lantern.title == "Lantern"  # from `name`, not `title`
    assert lantern.tagline.startswith("A reading light")
    assert "Category: Tools" in lantern.meta_line
    assert "Answers carry the passage" in lantern.extras


def test_holdout_is_marked_and_nothing_else_is():
    _, docs = documents()
    assert [d.key for d in docs if d.holdout] == ["the-quiet-part"]


def test_terms_are_read_as_pairs():
    _, docs = documents()
    glossary = next(d for d in docs if d.terms)
    assert len(glossary.terms) == 3
    assert glossary.terms[0][0] == "Goodhart's law"
