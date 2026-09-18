from dragon import textutil as t


def test_frontmatter_splits_header_from_body():
    meta, body = t.frontmatter("---\ntitle: A Piece\nlang: en\n---\nFirst line.\n")
    assert meta == {"title": "A Piece", "lang": "en"}
    assert body.strip() == "First line."


def test_frontmatter_without_a_header_is_not_an_error():
    meta, body = t.frontmatter("No header here.\n")
    assert meta == {}
    assert body.startswith("No header")


def test_clean_strips_markup_but_keeps_the_words():
    src = (
        "<Figure src='x.png' />\n\n"
        "![alt](/img/a.png)\n\n"
        "A [link](https://example.com) and an &amp; entity.[^1]\n\n"
        "[^1]: A footnote.\n"
    )
    out = t.clean(src)
    assert "link" in out and "https://example.com" not in out
    assert "&amp;" not in out and "&" in out
    assert "footnote" not in out
    assert "img/a.png" not in out


def test_sections_splits_on_headings():
    body = "Opening words.\n\n## One\n\nFirst.\n\n## Two\n\nSecond."
    assert t.sections(body) == [
        (None, "Opening words."),
        ("One", "First."),
        ("Two", "Second."),
    ]


def test_chunk_never_splits_a_paragraph():
    text = "\n\n".join(["word " * 30] * 4)
    pieces = t.chunk(text, max_words=60)
    assert len(pieces) == 2
    assert all(t.words(p) <= 60 for p in pieces)


def test_notes_are_a_gist_not_the_opening_clause():
    text = (
        "A number on a slide is a decision that has already been made. The work was in "
        "choosing it.\n\nThis is not an argument against measurement. It is an argument "
        "for knowing which number you picked and why."
    )
    notes = t.notes_from(text).splitlines()
    assert len(notes) == 2
    for note, para in zip(notes, t.paragraphs(text)):
        gist = note[2:]
        assert gist == gist.lower()
        assert not para.lower().startswith(gist[:20])  # cannot be copied as an opening
        assert all(w not in t.STOPWORDS for w in gist.split())  # content words only
    assert "slide decision already made" in notes[0]


def test_notes_skip_lists_and_quotes():
    assert t.notes_from("- a bullet point here\n\n> a quotation of some length") == ""
