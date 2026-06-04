"""
Tests for src/preprocessing.py (Elias — text extraction & chunking).

These run entirely offline: no Ollama, no downloaded dataset, no dependency on
Hugo's downloader. They exercise HTML extraction, text cleaning, and the
chunking logic on small in-memory samples.
"""

from __future__ import annotations

from src.preprocessing import Preprocessor
from src.utils import Chunk, Document


def _doc(**overrides) -> Document:
    base = dict(
        id="R01-01",
        title="Test Policy Document",
        url="https://example.eu/doc.pdf",
        publisher="EPRS",
        date="2025",
        theme="Trust",
        keywords="trust, fairness",
        citation="EPRS, 2025.",
    )
    base.update(overrides)
    return Document(**base)


# ----------------------------------------------------------------------------
# clean()
# ----------------------------------------------------------------------------
def test_clean_removes_soft_hyphens_and_ligatures():
    pre = Preprocessor()
    dirty = "trans­parency and eﬃciency in artiﬁcial intelligence"
    cleaned = pre.clean(dirty)
    assert "­" not in cleaned
    assert "transparency" in cleaned
    assert "efficiency" in cleaned
    assert "artificial" in cleaned


def test_clean_rejoins_hyphenated_line_breaks():
    pre = Preprocessor()
    cleaned = pre.clean("The regulation is transpar-\nency focused.")
    assert "transparency" in cleaned
    assert "transpar-" not in cleaned


def test_clean_collapses_blank_lines_but_keeps_paragraphs():
    pre = Preprocessor()
    cleaned = pre.clean("Para one.\n\n\n\n\nPara two.")
    # Multiple blank lines collapse to a single blank line (one paragraph break).
    assert cleaned == "Para one.\n\nPara two."


def test_clean_handles_empty():
    pre = Preprocessor()
    assert pre.clean("") == ""


# ----------------------------------------------------------------------------
# extract_html()
# ----------------------------------------------------------------------------
def test_extract_html_strips_boilerplate(tmp_path):
    html = """
    <html><head><style>body{color:red}</style></head>
    <body>
      <nav>Home | About | Menu</nav>
      <header>Site header</header>
      <main>
        <h1>AI Act Overview</h1>
        <p>The AI Act sets out rules for trustworthy artificial intelligence.</p>
      </main>
      <footer>Copyright 2025</footer>
      <script>console.log('tracking')</script>
    </body></html>
    """
    path = tmp_path / "page.html"
    path.write_text(html, encoding="utf-8")

    pre = Preprocessor()
    text = pre.extract_html(path)

    assert "AI Act sets out rules" in text
    assert "tracking" not in text          # script removed
    assert "Site header" not in text       # header removed
    assert "Copyright 2025" not in text    # footer removed
    assert "Menu" not in text              # nav removed


# ----------------------------------------------------------------------------
# chunk_text()
# ----------------------------------------------------------------------------
def test_chunk_text_basic_metadata_and_ids():
    pre = Preprocessor(chunk_size=50, chunk_overlap=10, min_chunk_chars=20)
    doc = _doc()
    # Three paragraphs of ~40 words each -> should pack into a couple of chunks.
    para = " ".join(["word%d" % i for i in range(40)])
    text = "\n\n".join([para, para, para])

    chunks = pre.chunk_text(doc, text)

    assert len(chunks) >= 1
    for i, ch in enumerate(chunks):
        assert isinstance(ch, Chunk)
        assert ch.id == f"{doc.id}::{i}"          # ids are contiguous
        assert ch.doc_id == doc.id
        assert ch.metadata["chunk_index"] == i
        for key in ("title", "publisher", "url", "theme"):
            assert key in ch.metadata
        assert len(ch.text) >= pre.min_chunk_chars


def test_chunk_text_respects_size_and_overlap():
    pre = Preprocessor(chunk_size=100, chunk_overlap=20, min_chunk_chars=20)
    doc = _doc()
    # One long paragraph of 350 words forces multiple chunks via sentence/word split.
    words = " ".join("token%d" % i for i in range(350))
    chunks = pre.chunk_text(doc, words)

    assert len(chunks) >= 3
    for ch in chunks:
        # No chunk should be wildly larger than chunk_size + overlap.
        assert len(ch.text.split()) <= pre.chunk_size + pre.chunk_overlap

    # Consecutive chunks should share some overlapping tokens.
    first_tail = set(chunks[0].text.split()[-pre.chunk_overlap:])
    second_words = set(chunks[1].text.split())
    assert first_tail & second_words, "expected overlap between consecutive chunks"


def test_chunk_text_drops_tiny_fragments():
    pre = Preprocessor(chunk_size=500, chunk_overlap=50, min_chunk_chars=100)
    doc = _doc()
    chunks = pre.chunk_text(doc, "Too short.")
    assert chunks == []


def test_chunk_text_empty_input():
    pre = Preprocessor()
    assert pre.chunk_text(_doc(), "") == []
    assert pre.chunk_text(_doc(), "   \n\n   ") == []
