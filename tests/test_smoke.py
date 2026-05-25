"""
Basic smoke tests — verify the skeleton imports cleanly and the shared
dataclasses behave. Each role should add their own tests in this directory.
"""

from pathlib import Path

import pytest


def test_all_modules_import():
    """The whole package should import without errors, even with stubs."""
    import src
    from src import chat, downloader, embeddings, preprocessing, utils, vector_db

    # Just touch the names so unused-import linters don't complain
    assert all([src, chat, downloader, embeddings, preprocessing, utils, vector_db])


def test_document_dataclass_roundtrip(tmp_path: Path):
    from src.utils import Document, load_json, save_json

    doc = Document(
        id="R01-01",
        title="Test Doc",
        url="https://example.com/test.pdf",
        publisher="EPRS",
        date="2025",
        theme="Trust",
        keywords="trust, fairness",
        citation="EPRS, 2025.",
        content_type="pdf",
        local_path="data/raw/R01-01.pdf",
        fetch_status="success",
    )
    path = tmp_path / "doc.json"
    save_json(doc, path)
    loaded = Document(**load_json(path))
    assert loaded == doc


def test_chunk_dataclass_roundtrip(tmp_path: Path):
    from src.utils import Chunk, load_json, save_json

    chunks = [
        Chunk(
            id=f"R01-01::{i}",
            doc_id="R01-01",
            text=f"chunk {i} text",
            metadata={"title": "Test", "publisher": "EPRS", "chunk_index": i},
        )
        for i in range(3)
    ]
    path = tmp_path / "chunks.json"
    save_json(chunks, path)
    loaded = [Chunk(**c) for c in load_json(path)]
    assert loaded == chunks


def test_paths_are_correct():
    """Sanity-check that utils' path constants point inside the project."""
    from src.utils import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, VECTOR_STORE_DIR

    assert PROJECT_ROOT.exists()
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DIR == DATA_DIR / "raw"
    assert PROCESSED_DIR == DATA_DIR / "processed"
    assert VECTOR_STORE_DIR == DATA_DIR / "vector_store"


def test_stubs_raise_not_implemented():
    """Stubs should raise NotImplementedError, not silently return None.

    This catches the case where someone deletes the `raise` and forgets
    to implement the method.
    """
    import numpy as np
    from src.downloader import Downloader
    from src.embeddings import EmbeddingClient
    from src.preprocessing import Preprocessor
    from src.utils import Document, Chunk
    from src.vector_db import VectorDB

    dl = Downloader(excel_path=Path("nonexistent.xlsx"))
    with pytest.raises(NotImplementedError):
        dl.load_manifest()

    emb = EmbeddingClient()
    with pytest.raises(NotImplementedError):
        emb.embed("test")

    pre = Preprocessor()
    with pytest.raises(NotImplementedError):
        pre.clean("test")

    db = VectorDB()
    with pytest.raises(NotImplementedError):
        db.add([], np.array([]))
    with pytest.raises(NotImplementedError):
        db.search(np.zeros(768))
    with pytest.raises(NotImplementedError):
        db.persist()
    with pytest.raises(NotImplementedError):
        db.load()


def test_chat_help_text_is_non_empty():
    """Helper that doesn't require Ollama — make sure the HELP_TEXT stays
    informative."""
    from src.chat import ChatSession

    assert "/help" in ChatSession.HELP_TEXT
    assert "/reset" in ChatSession.HELP_TEXT
    assert len(ChatSession.HELP_TEXT.splitlines()) >= 3
