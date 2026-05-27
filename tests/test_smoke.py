"""
Basic smoke tests — verify imports, dataclasses, and implemented modules work.
"""

from pathlib import Path

import numpy as np
import pytest


def test_all_modules_import():
    import src
    from src import chat, database, downloader, embeddings, preprocessing, utils, vector_db
    assert all([src, chat, database, downloader, embeddings, preprocessing, utils, vector_db])


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
    from src.utils import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, VECTOR_STORE_DIR

    assert PROJECT_ROOT.exists()
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DIR == DATA_DIR / "raw"
    assert PROCESSED_DIR == DATA_DIR / "processed"
    assert VECTOR_STORE_DIR == DATA_DIR / "vector_store"


def test_remaining_stubs_raise_not_implemented():
    """Hugo, Elias and Ksawery's modules are still stubs."""
    from src.downloader import Downloader
    from src.preprocessing import Preprocessor

    dl = Downloader(excel_path=Path("nonexistent.xlsx"))
    with pytest.raises(NotImplementedError):
        dl.load_manifest()

    pre = Preprocessor()
    with pytest.raises(NotImplementedError):
        pre.clean("test")


def test_vector_db_add_and_search():
    """VectorDB works in memory without Ollama."""
    from src.utils import Chunk
    from src.vector_db import VectorDB

    chunks = [
        Chunk(id="t1", doc_id="d1", text="AI transparency is important.", metadata={}),
        Chunk(id="t2", doc_id="d2", text="Algorithmic bias affects fairness.", metadata={}),
        Chunk(id="t3", doc_id="d3", text="Human oversight ensures safety.", metadata={}),
    ]
    embeddings = np.random.rand(3, 768).astype(np.float32)
    # L2-normalise so cosine = dot product
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    db = VectorDB()
    db.add(chunks, embeddings)

    assert len(db) == 3

    # Search with the first embedding — should return itself as top result
    results = db.search(embeddings[0], top_k=2)
    assert len(results) == 2
    assert results[0][0].id == "t1"
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_vector_db_persist_and_load(tmp_path: Path):
    """Saving and reloading the DB returns identical results."""
    from src.utils import Chunk
    from src.vector_db import VectorDB

    chunks = [Chunk(id=f"c{i}", doc_id="d1", text=f"text {i}", metadata={}) for i in range(5)]
    embeddings = np.random.rand(5, 768).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    db = VectorDB(store_dir=tmp_path)
    db.add(chunks, embeddings)
    db.persist()

    db2 = VectorDB(store_dir=tmp_path)
    db2.load()

    assert len(db2) == 5
    np.testing.assert_array_almost_equal(db.embeddings, db2.embeddings)
    assert [c.id for c in db2.chunks] == [c.id for c in chunks]


def test_chat_help_text_is_non_empty():
    from src.chat import ChatSession

    assert "/help" in ChatSession.HELP_TEXT
    assert "/reset" in ChatSession.HELP_TEXT
    assert len(ChatSession.HELP_TEXT.splitlines()) >= 3
