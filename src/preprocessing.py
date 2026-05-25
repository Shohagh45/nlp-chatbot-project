from __future__ import annotations

from pathlib import Path
from typing import Optional

from .utils import CHUNKS_PATH, MANIFEST_PATH, PROCESSED_DIR, Chunk, Document, get_logger, load_json, save_json

log = get_logger(__name__)


class Preprocessor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_chars: int = 100,
    ) -> None:
        # chunk_size is in words. 500 words ≈ 667 tokens, well within nomic-embed-text's limit.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        raise NotImplementedError

    def extract_html(self, path: Path) -> str:
        """Extract main content text from an HTML file using BeautifulSoup."""
        raise NotImplementedError

    def extract_text(self, doc: Document) -> str:
        if not doc.local_path:
            return ""
        path = Path(doc.local_path)
        if doc.content_type == "pdf":
            return self.extract_pdf(path)
        if doc.content_type == "html":
            return self.extract_html(path)
        log.warning("Unknown content_type for %s: %s", doc.id, doc.content_type)
        return ""

    def clean(self, text: str) -> str:
        """Normalise whitespace, fix ligatures, strip soft hyphens."""
        raise NotImplementedError

    def chunk_text(self, doc: Document, text: str) -> list[Chunk]:
        """Split cleaned text into overlapping chunks with metadata."""
        raise NotImplementedError

    def process_all(self, docs: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            if doc.fetch_status != "success":
                continue
            try:
                text = self.extract_text(doc)
                text = self.clean(text)
                chunks = self.chunk_text(doc, text)
                all_chunks.extend(chunks)
                log.info("[%s] %d chunks", doc.id, len(chunks))
            except Exception:
                log.exception("Failed to process %s", doc.id)
        return all_chunks


def main() -> None:
    raw_manifest = load_json(MANIFEST_PATH)
    docs = [Document(**d) for d in raw_manifest]
    pre = Preprocessor()
    chunks = pre.process_all(docs)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(chunks, CHUNKS_PATH)
    log.info("Wrote %d chunks -> %s", len(chunks), CHUNKS_PATH)


if __name__ == "__main__":
    main()
