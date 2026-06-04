from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .utils import CHUNKS_PATH, MANIFEST_PATH, PROCESSED_DIR, Chunk, Document, get_logger, load_json, save_json

log = get_logger(__name__)


# Common PDF ligature glyphs. These appear when a PDF's fonts are not embedded:
# the text "fi" is stored as a single code point (U+FB01) rather than the two
# letters "f" + "i". Embedding models treat them as out-of-vocabulary noise, so
# we expand them back into their plain ASCII equivalents.
LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

# Split on sentence-ending punctuation followed by whitespace. Deliberately
# simple — it does not need to be linguistically perfect, only good enough to
# avoid cutting a long paragraph mid-sentence when we have to fall back from
# paragraph-level splitting.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Preprocessor:
    """Turns raw downloaded files (PDF/HTML) into clean, overlapping text chunks.

    The pipeline per document is: extract_text -> clean -> chunk_text.
    Chunking is paragraph-preferred with a sentence-level fallback and a
    word-level last resort, producing ~`chunk_size`-word chunks that overlap by
    `chunk_overlap` words so that information straddling a boundary is never
    lost. See docs/nlp_approach.md (section 2) for the reasoning.
    """

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

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using pypdf, falling back to pdfplumber.

        pypdf is fast and handles most PDFs. When it returns very little text
        (< 200 chars — usually a scanned or awkwardly-encoded PDF), we retry
        with pdfplumber, which is slower but copes with more layouts.
        """
        text = ""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
        except Exception:
            log.exception("pypdf failed to read %s", path)
            text = ""

        if len(text.strip()) < 200:
            fallback = self._extract_pdf_pdfplumber(path)
            if len(fallback.strip()) > len(text.strip()):
                log.info("Using pdfplumber fallback for %s", path)
                text = fallback

        return text

    def _extract_pdf_pdfplumber(self, path: Path) -> str:
        """pdfplumber fallback for PDFs that pypdf handles poorly."""
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n\n".join(pages)
        except Exception:
            log.exception("pdfplumber fallback failed on %s", path)
            return ""

    def extract_html(self, path: Path) -> str:
        """Extract the main article text from an HTML file using BeautifulSoup.

        Strips boilerplate (scripts, styles, navigation, headers/footers) and
        prefers the semantic <main>/<article> region, falling back to <body>.
        """
        try:
            from bs4 import BeautifulSoup

            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(raw, "lxml")
            for tag in soup(
                ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
            ):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body
            return main.get_text(separator="\n") if main else ""
        except Exception:
            log.exception("Failed to extract HTML from %s", path)
            return ""

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

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean(self, text: str) -> str:
        """Normalise whitespace, fix ligatures, strip soft hyphens.

        Steps:
          1. Drop soft hyphens (U+00AD) — invisible word-break hints.
          2. Expand PDF ligatures (fi/fl/ff/ffi/ffl) back to plain letters.
          3. Re-join words hyphenated across a line break ("transpar-\\nency").
          4. Normalise non-breaking spaces/tabs, collapse runs of spaces.
          5. Strip each line, then collapse multiple blank lines into one so
             paragraph boundaries (a single blank line) survive for chunking.
        """
        if not text:
            return ""

        # 1. Soft hyphens.
        text = text.replace("­", "")

        # 2. Ligatures.
        for lig, repl in LIGATURES.items():
            text = text.replace(lig, repl)

        # 3. De-hyphenate words split across a line break. PDFs frequently break
        #    a word at the right margin as "transpar-\nency"; rejoin those.
        text = re.sub(r"(\w)-\n[ \t]*(\w)", r"\1\2", text)

        # 4. Normalise odd whitespace characters to plain spaces.
        text = text.replace(" ", " ").replace("\t", " ")

        # 5. Tidy line by line, then collapse blank-line runs.
        lines = [re.sub(r" {2,}", " ", ln).strip() for ln in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)  # 2+ blank lines -> one blank line

        return text.strip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def _split_paragraphs(self, text: str) -> list[str]:
        """Split cleaned text into paragraphs on blank lines."""
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        """Split a paragraph into sentences (best effort)."""
        return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]

    def _segment_words(self, text: str) -> list[list[str]]:
        """Break text into word-lists, each no larger than `chunk_size` words.

        Boundary preference (strongest first): paragraph -> sentence -> word.
        A paragraph that already fits becomes one segment. A paragraph that is
        too big is split into sentences; a single sentence that is *still* too
        big is hard-split on word boundaries as a last resort.
        """
        segments: list[list[str]] = []
        for para in self._split_paragraphs(text):
            words = para.split()
            if not words:
                continue
            if len(words) <= self.chunk_size:
                segments.append(words)
                continue

            # Paragraph too long: pack sentences up to chunk_size.
            buf: list[str] = []
            for sentence in self._split_sentences(para):
                sw = sentence.split()
                if len(sw) > self.chunk_size:
                    # A single very long sentence — flush, then hard-split it.
                    if buf:
                        segments.append(buf)
                        buf = []
                    for i in range(0, len(sw), self.chunk_size):
                        segments.append(sw[i : i + self.chunk_size])
                elif buf and len(buf) + len(sw) > self.chunk_size:
                    segments.append(buf)
                    buf = sw
                else:
                    buf.extend(sw)
            if buf:
                segments.append(buf)
        return segments

    def chunk_text(self, doc: Document, text: str) -> list[Chunk]:
        """Split cleaned text into overlapping chunks with metadata.

        Greedily packs whole segments (paragraphs/sentences) into chunks of
        about `chunk_size` words. When the next segment would overflow the
        current chunk, the chunk is emitted and the next one is seeded with the
        last `chunk_overlap` words of the previous chunk. Chunks shorter than
        `min_chunk_chars` (headers, page numbers, extraction debris) are dropped.
        """
        if not text or not text.strip():
            return []

        chunks: list[Chunk] = []
        current: list[str] = []

        def emit() -> None:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) < self.min_chunk_chars:
                return
            index = len(chunks)  # contiguous index; doubles as chunk_index
            chunks.append(
                Chunk(
                    id=f"{doc.id}::{index}",
                    doc_id=doc.id,
                    text=chunk_text,
                    metadata={
                        "title": doc.title,
                        "publisher": doc.publisher,
                        "url": doc.url,
                        "theme": doc.theme,
                        "date": doc.date,
                        "chunk_index": index,
                    },
                )
            )

        for segment in self._segment_words(text):
            if current and len(current) + len(segment) > self.chunk_size:
                emit()
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap else []
                current = overlap + segment
            else:
                current.extend(segment)

        if current:
            emit()

        return chunks

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
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