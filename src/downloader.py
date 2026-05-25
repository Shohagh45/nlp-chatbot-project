from __future__ import annotations

from pathlib import Path
from typing import Optional

from .utils import Document, MANIFEST_PATH, RAW_DIR, get_logger, save_json

log = get_logger(__name__)


class Downloader:
    def __init__(
        self,
        excel_path: Path,
        raw_dir: Path = RAW_DIR,
        timeout: int = 30,
        user_agent: str = "Mozilla/5.0 (NLP-Chatbot-Project Educational Crawler)",
    ) -> None:
        self.excel_path = Path(excel_path)
        self.raw_dir = Path(raw_dir)
        self.timeout = timeout
        self.user_agent = user_agent

    def load_manifest(self) -> list[Document]:
        """Read the Excel file and return one Document per row."""
        raise NotImplementedError

    def fetch_one(self, doc: Document) -> Document:
        """Download one document and set local_path, fetch_status, fetch_error."""
        raise NotImplementedError

    def download_all(self, docs: list[Document]) -> list[Document]:
        results = []
        for doc in docs:
            try:
                results.append(self.fetch_one(doc))
            except Exception as exc:
                doc.fetch_status = "failed"
                doc.fetch_error = repr(exc)
                results.append(doc)
                log.exception("Unhandled error fetching %s", doc.id)
        return results

    def save_manifest(self, docs: list[Document], path: Path = MANIFEST_PATH) -> None:
        save_json(docs, path)
        log.info("Wrote manifest with %d documents -> %s", len(docs), path)


def main(excel_path: Optional[str] = None) -> None:
    import sys
    excel_path = excel_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not excel_path:
        raise SystemExit("Usage: python -m src.downloader <path-to-excel>")
    dl = Downloader(Path(excel_path))
    docs = dl.load_manifest()
    docs = dl.download_all(docs)
    dl.save_manifest(docs)
    ok = sum(1 for d in docs if d.fetch_status == "success")
    log.info("Done: %d / %d documents fetched successfully.", ok, len(docs))


if __name__ == "__main__":
    main()
