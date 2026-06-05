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
        import pandas as pd
        import unicodedata

        df = pd.read_excel(self.excel_path, sheet_name=0, dtype=str)
        # Normalise column names to lowercase and strip whitespace
        df.columns = [c.strip().lower() for c in df.columns]

        docs: list[Document] = []
        for _, row in df.iterrows():
            def clean(val: object) -> str:
                if not isinstance(val, str) or val in ("nan", "NaN"):
                    return ""
                # Collapse internal newlines/tabs to a single space
                val = " ".join(val.split())
                # Replace non-standard Unicode hyphens/dashes with an ASCII hyphen
                val = "".join(
                    "-" if unicodedata.category(ch) in ("Pd",) and ch != "-" else ch
                    for ch in val
                )
                return val.strip()

            docs.append(Document(
                id=clean(row.get("id", "")),
                title=clean(row.get("title", "")),
                url=clean(row.get("url", "")),
                publisher=clean(row.get("publisher", "")),
                date=clean(row.get("date", "")),
                theme=clean(row.get("theme", "")),
                keywords=clean(row.get("keywords", "")),
                citation=clean(row.get("bronvermelding", "")),
            ))

        log.info("Loaded %d documents from %s", len(docs), self.excel_path)
        return docs

    def fetch_one(self, doc: Document) -> Document:
        """Download one document and set local_path, fetch_status, fetch_error."""
        import requests

        is_pdf = doc.url.lower().split("?")[0].endswith(".pdf")
        ext = "pdf" if is_pdf else "html"
        doc.content_type = ext
        local_path = self.raw_dir / f"{doc.id}.{ext}"
        doc.local_path = str(local_path)

        if local_path.exists():
            doc.fetch_status = "skipped"
            log.debug("Skipped (already exists): %s", local_path)
            return doc

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.headers["User-Agent"] = self.user_agent
        try:
            r = session.get(doc.url, timeout=self.timeout)
            r.raise_for_status()
            local_path.write_bytes(r.content)
            doc.fetch_status = "success"
            log.info("Fetched %s -> %s", doc.id, local_path.name)
        except Exception as exc:
            doc.fetch_status = "failed"
            doc.fetch_error = str(exc)
            log.warning("Failed %s: %s", doc.id, exc)
        return doc

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
