"""
src/database.py
===============
SQLite connection layer for document metadata.

OWNER: Role 5 — Wouter
PURPOSE: Load the Excel manifest into a local SQLite database so any module
can query document metadata (title, publisher, theme, URL, fetch status)
without re-reading the Excel file every time.

This is NOT the vector store (that stays in .npz + JSON as required).
This is a lightweight index of the 87 EU policy documents.

DATABASE
--------
    data/documents.db
    └── table: documents  (one row per EU policy document)

USAGE
-----
    # One-time load from Excel:
    python -m src.database 20260331_dataset2.xlsx

    # Query from anywhere in the codebase:
    from src.database import DocumentDB
    db = DocumentDB()
    docs = db.all()                          # all 87 documents
    docs = db.by_theme("Healthcare AI")      # filter by theme
    doc  = db.get("R01-01")                  # single document by ID
    stats = db.fetch_stats()                 # how many succeeded/failed
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import DATA_DIR, get_logger

log = get_logger(__name__)

DB_PATH = DATA_DIR / "documents.db"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIGATURES = str.maketrans({
    "ﬁ": "fi", "ﬀ": "ff", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
})


def _clean(value: object) -> str:
    """Coerce an Excel cell to a clean string."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    text = text.replace("­", "")    # soft hyphen
    text = text.replace("‑", "-")   # non-breaking hyphen
    text = text.replace(" ", " ")   # non-breaking space
    text = text.translate(_LIGATURES)
    return " ".join(text.split())        # collapse whitespace


# ---------------------------------------------------------------------------
# DocumentDB
# ---------------------------------------------------------------------------

class DocumentDB:
    """Thin wrapper around a SQLite database of EU policy document metadata."""

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS documents (
        id            TEXT PRIMARY KEY,
        title         TEXT,
        url           TEXT,
        publisher     TEXT,
        date          TEXT,
        theme         TEXT,
        keywords      TEXT,
        citation      TEXT,
        content_type  TEXT,
        local_path    TEXT,
        fetch_status  TEXT DEFAULT 'pending',
        fetch_error   TEXT
    )
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with row_factory so rows come back as dicts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_from_excel(self, excel_path: Path) -> int:
        """Read the Excel file and insert/replace all rows into SQLite.

        Returns the number of documents loaded.
        """
        excel_path = Path(excel_path)
        if not excel_path.exists():
            raise FileNotFoundError(
                f"Excel file not found: {excel_path}\n"
                "Place 20260331_dataset2.xlsx in the project root."
            )

        log.info("Reading Excel: %s", excel_path)
        df = pd.read_excel(excel_path, sheet_name=0, dtype=str)
        df.columns = [str(c).strip().lower() for c in df.columns]
        log.info("Columns: %s", list(df.columns))

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(self.CREATE_SQL)
            count = 0
            for _, row in df.iterrows():
                doc_id = _clean(row.get("id", ""))
                if not doc_id:
                    continue
                doc_id = doc_id.replace("‑", "-")  # non-breaking hyphen in IDs

                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                        (id, title, url, publisher, date, theme, keywords, citation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        _clean(row.get("title", "")),
                        _clean(row.get("url", "")).rstrip(),
                        _clean(row.get("publisher", "")),
                        _clean(row.get("date", "")),
                        _clean(row.get("theme", "")),
                        _clean(row.get("keywords", "")),
                        _clean(row.get("citation", row.get("bronvermelding", ""))),
                    ),
                )
                count += 1
            conn.commit()
            log.info("Loaded %d documents into %s", count, self.db_path)
            return count
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Update fetch status (called by downloader after fetching)
    # ------------------------------------------------------------------
    def update_fetch_status(
        self,
        doc_id: str,
        fetch_status: str,
        local_path: Optional[str] = None,
        content_type: Optional[str] = None,
        fetch_error: Optional[str] = None,
    ) -> None:
        """Update the fetch result for one document after downloading."""
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE documents
                SET fetch_status = ?, local_path = ?, content_type = ?, fetch_error = ?
                WHERE id = ?
                """,
                (fetch_status, local_path, content_type, fetch_error, doc_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def all(self) -> list[dict]:
        """Return all documents as a list of dicts."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get(self, doc_id: str) -> Optional[dict]:
        """Return one document by ID, or None if not found."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def by_theme(self, theme: str) -> list[dict]:
        """Return all documents matching a theme (case-insensitive partial match)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM documents WHERE theme LIKE ? ORDER BY id",
                (f"%{theme}%",),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def by_publisher(self, publisher: str) -> list[dict]:
        """Return all documents from a publisher (case-insensitive partial match)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM documents WHERE publisher LIKE ? ORDER BY id",
                (f"%{publisher}%",),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def fetch_stats(self) -> dict:
        """Return a summary of fetch statuses: {status: count}."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT fetch_status, COUNT(*) as n FROM documents GROUP BY fetch_status"
            ).fetchall()
            return {r["fetch_status"]: r["n"] for r in rows}
        finally:
            conn.close()

    def pending(self) -> list[dict]:
        """Return documents not yet downloaded."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM documents WHERE fetch_status = 'pending' ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def __len__(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        finally:
            conn.close()

    def __repr__(self) -> str:
        return f"<DocumentDB path={self.db_path} docs={len(self)}>"


# ---------------------------------------------------------------------------
# CLI — load the Excel into SQLite and print a summary
# ---------------------------------------------------------------------------
def main() -> None:
    import sys

    excel_path = sys.argv[1] if len(sys.argv) > 1 else "20260331_dataset2.xlsx"
    db = DocumentDB()
    count = db.load_from_excel(Path(excel_path))

    print(f"\n{'='*50}")
    print(f"  Loaded {count} documents into {db.db_path}")
    print(f"{'='*50}")

    # Show a sample
    docs = db.all()
    print(f"\n{'ID':<12} {'THEME':<30} {'PUBLISHER':<20} TITLE")
    print("-" * 90)
    for doc in docs[:10]:
        print(
            f"{doc['id']:<12} "
            f"{(doc['theme'] or '')[:28]:<30} "
            f"{(doc['publisher'] or '')[:18]:<20} "
            f"{(doc['title'] or '')[:40]}"
        )
    if count > 10:
        print(f"  ... and {count - 10} more.")

    # Theme breakdown
    print("\nThemes in the dataset:")
    themes: dict[str, int] = {}
    for doc in docs:
        t = doc["theme"] or "Unknown"
        themes[t] = themes.get(t, 0) + 1
    for theme, n in sorted(themes.items(), key=lambda x: -x[1]):
        print(f"  {n:>3}x  {theme}")


if __name__ == "__main__":
    main()
