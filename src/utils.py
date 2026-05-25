"""
src/utils.py
============
Shared helpers used across the project: logging, dataclasses, JSON I/O.

OWNER: Role 5 (Integration / Eval / Docs)
But every other role will add small helpers here as needed. Keep it small.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a consistently-formatted logger.

    Use this in every module instead of print() so we get timestamps and
    can adjust verbosity from app.py.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# Shared dataclasses — the contract between modules
# ----------------------------------------------------------------------------

@dataclass
class Document:
    """One source document from the Excel manifest.

    Populated incrementally:
      1. downloader.py reads metadata from the Excel  ->  fields 1..8 set
      2. downloader.py fetches the URL                ->  local_path, status set
      3. preprocessing.py extracts text               ->  uses local_path
    """
    id: str                       # e.g. "R01-01"
    title: str
    url: str
    publisher: str
    date: str
    theme: str
    keywords: str
    citation: str                 # "bronvermelding" column
    # Populated by the downloader:
    content_type: Optional[str] = None   # "pdf" | "html"
    local_path: Optional[str] = None     # path under data/raw/
    fetch_status: str = "pending"        # "success" | "failed" | "skipped"
    fetch_error: Optional[str] = None


@dataclass
class Chunk:
    """One retrievable unit of text.

    The metadata dict is intentionally flexible so the chunking team can
    add fields (page number, section heading, etc.) without breaking the
    embedding/vector-db code.
    """
    id: str                       # e.g. "R01-01::0"
    doc_id: str                   # back-pointer, e.g. "R01-01"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# JSON I/O — convenience around dataclasses
# ----------------------------------------------------------------------------

def save_json(obj: Any, path: Path) -> None:
    """Write any JSON-serialisable object (or list of dataclasses) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, list) and obj and hasattr(obj[0], "__dataclass_fields__"):
        payload = [asdict(item) for item in obj]
    elif hasattr(obj, "__dataclass_fields__"):
        payload = asdict(obj)
    else:
        payload = obj
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Common paths — one place to change if structure moves
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

MANIFEST_PATH = RAW_DIR / "manifest.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
