from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .utils import Chunk, CHUNKS_PATH, VECTOR_STORE_DIR, get_logger, load_json, save_json

log = get_logger(__name__)


class VectorDB:
    """In-memory vector index backed by .npz + JSON files on disk."""

    EMBEDDINGS_FILENAME = "embeddings.npz"
    CHUNKS_FILENAME = "chunks.json"

    def __init__(self, store_dir: Path = VECTOR_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Append chunks and their embeddings to the index."""
        raise NotImplementedError

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks with their similarity scores."""
        raise NotImplementedError

    def persist(self) -> None:
        """Save embeddings and chunks to disk."""
        raise NotImplementedError

    def load(self) -> None:
        """Load embeddings and chunks from disk."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        n = len(self)
        d = self.embeddings.shape[1] if self.embeddings is not None else "?"
        return f"<VectorDB n_chunks={n} dim={d} store={self.store_dir}>"


def build_vector_store() -> None:
    """Load chunks, embed them, and save the vector store."""
    raise NotImplementedError


if __name__ == "__main__":
    build_vector_store()
