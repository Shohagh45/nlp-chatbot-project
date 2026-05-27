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
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings"
            )
        embeddings = embeddings.astype(np.float32)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)
        log.info("Added %d chunks. Total: %d", len(chunks), len(self.chunks))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Cosine similarity = dot product (embeddings are pre-normalised)
        scores = self.embeddings @ query

        k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= score_threshold:
                results.append((self.chunks[idx], score))

        return results

    def persist(self) -> None:
        if self.embeddings is None:
            raise RuntimeError("Nothing to persist — call add() first.")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.store_dir / self.EMBEDDINGS_FILENAME,
            embeddings=self.embeddings,
        )
        save_json(self.chunks, self.store_dir / self.CHUNKS_FILENAME)
        log.info(
            "Persisted %d chunks to %s", len(self.chunks), self.store_dir
        )

    def load(self) -> None:
        emb_path = self.store_dir / self.EMBEDDINGS_FILENAME
        chunks_path = self.store_dir / self.CHUNKS_FILENAME

        if not emb_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.store_dir}. "
                "Run: python -m src.vector_db"
            )

        data = np.load(emb_path)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.chunks = [Chunk(**c) for c in load_json(chunks_path)]
        log.info(
            "Loaded %d chunks, embeddings shape %s",
            len(self.chunks),
            self.embeddings.shape,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        n = len(self)
        d = self.embeddings.shape[1] if self.embeddings is not None else "?"
        return f"<VectorDB n_chunks={n} dim={d} store={self.store_dir}>"


def build_vector_store() -> None:
    from .embeddings import EmbeddingClient

    log.info("Loading chunks from %s", CHUNKS_PATH)
    chunks = [Chunk(**c) for c in load_json(CHUNKS_PATH)]
    log.info("Loaded %d chunks.", len(chunks))

    embedder = EmbeddingClient()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts)

    db = VectorDB()
    db.add(chunks, embeddings)
    db.persist()
    log.info("Vector store built successfully.")


if __name__ == "__main__":
    build_vector_store()
