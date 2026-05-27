from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .utils import get_logger

log = get_logger(__name__)


class EmbeddingClient:
    """Wraps Ollama's nomic-embed-text model via the OpenAI-compatible SDK."""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    EXPECTED_DIM = 768

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        batch_size: int = 16,
        normalise: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.normalise = normalise
        self._dimension: int | None = None
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="ollama")
        return self._client

    def embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.EXPECTED_DIM, dtype=np.float32)
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=text)
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        if self.normalise:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        all_vecs: list[np.ndarray] = []
        batches = range(0, len(texts), self.batch_size)
        for i in tqdm(batches, desc="Embedding chunks", unit="batch"):
            batch = texts[i : i + self.batch_size]
            for text in batch:
                all_vecs.append(self.embed(text))
        return np.stack(all_vecs).astype(np.float32)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed("probe")
            self._dimension = int(probe.shape[0])
        return self._dimension

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
