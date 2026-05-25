from __future__ import annotations

import numpy as np

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
        """Lazy-initialise the OpenAI SDK client pointed at Ollama."""
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string, returns shape (D,)."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings, returns shape (N, D)."""
        raise NotImplementedError

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
