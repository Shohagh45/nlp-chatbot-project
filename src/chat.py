from __future__ import annotations

from typing import Optional

from .embeddings import EmbeddingClient
from .utils import Chunk, get_logger
from .vector_db import VectorDB

log = get_logger(__name__)


SYSTEM_PROMPT = """\
You are a reflective conversational assistant grounded in European Union
policy on artificial intelligence (the AI Act, EDPS/EDPB guidance, JRC
reports, and European Parliament briefings).

YOUR MANNER
- Calm, thoughtful, deliberately restrained. You do not lecture.
- You acknowledge what the user feels or experiences before discussing rules
  or evidence.
- You ask short clarifying questions when the user's intent is unclear.
- You are honest about your limits. You don't have memory across sessions.
  You don't have moral agency. You can be wrong.

WHEN ANSWERING
- You will be given excerpts retrieved from official EU policy documents.
  Treat them as authoritative on the policy questions they cover.
- If the excerpts do NOT address the user's question, say so plainly rather
  than inventing.
- Cite sources by document title and publisher. Inline citations like
  "(EDPS, 2025)" are fine; a "Sources" list at the end is also fine.
- Prefer short, substantive answers over long info dumps.
"""

USER_PROMPT_TEMPLATE = """\
The user said:
\"\"\"{user_message}\"\"\"

Relevant excerpts from EU AI policy documents (retrieved by semantic search):
{context_block}

Reply in your usual manner. Cite the sources you used.
"""


def format_context_block(retrieved: list[tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "(no relevant excerpts found — be honest about this)"
    blocks = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        title = chunk.metadata.get("title", "Unknown")
        publisher = chunk.metadata.get("publisher", "Unknown")
        blocks.append(
            f"[{i}] {title} — {publisher} (similarity={score:.2f})\n{chunk.text}"
        )
    return "\n\n".join(blocks)


class ChatSession:
    DEFAULT_LLM_MODEL = "granite4.1"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    HELP_TEXT = """\
Commands:
  /help    Show this message
  /reset   Forget the conversation so far (keeps the document index)
  /quit    Exit

This assistant is grounded in EU AI policy documents. Ask about trust,
fairness, transparency, accountability, deepfakes, workplace AI, healthcare
AI, and similar themes. The assistant will cite the documents it relied on.
"""

    def __init__(
        self,
        vector_db: VectorDB,
        embedder: EmbeddingClient,
        llm_model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        top_k: int = 6,
        history_window: int = 8,
        score_threshold: float = 0.25,
    ) -> None:
        self.db = vector_db
        self.embedder = embedder
        self.llm_model = llm_model
        self.base_url = base_url
        self.top_k = top_k
        self.history_window = history_window
        self.score_threshold = score_threshold
        self.history: list[dict[str, str]] = []
        self._llm = None

    def _get_llm(self):
        """Lazy-initialise the OpenAI SDK client pointed at Ollama."""
        raise NotImplementedError

    def _retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Embed the query and search the vector DB for relevant chunks."""
        raise NotImplementedError

    def _build_messages(
        self, user_message: str, retrieved: list[tuple[Chunk, float]]
    ) -> list[dict[str, str]]:
        """Assemble the full message list for the LLM call."""
        raise NotImplementedError

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM and return the reply text."""
        raise NotImplementedError

    def handle(self, user_message: str) -> tuple[str, list[tuple[Chunk, float]]]:
        retrieved = self._retrieve(user_message)
        messages = self._build_messages(user_message, retrieved)
        reply = self._generate(messages)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        return reply, retrieved

    def reset(self) -> None:
        self.history.clear()
        log.info("Conversation history cleared.")

    def help(self) -> str:
        return self.HELP_TEXT
