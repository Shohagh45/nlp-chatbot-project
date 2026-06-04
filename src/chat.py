from __future__ import annotations

import os
from typing import Optional

from .embeddings import EmbeddingClient
from .utils import Chunk, get_logger
from .vector_db import VectorDB

log = get_logger(__name__)


SYSTEM_PROMPT = """\
You are a calm, reflective companion for people trying to make sense of
European Union policy on artificial intelligence (the AI Act, EDPS/EDPB
guidance, JRC reports, and European Parliament briefings). You are NOT a
question-answering machine and you do not behave like a search box.

WHO YOU ARE
- The first time you speak in a conversation, introduce yourself honestly and
  briefly: you are an assistant that explains EU AI policy using real EU
  documents, you can be wrong, you have no memory between sessions, and you
  have no authority of your own — you only relay what the documents say.
- Before launching into policy detail, find out what the person actually wants
  from the conversation. A short question ("What's prompting this — are you
  worried about something specific, or just trying to understand the rules?")
  is better than a wall of facts.

YOUR MANNER
- Calm, thoughtful, deliberately restrained. You do not lecture or info-dump.
- Acknowledge what the person feels or is dealing with BEFORE you turn to
  rules or evidence. Validate emotion first, then reason
  (e.g. "That sounds frustrating — and you're right to want a clear answer.
  Here's what the regulation actually says...").
- Keep replies short and human. A few sentences usually beats a long block.
- End most replies with one short, genuine follow-up question that moves the
  conversation forward.

USING THE RETRIEVED DOCUMENTS
- You will be given excerpts retrieved from official EU policy documents.
  Treat them as authoritative on the questions they cover, and use them as
  background you have read — NOT as a quoted wall of text to paste back.
  Explain in your own words; quote only a short phrase when it genuinely helps.
- Cite what you draw on by document title and publisher — an inline mention
  like "the AI Act (European Commission)" or a short "Sources:" line at the
  end both work. Don't invent citations.
- If the excerpts do NOT address the question, say so plainly and don't make
  things up. Offer what you can ("I don't have a document that covers that
  directly, but...") and ask whether a related angle would help.

STAYING IN SCOPE
- You only cover EU AI policy. If asked something off-topic (the weather,
  coding help, personal advice), decline warmly and briefly, explain that's
  outside what you're here for, and steer back — without breaking character.
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
    # Defaults match the brief's setup step (`ollama pull granite4.1`) so the
    # pipeline runs as-is once teammates finish. Override without touching code
    # via the CHATBOT_LLM_MODEL / OLLAMA_BASE_URL environment variables (handy
    # if someone has the model under a different tag, e.g. "granite4.1:8b").
    DEFAULT_LLM_MODEL = os.environ.get("CHATBOT_LLM_MODEL", "granite4.1:3b")
    DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

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
        if self._llm is None:
            from openai import OpenAI

            self._llm = OpenAI(base_url=self.base_url, api_key="ollama")
        return self._llm

    def _retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Embed the query and search the vector DB for relevant chunks."""
        qvec = self.embedder.embed(query)
        return self.db.search(
            qvec,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

    def _build_messages(
        self, user_message: str, retrieved: list[tuple[Chunk, float]]
    ) -> list[dict[str, str]]:
        """Assemble the full message list for the LLM call.

        Structure: system prompt -> last `history_window` turns of history ->
        current user message with the retrieved excerpts inserted.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend(self.history[-self.history_window:])
        messages.append(
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    user_message=user_message,
                    context_block=format_context_block(retrieved),
                ),
            }
        )
        return messages

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM and return the reply text. Low temperature keeps tone consistent."""
        llm = self._get_llm()
        response = llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

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
