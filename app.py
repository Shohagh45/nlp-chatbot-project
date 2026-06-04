"""
app.py
======
CLI entry point. Wires the vector store + embedder + chat session together,
then runs an interactive prompt.

OWNER: Role 5 — Integration & Docs

USAGE
-----
    # First-time build (only once, or after re-downloading documents):
    python -m src.downloader 20260331_dataset2.xlsx
    python -m src.preprocessing
    python -m src.vector_db        # builds vector store

    # Then chat:
    python app.py

NOTE
----
This file is intentionally thin — it only deals with the CLI loop and
graceful startup/shutdown. All retrieval and generation logic lives in
src/chat.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.chat import ChatSession
from src.embeddings import EmbeddingClient
from src.utils import VECTOR_STORE_DIR, get_logger
from src.vector_db import VectorDB

log = get_logger("app")

BANNER = """\
============================================================
  EU AI Policy Chatbot
  Type your question.  /help for commands.  /quit to exit.
============================================================
"""


def run_cli() -> None:
    """Main interactive loop."""
    # ------------------------------------------------------------------
    # Startup: make sure the vector store is built
    # ------------------------------------------------------------------
    missing = [
        f for f in ["embeddings.npz", "chunks.json"]
        if not (VECTOR_STORE_DIR / f).exists()
    ]
    if missing:
        print(
            f"\n[!] Vector store not built yet (missing: {', '.join(missing)}).\n"
            "    Run the pipeline first:\n"
            "    python -m src.downloader 20260331_dataset2.xlsx\n"
            "    python -m src.preprocessing\n"
            "    python -m src.vector_db\n"
        )
        sys.exit(1)

    log.info("Loading vector store from %s", VECTOR_STORE_DIR)
    db = VectorDB()
    db.load()
    log.info("Loaded %d chunks.", len(db))

    embedder = EmbeddingClient()
    session = ChatSession(vector_db=db, embedder=embedder)

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------
    print(BANNER)
    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/quit", "/exit", "exit", "quit"}:
            print("Goodbye.")
            break

        if user_input.lower() == "/help":
            print(session.help())
            continue

        if user_input.lower() == "/reset":
            session.reset()
            print("(conversation reset)")
            continue

        try:
            reply, retrieved = session.handle(user_input)
        except Exception:
            log.exception("Error handling message")
            print("(something went wrong — see logs)")
            continue

        print(f"\nbot > {reply}\n")

        # Always print the sources we used, even if the model failed to cite.
        if retrieved:
            print("Sources used:")
            seen = set()
            for chunk, _score in retrieved:
                key = chunk.metadata.get("doc_id", chunk.doc_id)
                if key in seen:
                    continue
                seen.add(key)
                title = chunk.metadata.get("title", "Untitled")
                publisher = chunk.metadata.get("publisher", "Unknown")
                url = chunk.metadata.get("url", "")
                print(f"  - {title} ({publisher})  {url}")
            print()


if __name__ == "__main__":
    run_cli()
