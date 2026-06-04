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
from src.database import DB_PATH, DocumentDB
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


def _log_document_index() -> None:
    """Log a one-line summary of the document index at startup.

    Silently skips (with a friendly hint) when the SQLite db hasn't been
    built yet, and never crashes the CLI if the file is malformed.
    """
    if not DB_PATH.exists():
        log.info(
            "Document index not loaded. Run `python -m src.database "
            "20260331_dataset2.xlsx` to enable corpus stats."
        )
        return

    try:
        doc_db = DocumentDB()
        stats = doc_db.fetch_stats()
        total = len(doc_db)
        fetched = stats.get("success", 0)
        failed = stats.get("failed", 0)
        pending = stats.get("pending", 0)

        if pending:
            line = (
                f"Document index: {fetched}/{total} fetched "
                f"({failed} failed, {pending} pending)"
            )
        elif failed:
            line = f"Document index: {fetched}/{total} fetched ({failed} failed)"
        else:
            line = f"Document index: {fetched}/{total} fetched"
        log.info("%s", line)
    except Exception as exc:
        log.warning(
            "Document index present but unreadable (%s); skipping corpus stats.",
            exc,
        )


def _print_corpus_summary() -> None:
    """Print document-corpus statistics for the /docs command."""
    if not DB_PATH.exists():
        print(
            "(no document index loaded — run `python -m src.database "
            "20260331_dataset2.xlsx` to enable this)"
        )
        return

    try:
        doc_db = DocumentDB()
        stats = doc_db.fetch_stats()
        total = len(doc_db)
        fetched = stats.get("success", 0)
        failed = stats.get("failed", 0)
        pending = stats.get("pending", 0)

        print("Document corpus:")
        print(f"  {'Total:':<9}{total:>3}")
        print(f"  {'Fetched:':<9}{fetched:>3}")
        print(f"  {'Failed:':<9}{failed:>3}")
        print(f"  {'Pending:':<9}{pending:>3}")

        # Tally themes from the document rows.
        theme_counts: dict[str, int] = {}
        for doc in doc_db.all():
            theme = (doc.get("theme") or "").strip() or "Unknown"
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Sort by count descending, then alphabetically as a tiebreaker.
        ordered = sorted(theme_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = ordered[:15]

        print()
        print("  Themes:")
        if shown:
            name_width = max(len(name) for name, _ in shown)
            for name, count in shown:
                print(f"    {name:<{name_width}}  {count:>3}")
        if len(ordered) > 15:
            rest = ordered[15:]
            if all(count == 1 for _, count in rest):
                print(f"    ... and {len(rest)} more themes (1 doc each)")
            else:
                rest_docs = sum(count for _, count in rest)
                print(f"    ... and {len(rest)} more themes ({rest_docs} docs total)")
    except Exception as exc:
        print(f"(could not read document index: {exc})")


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

    _log_document_index()

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
            print("  /docs    Show indexed document corpus statistics")
            continue

        if user_input.lower() == "/reset":
            session.reset()
            print("(conversation reset)")
            continue

        if user_input.lower() == "/docs":
            _print_corpus_summary()
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
