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
    python app.py                    # CLI
    streamlit run app.py             # Web UI

NOTE
----
This file deals with both front-ends — the interactive CLI loop and the
optional Streamlit web UI — plus graceful startup/shutdown. The mode is
chosen at runtime by _is_streamlit_runtime(). All retrieval and generation
logic lives in src/chat.py; this file never touches it directly.
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


def _is_streamlit_runtime() -> bool:
    """True when this script is being run by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _streamlit_cache_resource(**kwargs):
    """Return st.cache_resource when streamlit is importable, else an identity
    decorator.

    This lets _load_db_and_embedder stay a module-level, decorated helper (so
    Streamlit caches it across reruns) without forcing a hard top-level
    streamlit import — `python app.py` must work even when streamlit is not
    installed.
    """
    try:
        import streamlit as st
        return st.cache_resource(**kwargs)
    except Exception:
        def _identity(fn):
            return fn
        return _identity


@_streamlit_cache_resource(show_spinner="Loading vector store...")
def _load_db_and_embedder():
    """Load the vector store and embedder once, cached across Streamlit reruns."""
    db = VectorDB()
    db.load()
    embedder = EmbeddingClient()
    return db, embedder


def _corpus_summary_data() -> dict | None:
    """Shared document-corpus stats source.

    Used by the CLI startup log line, the CLI `/docs` command, and the
    Streamlit sidebar so all three pull from one place.

    Returns a dict {"total", "fetched", "failed", "pending", "themes"} where
    "themes" is a list of (name, count) sorted by count descending then name.
    Returns None when the SQLite index does not exist yet. Raises if the file
    exists but cannot be read — callers decide how to report that.
    """
    if not DB_PATH.exists():
        return None

    doc_db = DocumentDB()
    stats = doc_db.fetch_stats()
    total = len(doc_db)

    theme_counts: dict[str, int] = {}
    for doc in doc_db.all():
        theme = (doc.get("theme") or "").strip() or "Unknown"
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
    # Sort by count descending, then alphabetically as a tiebreaker.
    themes = sorted(theme_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    return {
        "total": total,
        "fetched": stats.get("success", 0),
        "failed": stats.get("failed", 0),
        "pending": stats.get("pending", 0),
        "themes": themes,
    }


def _dedupe_sources(retrieved) -> list[str]:
    """Return deduplicated source lines (by doc_id) as markdown bullets.

    Same dedupe logic as the CLI's "Sources used" block, formatted for the web UI.
    """
    lines: list[str] = []
    seen: set = set()
    for chunk, score in retrieved:
        key = chunk.metadata.get("doc_id", chunk.doc_id)
        if key in seen:
            continue
        seen.add(key)
        title = chunk.metadata.get("title", "Untitled")
        publisher = chunk.metadata.get("publisher", "Unknown")
        url = chunk.metadata.get("url", "")
        lines.append(
            f"- **{title}** — {publisher} [{url}]({url}) (similarity={score:.2f})"
        )
    return lines


def _log_document_index() -> None:
    """Log a one-line summary of the document index at startup.

    Silently skips (with a friendly hint) when the SQLite db hasn't been
    built yet, and never crashes the CLI if the file is malformed.
    """
    try:
        data = _corpus_summary_data()
    except Exception as exc:
        log.warning(
            "Document index present but unreadable (%s); skipping corpus stats.",
            exc,
        )
        return

    if data is None:
        log.info(
            "Document index not loaded. Run `python -m src.database "
            "20260331_dataset2.xlsx` to enable corpus stats."
        )
        return

    total = data["total"]
    fetched = data["fetched"]
    failed = data["failed"]
    pending = data["pending"]

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


def _print_corpus_summary() -> None:
    """Print document-corpus statistics for the /docs command."""
    try:
        data = _corpus_summary_data()
    except Exception as exc:
        print(f"(could not read document index: {exc})")
        return

    if data is None:
        print(
            "(no document index loaded — run `python -m src.database "
            "20260331_dataset2.xlsx` to enable this)"
        )
        return

    total = data["total"]
    fetched = data["fetched"]
    failed = data["failed"]
    pending = data["pending"]
    ordered = data["themes"]

    print("Document corpus:")
    print(f"  {'Total:':<9}{total:>3}")
    print(f"  {'Fetched:':<9}{fetched:>3}")
    print(f"  {'Failed:':<9}{failed:>3}")
    print(f"  {'Pending:':<9}{pending:>3}")

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


def _render_sidebar(chat_session=None) -> None:
    """Render the sidebar. chat_session=None means the vector store isn't loaded;
    in that case the Conversation block (Reset button) is hidden and the
    Configuration block falls back to ChatSession's defaults."""
    import streamlit as st

    with st.sidebar:
        # Conversation block — only when there is a session to reset.
        if chat_session is not None:
            st.subheader("Conversation")
            if st.button("Reset conversation", use_container_width=True):
                st.session_state.chat_session.reset()
                st.session_state.messages = []
                st.rerun()
            st.divider()

        st.subheader("Document corpus")
        corpus = _corpus_summary_data()
        if corpus is None:
            st.info(
                "No document index loaded. Run `python -m src.database "
                "20260331_dataset2.xlsx` to enable corpus stats."
            )
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", corpus["total"])
            col2.metric("Fetched", corpus["fetched"])
            col3.metric("Failed", corpus["failed"])
            top_themes = corpus["themes"][:15]
            if top_themes:
                st.dataframe(
                    [
                        {"Theme": name, "Documents": count}
                        for name, count in top_themes
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

        st.divider()

        st.subheader("Configuration")
        if chat_session is not None:
            llm = chat_session.llm_model
            top_k = chat_session.top_k
            threshold = chat_session.score_threshold
        else:
            import inspect

            sig = inspect.signature(ChatSession)
            llm = ChatSession.DEFAULT_LLM_MODEL
            top_k = sig.parameters["top_k"].default
            threshold = sig.parameters["score_threshold"].default
        st.text(f"LLM:        {llm}")
        st.text(f"Embedding:  {EmbeddingClient.DEFAULT_MODEL}")
        st.text(f"Top-k:      {top_k}")
        st.text(f"Threshold:  {threshold}")


def run_streamlit() -> None:
    """Render the Streamlit web UI. Invoked under `streamlit run app.py`.

    Behaves equivalently to the CLI (same backend, same vector-store check,
    same source dedupe) but in a browser with a nicer presentation. The
    sidebar renders even when the vector store is missing, so the page never
    looks bare.
    """
    import streamlit as st

    st.set_page_config(
        page_title="EU AI Policy Chatbot",
        page_icon=None,
        layout="wide",
    )

    st.title("EU AI Policy Chatbot")
    st.caption(
        "Grounded in EU AI Act, EDPS/EDPB guidance, JRC reports, and EP briefings."
    )

    # Detect whether the vector store is available.
    missing = [
        f for f in ["embeddings.npz", "chunks.json"]
        if not (VECTOR_STORE_DIR / f).exists()
    ]

    if missing:
        # Render the sidebar with no chat session — Reset hidden,
        # Configuration shows defaults.
        _render_sidebar(chat_session=None)

        st.error("Vector store not built yet.")
        st.code(
            "python -m src.downloader 20260331_dataset2.xlsx\n"
            "python -m src.preprocessing\n"
            "python -m src.vector_db",
            language="bash",
        )
        st.info(
            "The sidebar shows what's available from the local document "
            "index. Build the vector store to enable chatting."
        )
        return

    # Vector store available — full flow.
    db, embedder = _load_db_and_embedder()

    # Per-browser-session state: the ChatSession (history) is NOT cached.
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(vector_db=db, embedder=embedder)
    if "messages" not in st.session_state:
        # list of dicts: {"role": "user"|"assistant", "content": str, "sources": list|None}
        st.session_state.messages = []

    chat_session = st.session_state.chat_session

    _render_sidebar(chat_session=chat_session)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources")
            if sources:
                with st.expander("Sources used"):
                    for line in _dedupe_sources(sources):
                        st.markdown(line)

    # ------------------------------------------------------------------
    # Chat input
    # ------------------------------------------------------------------
    user_input = st.chat_input("Ask about EU AI policy...")
    if user_input:
        # 1. Render the user message immediately.
        with st.chat_message("user"):
            st.markdown(user_input)
        # 2. Call the backend with a spinner.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply, retrieved = st.session_state.chat_session.handle(user_input)
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                    st.stop()
            st.markdown(reply)
            if retrieved:
                with st.expander("Sources used"):
                    for line in _dedupe_sources(retrieved):
                        st.markdown(line)
        # 3. Persist both turns so they survive the next rerun.
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "sources": None}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "sources": retrieved}
        )


if __name__ == "__main__":
    if _is_streamlit_runtime():
        run_streamlit()
    else:
        run_cli()
