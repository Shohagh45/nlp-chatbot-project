"""
Dev test for Ksawery — runs the full chat loop with fake chunks.
No Hugo/Elias code needed. Requires Ollama running with granite4.1:8b.

Run with:
    python tests/test_chat_dev.py
"""

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from src.utils import Chunk
from src.embeddings import EmbeddingClient
from src.vector_db import VectorDB
from src.chat import ChatSession


def build_dev_db() -> tuple[VectorDB, EmbeddingClient]:
    """Build a tiny in-memory vector store from fake chunks."""
    embedder = EmbeddingClient()

    fake_chunks = [
        Chunk("t1", "d1",
              "AI systems must be transparent and explainable to users and regulators.",
              {"title": "AI Act", "publisher": "EU Commission", "url": "https://example.com"}),
        Chunk("t2", "d2",
              "Algorithmic bias can harm fairness in hiring, credit scoring, and law enforcement.",
              {"title": "EDPS Guidance", "publisher": "EDPS", "url": "https://example.com"}),
        Chunk("t3", "d3",
              "Human oversight is required for all high-risk AI systems under the AI Act.",
              {"title": "JRC Report", "publisher": "JRC", "url": "https://example.com"}),
        Chunk("t4", "d4",
              "Deepfakes and synthetic media must be clearly labelled under EU transparency rules.",
              {"title": "Transparency Guidelines", "publisher": "EU Commission", "url": "https://example.com"}),
        Chunk("t5", "d5",
              "Healthcare AI must meet strict safety and accuracy standards before deployment.",
              {"title": "Healthcare AI Guidelines", "publisher": "EDPB", "url": "https://example.com"}),
    ]

    print("Embedding fake chunks...")
    texts = [c.text for c in fake_chunks]
    embeddings = embedder.embed_batch(texts)

    db = VectorDB()
    db.add(fake_chunks, embeddings)
    return db, embedder


def main():
    print("Building dev vector store with fake chunks...")
    db, embedder = build_dev_db()

    session = ChatSession(vector_db=db, embedder=embedder)
    print("\nChatbot ready. Type your question (or 'quit' to exit).\n")
    print("=" * 50)

    while True:
        user_input = input("you > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "/quit"}:
            break

        reply, retrieved = session.handle(user_input)
        print(f"\nbot > {reply}\n")

        if retrieved:
            print("Sources:")
            for chunk, score in retrieved:
                print(f"  - {chunk.metadata['title']} ({chunk.metadata['publisher']}) score={score:.2f}")
        print()


if __name__ == "__main__":
    main()
