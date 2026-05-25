# NLP Policy Chatbot

A retrieval-augmented chatbot that answers questions about AI ethics —
trust, fairness, responsibility, transparency, accountability, and related
themes — grounded in primary EU policy sources (the AI Act, EDPS/EDPB
guidance, JRC reports, European Parliament briefings).

> **Status:** scaffolding. See `docs/task_assignments.md` for who owns what.

---

## Quick start

```bash
# 1. Set up the environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Make sure Ollama is running and the models are pulled
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 3. Chat (vector store is shipped with the submission)
python app.py
```

To rebuild the vector store from scratch:

```bash
python -m src.downloader 20260331_dataset2.xlsx   # fetches PDFs / HTML
python -m src.preprocessing                        # extracts + chunks
python -m src.vector_db                            # embeds + persists
```

---

## Project structure

```
nlp_chatbot_project/
├── README.md
├── requirements.txt
├── app.py                     # CLI entry point
├── data/
│   ├── raw/                   # downloaded PDFs/HTML (rebuilt from Excel)
│   ├── processed/             # cleaned text + chunks.json  (shipped)
│   └── vector_store/          # embeddings.npz + chunks.json (shipped)
├── docs/
│   ├── architecture.md        # how the modules connect
│   └── task_assignments.md    # who owns what
├── src/
│   ├── downloader.py          # Excel + HTTP fetch
│   ├── preprocessing.py       # text extraction, cleaning, chunking
│   ├── embeddings.py          # Ollama nomic-embed-text wrapper
│   ├── vector_db.py           # from-scratch vector DB
│   ├── chat.py                # persona, prompt, RAG orchestration
│   └── utils.py               # shared dataclasses + helpers
└── tests/
```

---

## NLP approach

> **This section is graded.** Fill it out with substance — diagrams welcome.
> Below is a skeleton to expand. Each subsection should be 2–4 paragraphs.

### 1. Document acquisition

We parse an Excel manifest of 87 EU policy sources, distinguishing PDFs from
HTML pages by URL suffix and by `Content-Type` after fetch. Failures are
captured (not raised) so a single dead URL never kills the pipeline. The
fetch result is written to `data/raw/manifest.json` so downstream stages
work from a single source of truth.

### 2. Text extraction & cleaning

PDFs are extracted with `pypdf`. HTML pages are reduced to their `<main>`
or `<article>` content with BeautifulSoup, stripping navigation, scripts,
and styles. After extraction we normalise: ligatures (`ﬁ` → `fi`), soft
and non-breaking hyphens, smart quotes, and collapsed whitespace. *Discuss
why these matter for embedding quality: tokenisers split `ﬁle` differently
from `file`, harming retrieval recall.*

### 3. Chunking

We use overlapping paragraph-aware chunks of ~500 tokens with ~50 tokens
of overlap. *Discuss the tradeoff: smaller chunks improve precision (each
chunk has a tight topic) but lose surrounding context; larger chunks
preserve context but dilute the embedding signal. Overlap protects against
information being split awkwardly across chunk boundaries.*

### 4. Embeddings — why nomic-embed-text

We use `nomic-embed-text` (768 dimensions) served by Ollama. *Discuss:
- Trained on a large corpus of diverse English text — matches our policy
  document register reasonably well.
- 768 dimensions is a typical sweet spot: high enough to capture nuance,
  low enough to keep the vector store small and search fast.
- L2-normalised at insertion time, so cosine similarity becomes a simple
  matrix-vector product.
- Why not `text-embedding-ada-002`? It performs comparably but requires an
  external API and money; nomic runs locally.*

### 5. Vector storage & similarity search

A pure-numpy index. Embeddings are stored as a single `float32` matrix in
`embeddings.npz`; chunks and their metadata live alongside in `chunks.json`,
row-aligned with the matrix. At query time we embed the user's message,
compute `scores = embeddings @ q_vec` (valid because everything is unit
length), then take the top-k with `np.argpartition`. This is exact cosine
similarity, O(N·D) per query. *Discuss when this stops working — ANN
indexes (HNSW, IVF) become necessary above ~10⁶ vectors. We have ~2000,
so brute force is the right call.*

### 6. RAG prompt design

Each user message triggers retrieval of the top-k chunks (k=6 by default,
filtered by a similarity threshold to suppress noise). Retrieved chunks
are injected into the prompt with their titles and publishers. The system
prompt encodes the persona (calm, reflective, honest about limits) and
explicit instructions on citation, honesty about coverage gaps, and
following up rather than info-dumping. *Discuss the prompt-engineering
tradeoffs: temperature, citation format, when to ask vs answer.*

### 7. Conversation context

History is kept as a rolling window of the last N turns (default 8).
llama3.1:8b's 128K context window means we don't need aggressive
summarisation. For very long conversations we can add a summarisation
pass — see `chat.py` for the hook.

### 8. Source attribution

Every reply is followed by a deduplicated list of the documents the
retrieval step pulled, so users can verify even when the model forgets to
cite inline. This is independent of the model's behaviour, so it never
fails silently.

### 9. Error handling

Network errors during fetch are logged and recorded but don't stop the
pipeline. Empty or malformed PDFs produce empty extraction, which is
filtered out before chunking. Missing vector store at app startup prints
actionable instructions rather than crashing.

---

## Limitations

- English only. Many EU sources are also published in other languages but
  this build does not retrieve them.
- Snapshot in time. The corpus reflects EU policy as of early 2026; later
  updates require re-running the downloader.
- No web fallback. If a question is outside the corpus, the bot will say so
  rather than guess — and will not search the web at runtime.

## Authors

See `docs/task_assignments.md` for the team breakdown.
