# NLP Policy Chatbot

Submission Date: 2026-06-05
Team Members:
- Hugo (707828)
- Elias (685288)
- Sol (700695)
- Ksawery (715265)
- Wouter (681115)

## Project Overview

A retrieval-augmented chatbot that answers questions about EU artificial
intelligence policy — trust, fairness, responsibility, transparency,
accountability, deepfakes, healthcare AI, and related themes. Every answer
is grounded in primary EU sources: the AI Act, EDPS/EDPB guidance, JRC
reports, and European Parliament briefings. The bot does not generate
opinions or invent facts; it retrieves relevant passages from real documents
and uses them to inform its replies, always citing the source.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Copy the environment template and adjust to your machine:

```bash
cp .env.example .env       # Mac/Linux
copy .env.example .env     # Windows
```

Open `.env` and pick the LLM tag that matches your hardware:
- `granite4.1:3b` for CPU-only machines (faster, lower quality)
- `granite4.1:8b` for machines with a GPU (slower on CPU, better quality)

Install [Ollama](https://ollama.com/download) and pull the models:

```bash
ollama pull granite4.1:8b      # or granite4.1:3b for CPU-only machines (the default)
ollama pull nomic-embed-text
```

---

## Configuration

Both the LLM and the embedding model are served locally via Ollama.
Defaults are sensible, but you can override them per machine via `.env`:

| Variable             | Default                          | Purpose                                  |
|----------------------|----------------------------------|------------------------------------------|
| `CHATBOT_LLM_MODEL`  | `granite4.1:3b` (see chat.py)    | Ollama tag for the chat model            |
| `OLLAMA_BASE_URL`    | `http://localhost:11434/v1`      | Where the local Ollama server is bound   |

The embedding model is fixed to `nomic-embed-text` (768 dims).

---

## Usage

```bash
# Start the chatbot (CLI). The submission zip ships with the prebuilt vector store under data/vector_store/.
# On a fresh clone, rebuild it first using the steps below.
python app.py

# Or start the optional web UI:
streamlit run app.py
```

To rebuild the knowledge base from scratch:

```bash
python -m src.downloader 20260331_dataset2.xlsx
python -m src.preprocessing
python -m src.vector_db
python app.py
```

Available commands inside the chat:
- `/help` — show available commands
- `/reset` — clear conversation history
- `/quit` — exit

---

## Project Structure

```
nlp_chatbot_project/
├── app.py                     # CLI entry point
├── requirements.txt
├── data/
│   ├── raw/                   # downloaded PDFs and HTML files
│   ├── processed/             # chunks.json (shipped with submission)
│   └── vector_store/          # embeddings.npz + chunks.json (shipped)
├── src/
│   ├── downloader.py          # reads Excel manifest, downloads documents
│   ├── preprocessing.py       # text extraction, cleaning, chunking
│   ├── embeddings.py          # nomic-embed-text wrapper
│   ├── vector_db.py           # from-scratch vector database
│   ├── chat.py                # persona, prompt engineering, RAG pipeline
│   ├── database.py            # SQLite metadata layer
│   └── utils.py               # shared dataclasses and helpers
└── tests/
    └── test_smoke.py
```

---

## NLP Approach

_For a deeper technical walk-through (embedding math, chunking trade-offs, alternatives we considered), see [docs/nlp_approach.md](docs/nlp_approach.md)._

### 1. Document Acquisition

We parse an Excel manifest of 87 EU policy publications. Each row provides
a document ID, title, publisher, URL, theme, and citation. The downloader
detects whether a URL points to a PDF or an HTML page using the URL suffix
and the HTTP `Content-Type` response header, then saves each file to
`data/raw/` using the document ID as the filename. Network failures are
caught and recorded per document — a single broken URL never stops the
pipeline. The full manifest including fetch status is saved as
`data/raw/manifest.json`, giving downstream stages a single source of truth
without needing to re-examine the raw files.

### 2. Text Extraction & Cleaning

PDFs are extracted with `pypdf`, concatenating pages with a separator so
page boundaries are preserved. When `pypdf` returns suspiciously little text
(fewer than 200 characters), we fall back to `pdfplumber`, which handles
scanned or complex layout PDFs more reliably. HTML pages are reduced to
their main content using `BeautifulSoup`: navigation bars, footers, scripts,
and styles are stripped, and the `<main>` or `<article>` element is
preferred when present.

After extraction we apply text normalisation: PDF ligatures (`ﬁ` → `fi`,
`ﬀ` → `ff`), soft hyphens, non-breaking hyphens, and collapsed whitespace
are all standardised. These steps matter for embedding quality because a
tokeniser treats `ﬁle` and `file` as different tokens, which would cause
semantic mismatch between a user's query and the indexed content.

### 3. Chunking Strategy

We split documents into overlapping chunks of approximately 500 words with
a 50-word overlap between consecutive chunks. Splits prefer paragraph
boundaries over sentence boundaries, and sentence boundaries over hard cuts
mid-word. Chunks shorter than 100 characters are discarded as likely noise.

The 500-word target balances two competing concerns: shorter chunks have
tighter, more focused topics (which improves retrieval precision), but lose
surrounding context that the model needs to give a coherent answer. The
50-word overlap ensures that information sitting near a chunk boundary is
captured in both adjacent chunks, so no sentence is permanently lost to a
split. Each chunk carries full source metadata — document ID, title,
publisher, URL, theme, and chunk index — so every retrieved passage can be
attributed back to its original document.

### 4. Text Embeddings

We use `nomic-embed-text` (768 dimensions) served locally via Ollama. This
model was trained on a large, diverse corpus of English text and handles the
formal register of EU policy documents well. 768 dimensions provides a
useful balance: high enough to represent fine-grained semantic distinctions
between policy concepts, low enough that the full matrix (~2000 chunks ×
768 floats) fits comfortably in RAM and search remains fast.

All embeddings are L2-normalised at insertion time, which means the vector
for every chunk has length exactly 1. This transformation is the key
optimisation in our search step: once vectors are unit-length, cosine
similarity between any two vectors equals their dot product. That reduces
the retrieval step to a single matrix multiplication with no division.

We chose a locally-hosted model over a cloud embedding API (such as
`text-embedding-ada-002`) for two reasons: it runs without internet access
or API keys, and it keeps the entire system self-contained — a requirement
for reproducible academic work.

### 5. Vector Database & Similarity Search

The vector store is implemented from scratch using only NumPy. Embeddings
are stored as a single `float32` matrix in `embeddings.npz`
(shape `N × 768`); the corresponding chunk metadata lives in `chunks.json`,
row-aligned with the matrix so the i-th row always corresponds to the i-th
chunk.

At query time we embed the user's message with the same model, normalise
the query vector, then compute `scores = embeddings_matrix @ query_vector`.
This one line gives the cosine similarity between the query and every chunk
simultaneously. We use `np.argpartition` to extract the top-k results
without sorting the full array, then sort only those k entries. Chunks
scoring below a threshold of 0.25 are filtered out to avoid passing
irrelevant context to the language model.

This brute-force approach is O(N·D) per query — appropriate for our corpus
of approximately 2000 chunks. Approximate nearest-neighbour indexes (HNSW,
IVF) would be necessary above roughly one million vectors; at our scale they
would add complexity with no measurable benefit.

### 6. Prompt Engineering & RAG Design

Each conversation turn triggers retrieval of the top-6 chunks above the
similarity threshold. Retrieved chunks are formatted into a numbered context
block and injected into the user-side message alongside the original
question, so the model sees both what was asked and what the documents say.

The system prompt encodes a specific persona: a calm, reflective assistant
that validates a user's concern before discussing policy, uses retrieved
text as background rather than quoting it verbatim, ends replies with a
short follow-up question, and declines off-topic requests without breaking
character. This design follows the conversation examples in the assignment
scenario files. The temperature is set to 0.3 to keep the tone consistent
across turns while allowing natural variation in phrasing.

### 7. Conversation Management

Conversation history is maintained as a rolling window of the last 8 turns
(4 exchanges). `granite4.1` supports a 128 K-token context window, so
there is no need for aggressive summarisation at typical conversation
lengths. The rolling window means the model retains enough context to
resolve follow-up references ("and what about children?", "can they appeal
that?") without the prompt growing indefinitely. History stores the raw user
message and the assistant reply — not the retrieval-augmented prompt — to
keep the context compact.

### 8. Source Attribution

After every reply, the application prints a deduplicated list of the
documents the retrieval step drew from, regardless of whether the model
cited them inline. This is implemented in `app.py` independently of the
language model, so attribution never fails silently even if the model
omits a citation.

### 9. Error Handling

Network failures during document download are caught per URL, logged, and
recorded in the manifest as `fetch_status="failed"`. The preprocessing
stage skips any document not marked `success`. Empty or near-empty
extractions are discarded before chunking. If the vector store is missing
at startup, the application prints actionable rebuild instructions instead
of raising a traceback. These measures mean the system degrades gracefully
when individual documents are unavailable.

---

## Known Limitations

- **English only.** EU sources are published in all official EU languages
  but this build indexes only English-language documents.
- **Static corpus.** The knowledge base reflects EU AI policy as of early
  2026. Documents published after the download date are not included.
- **No live web search.** When a question falls outside the corpus the bot
  says so rather than searching the internet.
- **Local models only.** Both the embedding model and the language model run
  on the user's machine via Ollama. A machine without a GPU will be slower.
