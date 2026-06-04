# Team — Work Distribution

## Members

| Person | Module | Files | Grade weight |
|--------|--------|-------|-------------|
| Hugo | Data Acquisition | `src/downloader.py` | 10% |
| Elias | Text Processing & Chunking | `src/preprocessing.py` | 10% |
| Sol | Embeddings & Vector Store | `src/embeddings.py`, `src/vector_db.py` | 25% |
| Ksawery | Chat & Prompt Engineering | `src/chat.py` | 25% |
| Wouter | Integration, Database & Docs | `app.py`, `src/database.py`, `README.md`, `tests/` | 20% |

---

## Hugo — Data Acquisition

**File:** `src/downloader.py`

Implement `load_manifest()` to parse the Excel file into Document objects,
and `fetch_one()` to download each document to `data/raw/`. Failed downloads
should be recorded in the manifest as `fetch_status="failed"` — do not raise
exceptions. Re-running the script should skip files already on disk.

**Done when:**
- `python -m src.downloader 20260331_dataset2.xlsx` runs end-to-end
- `data/raw/manifest.json` contains all 87 entries with fetch status
- Broken URLs appear as failed, not as crashes

---

## Elias — Text Processing & Chunking

**File:** `src/preprocessing.py`

Implement `extract_pdf()` and `extract_html()` to get plain text from raw
files, `clean()` to normalise the text, and `chunk_text()` to split it into
overlapping passages of ~500 words with full source metadata attached.

**Done when:**
- `python -m src.preprocessing` produces `data/processed/chunks.json`
- Chunk count is between 1500 and 3000
- Every chunk has `title`, `publisher`, `url`, `theme`, `chunk_index` in metadata

---

## Sol — Embeddings & Vector Store

**Files:** `src/embeddings.py`, `src/vector_db.py`

Implement `EmbeddingClient` to convert text to 768-dimensional vectors via
`nomic-embed-text`, and `VectorDB` to store, search, persist, and reload
those vectors. Embeddings must be L2-normalised so cosine similarity equals
dot product at search time.

**Done when:**
- `EmbeddingClient().embed("hello").shape == (768,)`
- `python -m src.vector_db` builds `data/vector_store/embeddings.npz`
- Searching "deepfakes" returns deepfake chunks, not healthcare ones

---

## Ksawery — Chat & Prompt Engineering

**File:** `src/chat.py`

Implement the four methods in `ChatSession`: LLM client initialisation,
query retrieval, message assembly, and generation. Tune the `SYSTEM_PROMPT`
to match the calm, reflective persona from the scenario files. The bot
should validate emotion before reasoning, cite its sources, and decline
off-topic questions without breaking character.

**Done when:**
- A 5-turn conversation stays on topic and consistent in tone
- Follow-up questions resolve against earlier turns
- Off-topic questions are declined cleanly

---

## Wouter — Integration, Database & Documentation

**Files:** `app.py`, `src/database.py`, `README.md`, `tests/`

Keep the pipeline running as teammates merge. Build the Streamlit web
interface. Own the SQLite document metadata database. Write and polish the
README. Build an evaluation test set of ~10 queries drawn from the scenario
files. Prepare the submission zip.

**Done when:**
- `python app.py` and `streamlit run app.py` both work on a fresh checkout
- README is complete with team names, student numbers, and NLP approach
- All 10 evaluation queries return relevant sources
- Submission zip excludes `.venv` and `data/raw/`

---

## Conventions

- Use `get_logger(__name__)` from `utils.py` — not `print()`
- Keep the shared `Document` and `Chunk` dataclasses — do not create alternatives
- One PR per module — keep them small and reviewable
- Wouter merges to master after review
