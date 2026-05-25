# Task assignments — 5 people, 4 weeks

## Team

| Person | Role | Files |
|--------|------|-------|
| **Hugo** | Role 1 — Data Acquisition | `src/downloader.py` |
| **Elias** | Role 2 — Text Processing & Chunking | `src/preprocessing.py` |
| **Sol** | Role 3 — Embeddings & Vector Store | `src/embeddings.py`, `src/vector_db.py` |
| **Ksawery** | Role 4 — Chat & Prompt Engineering | `src/chat.py` |
| **Wouter** | Role 5 — Integration, Evaluation & Docs | `app.py`, `src/utils.py`, `README.md`, `tests/` |

## How we split the work

Five roles, one per person, aligned with the module boundaries in `src/` and
the grading rubric. Each role has clear ownership, clear deliverables, and
**a defined interface to the next person**, so we can work in parallel.

> If you don't know what your interface looks like, read `docs/architecture.md`
> and the docstrings at the top of the file you own.

---

## Role 1 — Data Acquisition · **Hugo**

**Files owned:** `src/downloader.py`
**Grade weight:** ~10% (Document acquisition pipeline) + share of Error handling
**Estimated effort:** medium (mostly engineering, few design decisions)

**Deliverables**
- `data/raw/<id>.pdf` and `data/raw/<id>.html` for every fetchable URL
- `data/raw/manifest.json` — every Document object with `fetch_status` filled in
- A short paragraph for the README §9 on how failures are handled

**Done when**
- `python -m src.downloader 20260331_dataset2.xlsx` runs end-to-end on a fresh
  checkout
- Failed URLs are recorded, not raised
- Re-running the script doesn't re-download already-fetched files

**Quality checks**
- The arxiv URL R18-03 (`https://arxiv.org/pdf/2603.22920`) is bogus — it must
  appear as `fetch_status="failed"`, not crash anything
- Excel-serial dates and non-breaking hyphens in IDs are normalised

---

## Role 2 — Text Processing & Chunking · **Elias**

**Files owned:** `src/preprocessing.py`
**Grade weight:** 10% (Preprocessing & chunking)
**Estimated effort:** medium-high (lots of judgement calls: extraction
quality, chunk size, cleaning rules)

**Deliverables**
- `data/processed/chunks.json` — every retrievable chunk with metadata
- A README §2–3 contribution explaining the chunking strategy and why
- A small evaluation: pick 5–10 chunks at random and confirm they're
  coherent and self-contained

**Done when**
- `python -m src.preprocessing` runs and produces `chunks.json`
- Total chunk count is in a reasonable range (~1500–3000 for our 87 docs)
- No chunk shorter than the min threshold survives
- Chunks carry `doc_id`, `title`, `publisher`, `url`, `theme`, `chunk_index`
  in metadata (this is what the chat module cites from)

**Quality checks**
- Open 3 random PDFs and 3 HTML pages, compare extracted text vs original
- A chunk in the middle of `chunks.json` should be readable in isolation

---

## Role 3 — Embeddings & Vector Store · **Sol**

**Files owned:** `src/embeddings.py`, `src/vector_db.py`
**Grade weight:** 15% + 10% = 25% (Embeddings + Similarity search)
**Estimated effort:** high (technical core, performance matters)

**Deliverables**
- `EmbeddingClient` wrapping `nomic-embed-text` via the OpenAI SDK
- `VectorDB` class with `.add`, `.search`, `.persist`, `.load`
- `data/vector_store/embeddings.npz` + `chunks.json` (built once, shipped)
- A README §4–5 contribution explaining the embedding model and the
  similarity-search algorithm

**Done when**
- `EmbeddingClient().embed("hello").shape == (768,)`
- `VectorDB` can be persisted, deleted-from-memory, reloaded, and returns
  identical search results
- Search returns the top-k by cosine similarity, exact algorithm

**Quality checks**
- Manually try 5 queries (e.g. "deepfakes", "algorithmic discrimination",
  "AI in healthcare", "human oversight", "watermarking") and confirm the
  top-3 retrieved chunks look thematically right
- Pickle/np.savez round-trip is bit-exact

---

## Role 4 — Chat, Persona & Prompt Engineering · **Ksawery**

**Files owned:** `src/chat.py`, the `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`
**Grade weight:** 10% + 10% + 5% = 25% (Prompt engineering + Response quality +
Follow-up handling)
**Estimated effort:** high (lots of iteration; the persona is hard)

**Deliverables**
- Working `ChatSession` that retrieves, builds prompts, calls Ollama, returns
  replies and citations
- Tuned `SYSTEM_PROMPT` matching the persona in the two scenario docx files
- `/help` and `/reset` commands
- A README §6–7 contribution explaining the prompt structure

**Done when**
- A user can carry a 5-turn conversation that stays on-topic and consistent
  in tone
- Follow-up questions (e.g. "and what about children?") work — pronouns
  resolve against earlier turns
- The bot is honest when retrieved context doesn't cover the question

**Quality checks**
- Try the scenarios in the docx files. Compare the bot's replies in tone
  and shape to Alex/Ada's. Iterate until the gap closes.
- Try adversarial questions ("what's the weather?") — the bot should
  decline cleanly without breaking persona

---

## Role 5 — Integration, Evaluation & Documentation · **Wouter**

**Files owned:** `app.py`, `src/utils.py`, `README.md`, `tests/`
**Grade weight:** 10% (Retrieval effectiveness/source attribution, shared
with Role 4) + share of Error handling + README is graded
**Estimated effort:** medium-high (integration is finicky and docs take time)

**Deliverables**
- `python app.py` runs end-to-end cleanly on a fresh checkout
- README §1, §8, §9, and final polish on §2–7 contributions from others
- Eval harness in `tests/` — a set of ~10 evaluation queries (drawn from the
  scenarios + invented ones) checked against retrieval and against the
  scenarios' tone
- The submission zip — no `.venv`, no `data/raw/`, yes `data/processed/`
  and `data/vector_store/`

**Done when**
- Fresh-checkout test passes: clone, venv, pip install, `python app.py`
  works without any other step
- All `/help` / `/reset` / `/quit` work
- Source attribution prints after every reply

**Quality checks**
- Read the README out loud. If a sentence is fluff, cut it.
- Run all evaluation queries and screenshot the results for the submission

---

## Timeline

### Week 1 — foundation, parallel work
- Role 1: downloader done, raw/ populated
- Role 2: extraction + cleaning working on ~5 sample docs
- Role 3: embedding + vector DB working on a toy corpus (10 chunks)
- Role 4: SYSTEM_PROMPT first draft, mock the retrieval to iterate prompts
- Role 5: utils + skeleton tests + README skeleton

### Week 2 — integration
- All roles run their modules end-to-end on the real corpus
- First full pipeline run: Excel → raw → processed → vector_store → app.py
- Role 4 starts iterating against real retrieval

### Week 3 — quality pass
- Role 2 tunes chunk size against retrieval quality
- Role 3 inspects retrieval failures with Role 4 — adjust threshold, top_k
- Role 4 tunes prompt + temperature
- Role 5 builds the eval harness

### Week 4 — polish, freeze, submit
- No code changes in last 2 days — only README, tests, and the submission zip
- Verify clean-install run on a fresh machine

---

## Cross-role conventions

- **Use `get_logger(__name__)` from `utils.py`, not `print()`** — except in
  `app.py` (CLI output is for users)
- **Talk through `Document` and `Chunk` dataclasses**, don't invent your own
  shapes — if you need new fields, add to those dataclasses
- **PRs are small** — one module's worth of progress per PR
- **One person owns the merge to `main`** (suggest Role 5) so the pipeline
  stays runnable
