# Architecture

A linear pipeline at build time; one feedback loop at query time.

## Build time (run once)

```
20260331_dataset2.xlsx
        │
        ▼
   downloader.py  ─►  data/raw/<id>.pdf | .html
                     data/raw/manifest.json   (list[Document])
        │
        ▼
  preprocessing.py  ─►  data/processed/chunks.json   (list[Chunk])
        │
        ▼
    embeddings.py
        │  (embed every chunk)
        ▼
    vector_db.py   ─►  data/vector_store/embeddings.npz
                       data/vector_store/chunks.json
```

Each arrow is a process boundary — the upstream stage writes to disk, the
downstream stage reads. This means **any one stage can be re-run without
re-running the others**, which is essential when you're iterating on
chunking strategy without wanting to re-download 87 documents.

## Query time (every user message)

```
user message
     │
     ▼
chat.py ──► embeddings.py.embed()
     │            │
     │            ▼
     │      query vector (1, 768)
     │            │
     │            ▼
     │      vector_db.py.search()
     │            │
     │            ▼
     │      top-k chunks  ──┐
     │                      │
     ▼                      ▼
chat.py.build_messages()  → context block
     │
     ▼
chat.py.generate()  ──►  Ollama  ──►  reply
     │
     ▼
reply + sources to user
```

## Module contracts (what each piece promises)

### Document  (defined in `utils.py`)
The output of `downloader.py`. Carries metadata + `local_path` to the raw
file + `fetch_status`. `preprocessing.py` skips anything where
`fetch_status != "success"`.

### Chunk  (defined in `utils.py`)
The output of `preprocessing.py`. Carries `text` + a `metadata` dict. The
metadata dict **must** include `doc_id`, `title`, `publisher`, `url`,
`theme`, and `chunk_index` for source attribution. Adding more keys is fine.

### Embeddings array
A 2D `np.ndarray` of shape `(N, 768)`, dtype `float32`, **L2-normalised
per row**. Normalising at embedding time means cosine similarity reduces
to a dot product at search time — half the work, same result.

### Vector store on disk
Two files, kept in lockstep:
- `embeddings.npz` — `np.savez_compressed`, single key `embeddings`
- `chunks.json` — list of Chunk dicts, **same order** as the array rows

The row order is the API. Don't sort either file independently.

### ChatSession
Holds the conversation history. Stateless across instances — the vector
DB is shared and read-only.

## Why no SQLite or LangChain

The assignment forbids pre-built RAG / vector libraries. Using SQLite for
the chunk store would be defensible but adds a dependency and obscures the
"from scratch" intent. NumPy + JSON keeps it transparent and grade-able.

## Where the 40% NLP grade lives

| Rubric item                                  | Module                | %    |
| -------------------------------------------- | --------------------- | ---- |
| Embeddings understanding                     | embeddings.py         | 15   |
| RAG implementation                           | vector_db.py + chat.py| 15   |
| Prompt engineering & context handling        | chat.py               | 10   |
| Document acquisition pipeline                | downloader.py         | 10   |
| Preprocessing & chunking                     | preprocessing.py      | 10   |
| Similarity search                            | vector_db.py          | 10   |
| Retrieval effectiveness & source attribution | chat.py + app.py      | 10   |
| Response quality                             | chat.py               | 10   |
| Follow-up handling                           | chat.py               | 5    |
| Error handling                               | all                   | 5    |

Three modules carry most of the grade: `chat.py` (~35%), `vector_db.py +
embeddings.py` (~25%), and the data pipeline (`downloader.py +
preprocessing.py`, ~20%).
