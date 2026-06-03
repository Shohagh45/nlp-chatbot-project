# What the Chat Layer Needs From You

*From: Ksawery (src/chat.py — Chat & Prompt Engineering)*
*To: Hugo, Elias, Sol, Wouter*

My part (`src/chat.py`) sits at the end of the pipeline, so it only works once
your stages produce the right outputs **in the right shape**. My code is done
and tested — this doc lists exactly what I rely on from each of you so we don't
hit surprises at integration. The full description of your task is in the
project brief; this is just the contract between your output and mine.

The whole chain:

```
Hugo            Elias               Sol                       Me (Ksawery)        Wouter
downloader.py → preprocessing.py →  embeddings + vector_db →  chat.py          →  app.py / Streamlit
data/raw/*      chunks.json         vector_store/*.npz        handle(question)    UI
```

---

## Hugo — `src/downloader.py`

**What I ultimately need:** clean `Document` metadata, because it flows through
Elias's chunks and ends up as the **citations my bot shows the user**.

- [ ] Produce `data/raw/manifest.json` with all 87 entries and the files in `data/raw/`.
- [ ] Keep the **shared `Document` dataclass** from `src/utils.py` — don't invent
      your own fields. The fields that matter downstream are `id`, `title`,
      `url`, `publisher`, `theme`.
- [ ] **Clean the metadata as you load it** (the brief asks for this): strip
      whitespace, fix the non-standard hyphens in IDs. If `title`/`publisher`
      come through as `"nan"`, blank, or full of junk, my bot will cite garbage
      like *"Sources: nan (nan)"*. Garbage in here = garbage citations out.
- [ ] Failed downloads should be marked `fetch_status='failed'` in the manifest,
      not crash — so Elias's stage can skip them cleanly.

**I don't call your code directly.** I depend on you only through the quality of
the metadata that survives into the chunks.

---

## Elias — `src/preprocessing.py`  ⭐ most important for me

This is the contract I care about most, because **my citations and the Sources
panel read directly from `chunk.metadata`.**

- [ ] Output `data/processed/chunks.json`, each entry a **shared `Chunk`** from
      `src/utils.py` (`id`, `doc_id`, `text`, `metadata`). Don't invent your own.
- [ ] **`metadata` MUST contain these keys** — my code and Wouter's UI read them
      by name:

      | key            | who reads it                         | what breaks if missing |
      |----------------|--------------------------------------|------------------------|
      | `title`        | my `format_context_block()` + Sources | citations say "Unknown" |
      | `publisher`    | my `format_context_block()` + Sources | citations say "Unknown" |
      | `url`          | Wouter's Sources expander            | no link shown |
      | `theme`        | (nice to have, used for filtering)   | — |
      | `chunk_index`  | debugging / ordering                 | — |

      My code calls `chunk.metadata.get("title")` and `.get("publisher")`, so a
      missing key won't crash — it just degrades every citation to "Unknown",
      which looks bad in the demo.
- [ ] `text` should read as a **coherent passage** (~500 words, split on
      paragraph boundaries). My prompt feeds these excerpts to the model as
      background; if a chunk is mid-sentence noise, the model's answer gets worse.
- [ ] No empty / sub-100-char chunks — they retrieve poorly and waste a slot in
      my `top_k`.

**Quick self-check before you hand off:**
```python
import json
chunks = json.load(open("data/processed/chunks.json", encoding="utf-8"))
assert 1500 <= len(chunks) <= 3000
c = chunks[len(chunks)//2]
assert {"title", "publisher", "url"} <= set(c["metadata"])   # keys I depend on
print("OK —", c["metadata"]["title"], "|", c["metadata"]["publisher"])
```

---

## Sol — `src/embeddings.py` + `src/vector_db.py`

Your code is implemented and my tests pass against it — I just need the store
**actually built** and a couple of behavioural guarantees.

- [ ] Run `python -m src.vector_db` so `data/vector_store/embeddings.npz` and
      `chunks.json` exist. (Right now the folder is empty — nothing for me to
      search.)
- [ ] **L2-normalise the embeddings** (you already do, `normalise=True`). My
      retrieval uses `score_threshold=0.25`, which only makes sense if scores are
      cosine similarities in roughly `[-1, 1]`. If normalisation is off, the
      threshold silently filters everything or nothing.
- [ ] Keep these exact signatures — I call them verbatim:
      ```python
      EmbeddingClient().embed(text)            -> np.ndarray shape (768,)
      VectorDB().search(qvec, top_k=..., score_threshold=...) -> list[(Chunk, float)]
      ```
- [ ] The `Chunk` objects that come back from `search()` must still carry the
      full `metadata` dict (so my citations work). Your `load()` already
      reconstructs `Chunk(**c)`, so this is fine as long as Elias's metadata is intact.
- [ ] Sanity check the brief asks for: searching `"deepfakes"` should return
      deepfake chunks, not healthcare. If that's wrong, my answers will be
      confidently off-topic and no prompt tuning of mine can fix it.

**Heads-up on models:** retrieval needs `nomic-embed-text` pulled in Ollama, and
my generation needs `granite4.1` (exactly the tag the brief tells everyone to
pull). Neither is on the integration machine yet — do
`ollama pull granite4.1 nomic-embed-text` before the first full run. If you keep
the model under a different tag, set `CHATBOT_LLM_MODEL` instead of editing code.

---

## Wouter — `app.py` + Streamlit (you consume me)

You call my layer; here's the stable interface you can build against — it won't change:

```python
from src.chat import ChatSession
session = ChatSession(vector_db=db, embedder=embedder)   # db.load() first
reply, retrieved = session.handle(user_message)
# reply     : str               -> show as the assistant message
# retrieved : list[(Chunk, float)] -> for your Sources expander
```

- Per source row, read: `chunk.metadata.get("title")`, `.get("publisher")`,
  `.get("url")`, and the float score. (These all trace back to Elias's metadata
  contract above.)
- `session.reset()` clears history; `session.help()` returns the help text.
- `handle()` already de-dups nothing — if you want unique documents in the
  Sources panel, de-dup by `chunk.doc_id` (the existing CLI in `app.py` does this).

---

## TL;DR — the three things that will actually bite us

1. **Elias:** every chunk's `metadata` must have `title`, `publisher`, `url`.
   This is the #1 integration risk — it silently degrades my citations.
2. **Sol:** build the store (`python -m src.vector_db`) and keep embeddings
   L2-normalised, or my `0.25` threshold misbehaves.
3. **Everyone:** `ollama pull granite4.1 nomic-embed-text` before the full run,
   and keep the shared `Document` / `Chunk` dataclasses — don't fork them.
