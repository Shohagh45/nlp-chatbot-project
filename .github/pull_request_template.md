## What does this PR do?
<!-- Briefly describe what you implemented -->

## Which module?
- [ ] Hugo — downloader.py
- [ ] Elias — preprocessing.py
- [ ] Sol — embeddings.py / vector_db.py
- [ ] Ksawery — chat.py
- [ ] Wouter — app.py / database.py / README

## How to test it
<!-- What command can the reviewer run to verify your work? -->
```
python -m src.???
```

## Checklist
- [ ] `python -m pytest tests/test_smoke.py` passes
- [ ] I tested my module in isolation
- [ ] No print() statements — using get_logger() instead
