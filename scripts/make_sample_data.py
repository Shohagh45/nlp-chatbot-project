"""
scripts/make_sample_data.py
===========================
Generate sample raw documents + a manifest so `python -m src.preprocessing`
can be run and tested WITHOUT Hugo's downloader, the real dataset, or Ollama.

This writes a few HTML files into data/raw/ and a data/raw/manifest.json that
points at them with fetch_status='success'. The text is long enough that the
chunker produces several overlapping chunks, so you can eyeball the output.

Usage:
    python scripts/make_sample_data.py        # create samples
    python -m src.preprocessing               # -> data/processed/chunks.json
    python scripts/make_sample_data.py --clean # remove the samples again
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import MANIFEST_PATH, RAW_DIR  # noqa: E402

# A reusable paragraph of EU-AI-policy-flavoured text (~120 words).
_PARA = (
    "The Artificial Intelligence Act establishes a harmonised legal framework "
    "for the development, placing on the market, and use of artificial "
    "intelligence systems across the European Union. It follows a risk-based "
    "approach, classifying systems according to the level of risk they pose to "
    "health, safety, and fundamental rights. High-risk systems are subject to "
    "strict obligations including risk management, data governance, "
    "transparency, human oversight, and robustness before they can be placed on "
    "the market. Transparency is central: providers of systems intended to "
    "interact with people must ensure those people know they are dealing with a "
    "machine, and deepfakes must be clearly labelled so that citizens can make "
    "informed judgements about the content they encounter online every day."
)

_PARA2 = (
    "Accountability rests with the human actors who deploy or rely on a system, "
    "not with the system itself. Deployers of high-risk artificial intelligence "
    "must maintain meaningful human oversight, keep logs, and be able to explain "
    "the basis of automated decisions to the individuals those decisions affect. "
    "This reflects the human-centric philosophy that runs through European "
    "digital policy. The regulation also stresses AI literacy: nearly half of EU "
    "citizens lack basic digital skills, which creates real risks when "
    "responsibility for consequential decisions is quietly shifted onto opaque "
    "systems. Raising literacy among providers, deployers, and the public keeps "
    "human judgement central to how these technologies are actually used in work "
    "and in education across the Union today."
)


def _html(title: str, paragraphs: list[str]) -> str:
    body = "\n".join(f"      <p>{p}</p>" for p in paragraphs)
    return f"""<html><head><title>{title}</title><style>.x{{}}</style></head>
<body>
  <nav>Home | About | Contact</nav>
  <header>European Commission — boilerplate that should be stripped</header>
  <main>
      <h1>{title}</h1>
{body}
  </main>
  <footer>Copyright 2025 European Union</footer>
  <script>analytics.track('pageview')</script>
</body></html>
"""


# (id, title, theme, paragraphs) — doc 1 is long enough to force multiple chunks.
SAMPLES = [
    (
        "SAMPLE-01",
        "The EU Artificial Intelligence Act",
        "Trust",
        [_PARA, _PARA2, _PARA, _PARA2, _PARA],   # ~600 words -> 2+ chunks
    ),
    (
        "SAMPLE-02",
        "Deepfakes and Information Integrity",
        "Bias, Discrimination & Fairness",
        [_PARA2, _PARA],
    ),
]


def create() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for doc_id, title, theme, paras in SAMPLES:
        html_path = RAW_DIR / f"{doc_id}.html"
        html_path.write_text(_html(title, paras), encoding="utf-8")
        manifest.append(
            {
                "id": doc_id,
                "title": title,
                "url": f"https://example.eu/{doc_id.lower()}",
                "publisher": "European Commission",
                "date": "2025",
                "theme": theme,
                "keywords": "transparency, accountability, fairness",
                "citation": "European Commission, 2025.",
                "content_type": "html",
                "local_path": str(html_path),
                "fetch_status": "success",
                "fetch_error": None,
            }
        )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {len(manifest)} sample docs + manifest at {MANIFEST_PATH}")
    print("Now run:  python -m src.preprocessing")


def clean() -> None:
    removed = 0
    for doc_id, *_ in SAMPLES:
        p = RAW_DIR / f"{doc_id}.html"
        if p.exists():
            p.unlink()
            removed += 1
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
        removed += 1
    print(f"Removed {removed} sample file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="remove the sample files")
    args = parser.parse_args()
    clean() if args.clean else create()
