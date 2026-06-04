#!/usr/bin/env python3
"""Build the NLP chatbot submission zip.

Validates the working tree against an include/exclude policy, prints a
manifest, and writes dist/nlp_chatbot_project.zip with every file nested
under a top-level nlp_chatbot_project/ directory so extraction yields a
clean folder rather than loose files.

Usage:
    python make_submission.py            # validate + build (warn on missing optional data)
    python make_submission.py --dry-run  # validate + print manifest, do NOT write zip
    python make_submission.py --strict   # refuse to build if any REQUIRED file is missing

Stdlib only. No new dependencies.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PREFIX = "nlp_chatbot_project"
DIST_DIR = ROOT / "dist"
ZIP_PATH = DIST_DIR / "nlp_chatbot_project.zip"

# --- Output helpers --------------------------------------------------------

_YELLOW = "\033[33m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_RESET = "\033[0m"


def _color(text, code):
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


def warn(text):
    return _color(text, _YELLOW)


def err(text):
    return _color(text, _RED)


def ok(text):
    return _color(text, _GREEN)


# --- Policy ----------------------------------------------------------------

# Files that must always exist or the build aborts (exit 1) regardless of flags.
ALWAYS_PRESENT = [
    "README.md",
    "requirements.txt",
    "app.py",
    "src/utils.py",
    "src/chat.py",
    "src/embeddings.py",
    "src/vector_db.py",
]

# Files required for a complete submission. Missing -> WARN (default) or abort (--strict).
REQUIRED_DATA = [
    "data/processed/chunks.json",
    "data/vector_store/embeddings.npz",
    "data/vector_store/chunks.json",
]

# Standalone files to include verbatim.
INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "app.py",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/processed/chunks.json",
    "data/vector_store/.gitkeep",
    "data/vector_store/embeddings.npz",
    "data/vector_store/chunks.json",
]

# Directories to include by extension.
INCLUDE_DIRS = [
    ("src", ".py"),
    ("tests", ".py"),
    ("docs", ".md"),
]

# Paths that override exclusion rules (the .gitkeep files keep empty dirs alive).
WHITELIST = {
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/vector_store/.gitkeep",
}

EXCLUDE_DIR_NAMES = {
    ".venv",
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".vscode",
    ".idea",
    "dist",
}

EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".xlsx", ".xls", ".docx", ".log"}

EXCLUDE_NAMES = {
    ".DS_Store",
    ".env",
    "documents.db",
    "make_submission.py",
    "20260331_dataset2.xlsx",
}


def is_excluded(rel_path):
    """Return True if rel_path (posix string) must never appear in the zip.

    Whitelist entries override every exclude rule.
    """
    if rel_path in WHITELIST:
        return False

    parts = rel_path.split("/")

    # Any excluded directory component anywhere in the path.
    for part in parts[:-1]:
        if part in EXCLUDE_DIR_NAMES:
            return True

    name = parts[-1]
    if name in EXCLUDE_NAMES:
        return True

    suffix = Path(name).suffix.lower()
    if suffix in EXCLUDE_EXTS:
        return True

    # data/raw/* is excluded (downloaded PDFs/HTML), except the whitelisted .gitkeep.
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "raw":
        return True

    # data/documents.db is regenerable.
    if rel_path == "data/documents.db":
        return True

    return False


# --- Discovery -------------------------------------------------------------


def gather_candidates():
    """Return a sorted, de-duplicated list of relative posix paths to consider."""
    candidates = set()

    for rel in INCLUDE_FILES:
        candidates.add(rel)

    for dirname, ext in INCLUDE_DIRS:
        base = ROOT / dirname
        if not base.is_dir():
            continue
        for path in base.rglob(f"*{ext}"):
            if path.is_symlink():
                continue
            if not path.is_file():
                continue
            candidates.add(path.relative_to(ROOT).as_posix())

    return sorted(candidates)


def fmt_size(num_bytes):
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def category_for(rel):
    if rel.startswith("src/"):
        return "src/*.py"
    if rel.startswith("tests/"):
        return "tests/*.py"
    if rel.startswith("docs/"):
        return "docs/*.md"
    if rel.startswith("data/"):
        return "data/"
    return "root files"


CATEGORY_ORDER = ["root files", "src/*.py", "tests/*.py", "docs/*.md", "data/"]


# --- Validation ------------------------------------------------------------


def git_info():
    def run(cmd):
        try:
            out = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return "unknown"

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = run(["git", "rev-parse", "--short", "HEAD"])
    return branch, commit


REBUILD_HINT = (
    "  Rebuild the data artifacts (see README Usage):\n"
    "    python -m src.downloader 20260331_dataset2.xlsx\n"
    "    python -m src.preprocessing\n"
    "    python -m src.vector_db"
)


def main():
    parser = argparse.ArgumentParser(
        description="Build the nlp_chatbot_project submission zip."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the manifest but do not write the zip",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="abort (exit 1) if any required submission file is missing",
    )
    opts = parser.parse_args()

    branch, commit = git_info()
    print(f"{ZIP_PREFIX} — submission builder")
    print(f"branch: {branch}   commit: {commit}")
    print()

    # 1. Always-present files: abort immediately if any are missing.
    missing_core = [rel for rel in ALWAYS_PRESENT if not (ROOT / rel).is_file()]
    if missing_core:
        print(err("FATAL: core project files are missing:"))
        for rel in missing_core:
            print(err(f"  - {rel}"))
        print("Cannot build a submission without these. Aborting.")
        return 1

    # 2. README placeholder guard.
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    if "(Student Number)" in readme_text:
        print(err("FATAL: README.md still contains the placeholder '(Student Number)'."))
        print(err("Fill in the real student numbers before building. Aborting."))
        return 1

    # 3. Required-data validation.
    print("Validating required files...")
    # Echo the core files as OK (we already proved they exist).
    for rel in ["README.md", "requirements.txt", "app.py"]:
        print(f"  [OK]   {rel}")

    missing_required = [rel for rel in REQUIRED_DATA if not (ROOT / rel).is_file()]
    for rel in REQUIRED_DATA:
        if (ROOT / rel).is_file():
            print(f"  [OK]   {rel}")
        else:
            print(warn(f"  [WARN] {rel}  (missing — required for final submission)"))
    print()

    if missing_required and opts.strict:
        print(err("STRICT MODE: refusing to build an incomplete submission."))
        print(err("Missing required files:"))
        for rel in missing_required:
            print(err(f"  - {rel}"))
        print(REBUILD_HINT)
        return 1

    # 4. Build the candidate list, applying exclude rules as a safety net.
    included = []
    for rel in gather_candidates():
        if is_excluded(rel):
            continue
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            continue
        included.append(rel)

    # 5. Manifest grouped by category.
    groups = {}
    for rel in included:
        cat = category_for(rel)
        size = (ROOT / rel).stat().st_size
        bucket = groups.setdefault(cat, {"count": 0, "size": 0})
        bucket["count"] += 1
        bucket["size"] += size

    total_count = len(included)
    total_size = sum((ROOT / rel).stat().st_size for rel in included)

    print("Manifest:")
    ordered_cats = [c for c in CATEGORY_ORDER if c in groups]
    ordered_cats += [c for c in groups if c not in CATEGORY_ORDER]
    for cat in ordered_cats:
        b = groups[cat]
        print(f"  {b['count']:>3}  {cat:<28} ({fmt_size(b['size'])})")
    print(f"  total: {total_count} files, {fmt_size(total_size)}")
    print()

    if opts.dry_run:
        print("Dry run — no zip written.")
        if missing_required:
            print(
                warn(
                    f"Note: {len(missing_required)} required file(s) still missing; "
                    "rerun with --strict before final upload."
                )
            )
        return 0

    # 6. Build the zip (idempotent overwrite).
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building {ZIP_PATH.relative_to(ROOT)} ...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in included:
            arcname = f"{ZIP_PREFIX}/{rel}"
            zf.write(ROOT / rel, arcname)

    compressed = ZIP_PATH.stat().st_size
    print(f"Wrote {ZIP_PATH.relative_to(ROOT)} ({fmt_size(compressed)} compressed)")
    print()

    if missing_required:
        print(
            warn(
                f"WARNING: built with {len(missing_required)} required file(s) missing "
                "— this zip is NOT submission-ready."
            )
        )
        for rel in missing_required:
            print(warn(f"  - {rel}"))
        print()

    print("Next steps:")
    print(
        "  - Extract and run `pip install -r requirements.txt && python app.py` "
        "to sanity-check."
    )
    print("  - Re-run with `--strict` before final upload to Moodle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
