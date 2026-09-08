"""
Phase 1a: Loader.

Job: read raw files off disk, return a uniform in-memory representation
regardless of source format. Downstream code (chunker) should never need
to know whether a document originally came from .md, .pdf, or .html.

This is the "multi-format document loader" from the guide's Phase 1,
scoped to markdown for now since that's our corpus — but the Document
dataclass is format-agnostic on purpose, so adding a PDF loader later
is additive, not a rewrite.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawDocument:
    """One source file, loaded and minimally normalized."""
    source_path: str      # filename, used later for citation ("[1] delhi_digital_twin_README.md")
    text: str              # full raw text content


def load_markdown_file(path: Path) -> RawDocument:
    """Read a single .md file as UTF-8 plain text.

    No transformation here — markdown structure (headers) is preserved
    on purpose because the chunker needs it to split by section.
    """
    text = path.read_text(encoding="utf-8")
    return RawDocument(source_path=path.name, text=text)


def load_corpus(raw_dir: Path) -> list[RawDocument]:
    """Load every .md file in a directory.

    Returns them sorted by filename for reproducible ordering —
    matters when debugging ("chunk #14 came from which file, in what order").
    """
    md_files = sorted(raw_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {raw_dir}")
    return [load_markdown_file(p) for p in md_files]


if __name__ == "__main__":
    # Quick manual check: does loading actually work on our real corpus?
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from app.config import RAW_DOCS_DIR

    docs = load_corpus(RAW_DOCS_DIR)
    for d in docs:
        print(f"{d.source_path}: {len(d.text)} chars")
