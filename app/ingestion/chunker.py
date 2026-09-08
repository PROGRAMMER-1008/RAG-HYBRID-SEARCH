"""
Phase 1b: Chunker.

Job: turn a RawDocument's full text into a list of Chunk objects, each
small enough to embed meaningfully and each carrying enough metadata
to be cited back to its source later.

Strategy: structure-aware (split on markdown headers) as the primary
cut, with a fixed-size-with-overlap fallback applied *within* any
section that's still too long after the header split.

Why this order and not the reverse: a markdown header is a real
semantic boundary the author already drew for us — splitting there
first respects content structure. Fixed-size splitting only kicks in
as a fallback for sections that are long prose with no natural
sub-boundary; using it as the *primary* strategy on our corpus would
mean carelessly slicing straight through a table or a code block.
"""

import re
import hashlib
from dataclasses import dataclass, field

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.loader import RawDocument

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _mask_code_fences(text: str) -> str:
    """Blank out the content of fenced code blocks before header detection.

    Bash comments like `# 1. Install dependencies` inside a ```bash block
    start with '#' at line-start and would otherwise be misread as markdown
    headers by HEADER_RE. We replace fence interiors with spaces (preserving
    string length and line count) so section split offsets computed against
    the masked text still line up with the original text.
    """
    fence_positions = [m.start() for m in CODE_FENCE_RE.finditer(text)]
    if len(fence_positions) < 2:
        return text  # no complete fence pair, nothing to mask

    masked = list(text)
    # Fences toggle in/out of code; mask between every open/close pair
    for i in range(0, len(fence_positions) - 1, 2):
        start, end = fence_positions[i], fence_positions[i + 1]
        for j in range(start, end):
            if masked[j] not in ("\n",):
                masked[j] = " "
    return "".join(masked)


@dataclass
class Chunk:
    text: str
    source_file: str
    heading_path: str          # e.g. "Technical Architecture > Traffic Model (BPR)"
    chunk_index: int           # position within this section (0 if not sub-split)
    strategy: str = field(default="header")   # "header" | "fixed_size_fallback"

    @property
    def chunk_id(self) -> str:
        """Stable unique ID, used as the ChromaDB primary key later.

        Uses a short hash of the FULL heading_path rather than truncating
        it — truncating first and hashing/joining second was a real bug:
        long files where every heading_path shares a long common H1 prefix
        (e.g. every section under "Delhi Digital Twin — Decision
        Intelligence System > ...") collapsed to duplicate IDs after a
        naive 60-char slice, because the truncation happened before
        uniqueness was established, not after.
        """
        heading_hash = hashlib.sha1(self.heading_path.encode("utf-8")).hexdigest()[:10]
        return f"{self.source_file}::{heading_hash}::{self.chunk_index}"


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split raw markdown into (heading_path, section_body) pairs.

    heading_path accumulates parent headers so nested sections stay
    unambiguous, e.g. a "## Technical Architecture" followed by a
    "### Traffic Model" produces heading_path
    "Technical Architecture > Traffic Model", not just "Traffic Model".
    """
    masked = _mask_code_fences(text)
    matches = list(HEADER_RE.finditer(masked))

    sections: list[tuple[str, str]] = []
    if matches:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Introduction", preamble))
    else:
        return [("Full Document", text.strip())]

    heading_stack: list[tuple[int, str]] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        heading_stack = [(lvl, name) for lvl, name in heading_stack if lvl < level]
        heading_stack.append((level, title))
        heading_path = " > ".join(name for _, name in heading_stack)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if body:
            sections.append((heading_path, body))

    return sections


def _fixed_size_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback splitter for a single section that's still too long.

    Cuts near `chunk_size` characters, stepping back `overlap` characters
    each time so content right at a cut boundary appears in both
    neighboring chunks instead of being silently truncated mid-thought.

    Prefers cutting at a bullet-point boundary ("\\n- " or "\\n* ") close
    to the target size, rather than an arbitrary character offset. This
    was added after a real bug: a hard character cut split a single
    markdown bullet mid-sentence ("- **Backward root-cause an" | "over-
    confident framing\\n- **Backward root-cause analysis**...") across
    two chunks, and the chunk containing the actual detailed explanation
    started mid-word — likely hurting how a downstream reranker judged
    its coherence. Searching backward from the target cut point for the
    nearest bullet boundary avoids reproducing that failure mode.
    """
    if len(text) <= chunk_size:
        return [text]

    # How far back from the ideal cut point we're willing to look for a
    # clean bullet boundary before giving up and hard-cutting anyway.
    # Too small and we rarely find a boundary; too large and chunks
    # vary wildly in size instead of clustering near chunk_size.
    search_window = min(150, chunk_size // 3)

    pieces = []
    start = 0
    while start < len(text):
        ideal_end = min(start + chunk_size, len(text))

        if ideal_end == len(text):
            pieces.append(text[start:ideal_end])
            break

        # Look for the last bullet-start ("\n- " or "\n* ") within the
        # search window before ideal_end. rfind returns -1 if not found.
        window_start = max(start, ideal_end - search_window)
        search_region = text[window_start:ideal_end]
        bullet_pos = max(search_region.rfind("\n- "), search_region.rfind("\n* "))

        if bullet_pos != -1:
            end = window_start + bullet_pos  # cut right before the bullet marker
        else:
            end = ideal_end  # no clean boundary nearby, fall back to hard cut

        if end <= start:  # safety: never produce an empty or negative-length piece
            end = ideal_end

        pieces.append(text[start:end])
        start = max(end - overlap, end - overlap) if end - overlap > start else end
    return pieces


def chunk_document(doc: RawDocument, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Full pipeline for one document: header split, then fallback split per section."""
    sections = _split_by_headers(doc.text)
    chunks: list[Chunk] = []

    for heading_path, body in sections:
        pieces = _fixed_size_split(body, chunk_size, overlap)
        strategy = "header" if len(pieces) == 1 else "fixed_size_fallback"
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source_file=doc.source_path,
                    heading_path=heading_path,
                    chunk_index=idx,
                    strategy=strategy,
                )
            )
    return chunks


def chunk_corpus(docs: list[RawDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    from app.ingestion.loader import load_corpus
    from app.config import RAW_DOCS_DIR

    docs = load_corpus(RAW_DOCS_DIR)
    all_chunks = chunk_corpus(docs)

    print(f"Total chunks: {len(all_chunks)}\n")
    for c in all_chunks:
        print(f"[{c.strategy:20s}] {c.source_file:35s} | {c.heading_path[:50]:50s} | {len(c.text)} chars")