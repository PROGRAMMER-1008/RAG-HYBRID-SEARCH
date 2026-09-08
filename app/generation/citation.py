"""
Phase 3c: Citation verification.

Job: after generation, parse every [N] citation out of the answer,
pair it with the specific sentence/claim it's attached to, and ask
the LLM (in a SEPARATE call, not the one that generated the answer)
whether that cited chunk actually supports that specific claim.

This is the step the guide explicitly calls out as "the quality layer
most RAG systems skip entirely." The failure mode it catches: the LLM
cites a REAL chunk number (so the citation isn't obviously broken),
but attaches it to a claim the chunk doesn't actually support — subtler
and more dangerous than an outright hallucination with no citation at all.
"""

import re
from dataclasses import dataclass

from app.config import JUDGE_PROVIDER
from app.generation.llm_client import generate
from app.generation.prompt import GenerationPrompt, GenerationPrompt as _GP
from app.retrieval.reranker import RerankedResult

CITATION_RE = re.compile(r"\[\s*(?:[A-Za-z]+\s+)?(\d+)\s*\]")

VERIFY_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given ONE claim and ONE source passage. Your only job is to decide whether the source passage actually supports the claim.

Respond with EXACTLY one word: SUPPORTED, UNSUPPORTED, or PARTIAL.
- SUPPORTED: the passage directly and clearly supports the claim.
- PARTIAL: the passage is related but doesn't fully support the specific detail claimed (e.g. right topic, wrong number).
- UNSUPPORTED: the passage does not support the claim at all.

Respond with only that one word, nothing else."""


@dataclass
class CitationCheck:
    claim_sentence: str
    cited_indices: list[int]         # which [N] numbers this sentence cited
    verdicts: dict[int, str]          # {chunk_index: "SUPPORTED"|"PARTIAL"|"UNSUPPORTED"}


def _split_into_claims(answer: str) -> list[str]:
    """Split the answer into claim-sized units for per-claim checking.

    Two-level split, applied per LINE first, then per sentence within
    a line:
      1. Split on newlines first. This was added after a real bug: a
         markdown bullet list like
             * Accuracy ~98% [3]
             * Tversky Score ~0.88 [3]
         has no ". " between bullets, so a pure sentence-period split
         merged the entire multi-fact bulleted block (three different
         citations) into ONE "claim" — meaning verify_citations() was
         checking "does chunk 3 relate to this whole paragraph" instead
         of verifying each individual number against its own citation.
      2. Within each non-empty line, still split on sentence punctuation,
         since a single line can contain multiple citation-bearing
         sentences (e.g. two claims joined in one paragraph line).
      3. If a line contains ONLY a citation marker and no other text
         (e.g. a lone "[1]" on its own line, separated from the claim
         it supports by a blank line), merge it into the previous
         claim instead of treating it as an independent claim. Found
         via a real eval run: llama3.1:8b sometimes writes the formula/
         fact on one line and puts "[1]" alone on the next, which
         otherwise produces two broken units — an uncited factual claim,
         and a citation with no claim text attached (verify_citations()
         would nonsensically ask "does chunk 1 support the claim '[1]'?").

    Still not a linguistically perfect tokenizer (won't handle "Dr.
    Smith" correctly) but this corpus is technical READMEs with
    markdown structure, not that kind of prose — the line-first split
    is what actually matters for correctness here.
    """
    claims = []
    for line in answer.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip a leading bullet marker ("* " or "- ") so the claim text
        # itself doesn't retain the bullet, but keep everything else.
        line = re.sub(r"^[*-]\s+", "", line)

        # Orphaned citation-only line: merge into the previous claim
        # rather than emitting it as its own claim (see docstring point 3).
        if claims and re.fullmatch(r"(?:\[\s*(?:[A-Za-z]+\s+)?\d+\s*\]\s*)+", line):
            claims[-1] = claims[-1] + " " + line
            continue

        sentence_parts = re.split(r"(?<=[.!?])\s+", line)
        for part in sentence_parts:
            part = part.strip()
            if not part:
                continue
            # Split further on ", " ONLY when this comma-separated part
            # itself ends in a citation AND a later part also ends in a
            # citation — the specific "fact [N], fact [N], fact [N]"
            # compound-claim shape found via a real eval run (q06:
            # "corruption_detector [1], failure_forensics [4], ..."),
            # where each comma-segment is really an independent claim
            # about a different project. Deliberately narrow: ordinary
            # prose with a comma before a single TRAILING citation (e.g.
            # "Precision, recall, and F1 were all above 0.95 [1]") has
            # only one citation total and must NOT be split apart, since
            # splitting there would separate one claim's supporting
            # clauses from its own citation.
            comma_segments = part.split(", ")
            segments_with_own_citation = [
                seg for seg in comma_segments if CITATION_RE.search(seg)
            ]
            if len(segments_with_own_citation) >= 2:
                claims.extend(seg.strip() for seg in comma_segments if seg.strip())
            else:
                claims.append(part)
    return claims


def verify_citations(answer: str, numbered_chunks: list[RerankedResult]) -> list[CitationCheck]:
    claims = _split_into_claims(answer)
    checks = []

    for claim in claims:
        cited = [int(n) for n in CITATION_RE.findall(claim)]
        if not cited:
            continue   # sentence made no citation claim — nothing to verify

        verdicts = {}
        for idx in cited:
            if idx < 1 or idx > len(numbered_chunks):
                verdicts[idx] = "UNSUPPORTED"   # citation points to a chunk number that doesn't exist
                continue

            source_chunk = numbered_chunks[idx - 1].text
            verdict = _check_one_claim(claim, source_chunk)
            verdicts[idx] = verdict

        checks.append(CitationCheck(claim_sentence=claim, cited_indices=cited, verdicts=verdicts))

    return checks


def _check_one_claim(claim: str, source_text: str) -> str:
    verify_prompt = _GP(
        system_prompt=VERIFY_SYSTEM_PROMPT,
        user_prompt=f"Claim: {claim}\n\nSource passage:\n{source_text}\n\nVerdict:",
        numbered_chunks=[],
    )
    # Judge uses JUDGE_PROVIDER, NOT GENERATION_PROVIDER — see config.py's
    # JUDGE_PROVIDER docstring. A same-class self-judge (the model judging
    # its own output) was caught making a wrong call on an easy, clearly-
    # correct citation during real eval testing (EVAL_FINDINGS.md, run 4,
    # q02) — routing verification to a separate, stronger model is the fix.
    raw = generate(verify_prompt, provider=JUDGE_PROVIDER).strip().upper()

    # Be defensive: the LLM might not follow the "one word only" instruction
    # perfectly. Look for our three known verdicts anywhere in the response
    # rather than requiring an exact match, which would break on
    # "SUPPORTED." with trailing punctuation or similar minor deviations.
    for verdict in ("SUPPORTED", "PARTIAL", "UNSUPPORTED"):
        if verdict in raw:
            return verdict
    return "UNSUPPORTED"   # fail closed: if we can't parse a verdict, don't silently trust the citation


def citation_accuracy(checks: list[CitationCheck]) -> float:
    """Fraction of individual citations (not sentences) rated SUPPORTED.
    Used later in Phase 4's eval metrics."""
    total = 0
    supported = 0
    for check in checks:
        for verdict in check.verdicts.values():
            total += 1
            if verdict == "SUPPORTED":
                supported += 1
    return supported / total if total else 0.0


if __name__ == "__main__":
    # Real end-to-end test: needs Ollama running (two LLM calls: generate
    # the answer, then verify its citation). Runs on your machine only.
    from app.generation.prompt import build_generation_prompt
    from app.retrieval.pipeline import HybridRetrievalPipeline

    pipeline = HybridRetrievalPipeline()
    query = "What ROC-AUC does the corruption detector achieve?"

    trace = pipeline.retrieve(query)
    prompt = build_generation_prompt(query, trace.final_results)
    answer = generate(prompt)

    print(f"Query: {query}\n")
    print("=== ANSWER ===")
    print(answer)

    checks = verify_citations(answer, trace.final_results)
    print("\n=== CITATION VERIFICATION ===")
    for c in checks:
        print(f"Claim: {c.claim_sentence}")
        print(f"  Verdicts: {c.verdicts}")

    print(f"\nCitation accuracy: {citation_accuracy(checks):.0%}")
