"""
Phase 4: Evaluation runner.

Job: run every golden question through the FULL pipeline (retrieve ->
generate -> verify citations) and score it on the dimensions the guide
specifies: retrieval relevance, answer correctness, citation accuracy,
plus a specific check for whether "no_answer" questions are handled
honestly rather than hallucinated.

This is what turns "I built a RAG system" into "I built a RAG system
that achieves X% retrieval hit rate and Y% citation accuracy on a
12-question eval suite" — the guide's whole point about leading with
numbers, not vibes.
"""

import json
from dataclasses import dataclass, asdict, field

from app.config import GOLDEN_SET_PATH, JUDGE_PROVIDER
from app.generation.citation import citation_accuracy, verify_citations
from app.generation.llm_client import generate
from app.generation.prompt import GenerationPrompt, build_generation_prompt
from app.retrieval.pipeline import HybridRetrievalPipeline

JUDGE_SYSTEM_PROMPT = """You are grading whether a generated answer correctly conveys the same information as a golden reference answer. You will be given the QUESTION, the GOLDEN ANSWER, and the GENERATED ANSWER.

Respond with EXACTLY one word: CORRECT, PARTIAL, or INCORRECT.
- CORRECT: the generated answer conveys the same key facts as the golden answer (wording can differ).
- PARTIAL: the generated answer gets some but not all key facts right, or is vague where the golden answer is specific.
- INCORRECT: the generated answer contradicts the golden answer, or (for a question marked NOT ANSWERABLE) confidently invents an answer instead of declining.

Respond with only that one word, nothing else."""


@dataclass
class EvalResult:
    question_id: str
    question: str
    question_type: str
    golden_answer: str
    generated_answer: str
    retrieval_hit: bool          # did expected_source_file(s) appear in the final retrieved chunks?
    retrieved_files: list[str]
    correctness_verdict: str     # CORRECT | PARTIAL | INCORRECT
    citation_accuracy_score: float
    cited_chunk_details: list[dict] = field(default_factory=list)
    # Records exactly which chunk (heading_path + a text snippet) sat at
    # each [N] position for THIS run. Added after a real debugging gap:
    # a citation was scored UNSUPPORTED in a live eval run, but the exact
    # same claim/chunk pair tested in isolation consistently scored
    # SUPPORTED — with no record of which chunk was actually at position
    # [1] during the failing run, there was no way to confirm whether
    # retrieval put a DIFFERENT chunk there than assumed. This field
    # closes that gap for future debugging.


def _check_retrieval_hit(expected_source_file: str | None, retrieved_files: list[str]) -> bool:
    """For no_answer questions, expected_source_file is None — a 'hit' there
    means retrieval correctly did NOT confidently surface an unrelated file
    as if it were the answer. We don't penalize retrieval for returning
    SOME chunks (the pipeline always returns top-k) — that's generation's
    job to handle via the honest non-answer instruction, not retrieval's.
    So no_answer questions are excluded from the retrieval_hit metric
    entirely (see run_eval): it isn't the right lens for that case.
    """
    if expected_source_file is None:
        return True   # not meaningfully applicable; see docstring

    if expected_source_file.startswith("MULTIPLE:"):
        required_files = [f.strip() for f in expected_source_file.replace("MULTIPLE:", "").split(",")]
        return all(f in retrieved_files for f in required_files)

    return expected_source_file in retrieved_files


def _judge_correctness(question: str, golden_answer: str, generated_answer: str) -> str:
    judge_prompt = GenerationPrompt(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=f"QUESTION: {question}\n\nGOLDEN ANSWER: {golden_answer}\n\nGENERATED ANSWER: {generated_answer}\n\nVerdict:",
        numbered_chunks=[],
    )
    # JUDGE_PROVIDER, not GENERATION_PROVIDER — see config.py. Keeps eval
    # scoring on a consistent, stronger judge regardless of which model
    # generated the answer being scored, so correctness numbers stay
    # comparable across an Ollama-generated run and a Groq-generated run.
    raw = generate(judge_prompt, provider=JUDGE_PROVIDER).strip().upper()
    for verdict in ("CORRECT", "PARTIAL", "INCORRECT"):
        if verdict in raw:
            return verdict
    return "INCORRECT"   # fail closed, same principle as citation.py's parser


def run_eval() -> list[EvalResult]:
    with open(GOLDEN_SET_PATH) as f:
        golden_set = json.load(f)

    pipeline = HybridRetrievalPipeline()
    results = []

    for item in golden_set:
        print(f"Evaluating {item['id']}: {item['question']}")

        trace = pipeline.retrieve(item["question"])
        retrieved_files = list({r.source_file for r in trace.final_results})

        prompt = build_generation_prompt(item["question"], trace.final_results)
        answer = generate(prompt)

        checks = verify_citations(answer, trace.final_results)
        cit_acc = citation_accuracy(checks)

        verdict = _judge_correctness(item["question"], item["golden_answer"], answer)
        hit = _check_retrieval_hit(item["expected_source_file"], retrieved_files)

        # Record exactly which chunk sat at each [N] position this run —
        # the diagnostic data that was missing when q02's citation
        # verdict couldn't be reproduced in isolation. 1-indexed to match
        # the [1], [2]... numbering the LLM and judge both see.
        chunk_details = [
            {
                "index": i,
                "source_file": r.source_file,
                "heading_path": r.heading_path,
                "text_snippet": r.text[:150],
            }
            for i, r in enumerate(trace.final_results, start=1)
        ]

        results.append(
            EvalResult(
                question_id=item["id"],
                question=item["question"],
                question_type=item["type"],
                golden_answer=item["golden_answer"],
                generated_answer=answer,
                retrieval_hit=hit,
                retrieved_files=retrieved_files,
                correctness_verdict=verdict,
                citation_accuracy_score=cit_acc,
                cited_chunk_details=chunk_details,
            )
        )

    return results


def print_report(results: list[EvalResult]) -> None:
    total = len(results)
    # Exclude no_answer questions from the retrieval hit-rate denominator —
    # see _check_retrieval_hit's docstring for why that metric doesn't
    # meaningfully apply to them.
    scoreable_for_retrieval = [r for r in results if r.question_type != "no_answer"]
    retrieval_hits = sum(r.retrieval_hit for r in scoreable_for_retrieval)

    correct = sum(r.correctness_verdict == "CORRECT" for r in results)
    partial = sum(r.correctness_verdict == "PARTIAL" for r in results)
    incorrect = sum(r.correctness_verdict == "INCORRECT" for r in results)

    # Same exemption applies to citation accuracy: a correctly-declined
    # no_answer question makes ZERO citable claims by design, so
    # citation_accuracy() returning 0.0 for "nothing to check" would
    # penalize exactly the honest-non-answer behavior we want to see —
    # not a real citation failure. Found via a real eval run where q08/q09
    # (both correctly declined) dragged the average down despite the
    # generation being exactly right.
    scoreable_for_citations = [r for r in results if r.question_type != "no_answer"]
    avg_citation_acc = (
        sum(r.citation_accuracy_score for r in scoreable_for_citations) / len(scoreable_for_citations)
        if scoreable_for_citations else 0.0
    )

    print("\n" + "=" * 70)
    print("EVAL REPORT")
    print("=" * 70)
    print(f"Total questions: {total}")
    print(f"Retrieval hit rate: {retrieval_hits}/{len(scoreable_for_retrieval)} "
          f"({retrieval_hits/len(scoreable_for_retrieval):.0%}) [excludes no_answer questions]")
    print(f"Answer correctness: {correct} CORRECT, {partial} PARTIAL, {incorrect} INCORRECT "
          f"({correct/total:.0%} fully correct)")
    print(f"Average citation accuracy: {avg_citation_acc:.0%} [excludes no_answer questions]")

    print("\nPer-question breakdown:")
    for r in results:
        hit_marker = "HIT " if r.retrieval_hit else "MISS"
        print(f"  [{r.question_type:10s}] {hit_marker} | {r.correctness_verdict:10s} | "
              f"cit={r.citation_accuracy_score:.0%} | {r.question_id}: {r.question[:60]}")

        if r.correctness_verdict == "INCORRECT" or not r.retrieval_hit:
            print(f"      GOLDEN:    {r.golden_answer[:100]}")
            print(f"      GENERATED: {r.generated_answer[:100]}")


def save_results(results: list[EvalResult], path: str = "data/eval_results.json") -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nFull results saved to {path}")


if __name__ == "__main__":
    results = run_eval()
    print_report(results)
    save_results(results)