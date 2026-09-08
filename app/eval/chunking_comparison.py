"""
Phase 4: Chunking strategy comparison.

Job: build three separate indexes — the production structure-aware
chunker, a fixed-size baseline, and a semantic (embedding-boundary)
strategy — over ISOLATED ChromaDB collections and BM25 pickles (never
touching the real production index), then run the same golden eval
suite against each and report retrieval-hit-rate and citation-accuracy
side by side. This is what turns "structure-aware chunking is the
right call for this corpus" from an assumption into a measured claim.

Usage:
    python -m app.eval.chunking_comparison

Needs Hub access (embedding model) and Ollama/Groq running, same as
the regular evaluator — run on your machine, not the dev sandbox.
"""

import json
from pathlib import Path

from app.config import BASE_DIR, GOLDEN_SET_PATH, RAW_DOCS_DIR
from app.eval.chunking_strategies import fixed_size_chunk_corpus, semantic_chunk_corpus
from app.generation.citation import citation_accuracy, verify_citations
from app.generation.llm_client import generate
from app.generation.prompt import build_generation_prompt
from app.ingestion.chunker import chunk_corpus
from app.ingestion.indexer import build_indexes
from app.ingestion.loader import load_corpus
from app.eval.evaluator import _check_retrieval_hit, _judge_correctness

COMPARISON_DIR = BASE_DIR / "data" / "chunking_comparison"


def _run_eval_against_index(chroma_dir: Path, bm25_path: Path, collection_name: str, label: str) -> dict:
    """Run the golden set against ONE specific index build, using the
    same retrieval->generate->verify chain the main evaluator uses, but
    pointed at a strategy-specific isolated index instead of the
    production one — via explicit constructor parameters (see
    DenseRetriever/SparseRetriever/HybridRetrievalPipeline), not a
    module-attribute override.
    """
    from app.retrieval.pipeline import HybridRetrievalPipeline

    with open(GOLDEN_SET_PATH) as f:
        golden_set = json.load(f)

    pipeline = HybridRetrievalPipeline(
        chroma_dir=chroma_dir, collection_name=collection_name, bm25_path=bm25_path
    )

    hits, total_scoreable = 0, 0
    cit_scores = []
    correctness = {"CORRECT": 0, "PARTIAL": 0, "INCORRECT": 0}

    for item in golden_set:
        trace = pipeline.retrieve(item["question"])
        retrieved_files = list({r.source_file for r in trace.final_results})

        prompt = build_generation_prompt(item["question"], trace.final_results)
        answer = generate(prompt)

        checks = verify_citations(answer, trace.final_results)
        cit_acc = citation_accuracy(checks)

        verdict = _judge_correctness(item["question"], item["golden_answer"], answer)
        correctness[verdict] = correctness.get(verdict, 0) + 1

        if item["type"] != "no_answer":
            total_scoreable += 1
            if _check_retrieval_hit(item["expected_source_file"], retrieved_files):
                hits += 1
            cit_scores.append(cit_acc)

        print(f"  [{label}] {item['id']}: {verdict}")

    return {
        "label": label,
        "retrieval_hit_rate": hits / total_scoreable if total_scoreable else 0.0,
        "avg_citation_accuracy": sum(cit_scores) / len(cit_scores) if cit_scores else 0.0,
        "correctness": correctness,
        "total": len(golden_set),
    }


def run_comparison():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_corpus(RAW_DOCS_DIR)

    results = []

    print("\n=== Building structure-aware index (production chunker) ===")
    structure_chunks = chunk_corpus(docs)
    build_indexes(
        chunks=structure_chunks,
        chroma_dir=COMPARISON_DIR / "structure_aware_chroma",
        bm25_path=COMPARISON_DIR / "structure_aware_bm25.pkl",
        collection_name="cmp_structure_aware",
    )

    print("\n=== Building fixed-size baseline index ===")
    from app.config import CHUNK_SIZE, CHUNK_OVERLAP
    fixed_chunks = fixed_size_chunk_corpus(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    build_indexes(
        chunks=fixed_chunks,
        chroma_dir=COMPARISON_DIR / "fixed_size_chroma",
        bm25_path=COMPARISON_DIR / "fixed_size_bm25.pkl",
        collection_name="cmp_fixed_size",
    )

    print("\n=== Building semantic index ===")
    from sentence_transformers import SentenceTransformer
    from app.config import EMBEDDING_MODEL
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    semantic_chunks = semantic_chunk_corpus(docs, embed_model, CHUNK_SIZE)
    build_indexes(
        chunks=semantic_chunks,
        chroma_dir=COMPARISON_DIR / "semantic_chroma",
        bm25_path=COMPARISON_DIR / "semantic_bm25.pkl",
        collection_name="cmp_semantic",
    )

    print("\n=== Running golden eval suite against each strategy ===")
    results.append(_run_eval_against_index(
        COMPARISON_DIR / "structure_aware_chroma", COMPARISON_DIR / "structure_aware_bm25.pkl",
        "cmp_structure_aware", "structure_aware"))
    results.append(_run_eval_against_index(
        COMPARISON_DIR / "fixed_size_chroma", COMPARISON_DIR / "fixed_size_bm25.pkl",
        "cmp_fixed_size", "fixed_size"))
    results.append(_run_eval_against_index(
        COMPARISON_DIR / "semantic_chroma", COMPARISON_DIR / "semantic_bm25.pkl",
        "cmp_semantic", "semantic"))

    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<18} {'Retrieval Hit':<15} {'Citation Acc':<15} {'Correctness'}")
    for r in results:
        c = r["correctness"]
        print(f"{r['label']:<18} {r['retrieval_hit_rate']:<15.0%} {r['avg_citation_accuracy']:<15.0%} "
              f"{c.get('CORRECT',0)}C/{c.get('PARTIAL',0)}P/{c.get('INCORRECT',0)}I")

    with open(COMPARISON_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {COMPARISON_DIR / 'comparison_results.json'}")


if __name__ == "__main__":
    run_comparison()
