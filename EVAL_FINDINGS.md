# Evaluation Methodology & Findings

This document summarizes how the RAG pipeline was evaluated and what the
process actually found — including real bugs discovered and fixed along
the way, and limitations that are inherent to the model choice rather
than fixable in code.

## Golden dataset

12 hand-written question/answer pairs (`app/eval/golden_set.json`),
deliberately spanning four categories to stress different parts of the
pipeline:

| Type | Count | Tests |
|---|---|---|
| Lookup | 6 | Basic retrieval + generation accuracy |
| Multi-hop | 2 | Retrieval breadth across multiple source files |
| No-answer | 2 | Whether the system honestly declines rather than hallucinating |
| Ambiguous phrasing | 2 | Retrieval robustness to casual/vague wording |

Every golden answer is tied to a specific, verified fact in the corpus —
including one deliberately corrected value (the corruption detector's
ROC-AUC, corrected from an inflated 0.87+ in the original README to the
verified 0.681–0.683) — so correctness can be checked precisely, not
just plausibly.

## Metrics

- **Retrieval hit rate**: did the expected source file(s) actually appear
  in the top-5 reranked chunks? (excludes no-answer questions, where
  "the right file" isn't a meaningful concept)
- **Answer correctness**: LLM-as-judge comparing the generated answer
  against the golden answer (CORRECT / PARTIAL / INCORRECT)
- **Citation accuracy**: for every `[N]` citation in the generated
  answer, a separate LLM call checks whether the cited chunk actually
  supports that specific claim (excludes no-answer questions, where a
  correct decline makes zero citable claims)

## What four evaluation runs actually found

Running the same 12-question golden set repeatedly (same code changes
in between, same model) surfaced four distinct, real bugs in the
citation-verification pipeline — each found by reading actual model
output rather than trusting the aggregate score:

1. **Claim-blob merging**: a markdown bulleted list with no periods
   between items collapsed into one giant "claim" with multiple citations
   attached, so the verifier checked "does chunk 3 support this entire
   paragraph" instead of verifying each bullet independently. Fixed by
   splitting on lines before splitting on sentences.

2. **Orphaned citation lines**: the model sometimes puts a citation
   marker alone on its own line, separated from the claim it supports by
   a blank line (e.g. a formula, then a blank line, then `[1]`). This
   produced two broken units instead of one correct one. Fixed by
   merging a citation-only line into the preceding claim.

3. **Compound comma-separated claims**: a terse answer like
   `"project_a [1], project_b [2], project_c [3]"` became one claim with
   three citations, so each citation was checked against the *entire*
   three-part sentence rather than its own fact — which no single chunk
   could fully support. Fixed with a narrow comma-split rule (only when
   multiple comma-segments each carry their own citation) plus a prompt
   rule instructing the model not to produce this shape in the first
   place.

4. **Citation format drift**: the model wrote `[Cites 1]` instead of
   `[1]` in one run — a real citation, wrong format, silently invisible
   to the original strict `\[(\d+)\]` regex. Fixed by broadening the
   parser defensively and tightening the system prompt's format
   instructions.

Each fix was verified against a full regression suite of every
previously-fixed case before being shipped, specifically to avoid the
common failure mode of fixing today's bug by reintroducing yesterday's.

## A limitation that is NOT a code bug

Run 4 surfaced a case (`q02`) where a single, clearly-correct citation
— parsed correctly, format correct, chunk genuinely supports the claim
on inspection — was still scored `UNSUPPORTED` by the verification
judge. This is not a parsing bug; it's the judge model itself (Llama
3.1 8B) making an incorrect call on a straightforward question.

This mirrors a documented limitation in a related project
(`failure_forensics_README.md`'s "Known limitations" section): a small,
same-class model used as its own judge is not perfectly reliable, and
self-reported confidence has been observed to vary between runs on
identical input.

### Follow-up: implementing and testing a stronger judge

The natural fix — route citation/correctness judging to a separate,
larger model (Groq's `gpt-oss-120b`) instead of whatever model
generated the answer — was implemented as `config.JUDGE_PROVIDER`,
decoupled from `config.GENERATION_PROVIDER`. This is a real
architectural improvement regardless of what follows: it means
verification quality no longer depends on which model happens to be
generating answers.

However, re-running the eval suite with the stronger judge active
still produced the exact same q02 failure — the identical claim
("Focal Tversky Loss... handles class imbalance [1]") cited against
the identical, genuinely-correct chunk still scored `UNSUPPORTED` on
this run. This ruled out the "same-class self-judge" theory as the
root cause; a materially stronger, independent model exhibited the
exact same failure.

Isolated reproduction narrowed it further. Testing the exact claim/
chunk pair standalone (outside the full pipeline) against `gpt-oss-120b`
returned `SUPPORTED` consistently — 4 separate calls across two testing
sessions all agreed the claim was supported. New diagnostic logging
(`EvalResult.cited_chunk_details`, added specifically for this
investigation) confirmed the live failing run's citation `[1]` pointed
to the exact same chunk, verbatim, that the isolated test used — so
there was no hidden data, indexing, or retrieval bug making the two
tests non-equivalent.

**Conclusion: this is LLM sampling non-determinism in the judge call,
not a fixable defect.** The same prompt, sent to the same model, does
not guarantee the same output every time — 3-4 consistent isolated
runs reduced the probability of observing this but didn't rule it out,
and a live pipeline run happened to draw the less-likely outcome. No
prompt rewrite, no stronger model, and no parser change eliminates
sampling variance; the only real mitigations are architectural
(retry-and-vote across multiple judge calls per citation, or accepting
a small non-zero disagreement rate as a documented property of
LLM-as-judge systems generally). This project does not implement
retry-and-vote — it's noted here as the honest next step, not treated
as solved.

## Run-to-run variance

Because generation and judging both go through a non-deterministic LLM,
individual question verdicts shift somewhat between identical runs
(e.g. `q04`, `q08`, `q10`, `q12` each flipped between CORRECT and
PARTIAL across different runs with no code changes in between).
Aggregate metrics across four runs:

| Run | Correctness (fully correct) | Citation accuracy | New bug found |
|---|---|---|---|
| 1 | 75% | 70%* | Claim-blob merging |
| 2 | 92% | 90%* | Orphaned citation lines |
| 3 | 67% | 87% | Compound comma-claims |
| 4 | 83% | 88% | None (judge unreliability observed, not a code bug) |

\* Runs 1–2's citation accuracy numbers are inflated/deflated by the
no-answer exemption bug (fixed after run 2) and are not directly
comparable to runs 3–4.

**Retrieval hit rate was 100% (10/10 scoreable questions) across all
four runs, with zero variance** — this is the one metric in this eval
suite that is fully deterministic (embedding + BM25 + RRF + reranking
have no randomness), and it held steady throughout every code change
made to the generation/citation layer downstream of it.

## Chunking strategy comparison

Per the guide's Phase 4 spec, three chunking strategies were built and
run against the same 12-question golden set, each on its own isolated
index (never touching the production index):

| Strategy | Chunks (post-dedup) | Retrieval Hit | Citation Acc | Correctness |
|---|---|---|---|---|
| Structure-aware (production) | 106 | 100% | 92% | 11 CORRECT / 1 PARTIAL |
| Fixed-size baseline | 81 | 100% | 90% | 9 CORRECT / 3 PARTIAL |
| Semantic (embedding-boundary) | 140 (16 dropped as near-dupes) | 100% | 100% | 10 CORRECT / 2 PARTIAL |

**Honest result: semantic chunking scored highest on citation accuracy,
not the production structure-aware chunker.** This was not the
expected outcome and is reported as measured, not adjusted to match
the prior assumption.

**Why this is plausible**: semantic chunking produced more, smaller,
topically-tighter chunks (140 vs. 106). A citation verifier checking
"does this specific chunk support this specific claim" has an easier
time returning a clean SUPPORTED verdict against a small, single-topic
chunk than a larger structure-aware chunk that may span a header's
topic plus several loosely related bullets. Finer granularity can make
individual citations easier to verify, independent of whether it helps
retrieval — and on this corpus, retrieval hit rate was identical
(100%) across all three strategies, so retrieval wasn't the
differentiator here at all.

**The real trade-off semantic chunking pays**: it needed dedup to drop
16 near-duplicate chunks (vs. zero for structure-aware) — a real sign
of redundant/overlapping segments before cleanup, not a free
improvement.

**Why structure-aware remains the production choice despite this**:
it is the strategy this project's Phase 1-3 debugging actually
hardened — it's where the code-fence parsing bug and the bullet-
boundary-splitting bug were found and fixed through real testing, and
it ties on retrieval with zero chunk redundancy. The eval set is also
only 12 questions; an 8-point citation-accuracy gap on that sample
size is evidence worth taking seriously, not evidence strong enough to
justify switching production strategies on. Semantic chunking is a
legitimate, measured alternative worth a larger-scale follow-up
comparison, not a strategy this project adopted.
