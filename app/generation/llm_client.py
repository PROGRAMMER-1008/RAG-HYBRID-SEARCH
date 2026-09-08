"""
Phase 3b: Generation client.

Job: send a GenerationPrompt to an LLM and get back raw text.
Supports two providers behind one interface, per config.GENERATION_PROVIDER:
  - "ollama": local, free, what you'll use for day-to-day dev iteration
  - "groq": hosted, free tier, what the portfolio demo link runs on
    (so a recruiter clicking your link doesn't need YOUR GPU running)

Neither the retrieval pipeline nor the citation verifier needs to know
which provider is active — they just call generate(prompt) and get text.

The optional `provider` parameter lets a caller override the default
for ONE call without touching global config — this is how judging
(citation.py, evaluator.py) can force Groq for verification even when
GENERATION_PROVIDER is set to "ollama" for the actual answer. See
config.JUDGE_PROVIDER's docstring for why that separation matters.
"""

from app.config import GENERATION_PROVIDER, GROQ_MODEL, OLLAMA_MODEL
from app.generation.prompt import GenerationPrompt


def generate(prompt: GenerationPrompt, provider: str | None = None) -> str:
    active_provider = provider if provider is not None else GENERATION_PROVIDER
    if active_provider == "ollama":
        return _generate_ollama(prompt)
    elif active_provider == "groq":
        return _generate_groq(prompt)
    else:
        raise ValueError(f"Unknown provider: {active_provider!r}")


def _generate_ollama(prompt: GenerationPrompt) -> str:
    import ollama   # imported lazily so groq-only environments don't need this installed

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ],
    )
    return response["message"]["content"]


def _generate_groq(prompt: GenerationPrompt) -> str:
    import os
    import re as _re
    from groq import Groq   # imported lazily, mirrors the ollama import above

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key at https://console.groq.com and set it before "
            "switching config.GENERATION_PROVIDER to 'groq'."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ],
        # include_reasoning=False requires a current SDK (requirements.txt
        # now pins groq==1.7.0). This raised TypeError under the
        # previously-pinned groq==0.11.0, which predates this parameter —
        # if you still see that error, run `pip install -r requirements.txt`
        # again to pick up the updated pin, or `pip install --upgrade groq`.
        include_reasoning=False,
    )
    content = response.choices[0].message.content

    # Defensive fallback: strip any reasoning block that leaked through
    # despite include_reasoning=False (see comment above) — belt-and-
    # braces in case a future model change re-introduces this.
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
    return content


if __name__ == "__main__":
    # Real end-to-end test: needs Ollama running locally with the model
    # pulled (`ollama pull llama3.1:8b`), so this only runs on your
    # machine, not in the dev sandbox (no Ollama daemon here).
    from app.generation.prompt import build_generation_prompt
    from app.retrieval.pipeline import HybridRetrievalPipeline

    pipeline = HybridRetrievalPipeline()
    query = "What ROC-AUC does the corruption detector achieve?"

    trace = pipeline.retrieve(query)
    prompt = build_generation_prompt(query, trace.final_results)

    print(f"Query: {query}\n")
    print("Generating with provider:", GENERATION_PROVIDER)
    answer = generate(prompt)
    print("\n=== ANSWER ===")
    print(answer)
