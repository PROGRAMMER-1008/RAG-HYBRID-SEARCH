"""Provider adapter with explicit configuration and request timeouts."""

from __future__ import annotations

import re

from app.config import settings
from app.generation.prompt import GenerationPrompt


def generate(
    prompt: GenerationPrompt,
    provider: str | None = None,
    max_output_tokens: int | None = None,
) -> str:
    active_provider = (provider or settings.generation_provider).strip().lower()
    token_limit = max_output_tokens or settings.max_answer_tokens
    if active_provider == "ollama":
        return _generate_ollama(prompt, token_limit)
    if active_provider == "groq":
        return _generate_groq(prompt, token_limit)
    raise ValueError(f"Unsupported generation provider: {active_provider!r}")


def _generate_ollama(prompt: GenerationPrompt, max_output_tokens: int) -> str:
    import ollama

    client = ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout_seconds)
    response = client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ],
        options={"temperature": 0.1, "num_predict": max_output_tokens},
    )
    return response["message"]["content"]


def _generate_groq(prompt: GenerationPrompt, max_output_tokens: int) -> str:
    from groq import Groq

    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
        raise RuntimeError("The configured Groq provider has no API key.")

    client = Groq(
        api_key=settings.groq_api_key.get_secret_value(),
        timeout=settings.groq_timeout_seconds,
        max_retries=1,
    )
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ],
        temperature=0.1,
        include_reasoning=False,
        max_completion_tokens=max_output_tokens,
    )
    content = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
