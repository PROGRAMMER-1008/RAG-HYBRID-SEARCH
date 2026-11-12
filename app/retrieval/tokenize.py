"""Shared tokenization for technical lexical search."""

import re


def tokenize_technical_text(text: str) -> list[str]:
    """Normalize prose while preserving technical identifiers and numbers."""
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    words = re.findall(r"[\w]+(?:[.+#/-][\w]+)*", separated.casefold(), flags=re.UNICODE)
    expanded: list[str] = []
    for word in words:
        expanded.append(word)
        expanded.extend(part for part in re.split(r"[_/.-]+", word) if part and part != word)
    return expanded
