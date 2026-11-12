from pathlib import Path

import pytest

from app.ingestion.loader import load_document, sanitize_source_name
from app.config import settings


def test_known_text_format_is_extracted(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("# Retrieval\n\nDense search ranks passages.", encoding="utf-8")

    document = load_document(path, source_name="project/docs/notes.md")

    assert document.source_path == "project/docs/notes.md"
    assert "Dense search ranks passages." in document.text
    assert len(document.content_hash) == 64


def test_unknown_extension_is_allowed_only_for_readable_text(tmp_path: Path):
    text_path = tmp_path / "design.custom"
    text_path.write_text("A readable project note.", encoding="utf-8")
    binary_path = tmp_path / "payload.custom"
    binary_path.write_bytes(b"\x00\x01\x02\xff")

    assert load_document(text_path).text == "A readable project note."
    with pytest.raises(ValueError, match="binary"):
        load_document(binary_path)


def test_credentials_and_secret_patterns_are_rejected_without_echoing(tmp_path: Path):
    env_file = tmp_path / ".env.production"
    env_file.write_text("this would be a secret", encoding="utf-8")
    token_file = tmp_path / "notes.txt"
    token_file.write_text("Example key gsk_" + "a" * 30, encoding="utf-8")

    with pytest.raises(ValueError, match="Credential"):
        load_document(env_file)
    with pytest.raises(ValueError, match="Potential credentials") as error:
        load_document(token_file)
    assert "gsk_" not in str(error.value)


def test_relative_paths_are_preserved_and_traversal_is_rejected():
    assert sanitize_source_name("src\\rag\\main.py") == "src/rag/main.py"
    with pytest.raises(ValueError, match="invalid path"):
        sanitize_source_name("../../outside.txt")




def test_extracted_text_is_bounded_by_workspace_limit(tmp_path: Path, monkeypatch):
    path = tmp_path / "large.txt"
    path.write_text("this text is too large", encoding="utf-8")
    monkeypatch.setattr(settings, "max_workspace_bytes", 10)

    with pytest.raises(ValueError, match="workspace processing limit"):
        load_document(path)
