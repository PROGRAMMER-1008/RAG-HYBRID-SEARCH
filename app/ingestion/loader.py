"""Safe, extensible text extraction for common documents and code files."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import re
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from pathlib import PurePosixPath
from typing import Callable

from defusedxml import ElementTree

from app.config import RAW_DOCS_DIR, settings


@dataclass
class RawDocument:
    """Extracted text with stable source metadata."""

    source_path: str
    text: str
    content_hash: str = ""
    media_type: str = "text/plain"
    size_bytes: int = 0


Parser = Callable[[Path], str]
_PARSERS: dict[str, Parser] = {}
_SECRET_PATTERN = re.compile(
    r"(?:gsk_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|glpat-[A-Za-z0-9_-]{20,}|npm_[A-Za-z0-9]{30,}|"
    r"sk_live_[A-Za-z0-9]{20,}|SG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|"
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)",
    re.IGNORECASE,
)
_TEXT_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def extension_for_path(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or path.name.lower().lstrip(".")


def register_parser(*extensions: str) -> Callable[[Parser], Parser]:
    """Register a parser for one or more extensions (without leading dots)."""
    def decorator(parser: Parser) -> Parser:
        for extension in extensions:
            _PARSERS[extension.lower().lstrip(".")] = parser
        return parser
    return decorator


def supported_extensions() -> frozenset[str]:
    return frozenset(_PARSERS)


def sanitize_source_name(name: str) -> str:
    """Keep safe relative folder names for source citations, never filesystem paths."""
    normalized = str(name or "").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if parts and re.fullmatch(r"[A-Za-z]:", parts[0]):
        parts = parts[1:]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("The uploaded filename contains an invalid path.")
    cleaned = []
    for part in parts:
        value = "".join(char for char in part if char.isprintable() and char not in "<>:\"|?*\x00")
        value = value.strip().rstrip(".")
        if not value:
            raise ValueError("The uploaded filename contains an invalid path.")
        cleaned.append(value[:120])
    result = PurePosixPath(*cleaned).as_posix()
    if len(result) > 240:
        raise ValueError("The uploaded filename is too long.")
    return result


def _parse_plain_text_fallback(path: Path) -> str:
    """Read unknown extensions only when bytes strongly resemble plain text."""
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ValueError("This binary file type is not supported.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ValueError("This file is not a supported text or document format.") from exc
    sample = text[:8192]
    if sample and len(_TEXT_CONTROL_PATTERN.findall(sample)) / len(sample) > 0.01:
        raise ValueError("This binary file type is not supported.")
    return text


def _decode_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ValueError("The file appears to contain binary data, not readable text.")
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Text encoding is unsupported. Save the file as UTF-8 and try again.")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _validate_archive(path: Path) -> None:
    """Reject malformed and unusually large Office archives before parsing."""
    if not zipfile.is_zipfile(path):
        raise ValueError("This Office document is not a valid Open XML archive.")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            expanded_bytes = sum(item.file_size for item in members)
            if len(members) > 2000 or expanded_bytes > 80 * 1024 * 1024:
                raise ValueError("The Office document expands beyond the safe processing limit.")
            for item in members:
                if item.file_size > 25 * 1024 * 1024:
                    raise ValueError("An Office document part is too large to process safely.")
                if item.file_size > 1024 * 1024 and item.compress_size == 0:
                    raise ValueError("The Office document has an invalid compression profile.")
                if item.compress_size and item.file_size / item.compress_size > 150:
                    raise ValueError("The Office document has an unsafe compression ratio.")
    except zipfile.BadZipFile as exc:
        raise ValueError("This Office document is not a valid Open XML archive.") from exc


@register_parser("txt", "md", "markdown", "rst", "log", "csv", "tsv", "json", "jsonl",
                 "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "py", "ipynb",
                 "js", "jsx", "ts", "tsx", "html", "htm", "css", "scss", "java", "kt",
                 "c", "h", "cpp", "hpp", "cs", "go", "rs", "php", "rb", "swift", "sh",
                 "bash", "ps1", "sql", "dockerfile", "makefile", "bat", "properties",
                 "gitignore")
def parse_textual(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    text = _decode_text(path)
    if suffix == "html" or suffix == "htm":
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            element.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        body = soup.get_text("\n", strip=True)
        return f"# {title}\n\n{body}" if title else body
    if suffix == "csv" or suffix == "tsv":
        delimiter = "\t" if suffix == "tsv" else ","
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        lines = []
        for row_number, row in enumerate(reader, start=1):
            if row_number > 100_000 or len(row) > 512:
                raise ValueError("CSV exceeds the 100,000-row or 512-column processing limit.")
            lines.append(" | ".join(cell.strip() for cell in row))
        return "\n".join(lines)
    if suffix == "json":
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return text
    if suffix == "jsonl":
        return "\n".join(
            json.dumps(json.loads(line), ensure_ascii=False) if line.strip() else ""
            for line in text.splitlines()
        )
    if suffix == "xml":
        try:
            root = ElementTree.fromstring(text)
            return "\n".join(value.strip() for value in root.itertext() if value.strip())
        except Exception as exc:
            raise ValueError("The XML document is malformed.") from exc
    if suffix == "ipynb":
        try:
            notebook = json.loads(text)
            parts = []
            for cell in notebook.get("cells", []):
                source = "".join(cell.get("source", []))
                if source.strip():
                    parts.append(f"## {cell.get('cell_type', 'cell').title()}\n{source}")
            return "\n\n".join(parts)
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ValueError("The notebook file is malformed.") from exc
    return text


@register_parser("pdf")
def parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    ocr_document = None
    try:
        reader = PdfReader(str(path), strict=True)
        if reader.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        if len(reader.pages) > settings.max_pages_per_file:
            raise ValueError(f"PDF exceeds the {settings.max_pages_per_file}-page limit.")
        pages = []
        ocr_page_count = 0
        for index, page in enumerate(reader.pages, start=1):
            content = page.extract_text() or ""
            if not content.strip() and settings.enable_ocr:
                ocr_page_count += 1
                if ocr_page_count > settings.max_ocr_pages_per_file:
                    raise ValueError(f"Scanned PDF exceeds the {settings.max_ocr_pages_per_file}-page OCR limit.")
                try:
                    if ocr_document is None:
                        import pypdfium2 as pdfium
                        ocr_document = pdfium.PdfDocument(str(path))
                    import pytesseract

                    pdf_page = ocr_document[index - 1]
                    if pdf_page.get_width() * pdf_page.get_height() * 2.25 > 40_000_000:
                        raise ValueError("A scanned PDF page exceeds the 40-megapixel OCR limit.")
                    bitmap = pdf_page.render(scale=1.5)
                    image = bitmap.to_pil()
                    try:
                        content = pytesseract.image_to_string(image, timeout=20)
                    finally:
                        image.close()
                        bitmap.close()
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError(
                        "OCR is unavailable for this scanned PDF. Install Tesseract OCR or disable OCR."
                    ) from exc
            if content.strip():
                pages.append(f"## Page {index}\n{content}")
        if not pages:
            raise ValueError("No selectable text was found in this PDF. Scanned PDFs need OCR first.")
        return "\n\n".join(pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be read. Check that it is valid and not damaged.") from exc
    finally:
        if ocr_document is not None:
            ocr_document.close()


@register_parser("png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp")
def parse_image(path: Path) -> str:
    try:
        from PIL import Image
        import pytesseract

        Image.MAX_IMAGE_PIXELS = 40_000_000
        with Image.open(path) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("Image exceeds the 40-megapixel OCR limit.")
            image.load()
            with image.convert("RGB") as rgb_image:
                text = pytesseract.image_to_string(rgb_image, timeout=25)
        if not text.strip():
            raise ValueError("OCR did not find readable text in this image.")
        return text
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Image OCR failed. Check that Tesseract is installed and the image is valid.") from exc


@register_parser("docx")
def parse_docx(path: Path) -> str:
    from docx import Document

    _validate_archive(path)
    try:
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table_index, table in enumerate(doc.tables, start=1):
            rows = [" | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells) for row in table.rows]
            if rows:
                paragraphs.append(f"## Table {table_index}\n" + "\n".join(rows))
        if not paragraphs:
            raise ValueError("No readable text was found in this Word document.")
        return "\n\n".join(paragraphs)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The Word document could not be read.") from exc


@register_parser("pptx")
def parse_pptx(path: Path) -> str:
    from pptx import Presentation

    _validate_archive(path)
    try:
        presentation = Presentation(str(path))
        if len(presentation.slides) > settings.max_pages_per_file:
            raise ValueError(f"Presentation exceeds the {settings.max_pages_per_file}-slide limit.")
        slides = []
        for index, slide in enumerate(presentation.slides, start=1):
            items = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text.strip():
                    items.append(shape.text.strip())
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        items.append(" | ".join(cell.text.strip() for cell in row.cells))
            if items:
                slides.append(f"## Slide {index}\n" + "\n".join(items))
        if not slides:
            raise ValueError("No readable text was found in this presentation.")
        return "\n\n".join(slides)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PowerPoint file could not be read.") from exc


@register_parser("xlsx", "xlsm")
def parse_spreadsheet(path: Path) -> str:
    from openpyxl import load_workbook

    _validate_archive(path)
    workbook = None
    try:
        workbook = load_workbook(path, read_only=True, data_only=True, keep_links=False)
        if len(workbook.worksheets) > 200:
            raise ValueError("Spreadsheet exceeds the 200-sheet processing limit.")
        sections = []
        row_count = 0
        for sheet in workbook.worksheets:
            if (sheet.max_row or 0) > 100_000 or (sheet.max_column or 0) > 256:
                raise ValueError("Spreadsheet exceeds the 100,000-row or 256-column limit per sheet.")
            lines = []
            for row in sheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                if any(value.strip() for value in values):
                    lines.append(" | ".join(values))
                    row_count += 1
                    if row_count > 100_000:
                        raise ValueError("Spreadsheet exceeds the 100,000 non-empty row limit.")
            if lines:
                sections.append(f"## Sheet: {sheet.title}\n" + "\n".join(lines))
        if not sections:
            raise ValueError("No readable cell values were found in this spreadsheet.")
        return "\n\n".join(sections)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The spreadsheet could not be read.") from exc
    finally:
        if workbook is not None:
            workbook.close()


@register_parser("eml")
def parse_email(path: Path) -> str:
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    parts = [f"Subject: {message.get('subject', '')}", f"From: {message.get('from', '')}"]
    body = ""
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                body += part.get_content()
    elif message.get_content_type() == "text/plain":
        body = message.get_content()
    parts.append(body)
    return "\n\n".join(part for part in parts if part.strip())


def load_document(path: Path, *, source_name: str | None = None) -> RawDocument:
    """Extract one file; callers must enforce upload and workspace quotas."""
    size = path.stat().st_size
    if size < 1:
        raise ValueError("Empty files cannot be indexed.")
    if size > settings.max_upload_file_bytes:
        raise ValueError(f"File exceeds the {settings.max_upload_file_bytes // (1024 * 1024)} MiB upload limit.")
    display_name = sanitize_source_name(source_name or path.name)
    lowered_parts = [part.casefold() for part in PurePosixPath(display_name).parts]
    lowered_name = lowered_parts[-1]
    if (any(part in {".git", ".ssh", ".aws", ".azure", ".gnupg"}
            or part == ".env"
            or (part.startswith(".env.") and part not in {".env.example", ".env.sample"})
            for part in lowered_parts)
            or lowered_name in {"id_rsa", "id_ed25519", "credentials", "secrets.json"}
            or lowered_name in {".netrc", ".npmrc", ".pypirc", ".git-credentials"}
            or PurePosixPath(display_name).suffix.casefold() in {".pem", ".key", ".p12", ".pfx", ".p8"}):
        raise ValueError("Credential and private-key files cannot be uploaded.")
    extension = extension_for_path(Path(display_name))
    parser = _PARSERS.get(extension)
    try:
        text = _clean_text(parser(path) if parser else _parse_plain_text_fallback(path))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not extract text from {source_name or path.name}.") from exc
    if not text:
        raise ValueError("No readable text was found in this file.")
    if len(text.encode("utf-8")) > settings.max_workspace_bytes:
        raise ValueError("Extracted text exceeds the configured workspace processing limit.")
    if _SECRET_PATTERN.search(text):
        raise ValueError("Potential credentials were detected. Remove secrets before uploading this document.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return RawDocument(
        source_path=display_name,
        text=text,
        content_hash=digest,
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        size_bytes=size,
    )


def load_corpus(raw_dir: Path = RAW_DOCS_DIR) -> list[RawDocument]:
    """Load supported files recursively in a stable order."""
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Document folder not found: {raw_dir}")
    resolved_root = raw_dir.resolve()
    files = sorted(
        (
            path for path in raw_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
            and extension_for_path(path) in _PARSERS
            and path.resolve().is_relative_to(resolved_root)
        ),
        key=lambda path: path.relative_to(raw_dir).as_posix().casefold(),
    )
    if not files:
        raise FileNotFoundError(f"No supported documents found in {raw_dir}")
    documents = []
    for path in files:
        documents.append(load_document(path, source_name=path.relative_to(raw_dir).as_posix()))
    return documents
