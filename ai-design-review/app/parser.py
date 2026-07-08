import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".text"}
MARKDOWN_CONTENT_TYPES = {
    "text/markdown",
    "text/x-markdown",
    "application/markdown",
}


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    title: str = ""


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    format: str
    pages: list[ParsedPage]


class DocumentParser:
    """Extract plain text from uploaded documents. More formats plug in here."""

    def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix in {".md", ".markdown"}:
            return self._parse_markdown(path)
        if suffix in {".txt", ".text"}:
            return self._parse_plain_text(path, doc_format="text")
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(ParsedPage(page_number=index, text=text))

        if not pages:
            logger.warning("No extractable text in PDF: %s", path.name)
            pages = [ParsedPage(page_number=1, text="")]

        return ParsedDocument(filename=path.name, format="pdf", pages=pages)

    def _parse_markdown(self, path: Path) -> ParsedDocument:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return ParsedDocument(
                filename=path.name,
                format="markdown",
                pages=[ParsedPage(page_number=1, text="", title=path.stem)],
            )

        sections = re.split(r"(?=^##\s)", content, flags=re.MULTILINE)
        pages: list[ParsedPage] = []

        for index, section in enumerate(sections, start=1):
            section = section.strip()
            if not section:
                continue
            pages.append(
                ParsedPage(
                    page_number=index,
                    text=section,
                    title=_markdown_section_title(section, fallback=path.stem),
                )
            )

        logger.info("Parsed markdown %s into %d sections", path.name, len(pages))
        return ParsedDocument(filename=path.name, format="markdown", pages=pages)

    def _parse_plain_text(self, path: Path, *, doc_format: str) -> ParsedDocument:
        text = path.read_text(encoding="utf-8").strip()
        return ParsedDocument(
            filename=path.name,
            format=doc_format,
            pages=[ParsedPage(page_number=1, text=text, title=path.stem)],
        )


def is_markdown_upload(filename: str | None, content_type: str | None) -> bool:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".md", ".markdown"}:
        return True
    if content_type:
        return content_type.split(";", 1)[0].strip().lower() in MARKDOWN_CONTENT_TYPES
    return False


def normalize_upload_filename(filename: str | None, content_type: str | None) -> str:
    name = Path(filename or "upload").name
    if Path(name).suffix:
        return name
    if is_markdown_upload(name, content_type):
        return f"{name}.md"
    if content_type and content_type.split(";", 1)[0].strip().lower() == "text/plain":
        return f"{name}.txt"
    return f"{name}.txt"


def _markdown_section_title(section: str, *, fallback: str) -> str:
    for pattern in (r"^##\s+(.+)$", r"^#\s+(.+)$"):
        match = re.search(pattern, section, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return fallback.replace("_", " ")
