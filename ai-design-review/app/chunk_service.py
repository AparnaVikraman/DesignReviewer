import logging
import re
from dataclasses import dataclass

from app.parser import ParsedDocument

logger = logging.getLogger(__name__)

# ~500-800 tokens: ~600 words is a simple equivalent (no overlap).
DEFAULT_CHUNK_WORDS = 600


@dataclass(frozen=True)
class TextChunk:
    content: str
    page_number: int
    chunk_number: int


class ChunkService:
    def __init__(self, *, chunk_words: int = DEFAULT_CHUNK_WORDS) -> None:
        self.chunk_words = chunk_words

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        next_chunk_number = 0

        for page in document.pages:
            if not page.text.strip():
                continue
            page_chunks = self.chunk_text(
                page.text,
                page_number=page.page_number,
                start_chunk_number=next_chunk_number,
                preserve_line_breaks=document.format == "markdown",
            )
            chunks.extend(page_chunks)
            next_chunk_number += len(page_chunks)

        logger.info(
            "Chunked %s into %d chunks across %d pages",
            document.filename,
            len(chunks),
            len(document.pages),
        )
        return chunks

    def chunk_text(
        self,
        text: str,
        *,
        page_number: int = 1,
        start_chunk_number: int = 0,
        preserve_line_breaks: bool = False,
    ) -> list[TextChunk]:
        if preserve_line_breaks:
            normalized = re.sub(r"[ \t]+", " ", text.strip())
            normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        else:
            normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []

        words = normalized.split(" ")
        chunks: list[TextChunk] = []
        chunk_number = start_chunk_number

        for start in range(0, len(words), self.chunk_words):
            segment = words[start : start + self.chunk_words]
            if not segment:
                continue
            chunks.append(
                TextChunk(
                    content=" ".join(segment),
                    page_number=page_number,
                    chunk_number=chunk_number,
                )
            )
            chunk_number += 1

        return chunks
