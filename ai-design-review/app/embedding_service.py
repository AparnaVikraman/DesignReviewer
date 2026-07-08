import logging
import os
import re
import uuid
from pathlib import Path

from openai import OpenAI

from app.chunk_service import ChunkService, TextChunk
from app.config import get_settings
from app.parser import ParsedDocument, ParsedPage
from app.vector_store import StoredChunk, StoredDocument, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"


class EmbeddingService:
    def __init__(
        self,
        *,
        documents_dir: Path | None = None,
        vector_store: VectorStore | None = None,
        chunk_service: ChunkService | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.documents_dir = documents_dir or DEFAULT_DOCUMENTS_DIR
        self.vector_store = vector_store or get_vector_store()
        self.chunk_service = chunk_service or ChunkService(
            chunk_words=settings.max_chunk_words
        )
        self.embedding_model = embedding_model or settings.embedding_model
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        )

    def index_documents(self, *, clear_existing: bool = True) -> int:
        if clear_existing:
            self.vector_store.clear()

        markdown_files = sorted(self.documents_dir.glob("*.md"))
        if not markdown_files:
            logger.warning("No markdown files found in %s", self.documents_dir)
            return 0

        total_chunks = 0
        for path in markdown_files:
            content = path.read_text(encoding="utf-8")
            document_id = str(uuid.uuid4())
            parsed = ParsedDocument(
                filename=path.name,
                format="markdown",
                pages=[ParsedPage(page_number=1, text=content)],
            )
            text_chunks = self._chunk_markdown(path.name, content, parsed)
            if not text_chunks:
                text_chunks = self.chunk_service.chunk_document(parsed)

            stored = self._embed_chunks(
                document_id=document_id,
                source_file=path.name,
                chunks=text_chunks,
            )
            document = StoredDocument(
                id=document_id,
                filename=path.name,
                format="markdown",
                page_count=1,
                chunk_count=len(stored),
            )
            self.vector_store.upsert_document(document, stored)
            total_chunks += len(stored)
            logger.info("Indexed %d chunks from %s", len(stored), path.name)

        return total_chunks

    def generate_embedding(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _chunk_markdown(self, filename: str, content: str, parsed: ParsedDocument) -> list[TextChunk]:
        sections = re.split(r"(?=^##\s)", content, flags=re.MULTILINE)
        chunks: list[TextChunk] = []
        chunk_number = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            title_match = re.match(r"^#\s*(.+)", section)
            section_title = title_match.group(1).strip() if title_match else filename
            section_chunks = self.chunk_service.chunk_text(
                section,
                page_number=1,
                start_chunk_number=chunk_number,
            )
            for chunk in section_chunks:
                chunks.append(
                    TextChunk(
                        content=f"# {section_title}\n{chunk.content}",
                        page_number=chunk.page_number,
                        chunk_number=chunk.chunk_number,
                    )
                )
            chunk_number += len(section_chunks)

        return chunks

    def _embed_chunks(
        self,
        *,
        document_id: str,
        source_file: str,
        chunks: list[TextChunk],
    ) -> list[StoredChunk]:
        stored: list[StoredChunk] = []
        for chunk in chunks:
            title_match = re.match(r"^#\s*(.+)", chunk.content)
            title = title_match.group(1).strip() if title_match else source_file
            embedding = self.generate_embedding(chunk.content)
            stored.append(
                StoredChunk(
                    id=None,
                    document_id=document_id,
                    source_file=source_file,
                    title=title,
                    page_number=chunk.page_number,
                    chunk_number=chunk.chunk_number,
                    content=chunk.content,
                    embedding=embedding,
                )
            )
        return stored
