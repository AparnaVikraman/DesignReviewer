import logging
import os
import re
from pathlib import Path

from openai import OpenAI

from app.vector_store import StoredChunk, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


class EmbeddingService:
    def __init__(
        self,
        *,
        documents_dir: Path | None = None,
        vector_store: VectorStore | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.documents_dir = documents_dir or DEFAULT_DOCUMENTS_DIR
        self.vector_store = vector_store or get_vector_store()
        self.embedding_model = embedding_model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

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
            chunks = self._split_document(path.name, content)
            stored = []
            for index, (title, chunk_text) in enumerate(chunks):
                embedding = self.generate_embedding(chunk_text)
                stored.append(
                    StoredChunk(
                        id=None,
                        source_file=path.name,
                        title=title,
                        chunk_index=index,
                        content=chunk_text,
                        embedding=embedding,
                    )
                )
            self.vector_store.upsert_chunks(stored)
            total_chunks += len(stored)
            logger.info("Indexed %d chunks from %s", len(stored), path.name)

        return total_chunks

    def generate_embedding(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _split_document(self, filename: str, content: str) -> list[tuple[str, str]]:
        sections = re.split(r"(?=^##\s)", content, flags=re.MULTILINE)
        chunks: list[tuple[str, str]] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            title_match = re.match(r"^#\s*(.+)", section)
            title = title_match.group(1).strip() if title_match else filename

            if len(section) <= CHUNK_SIZE:
                chunks.append((title, section))
                continue

            start = 0
            while start < len(section):
                end = min(start + CHUNK_SIZE, len(section))
                chunk_text = section[start:end].strip()
                if chunk_text:
                    chunks.append((title, chunk_text))
                if end >= len(section):
                    break
                start = end - CHUNK_OVERLAP

        return chunks
