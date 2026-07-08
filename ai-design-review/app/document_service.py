import logging
import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.chunk_service import ChunkService
from app.config import get_settings
from app.embedding_service import EmbeddingService
from app.models import DocumentRecord
from app.parser import DocumentParser, SUPPORTED_EXTENSIONS, normalize_upload_filename
from app.vector_store import StoredChunk, StoredDocument, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

DEFAULT_UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"


class DocumentService:
    def __init__(
        self,
        *,
        uploads_dir: Path | None = None,
        parser: DocumentParser | None = None,
        chunk_service: ChunkService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.uploads_dir = uploads_dir or DEFAULT_UPLOADS_DIR
        self.parser = parser or DocumentParser()
        settings = get_settings()
        self.chunk_service = chunk_service or ChunkService(
            chunk_words=settings.max_chunk_words
        )
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or EmbeddingService(
            vector_store=self.vector_store
        )

    def save_upload(self, upload: UploadFile) -> Path:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        raw_name = normalize_upload_filename(upload.filename, upload.content_type)
        filename = _safe_filename(raw_name)
        destination = self.uploads_dir / filename
        destination.write_bytes(upload.file.read())
        logger.info("Saved upload to %s", destination)
        return destination

    def process_file(self, path: Path) -> DocumentRecord:
        parsed = self.parser.parse(path)
        text_chunks = self.chunk_service.chunk_document(parsed)
        page_titles = {page.page_number: page.title for page in parsed.pages}

        document_id = str(uuid.uuid4())
        stored_chunks: list[StoredChunk] = []
        for chunk in text_chunks:
            embedding = self.embedding_service.generate_embedding(chunk.content)
            stored_chunks.append(
                StoredChunk(
                    id=None,
                    document_id=document_id,
                    source_file=parsed.filename,
                    title=_chunk_title(
                        parsed.filename,
                        chunk.page_number,
                        page_title=page_titles.get(chunk.page_number, ""),
                    ),
                    page_number=chunk.page_number,
                    chunk_number=chunk.chunk_number,
                    content=chunk.content,
                    embedding=embedding,
                )
            )

        document = StoredDocument(
            id=document_id,
            filename=parsed.filename,
            format=parsed.format,
            page_count=len(parsed.pages),
            chunk_count=len(stored_chunks),
        )
        self.vector_store.upsert_document(document, stored_chunks)
        logger.info(
            "Processed %s: %d pages, %d chunks",
            parsed.filename,
            document.page_count,
            document.chunk_count,
        )
        return DocumentRecord(
            document_id=document.id,
            filename=document.filename,
            format=document.format,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
        )

    def get_document(self, *, document_id: str | None = None, filename: str | None = None):
        if document_id:
            document = self.vector_store.get_document(document_id)
            if document is None:
                raise FileNotFoundError(f"Document not found: {document_id}")
            return document
        if filename:
            safe_name = _safe_filename(filename)
            document = self.vector_store.get_document_by_filename(safe_name)
            if document is None:
                raise FileNotFoundError(f"Document not found: {safe_name}")
            return document
        raise ValueError("Provide document_id or filename")

    def read_design_text(
        self, *, document_id: str | None = None, filename: str | None = None
    ) -> tuple[str, StoredDocument]:
        document = self.get_document(document_id=document_id, filename=filename)
        path = self.uploads_dir / document.filename
        if not path.exists():
            raise FileNotFoundError(
                f"Uploaded file missing on disk: {document.filename}"
            )

        parsed = self.parser.parse(path)
        pages = [page.text for page in parsed.pages if page.text.strip()]
        if not pages:
            raise ValueError(f"No readable text in document: {document.filename}")

        return "\n\n".join(pages), document

    def upload_and_process(self, upload: UploadFile) -> DocumentRecord:
        path = self.save_upload(upload)
        return self.process_file(path)

    def upload_and_process_many(self, uploads: list[UploadFile]) -> list[DocumentRecord]:
        return [self.upload_and_process(upload) for upload in uploads]


def _chunk_title(filename: str, page_number: int, *, page_title: str = "") -> str:
    if page_title:
        return page_title
    stem = Path(filename).stem.replace("_", " ")
    if page_number > 1:
        return f"{stem} (section {page_number})"
    return stem


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    cleaned = re.sub(r"[^\w.\-]", "_", name).strip("._")
    if not cleaned:
        raise ValueError("Invalid filename")
    suffix = Path(cleaned).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return cleaned
