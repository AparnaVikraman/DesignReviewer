import logging
import time
from dataclasses import dataclass

from app.config import get_settings
from app.embedding_service import EmbeddingService
from app.models import RetrievedChunk
from app.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrieveResult:
    chunks: list[RetrievedChunk]
    embedding_ms: float
    search_ms: float


class Retriever:
    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or EmbeddingService(
            vector_store=self.vector_store
        )
        self.top_k = top_k if top_k is not None else settings.top_k

    def retrieve(
        self,
        design_doc: str,
        *,
        exclude_document_id: str | None = None,
    ) -> RetrieveResult:
        embed_start = time.perf_counter()
        query_embedding = self.embedding_service.generate_embedding(design_doc)
        embedding_ms = round((time.perf_counter() - embed_start) * 1000, 2)

        search_start = time.perf_counter()
        results = self.vector_store.similarity_search(
            query_embedding, top_k=self.top_k
        )
        if exclude_document_id:
            results = [
                result
                for result in results
                if result.document_id != exclude_document_id
            ]
        search_ms = round((time.perf_counter() - search_start) * 1000, 2)

        chunks = [
            RetrievedChunk(
                source_file=result.source_file,
                title=result.title,
                content=result.content,
                score=round(result.score, 4),
                page_number=result.page_number,
                chunk_number=result.chunk_number,
            )
            for result in results
        ]
        logger.info(
            "Retrieved %d chunks embedding_ms=%.2f search_ms=%.2f top_score=%.4f",
            len(chunks),
            embedding_ms,
            search_ms,
            chunks[0].score if chunks else 0.0,
        )
        return RetrieveResult(
            chunks=chunks,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
        )
