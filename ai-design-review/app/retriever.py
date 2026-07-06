import logging

from app.embedding_service import EmbeddingService
from app.models import RetrievedChunk
from app.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        top_k: int = 5,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or EmbeddingService(
            vector_store=self.vector_store
        )
        self.top_k = top_k

    def retrieve(self, design_doc: str) -> list[RetrievedChunk]:
        query_embedding = self.embedding_service.generate_embedding(design_doc)
        results = self.vector_store.similarity_search(
            query_embedding, top_k=self.top_k
        )

        chunks = [
            RetrievedChunk(
                source_file=result.source_file,
                title=result.title,
                content=result.content,
                score=round(result.score, 4),
                chunk_index=result.chunk_index,
            )
            for result in results
        ]
        logger.info(
            "Retrieved %d chunks for design query (top score=%.4f)",
            len(chunks),
            chunks[0].score if chunks else 0.0,
        )
        return chunks
