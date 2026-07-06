import json
import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536


@dataclass(frozen=True)
class StoredChunk:
    id: int | None
    source_file: str
    title: str
    chunk_index: int
    content: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    source_file: str
    title: str
    content: str
    score: float
    chunk_index: int


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore(ABC):
    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[StoredChunk]) -> None: ...

    @abstractmethod
    def similarity_search(
        self, embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]: ...


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []
        self._next_id = 1

    def initialize(self) -> None:
        self._chunks = []
        self._next_id = 1

    def clear(self) -> None:
        self.initialize()

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        for chunk in chunks:
            chunk_id = chunk.id if chunk.id is not None else self._next_id
            self._next_id = max(self._next_id, (chunk_id or 0) + 1)
            self._chunks.append(
                StoredChunk(
                    id=chunk_id,
                    source_file=chunk.source_file,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    embedding=chunk.embedding,
                )
            )

    def similarity_search(
        self, embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        scored = [
            SearchResult(
                source_file=chunk.source_file,
                title=chunk.title,
                content=chunk.content,
                score=cosine_similarity(embedding, chunk.embedding),
                chunk_index=chunk.chunk_index,
            )
            for chunk in self._chunks
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class PgVectorStore(VectorStore):
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self._database_url)

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    title TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(%s) NOT NULL,
                    UNIQUE (source_file, chunk_index)
                )
                """
                % EMBEDDING_DIMENSION
            )
            conn.commit()
        logger.info("PgVectorStore initialized")

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("TRUNCATE document_chunks")
            conn.commit()

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO document_chunks
                        (source_file, title, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (source_file, chunk_index)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk.source_file,
                        chunk.title,
                        chunk.chunk_index,
                        chunk.content,
                        _vector_literal(chunk.embedding),
                    ),
                )
            conn.commit()

    def similarity_search(
        self, embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_file, title, chunk_index, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vector_literal(embedding), _vector_literal(embedding), top_k),
            ).fetchall()
        return [
            SearchResult(
                source_file=row[0],
                title=row[1],
                chunk_index=row[2],
                content=row[3],
                score=float(row[4]),
            )
            for row in rows
        ]


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        database_url = os.environ.get("DATABASE_URL")
        if database_url:
            _vector_store = PgVectorStore(database_url)
        else:
            logger.warning("DATABASE_URL not set; using in-memory vector store")
            _vector_store = InMemoryVectorStore()
        _vector_store.initialize()
    return _vector_store
