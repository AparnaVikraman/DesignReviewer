import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536


@dataclass(frozen=True)
class StoredDocument:
    id: str
    filename: str
    format: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True)
class StoredChunk:
    id: int | None
    document_id: str | None
    source_file: str
    title: str
    page_number: int
    chunk_number: int
    content: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    document_id: str | None
    source_file: str
    title: str
    content: str
    score: float
    page_number: int
    chunk_number: int


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
    def upsert_document(
        self, document: StoredDocument, chunks: list[StoredChunk]
    ) -> None: ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[StoredChunk]) -> None: ...

    @abstractmethod
    def list_documents(self) -> list[StoredDocument]: ...

    @abstractmethod
    def get_document(self, document_id: str) -> StoredDocument | None: ...

    @abstractmethod
    def get_document_by_filename(self, filename: str) -> StoredDocument | None: ...

    @abstractmethod
    def similarity_search(
        self, embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]: ...


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._documents: dict[str, StoredDocument] = {}
        self._chunks: list[StoredChunk] = []
        self._next_id = 1

    def initialize(self) -> None:
        self._documents = {}
        self._chunks = []
        self._next_id = 1

    def clear(self) -> None:
        self.initialize()

    def upsert_document(
        self, document: StoredDocument, chunks: list[StoredChunk]
    ) -> None:
        self._documents[document.id] = document
        self.upsert_chunks(chunks)

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        for chunk in chunks:
            chunk_id = chunk.id if chunk.id is not None else self._next_id
            self._next_id = max(self._next_id, (chunk_id or 0) + 1)
            self._chunks.append(
                StoredChunk(
                    id=chunk_id,
                    document_id=chunk.document_id,
                    source_file=chunk.source_file,
                    title=chunk.title,
                    page_number=chunk.page_number,
                    chunk_number=chunk.chunk_number,
                    content=chunk.content,
                    embedding=chunk.embedding,
                )
            )

    def list_documents(self) -> list[StoredDocument]:
        return sorted(self._documents.values(), key=lambda doc: doc.filename)

    def get_document(self, document_id: str) -> StoredDocument | None:
        return self._documents.get(document_id)

    def get_document_by_filename(self, filename: str) -> StoredDocument | None:
        for document in self._documents.values():
            if document.filename == filename:
                return document
        return None

    def similarity_search(
        self, embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        scored = [
            SearchResult(
                document_id=chunk.document_id,
                source_file=chunk.source_file,
                title=chunk.title,
                content=chunk.content,
                score=cosine_similarity(embedding, chunk.embedding),
                page_number=chunk.page_number,
                chunk_number=chunk.chunk_number,
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
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    format TEXT NOT NULL,
                    page_count INT NOT NULL,
                    chunk_count INT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id),
                    source_file TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_number INT NOT NULL DEFAULT 1,
                    chunk_number INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(%s) NOT NULL,
                    UNIQUE (source_file, chunk_number)
                )
                """
                % EMBEDDING_DIMENSION
            )
            conn.commit()
        logger.info("PgVectorStore initialized")

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("TRUNCATE document_chunks, documents")
            conn.commit()

    def upsert_document(
        self, document: StoredDocument, chunks: list[StoredChunk]
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, format, page_count, chunk_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    format = EXCLUDED.format,
                    page_count = EXCLUDED.page_count,
                    chunk_count = EXCLUDED.chunk_count
                """,
                (
                    document.id,
                    document.filename,
                    document.format,
                    document.page_count,
                    document.chunk_count,
                ),
            )
            conn.commit()
        self.upsert_chunks(chunks)

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO document_chunks
                        (document_id, source_file, title, page_number,
                         chunk_number, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (source_file, chunk_number)
                    DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        title = EXCLUDED.title,
                        page_number = EXCLUDED.page_number,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk.document_id,
                        chunk.source_file,
                        chunk.title,
                        chunk.page_number,
                        chunk.chunk_number,
                        chunk.content,
                        _vector_literal(chunk.embedding),
                    ),
                )
            conn.commit()

    def list_documents(self) -> list[StoredDocument]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, format, page_count, chunk_count
                FROM documents
                ORDER BY filename
                """
            ).fetchall()
        return [_row_to_document(row) for row in rows]

    def get_document(self, document_id: str) -> StoredDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, filename, format, page_count, chunk_count
                FROM documents
                WHERE id = %s
                """,
                (document_id,),
            ).fetchone()
        return _row_to_document(row) if row else None

    def get_document_by_filename(self, filename: str) -> StoredDocument | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, filename, format, page_count, chunk_count
                FROM documents
                WHERE filename = %s
                """,
                (filename,),
            ).fetchone()
        return _row_to_document(row) if row else None

    def similarity_search(
        self, embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, source_file, title, page_number, chunk_number,
                       content, 1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vector_literal(embedding), _vector_literal(embedding), top_k),
            ).fetchall()
        return [
            SearchResult(
                document_id=row[0],
                source_file=row[1],
                title=row[2],
                page_number=row[3],
                chunk_number=row[4],
                content=row[5],
                score=float(row[6]),
            )
            for row in rows
        ]


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def _row_to_document(row: tuple) -> StoredDocument:
    return StoredDocument(
        id=row[0],
        filename=row[1],
        format=row[2],
        page_count=row[3],
        chunk_count=row[4],
    )


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
