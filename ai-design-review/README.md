# AI Design Review Assistant

FastAPI service that reviews backend system design documents using OpenAI, with optional RAG over an internal engineering knowledge base and uploaded documents.

## Setup

```bash
cd ai-design-review
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
# Optional: set DATABASE_URL for pgvector (defaults to in-memory store)
```

## Index the knowledge base

```bash
python scripts/index_documents.py
# or
curl -X POST http://127.0.0.1:8000/index
```

Documents live in `documents/`:
- `retry_strategy.md`
- `api_guidelines.md`
- `kafka_best_practices.md`
- `observability.md`

## Upload design documents

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "files=@payment_design.pdf" \
  -F "files=@inventory.md"
```

Supported formats: PDF, Markdown, plain text. Files are saved under `uploads/`, parsed, chunked, embedded, and indexed with document metadata (`page_number`, `chunk_number`).

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open API docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + review counts |
| GET | `/metrics` | Latency, token, and cost metrics |
| POST | `/documents` | Upload PDF/Markdown/Text (`multipart/form-data`) |
| POST | `/index` | Index `documents/` into vector store |
| POST | `/review` | Design review (`use_retrieval` defaults to `true`) |
| POST | `/review/compare` | Compare review with vs without retrieval |
| POST | `/review/stream` | Stream review via SSE |

### POST /review (with RAG)

Pass design text directly:

```bash
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"design_doc": "Order service uses PostgreSQL and Kafka."}'
```

Or review an uploaded document by ID or filename:

```bash
# 1. Upload
curl -X POST http://127.0.0.1:8000/documents \
  -F "files=@samples/order_service_design.md"
# Response includes document_id

# 2. Review by document_id
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"document_id": "YOUR-DOCUMENT-ID-HERE"}'

# Or review by filename
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"filename": "order_service_design.md"}'
```

Provide exactly one of: `design_doc`, `document_id`, or `filename`.

Review metadata includes latency breakdown, token usage, and estimated cost:

```json
{
  "metadata": {
    "latency_ms": 3200.0,
    "latency": {
      "embedding_ms": 320.0,
      "retrieval_ms": 18.0,
      "llm_ms": 2800.0,
      "total_ms": 3200.0
    },
    "token_usage": { "input_tokens": 1450, "output_tokens": 640, "total_tokens": 2090 },
    "estimated_cost_usd": 0.001024
  }
}
```

Errors return structured JSON (not stack traces):

```json
{
  "error": "LLM request timed out",
  "error_type": "timeout",
  "request_id": "...",
  "retryable": true
}
```

### Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gpt-4.1-mini` | LLM model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TOP_K` | `5` | Retrieved chunks |
| `MAX_CHUNK_SIZE` | `600` | Chunk size in words |
| `LLM_TIMEOUT` | `60` | Request timeout (seconds) |
| `LLM_MAX_RETRIES` | `3` | Retries for transient failures |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Metrics dashboard (simple)

```bash
curl http://127.0.0.1:8000/metrics
```

Returns aggregate review latency, token totals, cost, and error counts.

Disable retrieval:

```bash
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"design_doc": "...", "use_retrieval": false}'
```

### POST /review/compare

```bash
python scripts/compare_review.py
# or
curl -X POST http://127.0.0.1:8000/review/compare \
  -H "Content-Type: application/json" \
  -d '{"design_doc": "Order service uses PostgreSQL and Kafka."}'
```

## Tests

```bash
pytest tests/ -v
```

## Project layout

```
ai-design-review/
├── app/
│   ├── main.py
│   ├── models.py
│   ├── review_service.py
│   ├── document_service.py
│   ├── parser.py
│   ├── chunk_service.py
│   ├── embedding_service.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── vector_store.py
│   ├── review_input.py
│   ├── config.py
│   ├── metrics.py
│   ├── errors.py
│   ├── observability.py
│   └── llm_client.py
├── documents/              # Knowledge base
├── uploads/                # Uploaded design docs (gitignored)
├── prompts/                # Prompt templates
├── scripts/
│   ├── index_documents.py
│   └── compare_review.py
├── tests/
│   ├── test_review_api.py
│   ├── test_rag.py
│   ├── test_documents.py
│   ├── test_documents.py
│   ├── test_chunk_service.py
│   └── test_observability.py
├── .env.example
├── requirements.txt
└── README.md
```

## RAG pipeline

```
documents/*.md or uploads/*
    ↓ DocumentParser (extract text)
    ↓ ChunkService (~600 words per chunk)
    ↓ EmbeddingService (embed → store with metadata)
pgvector / in-memory store
    ↓ Retriever (design → embed → top 5 chunks)
PromptBuilder (inject guidance + page/chunk numbers)
    ↓
LLM review with citations
```

## Week 4 responsibilities

| Module | Responsibility |
|--------|----------------|
| `parser.py` | Extract text from PDF/Markdown/Text |
| `chunk_service.py` | Split text into fixed-size chunks |
| `document_service.py` | Upload, parse, chunk, embed, store |
| `embedding_service.py` | Generate embeddings |
| `vector_store.py` | Store documents + chunks + search |
| `retriever.py` | Semantic search at review time |
| `prompts.py` | Build review prompts |
| `review_service.py` | Orchestrate retrieval → review |
