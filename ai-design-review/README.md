# AI Design Review Assistant

FastAPI service that reviews backend system design documents using OpenAI, with optional RAG over an internal engineering knowledge base.

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

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open API docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/index` | Index `documents/` into vector store |
| POST | `/review` | Design review (`use_retrieval` defaults to `true`) |
| POST | `/review/compare` | Compare review with vs without retrieval |
| POST | `/review/stream` | Stream review via SSE |

### POST /review (with RAG)

```bash
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"design_doc": "Order service uses PostgreSQL and Kafka."}'
```

Disable retrieval:

```bash
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{"design_doc": "...", "use_retrieval": false}'
```

### POST /review/compare (Day 5)

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
│   ├── embedding_service.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── vector_store.py
│   └── llm_client.py
├── documents/              # Knowledge base
├── prompts/                # Prompt templates
├── scripts/
│   ├── index_documents.py
│   └── compare_review.py
├── tests/
│   ├── test_review_api.py
│   └── test_rag.py
├── .env.example
├── requirements.txt
└── README.md
```

## RAG pipeline

```
documents/*.md
    ↓ EmbeddingService (chunk → embed → store)
pgvector / in-memory store
    ↓ Retriever (design → embed → top 5 chunks)
PromptBuilder (inject guidance into prompt)
    ↓
LLM review
```
