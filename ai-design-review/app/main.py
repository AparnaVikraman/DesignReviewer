import json
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.embedding_service import EmbeddingService
from app.models import CompareReviewResponse, ReviewRequest, ReviewResponse
from app.review_service import compare_reviews, review_design, stream_review_design

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(title="AI Design Review Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index_documents():
    count = EmbeddingService().index_documents()
    return {"status": "ok", "chunks_indexed": count}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest):
    return review_design(request.design_doc, use_retrieval=request.use_retrieval)


@app.post("/review/compare", response_model=CompareReviewResponse)
def review_compare(request: ReviewRequest):
    return compare_reviews(request.design_doc)


@app.post("/review/stream")
def review_stream(request: ReviewRequest):
    def event_generator():
        for stream_event in stream_review_design(
            request.design_doc, use_retrieval=request.use_retrieval
        ):
            payload = json.dumps(stream_event.data)
            yield f"event: {stream_event.event}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
