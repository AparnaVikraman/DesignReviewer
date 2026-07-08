import threading
from dataclasses import dataclass, field


@dataclass
class _LatencyStats:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, latency_ms: float) -> None:
        self.count += 1
        self.total_ms += latency_ms
        self.max_ms = max(self.max_ms, latency_ms)

    def avg_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return round(self.total_ms / self.count, 2)


@dataclass
class MetricsStore:
    reviews_total: int = 0
    reviews_failed: int = 0
    embedding: _LatencyStats = field(default_factory=_LatencyStats)
    retrieval: _LatencyStats = field(default_factory=_LatencyStats)
    llm: _LatencyStats = field(default_factory=_LatencyStats)
    total: _LatencyStats = field(default_factory=_LatencyStats)
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    estimated_cost_usd_total: float = 0.0
    retrieved_chunks_total: int = 0
    empty_retrieval_total: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "reviews_total": self.reviews_total,
            "reviews_failed": self.reviews_failed,
            "latency_ms": {
                "embedding_avg": self.embedding.avg_ms(),
                "embedding_max": round(self.embedding.max_ms, 2),
                "retrieval_avg": self.retrieval.avg_ms(),
                "retrieval_max": round(self.retrieval.max_ms, 2),
                "llm_avg": self.llm.avg_ms(),
                "llm_max": round(self.llm.max_ms, 2),
                "total_avg": self.total.avg_ms(),
                "total_max": round(self.total.max_ms, 2),
            },
            "tokens": {
                "input_total": self.input_tokens_total,
                "output_total": self.output_tokens_total,
            },
            "estimated_cost_usd_total": round(self.estimated_cost_usd_total, 6),
            "retrieved_chunks_total": self.retrieved_chunks_total,
            "empty_retrieval_total": self.empty_retrieval_total,
        }


_lock = threading.Lock()
_metrics = MetricsStore()


def get_metrics() -> MetricsStore:
    return _metrics


def record_review_success(
    *,
    embedding_ms: float | None,
    retrieval_ms: float | None,
    llm_ms: float,
    total_ms: float,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_usd: float,
    retrieved_chunk_count: int,
) -> None:
    with _lock:
        _metrics.reviews_total += 1
        if embedding_ms is not None:
            _metrics.embedding.record(embedding_ms)
        if retrieval_ms is not None:
            _metrics.retrieval.record(retrieval_ms)
        _metrics.llm.record(llm_ms)
        _metrics.total.record(total_ms)
        _metrics.input_tokens_total += input_tokens
        _metrics.output_tokens_total += output_tokens
        _metrics.estimated_cost_usd_total += estimated_cost_usd
        _metrics.retrieved_chunks_total += retrieved_chunk_count
        if retrieved_chunk_count == 0:
            _metrics.empty_retrieval_total += 1


def record_review_failure() -> None:
    with _lock:
        _metrics.reviews_failed += 1


def reset_metrics() -> None:
    global _metrics
    with _lock:
        _metrics = MetricsStore()
