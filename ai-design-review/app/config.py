import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    model_name: str
    embedding_model: str
    top_k: int
    max_chunk_words: int
    log_level: str
    llm_timeout: float
    llm_max_retries: int
    llm_temperature: float
    llm_max_tokens: int

    @staticmethod
    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        return int(raw)

    @staticmethod
    def _float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        return float(raw)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=os.environ.get("MODEL_NAME", "gpt-4.1-mini"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        top_k=Settings._int("TOP_K", 5),
        max_chunk_words=Settings._int("MAX_CHUNK_SIZE", 600),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        llm_timeout=Settings._float("LLM_TIMEOUT", 60.0),
        llm_max_retries=Settings._int("LLM_MAX_RETRIES", 3),
        llm_temperature=Settings._float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=Settings._int("LLM_MAX_TOKENS", 4096),
    )
