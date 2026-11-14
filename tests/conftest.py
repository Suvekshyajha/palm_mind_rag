"""Fake services so the API can be tested without Redis, Qdrant, Postgres or Groq."""
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import Booking
from app.dependencies import (
    get_bm25,
    get_embedder,
    get_memory,
    get_rag,
    get_session,
    get_vector_store,
)
from app.routes import chat, ingestion
from app.services.rag import RagResources
from app.services.retrieval import BM25Index
from app.typedefs import ChatMessage, ChunkPayload, LLMMessage, ScoredChunk


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://x",
        redis_url="redis://x",
        qdrant_url="http://x",
        upload_dir=str(tmp_path),
        _env_file=None,  # type: ignore[call-arg]
        **overrides,
    )


class FakeLLM:
    """Returns scripted replies, in order."""

    def __init__(self) -> None:
        self.json_replies: list[dict[str, Any]] = []
        self.text_replies: list[str] = []

    async def chat_json(self, messages: list[LLMMessage]) -> dict[str, Any]:
        return self.json_replies.pop(0)

    async def chat(self, messages: list[LLMMessage], temperature: float = 0.2) -> str:
        return self.text_replies.pop(0)


class FakeMemory:
    def __init__(self) -> None:
        self.history: dict[str, list[ChatMessage]] = {}
        self.bookings: dict[str, dict[str, str]] = {}
        self.fail = False

    async def session_exists(self, session_id: str) -> bool:
        self._check()
        return session_id in self.history or session_id in self.bookings

    async def get_history(self, session_id: str, limit: int) -> list[ChatMessage]:
        self._check()
        return self.history.get(session_id, [])[-limit:]

    async def add_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        self.history.setdefault(session_id, []).extend(messages)

    async def get_booking(self, session_id: str) -> dict[str, str]:
        return dict(self.bookings.get(session_id, {}))

    async def save_booking(self, session_id: str, details: dict[str, str]) -> None:
        self.bookings[session_id] = {"status": "collecting", **details}

    async def clear_booking(self, session_id: str) -> None:
        self.bookings.pop(session_id, None)

    def _check(self) -> None:
        if self.fail:
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("redis is down")


class FakeSession:
    """Stands in for the SQLAlchemy session."""

    def __init__(self) -> None:
        self.saved: list[Any] = []
        self.slot_taken = False

    def add(self, item: Any) -> None:
        self._pending = item

    async def commit(self) -> None:
        if self.slot_taken and isinstance(self._pending, Booking):
            from sqlalchemy.exc import IntegrityError

            raise IntegrityError("insert", {}, Exception("duplicate"))
        self.saved.append(self._pending)

    async def rollback(self) -> None:
        return None


class FakeEmbedder:
    dimension = 2

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Alternate two directions so semantic chunking sees topic changes.
        return [[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i in range(len(texts))]

    async def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


@dataclass
class FakeVectorStore:
    stored: list[ChunkPayload] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    fail_on_add: bool = False
    search_filters: list[list[str] | None] = field(default_factory=list)

    async def add_chunks(self, payloads: list[ChunkPayload], vectors: list[list[float]]) -> None:
        if self.fail_on_add:
            raise RuntimeError("qdrant exploded")
        self.stored.extend(payloads)

    async def search(
        self, vector: list[float], limit: int, document_ids: list[str] | None = None
    ) -> list[ScoredChunk]:
        self.search_filters.append(document_ids)
        return []

    async def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)


class FakeReranker:
    def __init__(self) -> None:
        self.results: list[ScoredChunk] = []

    async def rerank(self, question: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        return self.results


@dataclass
class Env:
    client: TestClient
    llm: FakeLLM
    memory: FakeMemory
    session: FakeSession
    vector_store: FakeVectorStore
    reranker: FakeReranker
    bm25: BM25Index
    settings: Settings
    upload_dir: Path


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    settings = make_settings(tmp_path, relevance_threshold=0.30)
    llm, memory, session = FakeLLM(), FakeMemory(), FakeSession()
    vector_store, reranker, bm25 = FakeVectorStore(), FakeReranker(), BM25Index()
    rag = RagResources(
        llm=llm,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        bm25=bm25,
        reranker=reranker,  # type: ignore[arg-type]
        settings=settings,
    )

    app = FastAPI()
    app.include_router(ingestion.router)
    app.include_router(chat.router)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_memory] = lambda: memory
    app.dependency_overrides[get_rag] = lambda: rag
    app.dependency_overrides[get_embedder] = lambda: rag.embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_bm25] = lambda: bm25

    with TestClient(app) as client:
        yield Env(client, llm, memory, session, vector_store, reranker, bm25, settings, tmp_path)


def tomorrow() -> str:
    return (date.today() + timedelta(days=1)).isoformat()