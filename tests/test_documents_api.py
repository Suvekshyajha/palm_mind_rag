from collections.abc import AsyncIterator
from datetime import date, time

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import Base, Booking, SlotTakenError, save_booking
from app.dependencies import get_session
from app.services.generation import NOT_FOUND
from app.typedefs import ScoredChunk
from tests.conftest import Env

TEXT = "Sera Shrestha is a backend engineer with five years of Python and FastAPI experience. " * 8


@pytest.fixture
async def sql() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A real (in-memory SQLite) database, so the SQL code is truly exercised."""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _use_sql(env: Env, factory: async_sessionmaker[AsyncSession]) -> None:
    async def override() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    env.client.app.dependency_overrides[get_session] = override  # type: ignore[attr-defined]


def test_list_and_delete_document(env: Env, sql: async_sessionmaker[AsyncSession]) -> None:
    _use_sql(env, sql)
    uploaded = env.client.post("/documents/upload", files={"file": ("resume.txt", TEXT.encode())}).json()
    document_id = uploaded["document_id"]

    listed = env.client.get("/documents").json()
    assert [d["document_id"] for d in listed] == [document_id]
    assert listed[0]["filename"] == "resume.txt"
    assert env.bm25.search("python", 1)

    assert env.client.delete(f"/documents/{document_id}").status_code == 204
    assert env.client.get("/documents").json() == []
    assert document_id in env.vector_store.deleted
    assert env.bm25.search("python", 1) == []
    assert not list(env.upload_dir.iterdir())
    assert env.client.delete(f"/documents/{document_id}").status_code == 404


def test_chat_can_be_limited_to_chosen_documents(env: Env) -> None:
    env.llm.json_replies = [{"intent": "document_question"}]
    env.client.post("/chat", json={"message": "what are the skills?", "document_ids": ["doc-1"]})
    assert env.vector_store.search_filters == [["doc-1"]]


def test_scoped_question_still_answers_when_scores_are_low(env: Env) -> None:
    weak = ScoredChunk(
        document_id="doc-1", chunk_id="c1", chunk_index=0, source="resume.txt", page=None,
        chunking_strategy="recursive", text="Backend engineer, Python.", rerank_score=0.05,
    )
    env.reranker.results = [weak]
    env.llm.json_replies = [{"intent": "document_question"}, {"intent": "document_question"}]
    env.llm.text_replies = ["She is a backend engineer."]

    unscoped = env.client.post("/chat", json={"message": "summarise the resume"}).json()
    assert unscoped["reply"] == NOT_FOUND  # nothing passed the 0.3 cut-off

    scoped = env.client.post(
        "/chat", json={"message": "summarise the resume", "document_ids": ["doc-1"]}
    ).json()
    assert scoped["reply"] == "She is a backend engineer."
    assert scoped["sources"][0]["source"] == "resume.txt"


async def test_same_interview_slot_cannot_be_booked_twice(sql: async_sessionmaker[AsyncSession]) -> None:
    def booking(name: str) -> Booking:
        return Booking(name=name, email=f"{name}@x.com", interview_date=date(2030, 1, 1), interview_time=time(10, 0))

    async with sql() as session:
        await save_booking(session, booking("sera"))
    async with sql() as session:
        with pytest.raises(SlotTakenError):
            await save_booking(session, booking("ram"))