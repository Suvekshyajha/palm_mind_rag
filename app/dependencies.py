from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.services.embedder import Embedder
from app.services.memory import Memory
from app.services.rag import RagResources
from app.services.retrieval import BM25Index
from app.services.vector_store import VectorStore


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One database session per request."""
    async with request.app.state.session_factory() as session:
        yield session


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_bm25(request: Request) -> BM25Index:
    return request.app.state.bm25


def get_memory(request: Request) -> Memory:
    return request.app.state.memory


def get_rag(request: Request) -> RagResources:
    return request.app.state.rag


# Short names the routes use as type hints.
SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
BM25Dep = Annotated[BM25Index, Depends(get_bm25)]
MemoryDep = Annotated[Memory, Depends(get_memory)]
RagDep = Annotated[RagResources, Depends(get_rag)]
SettingsDep = Annotated[Settings, Depends(get_settings)]