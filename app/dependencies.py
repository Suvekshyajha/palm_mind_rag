import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
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
    embedder: Embedder = request.app.state.embedder
    return embedder


def get_vector_store(request: Request) -> VectorStore:
    vector_store: VectorStore = request.app.state.vector_store
    return vector_store


def get_bm25(request: Request) -> BM25Index:
    bm25: BM25Index = request.app.state.bm25
    return bm25


def get_memory(request: Request) -> Memory:
    memory: Memory = request.app.state.memory
    return memory


def get_rag(request: Request) -> RagResources:
    rag: RagResources = request.app.state.rag
    return rag


async def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """If API_KEY is set in .env, every request must send it in the X-API-Key header."""
    if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


# Short names the routes use as type hints.
SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
BM25Dep = Annotated[BM25Index, Depends(get_bm25)]
MemoryDep = Annotated[Memory, Depends(get_memory)]
RagDep = Annotated[RagResources, Depends(get_rag)]
SettingsDep = Annotated[Settings, Depends(get_settings)]