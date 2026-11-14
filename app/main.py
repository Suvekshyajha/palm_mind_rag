import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db import Base
from app.routes import chat, ingestion
from app.services.embedder import Embedder
from app.services.llm import LLM
from app.services.memory import Memory
from app.services.rag import RagResources
from app.services.reranking import Reranker
from app.services.retrieval import BM25Index
from app.services.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the shared resources once at startup and close them at shutdown."""
    settings = get_settings()

    engine = create_async_engine(settings.database_url)
    app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Create the documents and bookings tables if they do not exist yet.
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    # Loading the models is slow, so it happens once here (in threads, not per request).
    embedder = await asyncio.to_thread(Embedder, settings.embedding_model)
    reranker = await asyncio.to_thread(Reranker, settings.reranker_model)

    vector_store = VectorStore(settings.qdrant_url, settings.qdrant_collection)
    await vector_store.ensure_collection(embedder.dimension)

    bm25 = BM25Index()
    await bm25.rebuild(vector_store)  # index every chunk already stored

    memory = Memory(settings)
    llm = LLM(settings.groq_api_key, settings.groq_model)

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.bm25 = bm25
    app.state.memory = memory
    app.state.rag = RagResources(
        llm=llm, embedder=embedder, vector_store=vector_store, bm25=bm25, reranker=reranker, settings=settings
    )

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    yield

    await memory.close()
    await vector_store.close()
    await engine.dispose()


app = FastAPI(title="Palm Mind RAG", lifespan=lifespan)
app.include_router(ingestion.router)
app.include_router(chat.router)