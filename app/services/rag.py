
import logging
from dataclasses import dataclass

from app.config import Settings

from app.services.embedder import Embedder
from app.services.generation import NOT_FOUND, generate_answer, rewrite_question
from app.services.llm import LLM
from app.services.reranking import Reranker, filter_relevant
from app.services.retrieval import BM25Index, hybrid_search
from app.services.vector_store import VectorStore
from app.typedefs import ChatMessage, SourceRef

logger = logging.getLogger("uvicorn.error")


@dataclass
class RagResources:
    """Everything the chat flow needs, created once at startup."""

    llm: LLM
    embedder: Embedder
    vector_store: VectorStore
    bm25: BM25Index
    reranker: Reranker
    settings: Settings


async def answer_question(
    question: str,
    history: list[ChatMessage],
    rag: RagResources,
    document_ids: list[str] | None = None,
) -> tuple[str, list[SourceRef]]:
    """The full RAG pipeline. Returns the answer and its sources."""

    settings = rag.settings

    # 1. Make follow-ups standalone.
    standalone = (
        await rewrite_question(rag.llm, question, history)
        if history
        else question
    )

    # 2. Dense + BM25 search, merged with RRF.
    candidates = await hybrid_search(
        standalone,
        rag.embedder,
        rag.vector_store,
        rag.bm25,
        settings,
        document_ids,
    )

    # 3. Rerank and keep relevant chunks.
    ranked = await rag.reranker.rerank(standalone, candidates)

    logger.info(
        "Question: %s | candidates: %d",
        standalone,
        len(candidates),
    )

    for c in ranked[:settings.final_top_k]:
        logger.info(
            "  score=%.3f page=%s %r",
            c["rerank_score"],
            c["page"],
            c["text"][:60],
        )

    chunks = filter_relevant(
        ranked,
        settings.relevance_threshold,
        settings.final_top_k,
    )

    if not chunks and document_ids:
        chunks = ranked[:settings.final_top_k]

    if not chunks:
        return NOT_FOUND, []

    # 4. Generate answer using retrieved chunks.
    reply = await generate_answer(
        rag.llm,
        standalone,
        chunks,
        history,
    )

    # 5. Prepare source references.
    sources = [
        SourceRef(
            document_id=c["document_id"],
            chunk_id=c["chunk_id"],
            source=c["source"],
            page=c["page"],
        )
        for c in chunks
    ]

    return reply, sources
