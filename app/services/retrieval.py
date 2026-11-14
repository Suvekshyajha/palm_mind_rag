import asyncio
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore

Chunk = dict[str, Any]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Keyword search kept in memory. Rebuilt from Qdrant at startup and after each upload."""

    def __init__(self) -> None:
        # Chunks and index are stored together so a search never sees a mismatched pair.
        self._state: tuple[list[Chunk], BM25Okapi] | None = None

    async def rebuild(self, vector_store: VectorStore) -> None:
        chunks = await vector_store.get_all_chunks()
        self._state = await asyncio.to_thread(self._build, chunks)

    @staticmethod
    def _build(chunks: list[Chunk]) -> tuple[list[Chunk], BM25Okapi] | None:
        if not chunks:
            return None
        return chunks, BM25Okapi([_tokenize(chunk["text"]) for chunk in chunks])

    def search(self, query: str, limit: int) -> list[Chunk]:
        if self._state is None:
            return []
        chunks, index = self._state
        scores = index.get_scores(_tokenize(query))
        best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:limit]
        # No score filter here: the reranker later decides what is truly relevant.
        return [chunks[i] for i in best]


def reciprocal_rank_fusion(result_lists: list[list[Chunk]], k: int) -> list[Chunk]:
    """Merge ranked lists. A chunk ranked high in several lists wins.

    The score is 1 / (k + rank), added up over the lists. It is NOT a similarity
    score, so it is only used for ordering and never compared to a threshold.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = chunk
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [{**chunks[chunk_id], "rrf_score": scores[chunk_id]} for chunk_id in ordered]


async def hybrid_search(
    question: str,
    embedder: Embedder,
    vector_store: VectorStore,
    bm25: BM25Index,
    settings: Settings,
) -> list[Chunk]:
    """Dense search + BM25 search, merged with RRF."""
    query_vector = await embedder.embed_query(question)
    dense = await vector_store.search(query_vector, settings.dense_top_k)
    sparse = bm25.search(question, settings.bm25_top_k)
    fused = reciprocal_rank_fusion([dense, sparse], settings.rrf_k)
    return fused[: settings.rerank_top_k]