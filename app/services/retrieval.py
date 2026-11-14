import asyncio
import re
from collections.abc import Sequence
from typing import cast

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.services.embedder import Embedder
from app.services.vector_store import VectorStore
from app.typedefs import ChunkPayload, ScoredChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Keyword search kept in memory.

    Built from Qdrant at startup. New uploads are added with `add`, which does
    not read Qdrant again. Note: the index lives in one process, so with several
    workers each one needs a restart (or a shared index) to see new documents.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tokens: list[list[str]] = []
        # Chunks and index are stored together so a search never sees a mismatched pair.
        self._state: tuple[list[ChunkPayload], BM25Okapi] | None = None

    async def rebuild(self, vector_store: VectorStore) -> None:
        """Read every chunk from Qdrant and index it (used at startup)."""
        chunks = await vector_store.get_all_chunks()
        async with self._lock:
            await self._set(chunks, [_tokenize(chunk["text"]) for chunk in chunks])

    async def add(self, new_chunks: Sequence[ChunkPayload]) -> None:
        """Add chunks from a new upload without reading Qdrant again."""
        async with self._lock:
            current = self._state[0] if self._state else []
            tokens = [*self._tokens, *(_tokenize(chunk["text"]) for chunk in new_chunks)]
            await self._set([*current, *new_chunks], tokens)

    async def _set(self, chunks: list[ChunkPayload], tokens: list[list[str]]) -> None:
        self._tokens = tokens
        if not chunks:
            self._state = None
            return
        index = await asyncio.to_thread(BM25Okapi, tokens)
        self._state = (chunks, index)

    async def remove(self, document_id: str) -> None:
        """Take a deleted document out of the index."""
        async with self._lock:
            if self._state is None:
                return
            keep = [i for i, chunk in enumerate(self._state[0]) if chunk["document_id"] != document_id]
            await self._set([self._state[0][i] for i in keep], [self._tokens[i] for i in keep])

    def search(self, query: str, limit: int, document_ids: list[str] | None = None) -> list[ChunkPayload]:
        if self._state is None:
            return []
        chunks, index = self._state
        scores = index.get_scores(_tokenize(query))
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        if document_ids:
            allowed = set(document_ids)
            ranked = [i for i in ranked if chunks[i]["document_id"] in allowed]
        # No score filter here: the reranker later decides what is truly relevant.
        return [chunks[i] for i in ranked[:limit]]


def reciprocal_rank_fusion(result_lists: Sequence[Sequence[ChunkPayload]], k: int) -> list[ScoredChunk]:
    """Merge ranked lists. A chunk ranked high in several lists wins.

    The score is 1 / (k + rank), added up over the lists. It is NOT a similarity
    score, so it is only used for ordering and never compared to a threshold.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, ChunkPayload] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = chunk
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    fused: list[ScoredChunk] = []
    for chunk_id in ordered:
        item = cast(ScoredChunk, dict(chunks[chunk_id]))
        item["rrf_score"] = scores[chunk_id]
        fused.append(item)
    return fused


async def hybrid_search(
    question: str,
    embedder: Embedder,
    vector_store: VectorStore,
    bm25: BM25Index,
    settings: Settings,
    document_ids: list[str] | None = None,
) -> list[ScoredChunk]:
    """Dense search + BM25 search, merged with RRF. Optionally limited to some documents."""
    query_vector = await embedder.embed_query(question)
    dense = await vector_store.search(query_vector, settings.dense_top_k, document_ids)
    sparse = bm25.search(question, settings.bm25_top_k, document_ids)
    fused = reciprocal_rank_fusion([dense, sparse], settings.rrf_k)
    return fused[: settings.rerank_top_k]