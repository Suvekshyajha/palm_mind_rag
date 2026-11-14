from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.typedefs import ChunkPayload, ScoredChunk


class VectorStore:
    """Small wrapper around Qdrant. One collection is shared by all documents."""

    def __init__(self, url: str, collection: str) -> None:
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection

    async def ensure_collection(self, dimension: int) -> None:
        """Create the collection on first run (cosine similarity)."""
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    async def add_chunks(self, payloads: list[ChunkPayload], vectors: list[list[float]]) -> None:
        """Store each chunk's vector together with its text and metadata."""
        points = [
            PointStruct(id=payload["chunk_id"], vector=vector, payload=cast(dict[str, Any], payload))
            for payload, vector in zip(payloads, vectors)
        ]
        for start in range(0, len(points), 100):  # send in batches of 100
            await self._client.upsert(collection_name=self._collection, points=points[start : start + 100])

    async def search(
        self, vector: list[float], limit: int, document_ids: list[str] | None = None
    ) -> list[ScoredChunk]:
        """Dense search. Returns payloads with the cosine score added.

        If `document_ids` is given, only chunks of those documents are searched.
        """
        query_filter = (
            Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))])
            if document_ids
            else None
        )
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [cast(ScoredChunk, {**(point.payload or {}), "score": point.score}) for point in result.points]

    async def get_all_chunks(self) -> list[ChunkPayload]:
        """Read every stored chunk (used to build the BM25 index)."""
        chunks: list[ChunkPayload] = []
        offset = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(cast(ChunkPayload, record.payload) for record in records if record.payload)
            if offset is None:
                return chunks

    async def delete_document(self, document_id: str) -> None:
        """Remove all chunks of one document (used to undo a failed upload)."""
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            ),
        )

    async def close(self) -> None:
        await self._client.close()