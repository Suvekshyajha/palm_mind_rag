from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)


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

    async def add_chunks(self, payloads: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        """Store each chunk's vector together with its text and metadata."""
        points = [
            PointStruct(id=payload["chunk_id"], vector=vector, payload=payload)
            for payload, vector in zip(payloads, vectors)
        ]
        for start in range(0, len(points), 100):  # send in batches of 100
            await self._client.upsert(collection_name=self._collection, points=points[start : start + 100])

    async def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        """Dense search. Returns payloads with the cosine score added."""
        result = await self._client.query_points(
            collection_name=self._collection, query=vector, limit=limit, with_payload=True
        )
        return [{**(point.payload or {}), "score": point.score} for point in result.points]

    async def get_all_chunks(self) -> list[dict[str, Any]]:
        """Read every stored chunk (used to build the BM25 index)."""
        chunks: list[dict[str, Any]] = []
        offset = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(record.payload for record in records if record.payload)
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