import asyncio
from typing import Any

from sentence_transformers import CrossEncoder

Chunk = dict[str, Any]


class Reranker:
    """Cross-encoder: reads the question and a chunk together and scores the match."""

    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name, max_length=512)

    async def rerank(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        """Add a 'rerank_score' (0 to 1) to each chunk, best first."""
        if not chunks:
            return []
        pairs = [(question, chunk["text"]) for chunk in chunks]
        # For this model, sentence-transformers already turns the raw output into 0..1.
        scores = await asyncio.to_thread(self._model.predict, pairs)
        scored = [{**chunk, "rerank_score": float(score)} for chunk, score in zip(chunks, scores)]
        return sorted(scored, key=lambda chunk: chunk["rerank_score"], reverse=True)


def filter_relevant(chunks: list[Chunk], threshold: float, limit: int) -> list[Chunk]:
    """Keep only chunks that scored high enough, at most `limit` of them."""
    return [chunk for chunk in chunks if chunk["rerank_score"] >= threshold][:limit]