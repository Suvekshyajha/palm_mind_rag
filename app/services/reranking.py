import asyncio
from typing import TYPE_CHECKING

from app.typedefs import ScoredChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder: reads the question and a chunk together and scores the match."""

    def __init__(self, model_name: str) -> None:
        # Imported here so tests and tooling do not need to load PyTorch.
        from sentence_transformers import CrossEncoder

        self._model: CrossEncoder = CrossEncoder(model_name, max_length=512)

    async def rerank(self, question: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Add a 'rerank_score' (0 to 1) to each chunk, best first."""
        if not chunks:
            return []
        pairs = [(question, chunk["text"]) for chunk in chunks]
        # For this model, sentence-transformers already turns the raw output into 0..1.
        scores = await asyncio.to_thread(self._model.predict, pairs)
        scored: list[ScoredChunk] = []
        for chunk, score in zip(chunks, scores):
            item = ScoredChunk(**chunk)
            item["rerank_score"] = float(score)
            scored.append(item)
        return sorted(scored, key=lambda chunk: chunk["rerank_score"], reverse=True)


def filter_relevant(chunks: list[ScoredChunk], threshold: float, limit: int) -> list[ScoredChunk]:
    """Keep only chunks that scored high enough, at most `limit` of them."""
    return [chunk for chunk in chunks if chunk["rerank_score"] >= threshold][:limit]