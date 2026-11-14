import asyncio

from sentence_transformers import SentenceTransformer


class Embedder:
    """Turns text into vectors. Used by both ingestion and RAG."""

    # BGE works better when search questions (not documents) start with this.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension() or 768

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Normalised vectors make cosine similarity work properly.
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # The model is heavy, so run it in a thread to keep the API responsive.
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, query: str) -> list[float]:
        vectors = await asyncio.to_thread(self._encode, [self.QUERY_PREFIX + query])
        return vectors[0]