from typing import Literal, NotRequired, TypedDict


class ChunkPayload(TypedDict):
    """One chunk as stored in Qdrant (text + metadata)."""

    document_id: str
    chunk_id: str
    chunk_index: int
    source: str
    page: int | None
    chunking_strategy: str
    text: str


class ScoredChunk(ChunkPayload, total=False):
    """A chunk after a search step. Each step adds its own score."""

    score: float  # dense cosine score
    rrf_score: float  # fusion score (ordering only)
    rerank_score: float  # cross-encoder score, 0..1


class SourceRef(TypedDict):
    document_id: str
    chunk_id: str
    source: str
    page: int | None


class ChatMessage(TypedDict):
    """One chat message as stored in Redis."""

    role: Literal["user", "assistant"]
    content: str
    sources: NotRequired[list[SourceRef]]


class LLMMessage(TypedDict):
    """One message sent to the LLM."""

    role: Literal["system", "user", "assistant"]
    content: str