from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# One chat message as stored in Redis: {"role", "content", optional "sources"}
ChatMessage = dict[str, Any]
BookingStatus = Literal["in_progress", "completed"]


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunking_strategy: ChunkingStrategy
    page_count: int | None  # only PDFs have pages
    chunk_count: int
    processing_time_seconds: float


class Intent(StrEnum):
    DOCUMENT_QUESTION = "document_question"
    DOCUMENT_FOLLOW_UP = "document_follow_up"
    INTERVIEW_BOOKING = "interview_booking"
    BOOKING_FOLLOW_UP = "booking_follow_up"
    GENERAL_CONVERSATION = "general_conversation"


class IntentResult(BaseModel):
    """What the LLM must return from intent detection (validated)."""

    intent: Intent


class Source(BaseModel):
    document_id: str
    chunk_id: str
    source: str  # original file name
    page: int | None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # Optional. If missing, a new session is created.
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[\w-]+$")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Intent
    sources: list[Source] = []
    booking_status: BookingStatus | None = None