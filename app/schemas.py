from enum import Enum

from pydantic import BaseModel, Field

# The typed dicts for chunks, sources and chat messages live in app/typedefs.py.


# ==========================================
# INTENT SCHEMAS
# ==========================================

class Intent(str, Enum):
    DOCUMENT_QUESTION = "document_question"
    DOCUMENT_FOLLOW_UP = "document_follow_up"
    INTERVIEW_BOOKING = "interview_booking"
    BOOKING_FOLLOW_UP = "booking_follow_up"
    GENERAL_CONVERSATION = "general_conversation"


class IntentResult(BaseModel):
    intent: Intent


# ==========================================
# BOOKING SCHEMAS
# ==========================================

class BookingStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


# ==========================================
# DOCUMENT INGESTION SCHEMAS
# ==========================================

class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentInfo(BaseModel):
    model_config = {"from_attributes": True}

    document_id: str = Field(
        validation_alias="id",
        serialization_alias="document_id",
    )
    filename: str
    file_type: str
    file_path: str
    chunking_strategy: str
    chunk_count: int
    page_count: int | None = None


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunking_strategy: ChunkingStrategy
    page_count: int | None = None
    chunk_count: int
    processing_time_seconds: float


# ==========================================
# CHAT API REQUEST AND RESPONSE
# ==========================================

class Source(BaseModel):
    document_id: str
    chunk_id: str
    source: str
    page: int | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    document_ids: list[str] | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Intent
    sources: list[Source] = Field(default_factory=list)
    booking_status: BookingStatus | None = None