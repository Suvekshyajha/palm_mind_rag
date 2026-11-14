from fastapi import APIRouter, HTTPException
from groq import APIError

from app.dependencies import MemoryDep, RagDep, SessionDep
from app.schemas import ChatRequest, ChatResponse
from app.services.router import handle_message

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, session: SessionDep, memory: MemoryDep, rag: RagDep) -> ChatResponse:
    """Ask about your documents, book an interview, or just talk."""
    try:
        return await handle_message(body=body, session=session, memory=memory, rag=rag)
    except APIError as error:
        raise HTTPException(status_code=502, detail="The language model is not available right now.") from error