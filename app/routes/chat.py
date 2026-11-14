from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import MemoryDep, RagDep, SessionDep, require_api_key
from app.errors import BACKEND_ERRORS, LLM_ERRORS, SessionNotFoundError
from app.schemas import ChatRequest, ChatResponse
from app.services.router import handle_message

router = APIRouter(tags=["Chat"], dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, session: SessionDep, memory: MemoryDep, rag: RagDep) -> ChatResponse:
    """Ask about your documents, book an interview, or just talk."""
    try:
        return await handle_message(body=body, session=session, memory=memory, rag=rag)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail="Unknown session_id. Leave it out to start a new chat.") from error
    except LLM_ERRORS as error:
        raise HTTPException(status_code=502, detail="The language model is not available right now.") from error
    except BACKEND_ERRORS as error:
        raise HTTPException(status_code=503, detail="A backend service (Redis, Qdrant or the database) is unavailable.") from error