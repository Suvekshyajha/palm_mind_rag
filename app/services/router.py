import uuid

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import ChatMessage, ChatRequest, ChatResponse, Intent, IntentResult, Source
from app.services.booking import FIELDS, handle_booking
from app.services.generation import general_reply
from app.services.llm import LLM
from app.services.memory import Memory
from app.services.rag import RagResources, answer_question

DOCUMENT_INTENTS = {Intent.DOCUMENT_QUESTION, Intent.DOCUMENT_FOLLOW_UP}
BOOKING_INTENTS = {Intent.INTERVIEW_BOOKING, Intent.BOOKING_FOLLOW_UP}


async def detect_intent(
    llm: LLM, message: str, history: list[ChatMessage], booking_state: dict[str, str]
) -> Intent:
    """Ask the LLM what the user wants. It also sees if a booking is in progress."""
    in_progress = bool(booking_state)
    missing = [f for f in FIELDS if f not in booking_state]
    system = (
        "You route messages for an assistant that answers questions about uploaded documents "
        "and books interviews. Pick exactly one intent for the latest message:\n"
        "- document_question: asks something about the uploaded documents\n"
        "- document_follow_up: asks a follow-up that depends on the earlier chat\n"
        "- interview_booking: wants to book or schedule an interview\n"
        "- booking_follow_up: gives booking details or answers a booking question "
        "(name, email, date, time)\n"
        "- general_conversation: greetings, thanks, small talk\n"
        f"Booking in progress: {'yes, still missing: ' + ', '.join(missing) if in_progress else 'no'}.\n"
        'Return JSON only, like {"intent": "document_question"}.'
    )
    recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history)
    user = f"Recent chat:\n{recent or '(none)'}\n\nLatest message: {message}"
    data = await llm.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        return IntentResult.model_validate(data).intent
    except ValidationError:  # the LLM gave something unusable, so fall back safely
        return Intent.BOOKING_FOLLOW_UP if in_progress else Intent.DOCUMENT_QUESTION


async def handle_message(
    *, body: ChatRequest, session: AsyncSession, memory: Memory, rag: RagResources
) -> ChatResponse:
    """The single entry point: load memory, detect intent, run one workflow, save."""
    session_id = body.session_id or str(uuid.uuid4())
    history = await memory.get_history(session_id, rag.settings.history_messages_in_prompt)
    booking_state = await memory.get_booking(session_id)

    intent = await detect_intent(rag.llm, body.message, history, booking_state)

    sources: list[Source] = []
    booking_status = None
    if intent in BOOKING_INTENTS:
        reply, booking_status = await handle_booking(
            message=body.message, session_id=session_id, llm=rag.llm, memory=memory, db_session=session
        )
    elif intent in DOCUMENT_INTENTS:
        reply, sources = await answer_question(body.message, history, rag)
    else:
        reply = await general_reply(rag.llm, body.message, history)

    await memory.add_messages(
        session_id,
        [
            {"role": "user", "content": body.message},
            {"role": "assistant", "content": reply, "sources": [s.model_dump() for s in sources]},
        ],
    )
    return ChatResponse(
        session_id=session_id, reply=reply, intent=intent, sources=sources, booking_status=booking_status
    )