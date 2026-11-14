from app.services.generation import NOT_FOUND
from app.typedefs import ScoredChunk
from tests.conftest import Env, make_settings, tomorrow


def _say(env: Env, message: str, session_id: str | None = None) -> dict[str, object]:
    body: dict[str, str] = {"message": message}
    if session_id:
        body["session_id"] = session_id
    response = env.client.post("/chat", json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_general_conversation_creates_session_and_saves_history(env: Env) -> None:
    env.llm.json_replies = [{"intent": "general_conversation"}]
    env.llm.text_replies = ["Hello! How can I help?"]
    data = _say(env, "hi")
    assert data["intent"] == "general_conversation"
    assert len(env.memory.history[str(data["session_id"])]) == 2


def test_unknown_session_id_is_rejected(env: Env) -> None:
    response = env.client.post("/chat", json={"message": "hi", "session_id": "made-up-id"})
    assert response.status_code == 404


def test_document_question_with_no_matches_says_not_found(env: Env) -> None:
    env.llm.json_replies = [{"intent": "document_question"}]
    data = _say(env, "what is the leave policy?")
    assert data["reply"] == NOT_FOUND
    assert data["sources"] == []


def test_document_question_returns_answer_and_sources(env: Env) -> None:
    chunk = ScoredChunk(
        document_id="d1", chunk_id="c1", chunk_index=0, source="policy.txt", page=None,
        chunking_strategy="recursive", text="Twenty days of paid leave.", rerank_score=0.9,
    )
    env.reranker.results = [chunk]
    env.llm.json_replies = [{"intent": "document_question"}]
    env.llm.text_replies = ["You get twenty days."]
    data = _say(env, "how many leave days?")
    assert data["reply"] == "You get twenty days."
    assert data["sources"] == [{"document_id": "d1", "chunk_id": "c1", "source": "policy.txt", "page": None}]


def test_multi_turn_booking_is_completed_and_saved(env: Env) -> None:
    env.llm.json_replies = [
        {"intent": "interview_booking"}, {"name": None, "email": None, "date": None, "time": None},
        {"intent": "booking_follow_up"},
        {"name": "Sera Shrestha", "email": "sera@gmail.com", "date": tomorrow(), "time": "10:30"},
    ]
    first = _say(env, "I want to book an interview")
    assert first["booking_status"] == "in_progress"

    second = _say(env, "Sera Shrestha, sera@gmail.com, tomorrow 10:30", str(first["session_id"]))
    assert second["booking_status"] == "completed"
    assert len(env.session.saved) == 1
    assert env.session.saved[0].email == "sera@gmail.com"
    assert env.memory.bookings == {}  # unfinished state cleared


def test_booking_asks_again_when_slot_is_taken(env: Env) -> None:
    env.session.slot_taken = True
    env.llm.json_replies = [
        {"intent": "interview_booking"},
        {"name": "Sera Shrestha", "email": "sera@gmail.com", "date": tomorrow(), "time": "10:30"},
    ]
    data = _say(env, "book me tomorrow 10:30, Sera Shrestha, sera@gmail.com")
    assert data["booking_status"] == "in_progress"
    assert "already booked" in str(data["reply"])
    assert "time" not in env.memory.bookings[str(data["session_id"])]


def test_redis_outage_returns_503(env: Env) -> None:
    env.memory.fail = True
    response = env.client.post("/chat", json={"message": "hi", "session_id": "abc"})
    assert response.status_code == 503


def test_api_key_is_required_when_configured(env: Env, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from app.config import get_settings

    secured = make_settings(tmp_path, api_key="secret")
    env.client.app.dependency_overrides[get_settings] = lambda: secured  # type: ignore[attr-defined]
    assert env.client.post("/chat", json={"message": "hi"}).status_code == 401
    env.llm.json_replies = [{"intent": "general_conversation"}]
    env.llm.text_replies = ["hello"]
    ok = env.client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200