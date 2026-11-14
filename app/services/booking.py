import json
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Booking, SlotTakenError, save_booking
from app.schemas import BookingStatus
from app.services.llm import LLM
from app.services.memory import Memory
from app.typedefs import LLMMessage

FIELDS = ("name", "email", "date", "time")
LABELS = {"name": "full name", "email": "email address", "date": "interview date", "time": "interview time"}
_EMAIL = TypeAdapter(EmailStr)


def local_now(timezone: str) -> datetime:
    """The current date and time in the booking timezone (no tzinfo, so it compares easily)."""
    return datetime.now(ZoneInfo(timezone)).replace(tzinfo=None)


async def extract_details(llm: LLM, message: str, collected: dict[str, str], today: date) -> dict[str, str]:
    """Ask the LLM to pull any booking details out of the message."""
    missing = [LABELS[f] for f in FIELDS if f not in collected]
    system = (
        "You extract interview booking details from a chat message.\n"
        f"Today is {today:%A, %Y-%m-%d}.\n"
        f"Already collected: {json.dumps(collected)}. Still missing: {', '.join(missing)}.\n"
        "The user may be answering a question about a missing detail, so a short message "
        "like 'Sera' or '3pm' can be the answer.\n"
        'Return JSON only: {"name": string or null, "email": string or null, '
        '"date": "YYYY-MM-DD" or null, "time": "HH:MM" in 24-hour format or null}.\n'
        "Use null for anything not given. Turn words like 'tomorrow' or 'next Friday' into real dates."
    )
    prompt: list[LLMMessage] = [{"role": "system", "content": system}, {"role": "user", "content": message}]
    data = await llm.chat_json(prompt)
    return {f: str(data[f]).strip() for f in FIELDS if data.get(f) not in (None, "", "null")}


def validate_details(details: dict[str, str], now: datetime | None = None) -> tuple[dict[str, str], list[str]]:
    """Check each detail. Returns (the valid ones, a list of problems to tell the user)."""
    now = now or datetime.now()
    valid: dict[str, str] = {}
    problems: list[str] = []

    name = details.get("name", "")
    if name:
        if len(name) >= 2 and not any(ch.isdigit() for ch in name):
            valid["name"] = name
        else:
            problems.append("That name doesn't look right.")

    if details.get("email"):
        try:
            valid["email"] = str(_EMAIL.validate_python(details["email"]))
        except ValidationError:
            problems.append("That email address doesn't look valid.")

    if details.get("date"):
        try:
            day = date.fromisoformat(details["date"])
            if day < now.date():
                problems.append("That date is in the past.")
            else:
                valid["date"] = day.isoformat()
        except ValueError:
            problems.append("I couldn't understand that date.")

    if details.get("time"):
        try:
            valid["time"] = time.fromisoformat(details["time"]).strftime("%H:%M")
        except ValueError:
            problems.append("I couldn't understand that time.")

    # If the interview is today, the time must still be ahead.
    if "date" in valid and "time" in valid and valid["date"] == now.date().isoformat():
        if time.fromisoformat(valid["time"]) <= now.time():
            problems.append("That time has already passed today.")
            del valid["time"]

    return valid, problems


def _ask_for_missing(problems: list[str], missing: list[str]) -> str:
    intro = " ".join(problems) if problems else "Happy to book your interview."
    return f"{intro} Please send your: {', '.join(LABELS[f] for f in missing)}."


async def handle_booking(
    *, message: str, session_id: str, llm: LLM, memory: Memory, db_session: AsyncSession, timezone: str = "UTC"
) -> tuple[str, BookingStatus]:
    """One step of the booking conversation. Returns the reply and the booking status."""
    saved = await memory.get_booking(session_id)
    collected = {f: saved[f] for f in FIELDS if f in saved}

    now = local_now(timezone)
    found = await extract_details(llm, message, collected, now.date())
    valid, problems = validate_details({**collected, **found}, now)
    missing = [f for f in FIELDS if f not in valid]

    if missing:  # keep what we have and ask for the rest
        await memory.save_booking(session_id, valid)
        return _ask_for_missing(problems, missing), "in_progress"

    interview_date = date.fromisoformat(valid["date"])
    interview_time = time.fromisoformat(valid["time"])
    try:
        await save_booking(
            db_session,
            Booking(
                name=valid["name"], email=valid["email"], interview_date=interview_date, interview_time=interview_time
            ),
        )
    except SlotTakenError:
        # Someone else has this slot. Keep the other details and ask for a new time.
        del valid["time"]
        await memory.save_booking(session_id, valid)
        return "That date and time is already booked. Please send a different interview time.", "in_progress"
    await memory.clear_booking(session_id)
    reply = (
        f"Your interview is booked, {valid['name']}! "
        f"{interview_date:%A, %d %B %Y} at {valid['time']} (email on file: {valid['email']})."
    )
    return reply, "completed"