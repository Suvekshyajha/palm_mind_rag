import json

import redis.asyncio as redis

from app.config import Settings
from app.schemas import ChatMessage


class Memory:
    """Redis storage: chat history (a List) and unfinished bookings (a Hash)."""

    def __init__(self, settings: Settings) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._max_messages = settings.history_messages_stored
        self._chat_ttl = settings.chat_ttl_seconds
        self._booking_ttl = settings.booking_ttl_seconds

    # ---- chat history ----
    async def get_history(self, session_id: str, limit: int) -> list[ChatMessage]:
        """The last `limit` messages, oldest first."""
        raw = await self._redis.lrange(f"chat:{session_id}", -limit, -1)
        return [json.loads(item) for item in raw]

    async def add_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        key = f"chat:{session_id}"
        await self._redis.rpush(key, *[json.dumps(message) for message in messages])
        await self._redis.ltrim(key, -self._max_messages, -1)  # keep only the newest
        await self._redis.expire(key, self._chat_ttl)  # session expires if unused

    # ---- booking in progress ----
    async def get_booking(self, session_id: str) -> dict[str, str]:
        """Empty dict means no booking is in progress."""
        return await self._redis.hgetall(f"booking:{session_id}")

    async def save_booking(self, session_id: str, details: dict[str, str]) -> None:
        """Replace the saved booking. The 'status' field marks that one has started."""
        key = f"booking:{session_id}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            pipe.hset(key, mapping={"status": "collecting", **details})
            pipe.expire(key, self._booking_ttl)
            await pipe.execute()

    async def clear_booking(self, session_id: str) -> None:
        await self._redis.delete(f"booking:{session_id}")

    async def close(self) -> None:
        await self._redis.aclose()