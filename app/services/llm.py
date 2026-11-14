import json
from typing import Any

from groq import AsyncGroq


class LLM:
    """Thin wrapper around Groq so the rest of the code never touches the SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """Normal text reply."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    async def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Reply as JSON (the prompt must mention JSON). Returns {} if it can't be parsed."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}