"""LLM service interface abstracting calls to Ollama/vLLM OpenAI-compatible API."""

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import settings


@dataclass
class LLMMessage:
    """A single message in a chat completion request."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM chat completion call."""

    content: str
    model: str
    usage: dict[str, int] | None = None


class LLMServiceProtocol(Protocol):
    """Protocol for LLM service implementations, enabling easy mocking."""

    async def chat_completion(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class LLMService:
    """LLM service that calls an OpenAI-compatible API (Ollama/vLLM)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self.api_key = api_key or settings.llm_api_key

    async def chat_completion(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the LLM API.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature override.
            max_tokens: Max tokens override.
            response_format: Optional response format (e.g. {"type": "json_object"}).

        Returns:
            LLMResponse with the generated content.

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.TimeoutException: If the request times out.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage")

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            usage=usage,
        )
