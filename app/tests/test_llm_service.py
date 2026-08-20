"""Tests for OpenAI-compatible LLM provider request construction."""

from typing import Any
from unittest.mock import patch

import pytest

from app.services.llm_service import LLMMessage, LLMService


class _FakeResponse:
    """Minimal HTTP response used by the provider request test."""

    def raise_for_status(self) -> None:
        """Match the HTTPX response method used by LLMService."""

    def json(self) -> dict[str, Any]:
        """Return a valid OpenAI-compatible completion response."""
        return {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "model": "ep-byteplus-test",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }


class _FakeAsyncClient:
    """Capture one request without making an external network call."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.json_payload: dict[str, Any] | None = None
        self.headers: dict[str, str] | None = None

    async def __aenter__(self) -> "_FakeAsyncClient":
        """Return the fake client for an async context manager."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Close the fake client context."""

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> _FakeResponse:
        """Capture the request and return a provider-shaped response."""
        self.url = url
        self.json_payload = json
        self.headers = headers
        return _FakeResponse()


@pytest.mark.asyncio
async def test_byteplus_request_uses_openai_compatible_contract() -> None:
    """BytePlus should receive the base URL, endpoint model, and Bearer key."""
    client = _FakeAsyncClient()

    with patch("app.services.llm_service.httpx.AsyncClient", return_value=client):
        service = LLMService(
            base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
            model="ep-byteplus-test",
            api_key="test-byteplus-key",  # pragma: allowlist secret
            provider="byteplus",
        )
        result = await service.chat_completion(
            [LLMMessage(role="user", content="Return JSON.")],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "test_result", "strict": True, "schema": {}},
            },
        )

    assert service.provider == "byteplus"
    assert client.url == "https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions"
    assert client.headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-byteplus-key",
    }
    assert client.json_payload is not None
    assert client.json_payload["model"] == "ep-byteplus-test"
    assert client.json_payload["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in client.json_payload
    assert "reasoning_format" not in client.json_payload
    assert result.content == "{}"
    assert result.finish_reason == "stop"


def test_auto_provider_detects_byteplus_host() -> None:
    """Auto mode should recognize the official BytePlus ModelArk host."""
    service = LLMService(
        base_url="https://ark.eu-west.bytepluses.com/api/v3/",
        model="ep-byteplus-test",
        api_key="test-byteplus-key",  # pragma: allowlist secret
    )

    assert service.provider == "byteplus"
