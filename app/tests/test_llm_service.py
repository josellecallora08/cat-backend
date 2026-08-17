"""Tests for provider compatibility in the shared LLM client."""

from unittest.mock import patch

import httpx
import pytest

from app.services.llm_service import LLMMessage, LLMService


class _FakeClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.payloads: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, _url: str, *, json: dict, headers: dict) -> httpx.Response:
        del headers
        self.payloads.append(json.copy())
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_falls_back_when_provider_rejects_json_schema() -> None:
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    client = _FakeClient(
        [
            httpx.Response(
                400,
                request=request,
                json={"error": {"message": "Model does not support response format json_schema"}},
            ),
            httpx.Response(
                200,
                request=request,
                json={
                    "choices": [{"message": {"content": "{}"}}],
                    "model": "test-model",
                },
            ),
        ]
    )
    service = LLMService(base_url="https://provider.test", model="test-model")

    with patch("app.services.llm_service.httpx.AsyncClient", return_value=client):
        response = await service.chat_completion(
            [LLMMessage(role="user", content="Return JSON")],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": {"type": "object"}},
            },
        )

    assert response.content == "{}"
    assert client.payloads[0]["response_format"]["type"] == "json_schema"
    assert client.payloads[1]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_uses_json_object_for_groq_without_a_rejected_schema_request() -> None:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client = _FakeClient(
        [
            httpx.Response(
                200,
                request=request,
                json={
                    "choices": [{"message": {"content": "{}"}}],
                    "model": "test-model",
                },
            )
        ]
    )
    service = LLMService(base_url="https://api.groq.com/openai/v1", model="test-model")

    with patch("app.services.llm_service.httpx.AsyncClient", return_value=client):
        await service.chat_completion(
            [LLMMessage(role="user", content="Return JSON")],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": {"type": "object"}},
            },
        )

    assert len(client.payloads) == 1
    assert client.payloads[0]["response_format"] == {"type": "json_object"}
