"""Focused tests for incremental delivery from the public TTS endpoint."""

from collections.abc import Iterator

from app.api import tts


async def test_elevenlabs_audio_is_streamed_without_full_buffering(client, monkeypatch) -> None:
    consumed: list[bytes] = []

    def chunks() -> Iterator[bytes]:
        for chunk in (b"first", b"second", b"third"):
            consumed.append(chunk)
            yield chunk

    monkeypatch.setattr(tts.settings, "tts_provider", "elevenlabs")
    monkeypatch.setattr(tts.settings, "elevenlabs_api_key", "test-key")
    monkeypatch.setattr(tts, "_synthesize_elevenlabs_stream", lambda _text, _voice: chunks())

    response = await client.post("/api/tts", json={"text": "Kumusta?"})

    assert response.status_code == 200
    assert response.content == b"firstsecondthird"
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["x-accel-buffering"] == "no"
    assert consumed == [b"first", b"second", b"third"]


async def test_empty_elevenlabs_stream_uses_gtts_fallback(client, monkeypatch) -> None:
    monkeypatch.setattr(tts.settings, "tts_provider", "elevenlabs")
    monkeypatch.setattr(tts.settings, "elevenlabs_api_key", "test-key")
    monkeypatch.setattr(tts, "_synthesize_elevenlabs_stream", lambda _text, _voice: iter(()))
    monkeypatch.setattr(tts, "_synthesize_gtts", lambda _text, _lang: tts.io.BytesIO(b"fallback"))

    response = await client.post("/api/tts", json={"text": "Kumusta?"})

    assert response.status_code == 200
    assert response.content == b"fallback"


async def test_tts_rejects_blank_text_before_contacting_provider(client) -> None:
    response = await client.post("/api/tts", json={"text": "   "})

    assert response.status_code == 400
