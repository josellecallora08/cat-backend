"""Text-to-Speech endpoint supporting ElevenLabs (primary) and gTTS (fallback).

ElevenLabs provides high-quality multilingual voices that handle Taglish naturally.
gTTS (Google Translate) is used as a free fallback when ElevenLabs is not configured.
"""

import asyncio
import io
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings


logger = logging.getLogger(__name__)

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    lang: str = "tl"  # Filipino/Tagalog for Taglish
    voice_id: str | None = None  # ElevenLabs voice ID override


def _synthesize_elevenlabs_stream(text: str, voice_id: str) -> Iterator[bytes]:
    """Return ElevenLabs' incremental MP3 iterator without buffering it."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    return client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        optimize_streaming_latency=2,
    )


def _next_audio_chunk(chunks: Iterator[bytes]) -> bytes | None:
    """Read one provider chunk in a worker thread, using None as EOF."""
    return next(chunks, None)


def _stream_with_first_chunk(first_chunk: bytes, chunks: Iterator[bytes]) -> Iterator[bytes]:
    """Preserve a preflighted chunk and lazily forward the rest of the provider stream."""
    yield first_chunk
    yield from chunks


def _synthesize_gtts(text: str, lang: str) -> io.BytesIO:
    """Synthesize speech using gTTS (Google Translate TTS)."""
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang, slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


@router.post("/tts")
async def synthesize_speech(body: TTSRequest):
    """Convert text to speech. Returns MP3 audio stream.

    Tries ElevenLabs first, then gTTS, then returns 503 so frontend uses browser TTS.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Try ElevenLabs
    use_elevenlabs = settings.tts_provider == "elevenlabs" or (
        settings.tts_provider == "auto" and settings.elevenlabs_api_key
    )

    if use_elevenlabs and settings.elevenlabs_api_key:
        try:
            voice_id = body.voice_id or settings.elevenlabs_voice_id
            audio_chunks = _synthesize_elevenlabs_stream(body.text, voice_id)
            # Fetch only the first chunk before returning headers. This preserves
            # the gTTS fallback for provider failures without buffering full audio.
            first_chunk = await asyncio.to_thread(_next_audio_chunk, audio_chunks)
            if not first_chunk:
                raise RuntimeError("ElevenLabs returned an empty audio stream")
            return StreamingResponse(
                _stream_with_first_chunk(first_chunk, audio_chunks),
                media_type="audio/mpeg",
                headers={
                    "Cache-Control": "no-store",
                    "Content-Disposition": "inline; filename=speech.mp3",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as e:
            logger.warning("ElevenLabs TTS failed: %s — trying gTTS fallback", e)

    # Try gTTS fallback
    try:
        audio_buffer = await asyncio.to_thread(_synthesize_gtts, body.text, body.lang)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"},
        )
    except Exception as e:
        logger.warning("gTTS also failed: %s — frontend will use browser TTS", e)

    # Both failed — return 503 so frontend falls back to browser TTS
    raise HTTPException(
        status_code=503,
        detail="TTS temporarily unavailable",
    )
