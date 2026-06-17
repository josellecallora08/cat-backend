"""Text-to-Speech endpoint supporting ElevenLabs (primary) and gTTS (fallback).

ElevenLabs provides high-quality multilingual voices that handle Taglish naturally.
gTTS (Google Translate) is used as a free fallback when ElevenLabs is not configured.
"""

import io
import logging

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


async def _synthesize_elevenlabs(text: str, voice_id: str) -> io.BytesIO:
    """Synthesize speech using ElevenLabs API.

    Returns MP3 audio buffer.
    """
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",  # Best for Taglish
        output_format="mp3_44100_128",
    )

    audio_buffer = io.BytesIO()
    for chunk in audio_generator:
        audio_buffer.write(chunk)
    audio_buffer.seek(0)
    return audio_buffer


def _synthesize_gtts(text: str, lang: str) -> io.BytesIO:
    """Synthesize speech using gTTS (Google Translate TTS) as fallback."""
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang, slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


@router.post("/tts")
async def synthesize_speech(body: TTSRequest):
    """Convert text to speech.

    Uses ElevenLabs if API key is configured (high-quality multilingual voices).
    Falls back to gTTS (Google Translate) if ElevenLabs is not available.

    Returns MP3 audio stream.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Choose TTS provider
        use_elevenlabs = (
            settings.tts_provider == "elevenlabs" or
            (settings.tts_provider == "auto" and settings.elevenlabs_api_key)
        )

        if use_elevenlabs and settings.elevenlabs_api_key:
            voice_id = body.voice_id or settings.elevenlabs_voice_id
            audio_buffer = await _synthesize_elevenlabs(body.text, voice_id)
            logger.debug("TTS via ElevenLabs: %d bytes", audio_buffer.getbuffer().nbytes)
        else:
            # gTTS — best for Taglish pronunciation
            audio_buffer = _synthesize_gtts(body.text, body.lang)
            logger.debug("TTS via gTTS: %d bytes", audio_buffer.getbuffer().nbytes)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"},
        )
    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        # If ElevenLabs fails, try gTTS as last resort
        try:
            audio_buffer = _synthesize_gtts(body.text, body.lang)
            return StreamingResponse(
                audio_buffer,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=speech.mp3"},
            )
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e2)}")
