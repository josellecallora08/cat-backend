"""Text-to-Speech endpoint using gTTS for natural Filipino/Taglish pronunciation."""

import io
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    lang: str = "tl"  # Filipino/Tagalog — works great for Taglish


@router.post("/tts")
async def synthesize_speech(body: TTSRequest):
    """Convert text to speech using gTTS (Google Translate TTS).

    Returns MP3 audio that the frontend can play directly.
    Uses 'tl' (Tagalog) language which handles Taglish naturally.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        from gtts import gTTS

        tts = gTTS(text=body.text, lang=body.lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"},
        )
    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
