"""Voice cloning, transcription, synthesis, and realtime call endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import wave
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.config import settings
from app.core.inference import inference_engine
from app.core.stt import stt_engine
from app.core.tts import voice_engine
from app.errors import api_error
from app.models.schemas import (
    SynthesisRequest,
    TranscriptionResponse,
    VoiceCapabilitiesResponse,
    VoiceProfileResponse,
)
from app.storage import store

router = APIRouter(prefix="/api/voice", tags=["voice"])

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}


def _profile_response(profile: dict) -> VoiceProfileResponse:
    return VoiceProfileResponse.model_validate(profile)


def _validate_audio(content: bytes, filename: str, *, reference: bool = False) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise api_error(
            400,
            "unsupported_audio_format",
            f"Unsupported audio type: {ext or '(none)'}.",
            hint=f"Use one of: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}.",
        )
    if not content:
        raise api_error(400, "empty_audio", "The uploaded audio is empty.")
    limit = (
        settings.max_voice_upload_bytes
        if reference
        else settings.max_call_utterance_bytes
    )
    if len(content) > limit:
        raise api_error(
            400,
            "audio_too_large",
            f"Audio exceeds the {limit // (1024 * 1024)} MB limit.",
        )

    if reference and ext == ".wav":
        try:
            with wave.open(io.BytesIO(content), "rb") as wav:
                duration = wav.getnframes() / max(wav.getframerate(), 1)
        except (wave.Error, EOFError) as exc:
            raise api_error(400, "invalid_audio", "The WAV file could not be decoded.") from exc
        if not 3 <= duration <= 30:
            raise api_error(
                400,
                "invalid_reference_duration",
                "Voice references must be between 3 and 30 seconds.",
                hint="Use a clean 8–15 second recording with one speaker and little background noise.",
            )
    return ext


@router.get("/capabilities", response_model=VoiceCapabilitiesResponse)
def capabilities() -> VoiceCapabilitiesResponse:
    return VoiceCapabilitiesResponse(
        tts_engine="Chatterbox",
        stt_engine="faster-whisper + Silero VAD",
        tts_model=settings.tts_model,
        stt_model=settings.stt_model,
        tts_installed=voice_engine.is_installed(),
        stt_installed=stt_engine.is_installed(),
        sample_rate_hz=24000,
        supported_voice_formats=sorted(SUPPORTED_AUDIO_EXTENSIONS),
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
) -> TranscriptionResponse:
    if not file.filename:
        raise api_error(400, "missing_filename", "No audio filename was provided.")
    content = await file.read()
    _validate_audio(content, file.filename)
    if language and language.lower() not in SUPPORTED_LANGUAGES:
        raise api_error(400, "unsupported_language", "Unsupported language code.")

    try:
        result = await asyncio.to_thread(
            stt_engine.transcribe,
            content,
            language=language.lower() if language else None,
        )
    except ValueError as exc:
        raise api_error(400, "no_speech", str(exc)) from exc
    except RuntimeError as exc:
        raise api_error(503, "stt_unavailable", str(exc)) from exc
    except Exception as exc:
        raise api_error(500, "transcription_failed", f"Transcription failed: {exc}") from exc
    return TranscriptionResponse.model_validate(result)


@router.get("/{persona_id}/profile", response_model=VoiceProfileResponse)
def get_voice_profile(persona_id: str) -> VoiceProfileResponse:
    persona = store.get_persona(persona_id)
    if not persona:
        raise api_error(404, "persona_not_found", "Persona not found.")
    profile = persona.get("voice_profile")
    if not profile:
        raise api_error(404, "voice_profile_not_found", "No cloned voice is configured.")
    return _profile_response(profile)


@router.post("/{persona_id}/profile", response_model=VoiceProfileResponse)
async def save_voice_profile(
    persona_id: str,
    file: UploadFile = File(...),
    consent_confirmed: bool = Form(default=False),
    language: str = Form(default="en"),
) -> VoiceProfileResponse:
    if not store.get_persona(persona_id):
        raise api_error(404, "persona_not_found", "Persona not found.")
    if not consent_confirmed:
        raise api_error(
            400,
            "consent_required",
            "Voice-owner consent must be confirmed before cloning.",
        )
    if not file.filename:
        raise api_error(400, "missing_filename", "No audio filename was provided.")
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        raise api_error(400, "unsupported_language", "Unsupported language code.")

    content = await file.read()
    _validate_audio(content, file.filename, reference=True)
    profile = store.save_voice_profile(
        persona_id,
        file.filename,
        content,
        language=language,
        consent_confirmed=True,
    )
    if not profile:
        raise api_error(404, "persona_not_found", "Persona not found.")
    return _profile_response(profile)


@router.delete("/{persona_id}/profile", status_code=204)
def delete_voice_profile(persona_id: str) -> Response:
    if not store.get_persona(persona_id):
        raise api_error(404, "persona_not_found", "Persona not found.")
    if not store.delete_voice_profile(persona_id):
        raise api_error(404, "voice_profile_not_found", "No cloned voice is configured.")
    return Response(status_code=204)


@router.post("/{persona_id}/synthesize")
async def synthesize_speech(persona_id: str, req: SynthesisRequest) -> Response:
    persona = store.get_persona(persona_id)
    if not persona:
        raise api_error(404, "persona_not_found", "Persona not found.")
    profile = persona.get("voice_profile")
    if not profile:
        raise api_error(
            400,
            "voice_profile_required",
            "Upload a consented voice reference before synthesizing speech.",
        )

    try:
        audio = await asyncio.to_thread(
            voice_engine.synthesize,
            req.text,
            profile["sample_path"],
            language=req.language,
        )
    except ValueError as exc:
        raise api_error(400, "synthesis_error", str(exc)) from exc
    except RuntimeError as exc:
        raise api_error(503, "tts_unavailable", str(exc)) from exc
    except Exception as exc:
        raise api_error(500, "synthesis_failed", f"Speech synthesis failed: {exc}") from exc

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="voice-preview.wav"'},
    )


async def _send_call_error(
    websocket: WebSocket,
    code: str,
    message: str,
    *,
    recoverable: bool,
) -> None:
    await websocket.send_json(
        {
            "type": "error",
            "code": code,
            "message": message,
            "recoverable": recoverable,
        }
    )


@router.websocket("/call/{persona_id}")
async def voice_call(websocket: WebSocket, persona_id: str) -> None:
    await websocket.accept()
    persona = store.get_persona(persona_id)
    if not persona:
        await _send_call_error(websocket, "persona_not_found", "Persona not found.", recoverable=False)
        await websocket.close(code=4404)
        return
    if not persona.get("adapter_path"):
        await _send_call_error(
            websocket,
            "persona_not_trained",
            "Train this persona before starting a voice call.",
            recoverable=False,
        )
        await websocket.close(code=4409)
        return
    profile = persona.get("voice_profile")
    if not profile:
        await _send_call_error(
            websocket,
            "voice_profile_required",
            "Upload a voice reference before starting a call.",
            recoverable=False,
        )
        await websocket.close(code=4409)
        return

    call_id = str(uuid.uuid4())
    history: list[dict[str, str]] = []
    await websocket.send_json(
        {
            "type": "ready",
            "call_id": call_id,
            "message": "Listening",
            "stt_model": settings.stt_model,
            "tts_model": settings.tts_model,
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _send_call_error(
                        websocket, "invalid_message", "Invalid call control message.", recoverable=True
                    )
                    continue
                if control.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                continue

            audio = message.get("bytes")
            if audio is None:
                continue
            if not audio:
                await _send_call_error(
                    websocket, "empty_audio", "The utterance was empty.", recoverable=True
                )
                continue
            if len(audio) > settings.max_call_utterance_bytes:
                await _send_call_error(
                    websocket,
                    "audio_too_large",
                    "The utterance exceeded the call audio limit.",
                    recoverable=True,
                )
                await websocket.send_json({"type": "ready", "call_id": call_id, "message": "Listening"})
                continue

            await websocket.send_json({"type": "state", "state": "transcribing"})
            try:
                transcript = await asyncio.to_thread(stt_engine.transcribe, audio)
            except ValueError as exc:
                await _send_call_error(websocket, "no_speech", str(exc), recoverable=True)
                await websocket.send_json({"type": "ready", "call_id": call_id, "message": "Listening"})
                continue
            except Exception as exc:
                await _send_call_error(websocket, "stt_failed", str(exc), recoverable=True)
                await websocket.send_json({"type": "ready", "call_id": call_id, "message": "Listening"})
                continue

            user_text = transcript["text"]
            history.append({"role": "user", "content": user_text})
            history = history[-20:]
            await websocket.send_json(
                {
                    "type": "user_transcript",
                    "text": user_text,
                    "language": transcript["language"],
                }
            )
            await websocket.send_json({"type": "state", "state": "thinking"})

            try:
                result = await asyncio.to_thread(
                    inference_engine.chat,
                    persona_id,
                    history,
                    max_tokens=240,
                    temperature=0.7,
                )
                assistant_text = result["response"]
                history.append({"role": "assistant", "content": assistant_text})
                history = history[-20:]
                await websocket.send_json(
                    {"type": "assistant_transcript", "text": assistant_text}
                )
                await websocket.send_json({"type": "state", "state": "speaking"})
                speech = await asyncio.to_thread(
                    voice_engine.synthesize,
                    assistant_text,
                    profile["sample_path"],
                    language=profile.get("language", "en"),
                )
                await websocket.send_json(
                    {"type": "audio", "content_type": "audio/wav", "size_bytes": len(speech)}
                )
                await websocket.send_bytes(speech)
            except Exception as exc:
                await _send_call_error(websocket, "response_failed", str(exc), recoverable=True)

            await websocket.send_json({"type": "ready", "call_id": call_id, "message": "Listening"})
    except WebSocketDisconnect:
        return
