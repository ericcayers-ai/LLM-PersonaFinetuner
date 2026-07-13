"""Voice cloning, transcription, synthesis, and realtime call endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
import wave
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

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
SUPPORTED_REFERENCE_EXTENSIONS = {".wav"}
SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}


def _profile_response(profile: dict) -> VoiceProfileResponse:
    return VoiceProfileResponse.model_validate(profile)


def _validate_audio(content: bytes, filename: str, *, reference: bool = False) -> str:
    ext = Path(filename).suffix.lower()
    if reference and ext not in SUPPORTED_REFERENCE_EXTENSIONS:
        raise api_error(
            400,
            "unsupported_reference_format",
            "Voice references must be uncompressed PCM WAV files.",
            hint="Record in the Voice step or convert the clip to WAV first.",
        )
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
                compression = wav.getcomptype()
        except (wave.Error, EOFError) as exc:
            raise api_error(400, "invalid_audio", "The WAV file could not be decoded.") from exc
        if compression != "NONE":
            raise api_error(
                400,
                "invalid_audio",
                "Voice references must use uncompressed PCM WAV encoding.",
            )
        if not 5 < duration <= 30:
            raise api_error(
                400,
                "invalid_reference_duration",
                "Voice references must be longer than 5 seconds and no more than 30 seconds.",
                hint="Use a clean 8–15 second recording with one speaker and little background noise.",
            )
    return ext


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise api_error(
                400,
                "audio_too_large",
                f"Audio exceeds the {limit // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


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
        supported_voice_formats=sorted(SUPPORTED_REFERENCE_EXTENSIONS),
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
) -> TranscriptionResponse:
    if not file.filename:
        raise api_error(400, "missing_filename", "No audio filename was provided.")
    content = await _read_limited(file, settings.max_call_utterance_bytes)
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
    if settings.tts_model == "turbo" and language != "en":
        raise api_error(
            400,
            "unsupported_language",
            "Chatterbox Turbo voice profiles must use English.",
            hint="Set PF_TTS_MODEL=multilingual-v3 to use another language.",
        )

    content = await _read_limited(file, settings.max_voice_upload_bytes)
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


def _websocket_origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    host = websocket.headers.get("host", "")
    parsed = urlsplit(origin)
    return (
        parsed.netloc == host
        or origin.rstrip("/") in {item.rstrip("/") for item in settings.allowed_origins}
    )


@router.websocket("/call/{persona_id}")
async def voice_call(websocket: WebSocket, persona_id: str) -> None:
    if not _websocket_origin_allowed(websocket):
        await websocket.close(code=4403, reason="Cross-origin voice calls are not allowed.")
        return
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
    turn_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=1)
    send_lock = asyncio.Lock()
    connected = True

    async def send_json(payload: dict) -> bool:
        nonlocal connected
        if not connected:
            return False
        try:
            async with send_lock:
                await websocket.send_json(payload)
            return True
        except Exception:
            connected = False
            return False

    async def send_error(code: str, message: str, *, recoverable: bool) -> bool:
        return await send_json(
            {
                "type": "error",
                "code": code,
                "message": message,
                "recoverable": recoverable,
            }
        )

    async def send_ready() -> bool:
        return await send_json(
            {"type": "ready", "call_id": call_id, "message": "Listening"}
        )

    async def receive_messages() -> None:
        nonlocal connected
        try:
            while connected:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    try:
                        control = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await send_error(
                            "invalid_message",
                            "Invalid call control message.",
                            recoverable=True,
                        )
                        continue
                    if control.get("type") == "ping":
                        await send_json({"type": "pong"})
                    continue

                audio = message.get("bytes")
                if audio is None:
                    continue
                if not audio:
                    await send_error(
                        "empty_audio", "The utterance was empty.", recoverable=True
                    )
                    await send_ready()
                    continue
                if len(audio) > settings.max_call_utterance_bytes:
                    await send_error(
                        "audio_too_large",
                        "The utterance exceeded the call audio limit.",
                        recoverable=True,
                    )
                    await send_ready()
                    continue
                if turn_queue.full():
                    await send_error(
                        "turn_in_progress",
                        "Wait for the current response before speaking again.",
                        recoverable=True,
                    )
                    continue
                await turn_queue.put(audio)
        except WebSocketDisconnect:
            pass
        finally:
            connected = False
            try:
                turn_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def process_turns() -> None:
        nonlocal history
        while connected:
            audio = await turn_queue.get()
            if audio is None or not connected:
                return

            if not await send_json({"type": "state", "state": "transcribing"}):
                return
            try:
                transcript = await asyncio.to_thread(stt_engine.transcribe, audio)
            except ValueError as exc:
                await send_error("no_speech", str(exc), recoverable=True)
                await send_ready()
                continue
            except Exception as exc:
                await send_error("stt_failed", str(exc), recoverable=True)
                await send_ready()
                continue
            if not connected:
                return

            user_text = transcript["text"]
            history.append({"role": "user", "content": user_text})
            history = history[-20:]
            if not await send_json(
                {
                    "type": "user_transcript",
                    "text": user_text,
                    "language": transcript["language"],
                }
            ):
                return
            if not await send_json({"type": "state", "state": "thinking"}):
                return

            try:
                result = await asyncio.to_thread(
                    inference_engine.chat,
                    persona_id,
                    history,
                    max_tokens=240,
                    temperature=0.7,
                )
                if not connected:
                    return
                assistant_text = result["response"]
                history.append({"role": "assistant", "content": assistant_text})
                history = history[-20:]
                if not await send_json(
                    {"type": "assistant_transcript", "text": assistant_text}
                ):
                    return
                if not await send_json({"type": "state", "state": "speaking"}):
                    return
                speech = await asyncio.to_thread(
                    voice_engine.synthesize,
                    assistant_text,
                    profile["sample_path"],
                    language=profile.get("language", "en"),
                )
                if not connected:
                    return
                if not await send_json(
                    {
                        "type": "audio",
                        "content_type": "audio/wav",
                        "size_bytes": len(speech),
                    }
                ):
                    return
                async with send_lock:
                    await websocket.send_bytes(speech)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await send_error("response_failed", str(exc), recoverable=True)

            if not await send_ready():
                return

    await websocket.send_json(
        {
            "type": "ready",
            "call_id": call_id,
            "message": "Listening",
            "stt_model": settings.stt_model,
            "tts_model": settings.tts_model,
        }
    )
    reader = asyncio.create_task(receive_messages())
    worker = asyncio.create_task(process_turns())
    done, pending = await asyncio.wait(
        {reader, worker},
        return_when=asyncio.FIRST_COMPLETED,
    )
    connected = False
    for task in pending:
        task.cancel()
    await asyncio.gather(*done, *pending, return_exceptions=True)
