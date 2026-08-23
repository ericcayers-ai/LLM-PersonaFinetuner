"""Voice cloning API: reference upload + synthesize (all 22 backends), and a
live adaptive session (mic streaming -> rolling clone -> lock-in) served over
a WebSocket.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.config import settings
from app.core.voice.pipeline import VoiceCloneError, load_audio
from app.core.voice.service import (
    SynthesisConfig,
    VoiceSessionManager,
    list_backends,
    run_pipeline,
)
from app.errors import api_error
from app.models.schemas import (
    VoiceBackendInfo,
    VoiceBackendsResponse,
    VoiceLiveStartRequest,
    VoiceLiveStatusResponse,
    VoiceReferenceResponse,
    VoiceSynthesizeRequest,
)

router = APIRouter(prefix="/api/voice", tags=["voice"])

_session_manager = VoiceSessionManager(settings.voice_sessions_dir)


def _voice_dir(voice_id: str) -> Path:
    return settings.voices_dir / voice_id


@router.get("/backends", response_model=VoiceBackendsResponse)
def get_backends() -> VoiceBackendsResponse:
    info = list_backends()
    return VoiceBackendsResponse(
        backends=[VoiceBackendInfo(name=name, weight=v["weight"], description=v["desc"]) for name, v in info.items()],
        default_backend=settings.default_voice_backend,
    )


@router.post("/reference", response_model=VoiceReferenceResponse)
async def upload_reference(file: UploadFile = File(...)) -> VoiceReferenceResponse:
    if not file.filename:
        raise api_error(400, "missing_filename", "No filename was provided with the upload.")

    content = await file.read()
    if not content:
        raise api_error(400, "empty_file", "The uploaded reference audio is empty.")
    if len(content) > settings.max_voice_upload_bytes:
        limit_mb = settings.max_voice_upload_bytes // (1024 * 1024)
        raise api_error(400, "file_too_large", f"Reference audio exceeds the {limit_mb} MB limit.")

    voice_id = uuid.uuid4().hex[:12]
    voice_dir = _voice_dir(voice_id)
    voice_dir.mkdir(parents=True, exist_ok=True)
    dest = voice_dir / "reference.wav"

    raw_dest = voice_dir / f"upload{Path(file.filename).suffix or '.wav'}"
    raw_dest.write_bytes(content)

    try:
        audio, sr = load_audio(raw_dest)
        duration = len(audio) / float(sr)
        from app.core.voice.pipeline import save_wav

        save_wav(dest, audio, sr)
    except VoiceCloneError as e:
        raise api_error(400, "invalid_audio", str(e)) from e
    except Exception as e:
        raise api_error(400, "invalid_audio", f"Could not read reference audio: {e}") from e

    return VoiceReferenceResponse(voice_id=voice_id, filename=file.filename, duration_seconds=round(duration, 2))


def _resolve_reference(req: VoiceSynthesizeRequest) -> Path:
    if req.voice_id:
        path = _voice_dir(req.voice_id) / "reference.wav"
        if not path.exists():
            raise api_error(404, "voice_not_found", f"No uploaded reference audio for voice_id={req.voice_id}")
        return path
    if req.session_id:
        session = _session_manager.get(req.session_id)
        if session is None:
            raise api_error(404, "session_not_found", f"No live voice session {req.session_id}")
        if session.reference_path is None:
            raise api_error(400, "session_not_ready", "Live session has not captured any reference audio yet.")
        return session.reference_path
    raise api_error(400, "missing_reference", "Provide voice_id or session_id to select a reference voice.")


@router.post("/synthesize")
def synthesize(req: VoiceSynthesizeRequest) -> FileResponse:
    if not req.text.strip():
        raise api_error(400, "empty_text", "Text to synthesize is empty.")

    reference = _resolve_reference(req)

    cfg = SynthesisConfig(
        backend=req.backend,
        device=req.device,
        language=req.language,
        ref_text=req.ref_text,
        upscale_backend=req.upscale_backend,
        skip_denoise=req.skip_denoise,
    )

    out_dir = settings.voices_dir / "_synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex[:12]}.wav"

    try:
        run_pipeline(text=req.text, reference_audio=reference, out=out_path, cfg=cfg)
    except VoiceCloneError as e:
        raise api_error(502, "synthesis_failed", str(e)) from e
    except Exception as e:
        raise api_error(500, "synthesis_failed", f"Voice synthesis failed: {e}") from e

    return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")


# ---------------------------------------------------------------------------
# Live adaptive session
# ---------------------------------------------------------------------------

@router.post("/live/start", response_model=VoiceLiveStatusResponse)
def start_live_session(req: VoiceLiveStartRequest) -> VoiceLiveStatusResponse:
    session = _session_manager.create(
        lock_after_seconds=req.lock_after_seconds,
        lock_after_convergence_seconds=req.lock_after_convergence_seconds,
        lock_after_convergence_seconds_min=req.lock_after_convergence_seconds_min,
        convergence_std_threshold=req.convergence_std_threshold,
        convergence_window=req.convergence_window,
    )
    state = session.state()
    return VoiceLiveStatusResponse(
        session_id=state.session_id, lock_after_seconds=state.lock_after_seconds,
        total_seconds=state.total_seconds, locked=state.locked, chunk_count=state.chunk_count,
        has_reference=state.reference_path is not None,
        convergence_score=state.convergence_score,
        convergence_std=state.convergence_std,
        embedding_history=state.embedding_history,
        locked_reason=state.locked_reason,
    )


@router.get("/live/{session_id}/status", response_model=VoiceLiveStatusResponse)
def live_status(session_id: str) -> VoiceLiveStatusResponse:
    session = _session_manager.get(session_id)
    if session is None:
        raise api_error(404, "session_not_found", f"No live voice session {session_id}")
    state = session.state()
    return VoiceLiveStatusResponse(
        session_id=state.session_id, lock_after_seconds=state.lock_after_seconds,
        total_seconds=state.total_seconds, locked=state.locked, chunk_count=state.chunk_count,
        has_reference=state.reference_path is not None,
        convergence_score=state.convergence_score,
        convergence_std=state.convergence_std,
        embedding_history=state.embedding_history,
        locked_reason=state.locked_reason,
    )


@router.post("/live/{session_id}/lock", response_model=VoiceLiveStatusResponse)
def live_force_lock(session_id: str) -> VoiceLiveStatusResponse:
    session = _session_manager.get(session_id)
    if session is None:
        raise api_error(404, "session_not_found", f"No live voice session {session_id}")
    state = session.force_lock()
    return VoiceLiveStatusResponse(
        session_id=state.session_id, lock_after_seconds=state.lock_after_seconds,
        total_seconds=state.total_seconds, locked=state.locked, chunk_count=state.chunk_count,
        has_reference=state.reference_path is not None,
        convergence_score=state.convergence_score,
        convergence_std=state.convergence_std,
        embedding_history=state.embedding_history,
        locked_reason=state.locked_reason,
    )


@router.delete("/live/{session_id}")
def live_drop(session_id: str) -> dict:
    _session_manager.drop(session_id)
    return {"ok": True}


@router.websocket("/live/{session_id}/stream")
async def live_stream(websocket: WebSocket, session_id: str) -> None:
    """Client sends binary WAV-encoded chunks; server replies with the
    current session status (JSON) after each chunk, including when it locks.
    """
    await websocket.accept()
    session = _session_manager.get(session_id)
    if session is None:
        await websocket.send_json({"error": "session_not_found"})
        await websocket.close(code=4404)
        return

    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                state = session.add_chunk(data)
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                continue
            await websocket.send_json({
                "session_id": state.session_id,
                "total_seconds": state.total_seconds,
                "locked": state.locked,
                "chunk_count": state.chunk_count,
                "lock_after_seconds": state.lock_after_seconds,
                "convergence_score": state.convergence_score,
                "convergence_std": state.convergence_std,
                "embedding_history": state.embedding_history,
                "locked_reason": state.locked_reason,
            })
            if state.locked:
                break
    except WebSocketDisconnect:
        pass
