"""Live call loop scaffolding: ASR -> LLM chat -> synthesize through locked voice.

This module wires together the existing pieces so a future websocket handler
can spawn `live_call_loop(session_id)` per active call. Each step is wrapped
so the loop survives missing pieces (no trained persona, no pocket_tts, no
playback backend) without crashing the websocket.

Flow per turn:
  1. Drain incoming WAV chunks off an asyncio queue (the websocket producer).
  2. ASR the buffered audio via faster-whisper.
  3. Chat against the persona via `app.api.inference.chat` (LLM call).
  4. Synthesize the reply via `app.core.voice.pipeline.run_pipeline` using
     the session's locked reference.

This is scaffolding -- it does NOT require a real LLM or pocket_tts to be
running to start. When a step fails (e.g. "persona not found", "no CUDA"),
the failure is logged and the loop moves on so the live call doesn't die
on the first hiccup.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional

from app.core.voice.pipeline import VoiceCloneError
from app.core.voice.service import SynthesisConfig, VoiceSessionManager

log = logging.getLogger(__name__)


class LiveCallState:
    """Per-call bookkeeping exposed back to the websocket handler."""

    def __init__(self, session_id: str, persona_id: Optional[str] = None):
        self.session_id = session_id
        self.persona_id = persona_id
        self.started_at = time.monotonic()
        self.turn_count = 0
        self.last_user_text: Optional[str] = None
        self.last_reply_text: Optional[str] = None
        self.last_reply_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "turn_count": self.turn_count,
            "last_user_text": self.last_user_text,
            "last_reply_text": self.last_reply_text,
            "last_reply_path": self.last_reply_path,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 2),
        }


async def live_call_loop(
    session_id: str,
    *,
    manager: VoiceSessionManager,
    queue: "asyncio.Queue[bytes | None]",
    persona_id: Optional[str] = None,
    silence_for_close_seconds: float = 1.5,
    max_turns: int = 50,
    on_reply_ready=None,
) -> LiveCallState:
    """Run a live call against an active VoiceSession.

    Args:
        session_id: id of an active VoiceSession (already receiving audio).
        manager: session manager that owns the session.
        queue: asyncio queue the websocket producer pushes WAV chunks into.
            A `None` sentinel ends the loop.
        persona_id: trained persona to chat against. None -> "no persona" mode
            (the LLM step is skipped).
        silence_for_close_seconds: how long to wait for new audio before
            treating the user's turn as complete and synthesizing a reply.
        max_turns: hard cap on turns (also ends the call).
        on_reply_ready: optional async callback(synth_path, reply_text) the
            websocket handler can use to stream the synthesized audio back.

    Returns the populated LiveCallState when the loop exits.
    """
    state = LiveCallState(session_id=session_id, persona_id=persona_id)
    log.info("[live_call %s] starting (persona=%s, max_turns=%d)", session_id, persona_id, max_turns)

    session = manager.get(session_id)
    if session is None:
        log.warning("[live_call %s] session not found, exiting", session_id)
        return state

    while state.turn_count < max_turns:
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=silence_for_close_seconds)
        except asyncio.TimeoutError:
            log.info("[live_call %s] silence for %.1fs, ending call", session_id, silence_for_close_seconds)
            break
        if chunk is None:
            log.info("[live_call %s] received shutdown sentinel", session_id)
            break

        session.add_chunk(chunk)
        if not session.locked:
            # Keep draining -- we only respond once the reference has stabilized.
            continue

        # Turn boundary: we have a chunk AND the session is locked.
        state.turn_count += 1
        log.info("[live_call %s] turn %d beginning", session_id, state.turn_count)

        user_text = await _safe_asr(session)
        if not user_text:
            log.info("[live_call %s] ASR returned no text, skipping turn", session_id)
            continue
        state.last_user_text = user_text

        reply_text = await _safe_chat(persona_id, user_text)
        if reply_text is None:
            continue
        state.last_reply_text = reply_text

        reply_path = await _safe_synthesize(session, reply_text)
        if reply_path is None:
            continue
        state.last_reply_path = reply_path

        if on_reply_ready is not None:
            try:
                result = on_reply_ready(reply_path, reply_text)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.warning("[live_call %s] on_reply_ready callback raised: %s", session_id, exc)

    log.info("[live_call %s] exiting (turns=%d)", session_id, state.turn_count)
    return state


# ---------------------------------------------------------------------------
# Step helpers (each one is best-effort -- a missing dep shouldn't kill the loop)
# ---------------------------------------------------------------------------


async def _safe_asr(session) -> Optional[str]:
    """Transcribe the session's accumulated audio via faster-whisper.

    Returns the transcript or None on failure.
    """
    if session.reference_path is None:
        return None
    try:
        from app.core.voice.service import transcribe_reference
        # faster-whisper is sync; offload to a thread.
        return await asyncio.to_thread(transcribe_reference, Path(session.reference_path), "base", "cpu")
    except Exception as exc:
        log.warning("ASR step failed: %s", exc)
        return None


async def _safe_chat(persona_id: Optional[str], user_text: str) -> Optional[str]:
    """Call the LLM chat endpoint. Returns the assistant reply or None.

    Skipped silently when no persona_id is provided.
    """
    if not persona_id:
        log.debug("chat step skipped (no persona_id)")
        return None
    try:
        from app.api.inference import chat
        from app.models.schemas import ChatMessage, ChatRequest

        req = ChatRequest(
            persona_id=persona_id,
            messages=[ChatMessage(role="user", content=user_text)],
            max_tokens=256,
            temperature=0.7,
        )
        result = await asyncio.to_thread(chat, req)
        return result.response
    except Exception as exc:
        log.warning("chat step failed: %s", exc)
        return None


async def _safe_synthesize(session, reply_text: str) -> Optional[str]:
    """Run the standard voice pipeline against the locked reference.

    Returns the path to the synthesized WAV or None on failure.
    """
    if session.reference_path is None:
        return None
    try:
        from app.core.voice.service import run_pipeline

        out_path = Path(session.reference_path).parent / f"reply_{int(time.time() * 1000)}.wav"
        cfg = SynthesisConfig(backend="pocket", skip_denoise=True, upscale_backend="resample")
        await asyncio.to_thread(
            run_pipeline,
            text=reply_text,
            reference_audio=Path(session.reference_path),
            out=out_path,
            cfg=cfg,
        )
        return str(out_path)
    except VoiceCloneError as exc:
        log.warning("synthesize step failed (VoiceCloneError): %s", exc)
        return None
    except Exception as exc:
        log.warning("synthesize step failed: %s", exc)
        return None
