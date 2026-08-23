"""High-level orchestration: full clone pipeline + live adaptive voice sessions.

Live adaptive cloning: a VoiceSession accumulates streamed reference audio
(e.g. from a mic, or from an ongoing "call") into a rolling buffer. After
each chunk it re-derives a working reference clip. Once the accumulated
duration reaches `lock_after_seconds`, the session locks: the reference is
frozen and the same clip is reused for every subsequent synthesis call
instead of continuing to drift.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.voice.backends import (
    BACKEND_FUNCS,
    BACKEND_INFO,
    BACKENDS_NEEDING_REF_TEXT,
    BackendOptions,
)
from app.core.voice.embedding import cosine_similarity, extract_embedding
from app.core.voice.pipeline import (
    Sanitizer,
    VoiceCloneError,
    concat_audio,
    denoise_audio,
    load_audio,
    optional_import,
    save_wav,
    split_text,
    upscale_audio,
)


@dataclass
class SynthesisConfig:
    backend: str = "pocket"
    device: str = "auto"
    language: Optional[str] = "en"
    ref_text: str = ""
    max_chars: int = 300
    gap_ms: int = 180
    sanitize_sr: int = 24000
    trim_db: float = 45.0
    sanitize_denoise: bool = False
    prop_decrease: float = 0.5
    raw_ref: bool = False
    fade_ms: float = 12.0
    upscale_backend: str = "resample"  # 'resample' is the safe zero-setup default
    skip_denoise: bool = True  # DeepFilterNet3 pulls a heavy extra download; opt-in
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    backend_options: BackendOptions = field(default_factory=BackendOptions)
    # Convergence-based early lock for live adaptive sessions. The duration
    # cap (`lock_after_seconds`) is still the hard fallback. If the speaker
    # embedding has been stable for >=`lock_after_convergence_seconds_min`
    # audio seconds, we lock as soon as the std-dev of cosine similarities
    # vs. the previous 5 embeddings drops below `convergence_std_threshold`.
    # Hard upper bound is `lock_after_convergence_seconds`.
    lock_after_convergence_seconds: float = 30.0
    lock_after_convergence_seconds_min: float = 4.0
    convergence_std_threshold: float = 0.02
    convergence_window: int = 5


def transcribe_reference(audio_path: Path, model_name: str, device: str = "cpu") -> str:
    fw = optional_import("faster_whisper", "faster-whisper")
    WhisperModel = fw.WhisperModel
    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(audio_path), beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    if not text:
        raise VoiceCloneError(
            "faster-whisper produced no transcript from the reference audio. "
            "It may be silent -- pass ref_text explicitly instead."
        )
    return text


def _get_ref_text(reference: Path, cfg: SynthesisConfig) -> str:
    if cfg.ref_text:
        return cfg.ref_text
    return transcribe_reference(reference, cfg.whisper_model, cfg.whisper_device)


def synthesize_chunk(backend: str, text: str, reference: Path, out: Path, cfg: SynthesisConfig) -> Path:
    if backend not in BACKEND_FUNCS:
        raise VoiceCloneError(f"Unknown backend '{backend}'. Available: {', '.join(sorted(BACKEND_FUNCS))}")

    ref_text = ""
    if backend in BACKENDS_NEEDING_REF_TEXT:
        ref_text = _get_ref_text(reference, cfg)

    opts = cfg.backend_options
    opts.device = cfg.device
    opts.language = cfg.language

    fn = BACKEND_FUNCS[backend]
    fn(text, reference, ref_text, out, opts)
    if not out.exists():
        raise VoiceCloneError(f"[{backend}] did not create {out.name}")
    return out


def run_pipeline(*, text: str, reference_audio: Path, out: Path, cfg: SynthesisConfig) -> Path:
    """SANITIZE -> synthesize (per-chunk) -> concat -> UPSCALE -> DENOISE -> out"""
    work = Path(tempfile.mkdtemp(prefix="voice_pipeline_"))
    try:
        sanitized = work / "reference_sanitized.wav"
        Sanitizer(
            sample_rate=cfg.sanitize_sr, trim_db=cfg.trim_db, denoise=cfg.sanitize_denoise,
            prop_decrease=cfg.prop_decrease, fade_ms=cfg.fade_ms, raw_ref=cfg.raw_ref,
        ).run(reference_audio, sanitized)

        chunks = split_text(text, cfg.max_chars)
        if not chunks:
            raise VoiceCloneError("No text to synthesize.")

        chunk_files: list[Path] = []
        for idx, chunk in enumerate(chunks, 1):
            chunk_out = work / f"tts_{idx:04d}.wav"
            synthesize_chunk(cfg.backend, chunk, sanitized, chunk_out, cfg)
            chunk_files.append(chunk_out)

        cloned = work / "cloned.wav"
        concat_audio(chunk_files, cloned, sr=24000, gap_ms=max(0, cfg.gap_ms))

        upscaled = work / "upscaled.wav"
        upscale_audio(cloned, upscaled, backend=cfg.upscale_backend, device=cfg.device)

        final = work / "final.wav"
        denoise_audio(upscaled, final, skip=cfg.skip_denoise)

        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(final, out)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def list_backends() -> dict[str, dict[str, str]]:
    return BACKEND_INFO


# ---------------------------------------------------------------------------
# Live adaptive voice session
# ---------------------------------------------------------------------------

@dataclass
class VoiceSessionState:
    session_id: str
    lock_after_seconds: float
    total_seconds: float = 0.0
    locked: bool = False
    chunk_count: int = 0
    reference_path: Optional[str] = None
    # Convergence-detection metrics (None until enough embeddings exist).
    convergence_score: Optional[float] = None
    convergence_std: Optional[float] = None
    embedding_history: Optional[list[float]] = None
    locked_reason: str = "duration"  # "duration" | "convergence" | "forced"


class VoiceSession:
    """Accumulates streamed reference audio and locks in a clone reference
    once `lock_after_seconds` of audio has been captured.

    Two lock-in strategies, evaluated after every chunk:
    1. Duration cap (default 12s, hard fallback).
    2. Convergence: if the std-dev of cosine similarities between the new
       embedding and the previous `convergence_window` embeddings drops below
       `convergence_std_threshold`, AND we've accumulated at least
       `lock_after_convergence_seconds_min` of audio, AND we're still under
       `lock_after_convergence_seconds`, we lock early.

    Not thread-safe across sessions (each session has its own lock), safe for
    one writer (the websocket handler) at a time.
    """

    def __init__(
        self,
        session_dir: Path,
        lock_after_seconds: float = 12.0,
        sample_rate: int = 24000,
        *,
        lock_after_convergence_seconds: float = 30.0,
        lock_after_convergence_seconds_min: float = 4.0,
        convergence_std_threshold: float = 0.02,
        convergence_window: int = 5,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.dir = session_dir / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock_after_seconds = lock_after_seconds
        self.sample_rate = sample_rate
        self.lock_after_convergence_seconds = lock_after_convergence_seconds
        self.lock_after_convergence_seconds_min = lock_after_convergence_seconds_min
        self.convergence_std_threshold = convergence_std_threshold
        self.convergence_window = max(2, int(convergence_window))
        self._lock = threading.Lock()
        self._segments: list = []  # list[np.ndarray]
        self._embedding_history: list[list[float]] = []
        self.total_seconds = 0.0
        self.locked = False
        self.chunk_count = 0
        self.reference_path: Optional[Path] = None
        self.locked_reason: str = "duration"

    def _current_raw_path(self) -> Path:
        return self.dir / "raw_accumulated.wav"

    def _locked_path(self) -> Path:
        return self.dir / "reference_locked.wav"

    def add_chunk(self, wav_bytes: bytes) -> VoiceSessionState:
        """Append a chunk of WAV-encoded audio (mono, any sample rate) to the
        rolling reference buffer. Re-sanitizes and re-derives the working
        reference clip after each chunk until the session locks.
        """
        np = optional_import("numpy", "numpy")
        sf = optional_import("soundfile", "soundfile")
        soxr = optional_import("soxr", "soxr")

        with self._lock:
            if self.locked:
                return self.state()

            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != self.sample_rate:
                audio = soxr.resample(audio, sr, self.sample_rate, quality="HQ").astype(np.float32)

            self._segments.append(audio)
            self.chunk_count += 1
            self.total_seconds = sum(len(s) for s in self._segments) / float(self.sample_rate)

            combined = np.concatenate(self._segments)
            save_wav(self._current_raw_path(), combined, self.sample_rate)

            sanitized_path = self.dir / "working_reference.wav"
            Sanitizer(sample_rate=self.sample_rate, trim_db=40.0, fade_ms=8.0).run(
                self._current_raw_path(), sanitized_path
            )
            self.reference_path = sanitized_path

            # Convergence check (uses the sanitized reference, not the raw buffer)
            sanitized_bytes = sanitized_path.read_bytes()
            new_embedding = extract_embedding(sanitized_bytes, self.sample_rate)
            self._update_convergence(new_embedding)

            # Lock decision order:
            #   1. duration cap (hard fallback)
            #   2. convergence window (early lock if embedding is stable)
            if self.total_seconds >= self.lock_after_seconds:
                self._lock_in(sanitized_path, reason="duration")
            elif self._should_lock_by_convergence():
                self._lock_in(sanitized_path, reason="convergence")

            return self.state()

    def _update_convergence(self, new_embedding: list[float]) -> None:
        """Append the new embedding and trim the rolling window."""
        self._embedding_history.append(new_embedding)
        if len(self._embedding_history) > self.convergence_window + 1:
            self._embedding_history = self._embedding_history[-(self.convergence_window + 1):]

    def _should_lock_by_convergence(self) -> bool:
        # Need at least `convergence_window + 1` embeddings (one new + N previous)
        if len(self._embedding_history) < self.convergence_window + 1:
            return False
        if self.total_seconds < self.lock_after_convergence_seconds_min:
            return False
        if self.total_seconds > self.lock_after_convergence_seconds:
            return False
        sims: list[float] = []
        prev = self._embedding_history[0]
        for cur in self._embedding_history[1:]:
            sims.append(cosine_similarity(prev, cur))
            prev = cur
        n = len(sims)
        if n == 0:
            return False
        mean = sum(sims) / n
        var = sum((s - mean) ** 2 for s in sims) / n
        std = var ** 0.5
        return std < self.convergence_std_threshold

    def _convergence_metrics(self) -> tuple[Optional[float], Optional[float], Optional[list[float]]]:
        """Return (latest_sim, std_dev, list_of_cosine_sims) or (None,None,None)."""
        if len(self._embedding_history) < 2:
            return None, None, None
        sims: list[float] = []
        prev = self._embedding_history[0]
        for cur in self._embedding_history[1:]:
            sims.append(cosine_similarity(prev, cur))
            prev = cur
        n = len(sims)
        mean = sum(sims) / n
        var = sum((s - mean) ** 2 for s in sims) / n
        std = var ** 0.5
        return sims[-1], std, sims

    def _lock_in(self, working_reference: Path, *, reason: str = "duration") -> None:
        shutil.copyfile(working_reference, self._locked_path())
        self.reference_path = self._locked_path()
        self.locked = True
        self.locked_reason = reason

    def force_lock(self) -> VoiceSessionState:
        with self._lock:
            if not self.locked and self.reference_path is not None:
                self._lock_in(self.reference_path, reason="forced")
            return self.state()

    def state(self) -> VoiceSessionState:
        last_sim, std, sims = self._convergence_metrics()
        return VoiceSessionState(
            session_id=self.id,
            lock_after_seconds=self.lock_after_seconds,
            total_seconds=round(self.total_seconds, 2),
            locked=self.locked,
            chunk_count=self.chunk_count,
            reference_path=str(self.reference_path) if self.reference_path else None,
            convergence_score=last_sim,
            convergence_std=std,
            embedding_history=sims,
            locked_reason=self.locked_reason,
        )


class VoiceSessionManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        lock_after_seconds: float = 12.0,
        *,
        lock_after_convergence_seconds: float = 30.0,
        lock_after_convergence_seconds_min: float = 4.0,
        convergence_std_threshold: float = 0.02,
        convergence_window: int = 5,
    ) -> VoiceSession:
        session = VoiceSession(
            self.base_dir,
            lock_after_seconds=lock_after_seconds,
            lock_after_convergence_seconds=lock_after_convergence_seconds,
            lock_after_convergence_seconds_min=lock_after_convergence_seconds_min,
            convergence_std_threshold=convergence_std_threshold,
            convergence_window=convergence_window,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[VoiceSession]:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            shutil.rmtree(session.dir, ignore_errors=True)
