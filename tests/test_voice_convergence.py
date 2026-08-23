"""Tests for the WS-8 voice convergence workstream.

Covering:
- sanitize_reference_audio matches the clonevoice.py reference
- VoiceSession converges early via embedding similarity before the duration cap
- VoiceSession still falls back to the duration cap when embeddings are noisy
- Live voice API responses expose convergence_score / embedding_history /
  convergence_std fields
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.voice.service import VoiceSession
from app.main import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_wav(path: Path, sr: int = 22050, duration: float = 1.0, freq: float = 220.0) -> None:
    """Write a mono int16 WAV with a sine tone at peak amplitude 0.85."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio = audio * (0.85 / peak)
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio_int16.tobytes())


def _make_wav_bytes(sr: int = 24000, duration: float = 0.5, freq: float = 220.0) -> bytes:
    """Return mono int16 WAV bytes of a sine tone (peak 0.5) at given sample rate."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Slice 1: sanitize
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_sanitize_reference_audio_matches_clonevoice(self, tmp_path: Path):
        from app.core.voice.sanitize import sanitize_reference_audio

        src = tmp_path / "input.wav"
        _write_synthetic_wav(src, sr=22050, duration=0.5, freq=440.0)

        out_path = sanitize_reference_audio(str(src))

        assert Path(out_path).exists()
        # clonevoice.py writes mono 16-bit PCM at 24000 Hz
        with wave.open(out_path, "rb") as f:
            assert f.getnchannels() == 1
            assert f.getframerate() == 24000
            assert f.getsampwidth() == 2
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

        peak = float(np.max(np.abs(data.astype(np.float32) / 32767.0)))
        # clonevoice.py peak-normalizes to 0.85
        assert math.isclose(peak, 0.85, rel_tol=0.05), f"peak was {peak}"

    def test_sanitize_streaming_chunk_returns_wav_bytes(self):
        from app.core.voice.sanitize import sanitize_streaming_chunk

        chunk = _make_wav_bytes(sr=48000, duration=0.25)
        out = sanitize_streaming_chunk(chunk, sample_rate=24000)

        assert isinstance(out, (bytes, bytearray))
        # Round-trip: should be a parseable 24kHz mono WAV
        with wave.open(io.BytesIO(out), "rb") as f:
            assert f.getnchannels() == 1
            assert f.getframerate() == 24000
            assert f.getsampwidth() == 2


# ---------------------------------------------------------------------------
# Slice 2: embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_cosine_similarity_identical_vectors(self):
        from app.core.voice.embedding import cosine_similarity

        a = [0.1, 0.2, 0.3, 0.4]
        assert math.isclose(cosine_similarity(a, a), 1.0, rel_tol=1e-5)

    def test_cosine_similarity_orthogonal_vectors(self):
        from app.core.voice.embedding import cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-6)

    def test_extract_embedding_is_stable(self):
        from app.core.voice.embedding import extract_embedding

        chunk = _make_wav_bytes(sr=24000, duration=0.5, freq=220.0)
        e1 = extract_embedding(chunk, sample_rate=24000)
        e2 = extract_embedding(chunk, sample_rate=24000)
        assert len(e1) == 64
        assert len(e2) == 64
        # Same input -> same vector
        for a, b in zip(e1, e2):
            assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)

    def test_extract_embedding_differs_for_different_audio(self):
        from app.core.voice.embedding import cosine_similarity, extract_embedding

        c1 = _make_wav_bytes(sr=24000, duration=0.5, freq=220.0)
        c2 = _make_wav_bytes(sr=24000, duration=0.5, freq=880.0)
        e1 = extract_embedding(c1, sample_rate=24000)
        e2 = extract_embedding(c2, sample_rate=24000)
        # Different content -> similarity below 1
        sim = cosine_similarity(e1, e2)
        assert sim < 0.99


# ---------------------------------------------------------------------------
# Slice 3: convergence lock
# ---------------------------------------------------------------------------


class TestConvergenceLock:
    def test_convergence_lock_kicks_in_before_duration_cap(self, tmp_path: Path):
        # Identical embeddings -> cosine_sim always 1.0 -> std-dev 0 -> lock fast.
        def fake_extract(audio_bytes: bytes, sample_rate: int) -> list[float]:
            return [0.1] * 64

        session = VoiceSession(tmp_path, lock_after_seconds=12.0, sample_rate=24000)
        chunk = _make_wav_bytes(sr=24000, duration=0.5)

        with patch("app.core.voice.service.extract_embedding", side_effect=fake_extract):
            state = None
            for _ in range(30):
                state = session.add_chunk(chunk)
                if state.locked:
                    break

        assert state is not None and state.locked
        # duration cap was 12s and each chunk is 0.5s; convergence should lock earlier
        assert state.total_seconds < 12.0

    def test_duration_lock_still_works_when_no_convergence(self, tmp_path: Path):
        # Each embedding is a different one-hot unit vector -> orthogonal ->
        # cosine similarities fluctuate with std-dev far above 0.02.
        counter = {"n": 0}

        def varying_extract(audio_bytes: bytes, sample_rate: int) -> list[float]:
            i = counter["n"]
            counter["n"] += 1
            vec = [0.0] * 64
            vec[i % 64] = 1.0
            return vec

        session = VoiceSession(tmp_path, lock_after_seconds=2.0, sample_rate=24000)
        chunk = _make_wav_bytes(sr=24000, duration=0.5)

        with patch("app.core.voice.service.extract_embedding", side_effect=varying_extract):
            state = None
            for _ in range(20):
                state = session.add_chunk(chunk)
                if state.locked:
                    break

        assert state is not None and state.locked
        # duration cap was 2s and the embeddings never converged
        assert state.total_seconds >= 2.0


# ---------------------------------------------------------------------------
# Slice 4: API surface
# ---------------------------------------------------------------------------


class TestVoiceAPISurface:
    def test_voice_response_includes_convergence_metrics(self):
        # /api/voice/live/start returns the new convergence fields
        res = client.post("/api/voice/live/start", json={"lock_after_seconds": 12.0})
        assert res.status_code == 200
        data = res.json()
        assert "convergence_score" in data
        assert "embedding_history" in data
        assert "convergence_std" in data
        # fields default to None / empty for a fresh session
        assert data["convergence_score"] is None
        assert data["embedding_history"] in (None, [])
        assert data["convergence_std"] is None
        assert data["locked"] is False

        sid = data["session_id"]
        # /api/voice/live/{id}/status mirrors them
        res2 = client.get(f"/api/voice/live/{sid}/status")
        assert res2.status_code == 200
        data2 = res2.json()
        assert "convergence_score" in data2
        assert "embedding_history" in data2
        assert "convergence_std" in data2

        # clean up
        client.delete(f"/api/voice/live/{sid}")
