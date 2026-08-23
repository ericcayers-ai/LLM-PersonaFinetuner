"""Reference-audio sanitization, ported from the user's clonevoice.py script.

`clonevoice.py` was a 120-line script using pocket_tts (Kyutai MIT, 100M params,
CPU-realtime, 24kHz). The two relevant pieces ported verbatim:

- `sanitize_reference_audio(input_path)` - reads a WAV, converts to mono,
  resamples to 24kHz, peak-normalizes to 0.85, writes 16-bit PCM WAV.
- `sanitize_streaming_chunk(buf, sample_rate)` - same algorithm but operates
  on raw WAV bytes so the live adaptive session can sanitize each chunk
  before recomputing the speaker embedding / re-deriving the reference.

Why: clonevoice.py's exact algorithm is what the rest of the live pipeline
(embedding + TTS) was tuned against. Keeping it bit-identical ensures no
behavior change for the existing 22-backend synthesis path.
"""

from __future__ import annotations

import io
import os
from typing import Optional


def _to_float32(data, np):
    """Mirror clonevoice.py: integer PCM -> float32 in [-1, 1]."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


def _mono(data, np):
    """Mean-down stereo to mono (clonevoice.py: `np.mean(data, axis=1)`)."""
    if data.ndim > 1 and data.shape[1] > 1:
        return np.mean(data, axis=1)
    return data


def _resample(data, sr: int, target_sr: int, scipy_signal):
    if sr == target_sr:
        return data
    num_samples = int(len(data) * target_sr / sr)
    return scipy_signal.resample(data, num_samples)


def _peak_normalize(data, np, target_peak: float = 0.85):
    max_val = float(np.max(np.abs(data))) if data.size else 0.0
    if max_val > 0:
        return (data / max_val) * target_peak
    return data


def _encode_wav_bytes(audio_int16, sample_rate: int, scipy_io) -> bytes:
    buf = io.BytesIO()
    scipy_io.write(buf, sample_rate, audio_int16)
    return buf.getvalue()


def sanitize_reference_audio(
    input_path: str,
    target_sr: int = 24000,
    output_path: Optional[str] = None,
) -> str:
    """Verbatim port of clonevoice.py `sanitize_reference_audio`.

    Cleans a reference WAV file using scipy/numpy:
      - Converts stereo to mono
      - Resamples to 24kHz
      - Peak-normalizes to 0.85 to prevent static/clipping
    Writes a 16-bit PCM WAV. If `output_path` is None the cleaned file is
    written alongside the input as `cleaned_voice.wav` (clonevoice's default).
    """
    np = _np()
    scipy_io = _scipy_io()
    scipy_signal = _scipy_signal()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_path), "cleaned_voice.wav")

    sr, data = scipy_io.read(input_path)
    data = _to_float32(data, np)
    data = _mono(data, np)
    if sr != target_sr:
        data = _resample(data, sr, target_sr, scipy_signal)
    data = _peak_normalize(data, np, target_peak=0.85)

    data_int16 = (data * 32767).astype(np.int16)
    scipy_io.write(output_path, target_sr, data_int16)
    return output_path


def sanitize_streaming_chunk(buf: bytes, sample_rate: int = 24000) -> bytes:
    """Same algorithm as `sanitize_reference_audio` but operates on raw WAV
    bytes (no disk write) for the live adaptive session. Returns sanitized
    16-bit PCM WAV bytes at `sample_rate` (default 24kHz).
    """
    np = _np()
    scipy_io = _scipy_io()
    scipy_signal = _scipy_signal()

    sr, data = scipy_io.read(io.BytesIO(buf))
    data = _to_float32(data, np)
    data = _mono(data, np)
    if sr != sample_rate:
        data = _resample(data, sr, sample_rate, scipy_signal)
    data = _peak_normalize(data, np, target_peak=0.85)

    data_int16 = (data * 32767).astype(np.int16)
    return _encode_wav_bytes(data_int16, sample_rate, scipy_io)


# ---------------------------------------------------------------------------
# Lazy optional imports (mirrors pipeline.py style so missing scipy doesn't
# block unrelated modules).
# ---------------------------------------------------------------------------


def _np():
    import importlib
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError("Missing 'numpy'. Install it with: pip install numpy") from exc


def _scipy_io():
    import importlib
    try:
        return importlib.import_module("scipy.io.wavfile")
    except ImportError as exc:
        raise RuntimeError("Missing 'scipy'. Install it with: pip install scipy") from exc


def _scipy_signal():
    import importlib
    try:
        return importlib.import_module("scipy.signal")
    except ImportError as exc:
        raise RuntimeError("Missing 'scipy'. Install it with: pip install scipy") from exc
