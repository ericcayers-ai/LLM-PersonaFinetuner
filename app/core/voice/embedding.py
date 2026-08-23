"""Speaker embedding extractor + cosine similarity.

Stub implementation: 64-dimensional spectral feature derived from the audio's
FFT magnitude bucketed into 64 bins and L2-normalized. The output is
deterministic for identical audio (same bytes -> same vector) and varies
with content (different sine freq -> different vector). This is enough to
drive convergence detection in `VoiceSession.add_chunk`.

Why a stub: a real resemblyzer / wespeaker / pyannote model would add a
heavy dependency that must be loaded on every chunk. The convergence lock
only needs stable-enough features to detect when successive sanitized
references stop changing meaningfully. Once a real model is wired in, this
module is the single point to swap.

Public surface:
    extract_embedding(audio_bytes, sample_rate) -> list[float]   # 64-dim
    cosine_similarity(a, b) -> float                            # [-1, 1]
"""

from __future__ import annotations

import io
import math

_EMBED_DIM = 64


def cosine_similarity(a, b) -> float:
    """Standard cosine similarity between two equal-length float vectors."""
    if len(a) != len(b):
        raise ValueError(f"cosine_similarity: length mismatch ({len(a)} vs {len(b)})")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def extract_embedding(audio_bytes: bytes, sample_rate: int = 24000) -> list[float]:
    """Compute a deterministic 64-dim spectral embedding from raw audio bytes.

    Accepts any format `soundfile` can decode (WAV, FLAC, OGG, ...). For WAV
    chunks the live adaptive session uses this is the typical case. Output
    is L2-normalized so cosine_similarity reduces to a dot product.
    """
    np = _np()
    sf = _soundfile()

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = _resample(audio, sr, sample_rate, np)

    if audio.size < 8:
        # Pathological: too short to characterize. Return a neutral unit
        # vector along dim 0 so cosine_similarity stays well-defined.
        vec = np.zeros(_EMBED_DIM, dtype=np.float32)
        vec[0] = 1.0
        return vec.tolist()

    # DC removal -> spectral magnitude -> bucket into 64 dims.
    audio = audio - float(np.mean(audio))
    spec = np.abs(np.fft.rfft(audio))
    spec = spec.astype(np.float32)

    n = spec.shape[0]
    if n >= _EMBED_DIM:
        # Pick _EMBED_DIM evenly-spaced indices across the spectrum.
        idx = (np.linspace(0.0, float(n - 1), _EMBED_DIM)).astype(np.int64)
        feat = spec[idx]
    else:
        feat = np.zeros(_EMBED_DIM, dtype=np.float32)
        feat[:n] = spec

    # Log-compress to reduce dynamic range dominance.
    feat = np.log1p(feat)

    norm = float(np.linalg.norm(feat))
    if norm > 1e-8:
        feat = feat / norm

    return feat.astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# Optional-import helpers
# ---------------------------------------------------------------------------


def _np():
    import importlib
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError("Missing 'numpy'. Install it with: pip install numpy") from exc


def _soundfile():
    import importlib
    try:
        return importlib.import_module("soundfile")
    except ImportError as exc:
        raise RuntimeError("Missing 'soundfile'. Install it with: pip install soundfile") from exc


def _resample(audio, sr: int, target_sr: int, np):
    # Use simple linear resample. We only need a stable spectral summary,
    # not studio-quality SRC, soxr isn't required here.
    if sr == target_sr or audio.size == 0:
        return audio
    duration = audio.size / float(sr)
    n_out = int(round(duration * target_sr))
    if n_out <= 1:
        return np.zeros(max(n_out, 1), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, audio.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)
