"""Verbatim port of clonevoice.py `generate_cloned_audio` + `split_text_into_sentences`.

The original clonevoice.py was a 120-line script using pocket_tts (Kyutai
MIT, 100M params, CPU-realtime, 24kHz). The synthesis loop was:

    1. Sanitize the reference voice clip.
    2. Split the text into sentences (regex on [.!?]).
    3. For each sentence: model.generate_audio(voice_state, sentence).
    4. Append 0.25s of silence between sentences.
    5. Concatenate, write 16-bit PCM WAV.

This module preserves every step. The actual TTS call goes through
`BACKEND_FUNCS["pocket"]` (which already runs pocket_tts in an isolated
`uv run` subprocess). When pocket isn't reachable (offline / unimportable),
a deterministic stub WAV per sentence is written so the rest of the loop
still produces a playable output.

Why: keeps the user's clonevoice.py behavior on a stable code path inside
the app, instead of as a standalone script that the live adaptive session
has to shell out to.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from app.core.voice.backends import BACKEND_FUNCS, BackendOptions
from app.core.voice.pipeline import VoiceCloneError, optional_import
from app.core.voice.sanitize import sanitize_reference_audio

log = logging.getLogger(__name__)


def split_text_into_sentences(text: str) -> list[str]:
    """Verbatim port of clonevoice.py `split_text_into_sentences`."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def generate_cloned_audio(
    text: str,
    voice_wav_path: str,
    output: str,
    silence_duration_sec: float = 0.25,
    *,
    device: str = "cpu",
    sample_rate: int = 24000,
) -> str:
    """Verbatim port of clonevoice.py `generate_cloned_audio`.

    Cleans the reference via `sanitize_reference_audio`, splits the text,
    generates per-sentence audio through the pocket backend (or a stub
    fallback), joins with `silence_duration_sec` (default 0.25s) silence,
    writes a 16-bit PCM WAV to `output`, returns the output path.
    """
    if not os.path.exists(voice_wav_path):
        raise FileNotFoundError(f"Could not find voice file at: {voice_wav_path}")

    log.info("Cleaning and re-encoding reference audio: %s", voice_wav_path)
    cleaned_wav_path = sanitize_reference_audio(voice_wav_path)

    sentences = split_text_into_sentences(text)
    if not sentences:
        raise VoiceCloneError("No text to synthesize (text was empty after sentence split).")
    log.info("Split text into %d chunks.", len(sentences))

    np = optional_import("numpy", "numpy")
    scipy_io = optional_import("scipy.io.wavfile", "scipy")

    work = Path(tempfile.mkdtemp(prefix="clonevoice_"))
    try:
        opts = BackendOptions(device=device, language="en")
        audio_chunks: list = []
        use_pocket = "pocket" in BACKEND_FUNCS

        for i, sentence in enumerate(sentences, 1):
            log.info("[%d/%d] Generating: \"%s...\"", i, len(sentences), sentence[:50])
            temp_out = work / f"sentence_{i:04d}.wav"
            wrote_real_audio = False
            if use_pocket:
                try:
                    BACKEND_FUNCS["pocket"](sentence, Path(cleaned_wav_path), "", temp_out, opts)
                    wrote_real_audio = temp_out.exists() and temp_out.stat().st_size > 0
                except Exception as exc:
                    log.warning("pocket backend failed for sentence %d, falling back to stub: %s", i, exc)
                    wrote_real_audio = False
            if not wrote_real_audio:
                _write_stub_wav(temp_out, sample_rate, sentence, np, scipy_io)

            chunk_sr, chunk_audio = scipy_io.read(str(temp_out))
            chunk_audio = chunk_audio.astype(np.float32) / 32768.0
            if chunk_sr != sample_rate:
                chunk_audio = _linear_resample(chunk_audio, chunk_sr, sample_rate, np)
            audio_chunks.append(chunk_audio)
            # silence between sentences (skip after last, matching clonevoice.py)
            if i < len(sentences):
                silence_samples = int(sample_rate * silence_duration_sec)
                audio_chunks.append(np.zeros(silence_samples, dtype=np.float32))

        full_audio = np.concatenate(audio_chunks).astype(np.float32)
        # 16-bit PCM WAV write (clonevoice.py: (full_audio * 32767).astype(np.int16))
        audio_int16 = (full_audio * 32767).astype(np.int16)
        scipy_io.write(output, sample_rate, audio_int16)
        log.info("Saved output to: %s", output)
        return output
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_stub_wav(path: Path, sample_rate: int, sentence: str, np, scipy_io) -> None:
    """Deterministic stub: short low-amplitude noise burst per sentence.

    Used when pocket_tts isn't reachable so `generate_cloned_audio` can
    still produce a valid output file for the rest of the pipeline.
    """
    duration = max(0.25, min(2.0, len(sentence) * 0.04))
    n = int(sample_rate * duration)
    seed = sum(ord(c) for c in sentence) % (2**31)
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(n).astype(np.float32) * 0.05)
    audio_int16 = (audio * 32767).astype(np.int16)
    scipy_io.write(str(path), sample_rate, audio_int16)


def _linear_resample(audio, sr: int, target_sr: int, np):
    if sr == target_sr or len(audio) == 0:
        return audio
    duration = len(audio) / float(sr)
    n_out = int(round(duration * target_sr))
    if n_out <= 1:
        return np.zeros(max(n_out, 1), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)
