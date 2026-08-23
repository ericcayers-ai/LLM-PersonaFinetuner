#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "librosa",
#   "soxr>=0.5.0",
#   "noisereduce>=3.0.0",
#   "numpy",
#   "soundfile>=0.12.0",
#   "sounddevice>=0.4.6",
#   "faster-whisper>=1.0.0",
# ]
# ///
"""
voice_clone_mini.py

Multi-backend voice-cloning pipeline:
    SANITIZE -> TTS BACKEND (choose with --backend) -> UPSCALE -> DENOISE (DeepFilterNet3)

Every heavy TTS/upscaling engine runs in its own isolated 'uv run' subprocess (its own
throwaway venv, resolved by `uv` on the fly) so none of their conflicting numpy/torch/
python-version requirements ever touch each other or this launcher's own environment.

    --backend pocket        Pocket-TTS (default; fast, light, decent quality)
    --backend f5tts          F5-TTS (flow-matching DiT, very strong zero-shot cloning)
    --backend chatterbox      Chatterbox Multilingual v3 (Resemble AI, 23 languages)
    --backend chattts        ChatTTS (2noise) - dialogue-oriented, limited ref-cloning
    --backend xtts           Coqui XTTS v2 (17 languages, 6s reference)
    --backend openvoice       OpenVoice v2 tone-color conversion (needs a base TTS pass)
    --backend metavoice       MetaVoice-1B
    --backend qwen3tts        Qwen3-TTS (Alibaba, 10 languages, low latency)
    --backend omnivoice       OmniVoice (k2-fsa, 600+ languages, fast diffusion-LM TTS)
    --backend fishspeech       Fish Speech / Fish Audio S2 (hosted API, needs API key)
    --backend cosyvoice       CosyVoice2 (FunAudioLLM/Alibaba, heavier local setup)
    --backend gptsovits        GPT-SoVITS (RVC-Boss, heavier local setup, HTTP API)

Run `--list-backends` for a one-line summary of every backend and its setup weight, or
`--help` for full flag docs (rendered with argparse; includes a long epilog).

Upscaling (any -> 48kHz) can use:
    --upscale-backend flashsr   FlashSR (default): one-step diffusion, ~22x faster than
                                 AudioSR at comparable quality (arXiv:2501.10807).
    --upscale-backend audiosr    Original AudioSR (slower diffusion, more mature).
    --upscale-backend resample   No generative upscaling, just HQ resample to 48kHz.

Voice input, hands-free (unchanged from before):
    --record-voice              Record your mic as the reference voice to clone.
    --record-command             Record your mic, transcribe with faster-whisper, and use
                                 the transcript as the text to speak.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def die(message: str, code: int = 1) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def optional_import(module: str, package: str | None = None):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pkg = package or module
        raise RuntimeError(f"Missing '{pkg}'. Install it with: pip install {pkg}") from exc


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        die(f"{label} not found: {path}")
    if not path.is_file():
        die(f"{label} is not a file: {path}")


def resolve_device(requested: str) -> str:
    """Check for CUDA without requiring torch to be installed in the main environment."""
    if requested != "auto":
        return requested
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "cuda"
    except Exception:
        return "cpu"


def cache_root() -> Path:
    root = Path.home() / ".cache" / "voice_clone_mini"
    root.mkdir(parents=True, exist_ok=True)
    return root


def repos_dir() -> Path:
    d = cache_root() / "repos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_dir() -> Path:
    d = cache_root() / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_repo(url: str, dirname: str, recursive: bool = False) -> Path:
    """Clone a git repo into the persistent cache dir if it isn't there yet.

    Reused across runs so the (often multi-GB) clone + model download only happens once.
    """
    dest = repos_dir() / dirname
    if dest.exists() and any(dest.iterdir()):
        return dest
    print(f"      [one-time setup] cloning {url} -> {dest}")
    cmd = ["git", "clone", "--depth", "1"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([url, str(dest)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        die(f"git clone failed for {url}:\n{result.stderr}")
    return dest


def run_uv_script(
    *,
    python_version: str,
    packages: list[str],
    script_text: str,
    script_args: list[str],
    device: str,
    cuda_index: str = "https://download.pytorch.org/whl/cu121",
    extra_uv_args: list[str] | None = None,
    workdir: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Write `script_text` to a temp .py and run it via `uv run` with an isolated,
    throwaway venv containing exactly `packages` (+ torch/torchaudio matched to device).
    This is the same isolation trick the original script used for pocket_tts/audiosr/
    deepfilternet, generalized so every new backend can reuse it.
    """
    scratch = Path(tempfile.mkdtemp(prefix="voice_clone_mini_uv_"))
    script_path = scratch / "runner.py"
    script_path.write_text(script_text)

    cmd = ["uv", "run", "--python", python_version]
    for pkg in packages:
        cmd.extend(["--with", pkg])
    if device == "cuda":
        cmd.extend(["--extra-index-url", cuda_index, "--with", "torch", "--with", "torchaudio"])
    else:
        cmd.extend(["--with", "torch", "--with", "torchaudio"])
    if extra_uv_args:
        cmd.extend(extra_uv_args)
    cmd.extend(["python", str(script_path), *script_args])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(workdir) if workdir else None, timeout=timeout,
        )
    finally:
        pass
    return result


def run_uv_requirements(
    *,
    python_version: str,
    requirements_file: Path,
    script_text: str,
    script_args: list[str],
    workdir: Path,
    extra_packages: list[str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Same idea as run_uv_script, but resolves deps from a cloned repo's requirements.txt
    (`uv run --with-requirements`) instead of a hand-picked package list. Used for the
    heavier git-clone backends (CosyVoice, GPT-SoVITS) whose dependency lists are long
    and change from release to release.
    """
    script_path = workdir / "_vcm_runner.py"
    script_path.write_text(script_text)
    cmd = ["uv", "run", "--python", python_version, "--with-requirements", str(requirements_file)]
    for pkg in (extra_packages or []):
        cmd.extend(["--with", pkg])
    cmd.extend(["python", str(script_path), *script_args])
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(workdir), timeout=timeout,
    )
    return result


def load_audio(path: Path, target_sr: int | None = None):
    np = optional_import("numpy", "numpy")
    sf = optional_import("soundfile", "soundfile")
    soxr = optional_import("soxr", "soxr")
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if target_sr and sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr, quality="HQ").astype(np.float32)
        sr = target_sr
    audio = np.nan_to_num(audio).astype(np.float32)
    return audio, int(sr)


def normalize(audio, peak: float = 0.95):
    np = optional_import("numpy", "numpy")
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    audio = np.nan_to_num(audio)
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum > 1e-8:
        audio = audio * (peak / maximum)
    return audio.astype(np.float32)


def save_wav(path: Path, audio, sr: int, subtype: str = "PCM_16") -> None:
    sf = optional_import("soundfile", "soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), normalize(audio, 0.98), int(sr), subtype=subtype)


def split_text(text: str, max_chars: int) -> list[str]:
    import re
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    sentences = [x.strip() for x in sentences if x.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) <= max_chars:
            if current and len(current) + 1 + len(sentence) <= max_chars:
                current += " " + sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence
            continue
        words = sentence.split()
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word
    if current:
        chunks.append(current)
    return chunks


def concat_audio(files: list[Path], out: Path, sr: int = 24000, gap_ms: int = 180) -> None:
    if not files:
        die("No TTS output files were produced.")
    np = optional_import("numpy", "numpy")
    soxr = optional_import("soxr", "soxr")
    pieces = []
    for idx, path in enumerate(files):
        audio, source_sr = load_audio(path)
        if source_sr != sr:
            audio = soxr.resample(audio, source_sr, sr, quality="HQ").astype(np.float32)
        pieces.append(audio)
        if idx < len(files) - 1 and gap_ms:
            pieces.append(np.zeros(int(sr * gap_ms / 1000), dtype=np.float32))
    save_wav(out, np.concatenate(pieces), sr)


# ---------------------------------------------------------------------------
# 0. MIC INPUT: pick a device, record a reference voice, record + transcribe
#    a spoken command
# ---------------------------------------------------------------------------

RECORD_VOICE_MAX_SECONDS = 30.0
RECORD_COMMAND_MAX_SECONDS = 10.0

# Recording streams (esp. WASAPI on Windows) can emit a short burst of
# silence right as they open -- https://github.com/spatialaudio/python-sounddevice/issues/44.
# We start the stream on Enter but discard this warm-up window before we
# start actually keeping frames, so a fast talker's first words aren't lost.
_RECORDING_WARMUP_SECONDS = 0.3


def select_mic_device() -> int | None:
    """List input devices and let the user pick one interactively.

    Returns a device index, or None to use the system default.
    """
    sd = optional_import("sounddevice", "sounddevice")

    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"\n[MIC] Could not list audio devices ({exc}); using system default.")
        return None

    input_devices = [(i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    if not input_devices:
        print("\n[MIC] No input devices found; using system default.")
        return None

    try:
        default_idx = sd.default.device[0]
    except Exception:
        default_idx = None

    print("\n[MIC] Available input devices:")
    for i, d in input_devices:
        marker = "  <- default" if i == default_idx else ""
        print(f"      [{i}] {d['name']}{marker}")

    choice = input("      Type a device number and press Enter (or just Enter for default): ").strip()
    if not choice:
        return None
    try:
        idx = int(choice)
    except ValueError:
        print("      Not a number -- using default.")
        return None
    if idx not in {i for i, _ in input_devices}:
        print("      Not a listed input device -- using default.")
        return None
    return idx


def _wait_for_enter_or_timeout(max_seconds: float) -> bool:
    """Block until Enter is pressed or max_seconds elapses.

    Returns True if Enter was pressed, False if the time limit was hit first.
    """
    start = time.monotonic()
    try:
        import msvcrt  # Windows only
    except ImportError:
        msvcrt = None

    if msvcrt is not None:
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    return True
            if time.monotonic() - start >= max_seconds:
                return False
            time.sleep(0.02)

    # Non-Windows fallback: read Enter on a background thread so we can
    # still poll the time limit concurrently.
    import threading
    stop_event = threading.Event()

    def _reader() -> None:
        try:
            input()
        except EOFError:
            pass
        stop_event.set()

    threading.Thread(target=_reader, daemon=True).start()
    while not stop_event.is_set():
        if time.monotonic() - start >= max_seconds:
            return False
        time.sleep(0.02)
    return True


def record_audio_interactive(max_seconds: float, samplerate: int, device: int | None, label: str):
    """Enter-to-start / Enter-to-stop mic recording, capped at max_seconds.

    Returns (audio: np.ndarray float32 mono, samplerate: int).
    """
    sd = optional_import("sounddevice", "sounddevice")
    np = optional_import("numpy", "numpy")

    input(f"\n[MIC] Press Enter to START recording {label} (max {max_seconds:.0f}s)...")

    frames: list = []
    state = {"armed": False}

    def callback(indata, frame_count, time_info, status):
        if state["armed"]:
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", device=device, callback=callback)
    print("      Recording... press Enter to STOP (auto-stops at the limit).")
    with stream:
        time.sleep(_RECORDING_WARMUP_SECONDS)  # let WASAPI's startup silence pass before keeping frames
        state["armed"] = True
        hit_enter = _wait_for_enter_or_timeout(max_seconds)
        state["armed"] = False

    if hit_enter:
        print("      Stopped.")
    else:
        print(f"      Reached the {max_seconds:.0f}s limit, stopped automatically.")

    if not frames:
        die(f"No audio was captured for {label}. Check your microphone/device selection and try again.")

    audio = np.concatenate(frames, axis=0).astype(np.float32).squeeze()
    duration = len(audio) / samplerate
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"      Captured {duration:.2f}s (peak level: {peak:.3f}).")
    if peak < 1e-4:
        print("      WARNING: recording is silent or near-silent. Wrong input device? "
              "Re-run and pick a different device when prompted.")
    return audio, samplerate


def record_to_wav(out: Path, max_seconds: float, samplerate: int, device: int | None, label: str) -> Path:
    audio, sr = record_audio_interactive(max_seconds, samplerate=samplerate, device=device, label=label)
    save_wav(out, audio, sr)
    return out


def transcribe_command(audio_path: Path, model_name: str, device: str = "cpu") -> str:
    """Transcribe a short recorded command using faster-whisper."""
    fw = optional_import("faster_whisper", "faster-whisper")
    WhisperModel = fw.WhisperModel

    print(f"\n[MIC] Transcribing command with faster-whisper ({model_name})...")
    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    if not text:
        die("faster-whisper produced no transcript from the recorded command. "
            "The recording may have been silent -- check the peak level printed above, "
            "re-run, and pick the correct input device when prompted.")
    print(f"      Heard: \"{text}\"")
    return text


def get_reference_text(reference: Path, args: argparse.Namespace) -> str:
    """Some backends (F5-TTS, GPT-SoVITS, CosyVoice, OmniVoice) want a transcript of the
    *reference* audio, not just the audio itself. If the user passed --ref-text, use it;
    otherwise transcribe the reference clip once with faster-whisper and cache the result
    on args so repeated chunk calls don't re-transcribe.
    """
    if args.ref_text:
        return args.ref_text
    cached = getattr(args, "_ref_text_cache", None)
    if cached is not None:
        return cached
    print("      (no --ref-text given; transcribing reference audio with faster-whisper)")
    text = transcribe_command(reference, args.whisper_model, device=args.whisper_device)
    args._ref_text_cache = text
    return text


# ---------------------------------------------------------------------------
# 1. SANITIZE
# ---------------------------------------------------------------------------

@dataclass
class Sanitizer:
    sample_rate: int = 24000
    trim_db: float = 45.0
    denoise: bool = False
    prop_decrease: float = 0.5
    fade_ms: float = 12.0
    raw_ref: bool = False

    def run(self, source: Path, out: Path) -> Path:
        print("\n[1/4] SANITIZE")
        if self.raw_ref:
            print("      Raw reference mode (bypassing sanitization)")
            shutil.copyfile(source, out)
            return out

        audio, sr = load_audio(source, self.sample_rate)

        if self.denoise and self.prop_decrease > 0:
            try:
                nr = optional_import("noisereduce", "noisereduce")
                print(f"      Gentle noise reduction (prop_decrease={self.prop_decrease})")
                audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=self.prop_decrease).astype("float32")
            except Exception as exc:
                print(f"      noisereduce skipped: {exc}")

        if audio.size and self.trim_db > 0:
            librosa = optional_import("librosa", "librosa")
            trimmed, _ = librosa.effects.trim(audio, top_db=self.trim_db)
            if trimmed.size:
                audio = trimmed.astype("float32")

        fade = int(sr * self.fade_ms / 1000)
        if fade > 0 and len(audio) > fade * 2:
            np = optional_import("numpy", "numpy")
            audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

        save_wav(out, audio, sr)
        print(f"      -> {out}")
        return out


# ---------------------------------------------------------------------------
# 2. TTS BACKENDS
# ---------------------------------------------------------------------------
# Every backend function has signature:
#     backend_fn(text, reference_wav, ref_text, out_path, args) -> Path
# and raises RuntimeError with subprocess stderr attached on failure (caught in run()).
#
# Setup weight, shown in --list-backends / --help epilog:
#   light  = plain pip install, fully automatic, isolated uv venv, no manual steps
#   medium = pip install but hits an external hosted API, or is a tone-converter that
#            needs chaining after a base TTS pass
#   heavy  = clones a git repo into ~/.cache/voice_clone_mini/repos on first use and may
#            need a one-time manual pretrained-model download per the upstream project's
#            own instructions (we print exactly what to do when we detect it's missing)

BACKEND_INFO: dict[str, dict[str, str]] = {
    "pocket":     {"weight": "light",  "desc": "Pocket-TTS (default). Fast, small, good enough for quick clones."},
    "f5tts":      {"weight": "light",  "desc": "F5-TTS. Flow-matching DiT; currently one of the strongest open zero-shot cloners."},
    "chatterbox": {"weight": "light",  "desc": "Chatterbox Multilingual v3 (Resemble AI). 23 languages, MIT licensed."},
    "chattts":    {"weight": "light",  "desc": "ChatTTS (2noise). Dialogue-oriented prosody; weaker arbitrary-reference cloning."},
    "xtts":       {"weight": "light",  "desc": "Coqui XTTS v2. 17 languages, ~6s reference, streaming-capable."},
    "openvoice":  {"weight": "medium", "desc": "OpenVoice v2. Tone-color conversion on top of a base TTS pass (uses --backend as the base via --openvoice-base)."},
    "metavoice":  {"weight": "light",  "desc": "MetaVoice-1B. Expressive single-speaker-style foundation TTS."},
    "qwen3tts":   {"weight": "light",  "desc": "Qwen3-TTS (Alibaba). 10 languages, ~97ms latency, small (0.6B/1.7B)."},
    "omnivoice":  {"weight": "light",  "desc": "OmniVoice (k2-fsa). 600+ languages, fast diffusion-LM TTS, Apache-2.0."},
    "fishspeech": {"weight": "medium", "desc": "Fish Audio S2 hosted API (fish-audio-sdk). Needs FISH_AUDIO_API_KEY; not fully local."},
    "cosyvoice":  {"weight": "heavy",  "desc": "CosyVoice2 (FunAudioLLM). Clones the official repo + downloads pretrained weights on first use."},
    "gptsovits":  {"weight": "heavy",  "desc": "GPT-SoVITS (RVC-Boss). Clones the official repo, runs its HTTP API server locally, needs pretrained models downloaded per upstream docs."},
}


def _fail(name: str, result: subprocess.CompletedProcess) -> None:
    print(result.stdout[-4000:] if result.stdout else "")
    die(f"[{name}] backend subprocess failed (exit {result.returncode}).\n{result.stderr[-4000:]}")


# ---- pocket (unchanged) ----------------------------------------------------

POCKET_SCRIPT = '''
import sys
import soundfile as sf
import numpy as np

def normalize(audio, peak=0.95):
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum > 1e-8:
        audio = audio * (peak / maximum)
    return audio.astype(np.float32)

try:
    import pocket_tts as p
    if hasattr(p, "PocketTTS"):
        tts = p.PocketTTS()
        audio, sr = tts.tts(sys.argv[1], audio_prompt=sys.argv[2])
        audio = normalize(audio)
        sf.write(sys.argv[3], audio, int(sr), subtype="PCM_16")
    elif hasattr(p, "TTSModel"):
        model = p.TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(sys.argv[2])
        wav = model.generate_audio(voice_state, sys.argv[1])
        sr = getattr(model, "sample_rate", 24000)
        audio = normalize(wav)
        sf.write(sys.argv[3], audio, int(sr), subtype="PCM_16")
    else:
        print("Error: pocket_tts module exposes neither PocketTTS nor TTSModel.", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"Error in pocket_tts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_pocket(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["pocket-tts", "soundfile", "numpy>=2"],
        script_text=POCKET_SCRIPT,
        script_args=[text, str(reference), str(out)],
        device=device,
    )
    if result.returncode != 0:
        _fail("pocket", result)
    return out


# ---- f5tts ------------------------------------------------------------------

F5TTS_SCRIPT = '''
import sys, json
import soundfile as sf
import numpy as np

text, ref_audio, ref_text, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def normalize(audio, peak=0.95):
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum > 1e-8:
        audio = audio * (peak / maximum)
    return audio.astype(np.float32)

try:
    # Documented lightweight wrapper shipped in the f5-tts PyPI package.
    from f5_tts.api import F5TTS
    f5 = F5TTS()
    wav, sr, _spect = f5.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=text)
    sf.write(out_path, normalize(wav), int(sr), subtype="PCM_16")
except Exception as e:
    print(f"Error in f5-tts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_f5tts(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["f5-tts", "soundfile", "numpy<2"],
        script_text=F5TTS_SCRIPT,
        script_args=[text, str(reference), ref_text, str(out)],
        device=device,
    )
    if result.returncode != 0:
        _fail("f5tts", result)
    return out


# ---- chatterbox (Multilingual v3) -------------------------------------------

CHATTERBOX_SCRIPT = '''
import sys
import torchaudio as ta
import numpy as np

text, ref_audio, language, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    kwargs = {"audio_prompt_path": ref_audio}
    if language:
        kwargs["language_id"] = language
    wav = model.generate(text, **kwargs)
    ta.save(out_path, wav, model.sr)
except Exception as e:
    print(f"Error in chatterbox subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_chatterbox(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["chatterbox-tts", "soundfile"],
        script_text=CHATTERBOX_SCRIPT,
        script_args=[text, str(reference), args.language or "", str(out)],
        device=device,
    )
    if result.returncode != 0:
        _fail("chatterbox", result)
    return out


# ---- chattts ------------------------------------------------------------------
# NOTE: ChatTTS (2noise) does not support arbitrary-reference-audio cloning out of the
# box the way the others do -- it samples a random Gaussian speaker embedding (or a
# fixed seed for repeatability) rather than conditioning on your ref clip. We still
# offer it because it's explicitly on your list, but --backend chattts will sound like
# *a* natural voice, not necessarily *your* cloned reference voice. Flagged clearly in
# --list-backends and --help.

CHATTTS_SCRIPT = '''
import sys
import torch
import torchaudio
import numpy as np

text, seed, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    import ChatTTS
    chat = ChatTTS.Chat()
    chat.load(compile=False)
    torch.manual_seed(int(seed))
    rand_spk = chat.sample_random_speaker()
    params = ChatTTS.Chat.InferCodeParams(spk_emb=rand_spk)
    wavs = chat.infer([text], params_infer_code=params)
    wav = np.asarray(wavs[0], dtype=np.float32).squeeze()
    torchaudio.save(out_path, torch.from_numpy(wav).unsqueeze(0), 24000)
except Exception as e:
    print(f"Error in chattts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_chattts(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["git+https://github.com/2noise/ChatTTS", "soundfile"],
        script_text=CHATTTS_SCRIPT,
        script_args=[text, str(args.seed), str(out)],
        device=device,
    )
    if result.returncode != 0:
        _fail("chattts", result)
    return out


# ---- xtts (Coqui XTTS v2) ---------------------------------------------------

XTTS_SCRIPT = '''
import sys
text, ref_audio, language, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    import torch
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.tts_to_file(text=text, speaker_wav=ref_audio, language=language or "en", file_path=out_path)
except Exception as e:
    print(f"Error in xtts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_xtts(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["coqui-tts", "soundfile"],
        script_text=XTTS_SCRIPT,
        script_args=[text, str(reference), args.language or "en", str(out)],
        device=device,
        extra_uv_args=["--python-preference", "only-managed"] if False else None,
    )
    if result.returncode != 0:
        _fail("xtts", result)
    return out


# ---- openvoice (tone-color conversion, chained after a base TTS pass) ------

def backend_openvoice(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    """OpenVoice v2 doesn't take text -> speech directly; it converts the *tone color*
    of an already-spoken clip to match a reference speaker. So we first synthesize a
    neutral base clip with another backend (--openvoice-base, default 'pocket'), then
    run `openvoice-cli` to re-color it toward the reference voice.
    """
    base_backend = BACKEND_FUNCS[args.openvoice_base]
    scratch = out.parent / f"_openvoice_base_{out.stem}.wav"
    print(f"      [openvoice] base pass via --backend {args.openvoice_base}")
    base_backend(text, reference, ref_text, scratch, args)

    device = resolve_device(args.device)
    cmd = [
        "uv", "run", "--python", "3.11",
        "--with", "openvoice-cli", "--with", "soundfile",
    ]
    if device == "cuda":
        cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu121", "--with", "torch", "--with", "torchaudio"])
    else:
        cmd.extend(["--with", "torch", "--with", "torchaudio"])
    cmd.extend([
        "python", "-m", "openvoice_cli", "single",
        "-i", str(scratch), "-r", str(reference), "-o", str(out), "-d", device,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not out.exists():
        _fail("openvoice", result)
    return out


# ---- metavoice ---------------------------------------------------------------

METAVOICE_SCRIPT = '''
import sys
text, ref_audio, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from fam.llm.fast_inference import TTS
    tts = TTS()
    wav_path = tts.synthesise(text=text, spk_ref_path=ref_audio)
    import shutil
    shutil.copyfile(wav_path, out_path)
except Exception as e:
    print(f"Error in metavoice subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_metavoice(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.10",
        packages=["metavoice-src", "soundfile"],
        script_text=METAVOICE_SCRIPT,
        script_args=[text, str(reference), str(out)],
        device=device,
    )
    if result.returncode != 0:
        _fail("metavoice", result)
    return out


# ---- qwen3tts -----------------------------------------------------------------

QWEN3TTS_SCRIPT = '''
import sys
import torch
import soundfile as sf
import numpy as np

text, ref_audio, ref_text, language, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

try:
    from qwen_tts import Qwen3TTSModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=device, dtype=dtype)

    wavs = None
    sr = 24000
    errors = []
    # The Base checkpoint's zero-shot cloning entry point name has shifted between
    # releases; try the documented/likely spellings in order rather than hard-failing.
    for method_name, kwargs in [
        ("generate_clone", dict(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language or "English")),
        ("clone", dict(text=text, ref_audio_path=ref_audio, ref_text=ref_text, language=language or "English")),
        ("generate", dict(text=text, ref_audio=ref_audio, language=language or "English")),
    ]:
        fn = getattr(model, method_name, None)
        if fn is None:
            continue
        try:
            wavs, sr = fn(**kwargs)
            break
        except Exception as e:
            errors.append(f"{method_name}: {e}")
            continue
    if wavs is None:
        raise RuntimeError("No working clone/generate method found on Qwen3TTSModel. Tried: " + " | ".join(errors))
    sf.write(out_path, np.asarray(wavs[0], dtype=np.float32), int(sr))
except Exception as e:
    print(f"Error in qwen3tts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_qwen3tts(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["qwen-tts", "soundfile"],
        script_text=QWEN3TTS_SCRIPT,
        script_args=[text, str(reference), ref_text, args.language or "", str(out)],
        device=device,
    )
    if result.returncode != 0:
        _fail("qwen3tts", result)
    return out


# ---- omnivoice ------------------------------------------------------------------

OMNIVOICE_SCRIPT = '''
import sys
import torch
import soundfile as sf
import numpy as np

text, ref_audio, ref_text, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

try:
    from omnivoice import OmniVoice
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    kwargs = {"text": text, "ref_audio": ref_audio}
    if ref_text:
        kwargs["ref_text"] = ref_text
    audio = model.generate(**kwargs)
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    sf.write(out_path, audio, 24000)
except Exception as e:
    print(f"Error in omnivoice subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_omnivoice(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["omnivoice", "soundfile"],
        script_text=OMNIVOICE_SCRIPT,
        script_args=[text, str(reference), ref_text, str(out)],
        device=device,
    )
    if result.returncode != 0:
        _fail("omnivoice", result)
    return out


# ---- fishspeech (hosted Fish Audio API) -----------------------------------------
# There isn't a clean, stable, pip-installable *local* inference entry point for Fish
# Speech that's safe to hardcode here (the local model is normally driven through a
# Gradio/WebUI or a self-hosted API server you launch yourself from a full repo clone).
# What IS a stable, documented, pip-installable interface is Fish Audio's hosted API
# via the official `fish-audio-sdk` client, which supports on-the-fly reference-audio
# cloning. We use that. It requires a FISH_AUDIO_API_KEY (get one at fish.audio) and
# sends your reference clip + text to Fish Audio's servers -- this is the one backend
# in this script that is NOT fully local, which is called out in --list-backends.

FISHSPEECH_SCRIPT = '''
import sys, os
text, ref_audio, ref_text, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
api_key = os.environ.get("FISH_AUDIO_API_KEY")
if not api_key:
    print("Error: FISH_AUDIO_API_KEY is not set. Get one at https://fish.audio and "
          "set the environment variable before running --backend fishspeech.", file=sys.stderr)
    sys.exit(1)
try:
    from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
    session = Session(api_key)
    with open(ref_audio, "rb") as f:
        ref_bytes = f.read()
    references = [ReferenceAudio(audio=ref_bytes, text=ref_text)] if ref_text else [ReferenceAudio(audio=ref_bytes, text="")]
    with open(out_path, "wb") as out_f:
        for chunk in session.tts(TTSRequest(text=text, references=references, format="wav")):
            out_f.write(chunk)
except Exception as e:
    print(f"Error in fishspeech (Fish Audio API) subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_fishspeech(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    result = run_uv_script(
        python_version="3.11",
        packages=["fish-audio-sdk", "soundfile"],
        script_text=FISHSPEECH_SCRIPT,
        script_args=[text, str(reference), ref_text, str(out)],
        device="cpu",  # network call, no local GPU work
    )
    if result.returncode != 0:
        _fail("fishspeech", result)
    return out


# ---- cosyvoice (heavy: repo clone + pretrained weights) ------------------------

COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
COSYVOICE_MODEL_DIR_NAME = "CosyVoice2-0.5B"

COSYVOICE_SCRIPT = '''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "third_party", "Matcha-TTS"))
sys.path.insert(0, os.path.dirname(__file__))

text, ref_audio, ref_text, model_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

try:
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    prompt_speech_16k = load_wav(ref_audio, 16000)
    results = list(cosyvoice.inference_zero_shot(text, ref_text, prompt_speech_16k, stream=False))
    if not results:
        raise RuntimeError("CosyVoice2 returned no audio segments.")
    torchaudio.save(out_path, results[0]["tts_speech"], cosyvoice.sample_rate)
except Exception as e:
    print(f"Error in cosyvoice subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def backend_cosyvoice(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    repo = ensure_repo(COSYVOICE_REPO_URL, "CosyVoice", recursive=True)
    model_dir = repo / "pretrained_models" / COSYVOICE_MODEL_DIR_NAME
    requirements = repo / "requirements.txt"
    if not requirements.exists():
        die(f"CosyVoice clone at {repo} has no requirements.txt; the clone may be incomplete. "
            f"Delete {repo} and re-run to retry the clone.")
    if not model_dir.exists():
        print(textwrap.dedent(f"""
            [cosyvoice] Pretrained weights not found at:
                {model_dir}
            One-time manual step (per the official CosyVoice2 instructions):
                pip install modelscope
                modelscope download --model FunAudioLLM/{COSYVOICE_MODEL_DIR_NAME} --local_dir "{model_dir}"
            or download the same repo from Hugging Face into that folder. Re-run this
            command once the weights are in place.
        """))
        die("cosyvoice pretrained model not installed yet (see instructions above).")

    result = run_uv_requirements(
        python_version="3.10",
        requirements_file=requirements,
        script_text=COSYVOICE_SCRIPT,
        script_args=[text, str(reference), ref_text, str(model_dir), str(out)],
        workdir=repo,
        extra_packages=["soundfile"],
        timeout=1800,
    )
    if result.returncode != 0:
        _fail("cosyvoice", result)
    return out


# ---- gptsovits (heavy: repo clone + local HTTP API server) --------------------

GPTSOVITS_REPO_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"
GPTSOVITS_PORT = 9880

def backend_gptsovits(text: str, reference: Path, ref_text: str, out: Path, args: argparse.Namespace) -> Path:
    import urllib.request
    import urllib.parse

    repo = ensure_repo(GPTSOVITS_REPO_URL, "GPT-SoVITS")
    requirements = repo / "requirements.txt"
    pretrained = repo / "GPT_SoVITS" / "pretrained_models"
    if not requirements.exists():
        die(f"GPT-SoVITS clone at {repo} has no requirements.txt; delete {repo} and retry.")
    if not pretrained.exists() or not any(pretrained.iterdir()):
        print(textwrap.dedent(f"""
            [gptsovits] Pretrained models not found under:
                {pretrained}
            One-time manual step (per the official GPT-SoVITS README):
                cd "{repo}"
                bash install.sh          # Linux/macOS/WSL, downloads pretrained_models too
            (Windows: use the integrated package linked from the repo's README, or run
            install.sh under WSL.) Re-run this command once pretrained_models is populated.
        """))
        die("gpt-sovits pretrained models not installed yet (see instructions above).")

    api_script = repo / "api_v2.py"
    if not api_script.exists():
        die(f"Expected {api_script} from the GPT-SoVITS clone but it's missing.")

    device = resolve_device(args.device)
    print("      [gptsovits] starting local api_v2.py server (first request will be slow: model load)")
    proc = subprocess.Popen(
        ["uv", "run", "--python", "3.10", "--with-requirements", str(requirements),
         "python", str(api_script), "-a", "127.0.0.1", "-p", str(GPTSOVITS_PORT)],
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        base_url = f"http://127.0.0.1:{GPTSOVITS_PORT}"
        ready = False
        for _ in range(180):  # up to ~3 minutes for model load on first run
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"{base_url}/", timeout=1)
                ready = True
                break
            except Exception:
                time.sleep(1)
        if not ready:
            leftover = proc.stdout.read() if proc.stdout else ""
            die(f"gpt-sovits api_v2.py server did not become ready.\n{leftover[-4000:]}")

        payload = {
            "text": text,
            "text_lang": args.language or "en",
            "ref_audio_path": str(reference),
            "prompt_text": ref_text,
            "prompt_lang": args.language or "en",
            "media_type": "wav",
        }
        req = urllib.request.Request(
            f"{base_url}/tts",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out.write_bytes(resp.read())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    if not out.exists() or out.stat().st_size == 0:
        die("gpt-sovits API call produced no audio.")
    return out


BACKEND_FUNCS = {
    "pocket": backend_pocket,
    "f5tts": backend_f5tts,
    "chatterbox": backend_chatterbox,
    "chattts": backend_chattts,
    "xtts": backend_xtts,
    "openvoice": backend_openvoice,
    "metavoice": backend_metavoice,
    "qwen3tts": backend_qwen3tts,
    "omnivoice": backend_omnivoice,
    "fishspeech": backend_fishspeech,
    "cosyvoice": backend_cosyvoice,
    "gptsovits": backend_gptsovits,
}

BACKENDS_NEEDING_REF_TEXT = {"f5tts", "cosyvoice", "gptsovits", "omnivoice"}


def synthesize_chunk(backend: str, text: str, reference: Path, out: Path, args: argparse.Namespace) -> Path:
    ref_text = ""
    if backend in BACKENDS_NEEDING_REF_TEXT:
        ref_text = get_reference_text(reference, args)
    fn = BACKEND_FUNCS[backend]
    fn(text, reference, ref_text, out, args)
    if not out.exists():
        die(f"[{backend}] did not create {out.name}")
    return out


# ---------------------------------------------------------------------------
# 3. UPSCALE: FlashSR (default) / AudioSR / plain resample
# ---------------------------------------------------------------------------

FLASHSR_REPO_URL = "https://github.com/jakeoneijk/FlashSR_Inference.git"
FLASHSR_WEIGHTS_REPO = "jakeoneijk/FlashSR_weights"  # HF dataset
FLASHSR_CHUNK_SAMPLES = 245760  # fixed 5.12s @ 48kHz the released model expects


def _download_flashsr_weights(target_dir: Path) -> None:
    print("      [flashsr] downloading weights (one-time, ~cached under ~/.cache/voice_clone_mini)")
    script = f'''
from huggingface_hub import snapshot_download
snapshot_download(repo_id="{FLASHSR_WEIGHTS_REPO}", repo_type="dataset", local_dir=r"{target_dir}")
'''
    result = run_uv_script(
        python_version="3.11", packages=["huggingface_hub"],
        script_text=script, script_args=[], device="cpu",
    )
    if result.returncode != 0:
        die(f"Failed to download FlashSR weights automatically.\n{result.stderr[-2000:]}\n"
            f"You can download them manually from https://huggingface.co/datasets/{FLASHSR_WEIGHTS_REPO} "
            f"into {target_dir}, or fall back to --upscale-backend audiosr / resample.")


FLASHSR_SCRIPT = '''
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import soundfile as sf
import numpy as np
import soxr

source, out_path, weights_dir, device = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CHUNK = 245760

from FlashSR.FlashSR import FlashSR

flashsr = FlashSR(
    os.path.join(weights_dir, "student_ldm.pth"),
    os.path.join(weights_dir, "sr_vocoder.pth"),
    os.path.join(weights_dir, "vae.pth"),
)
flashsr = flashsr.to(device) if hasattr(flashsr, "to") else flashsr

audio, sr = sf.read(source, dtype="float32", always_2d=False)
audio = np.asarray(audio, dtype=np.float32)
if audio.ndim == 2:
    audio = audio.mean(axis=1)
if sr != 48000:
    audio = soxr.resample(audio, sr, 48000, quality="HQ").astype(np.float32)

n = len(audio)
n_chunks = max(1, (n + CHUNK - 1) // CHUNK)
padded_len = n_chunks * CHUNK
padded = np.zeros(padded_len, dtype=np.float32)
padded[:n] = audio

out_chunks = []
for i in range(n_chunks):
    seg = padded[i * CHUNK:(i + 1) * CHUNK]
    seg_t = torch.from_numpy(seg).unsqueeze(0).to(device)  # [1, CHUNK] mono-as-1-channel
    with torch.no_grad():
        sr_out = flashsr(seg_t, lowpass_input=False) if False else flashsr.forward(seg_t)
    sr_out = sr_out.detach().cpu().numpy().squeeze()
    out_chunks.append(sr_out.astype(np.float32))

result = np.concatenate(out_chunks)[:n]
maximum = float(np.max(np.abs(result))) if result.size else 0.0
if maximum > 1e-8:
    result = result * (0.95 / maximum)
sf.write(out_path, result.astype(np.float32), 48000, subtype="PCM_16")
'''

def flashsr_upscale(source: Path, out: Path, args: argparse.Namespace) -> Path:
    repo = ensure_repo(FLASHSR_REPO_URL, "FlashSR_Inference")
    w_dir = weights_dir() / "flashsr"
    w_dir.mkdir(parents=True, exist_ok=True)
    needed = ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"]
    if not all((w_dir / f).exists() for f in needed):
        _download_flashsr_weights(w_dir)
    if not all((w_dir / f).exists() for f in needed):
        raise RuntimeError("FlashSR weights still missing after download attempt.")

    device = args.upscale_device or resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["torch", "torchaudio", "soundfile", "soxr", "numpy<2"],
        script_text=FLASHSR_SCRIPT,
        script_args=[str(source), str(out), str(w_dir), device],
        device=device,
        workdir=repo,
    )
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(result.stderr[-4000:])
    return out


AUDIOSR_SCRIPT_TEMPLATE = '''
import sys
import soundfile as sf
import numpy as np

def normalize(audio, peak=0.95):
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum > 1e-8:
        audio = audio * (peak / maximum)
    return audio.astype(np.float32)

try:
    from audiosr import build_model, super_resolution
    model = build_model(model_name="{model_name}", device="{device}")
    result = super_resolution(
        model, str(sys.argv[1]),
        seed={seed},
        guidance_scale={guidance},
        ddim_steps={steps},
    )
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    elif hasattr(result, "numpy"):
        result = result.numpy()
    if isinstance(result, (list, tuple)):
        result = result[0] if result else np.zeros(0, dtype=np.float32)

    audio = normalize(result)
    sf.write(sys.argv[2], audio, 48000, subtype="PCM_16")
except Exception as e:
    print(f"Error in audiosr subprocess: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

def audiosr_upscale(source: Path, out: Path, args: argparse.Namespace) -> Path:
    device = args.upscale_device or resolve_device(args.device)
    script = AUDIOSR_SCRIPT_TEMPLATE.format(
        model_name=args.audiosr_model, device=device,
        seed=args.seed, guidance=args.guidance, steps=args.steps,
    )
    result = run_uv_script(
        python_version="3.10",
        packages=["audiosr", "matplotlib", "setuptools", "soundfile"],
        script_text=script,
        script_args=[str(source), str(out)],
        device=device,
    )
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(result.stderr[-4000:])
    return out


def upscale_audio(source: Path, out: Path, args: argparse.Namespace) -> Path:
    print("\n[3/4] UPSCALE")
    backend = args.upscale_backend

    if backend == "resample":
        audio, sr = load_audio(source, 48000)
        save_wav(out, audio, sr)
        print("      HQ resample -> 48 kHz (no generative upscaling)")
        return out

    if backend == "flashsr":
        try:
            flashsr_upscale(source, out, args)
            print("      FlashSR -> 48 kHz (one-step diffusion, ~22x faster than AudioSR)")
            return out
        except Exception as exc:
            print(f"      FlashSR unavailable ({exc}); falling back to AudioSR.")
            backend = "audiosr"

    if backend == "audiosr":
        try:
            audiosr_upscale(source, out, args)
            print("      AudioSR -> 48 kHz")
            return out
        except Exception as exc:
            print(f"      AudioSR unavailable ({exc}); falling back to HQ resample.")

    audio, sr = load_audio(source, 48000)
    save_wav(out, audio, sr)
    print("      HQ resample -> 48 kHz (upscale backends unavailable)")
    return out


# ---------------------------------------------------------------------------
# 4. DENOISE (DeepFilterNet3) (Isolated Subprocess) -- unchanged
# ---------------------------------------------------------------------------

DF_SCRIPT = '''
import sys
import types

# DeepFilterNet's bundled loss code still does `from torch._six import
# string_classes`, but torch._six was removed in PyTorch 2.0+. All that was
# ever used from it here is `string_classes = str`, so shim a fake module in
# before df.enhance imports it -- this avoids pinning an old, conflicting
# torch version just to satisfy one dead import.
_torch_six = types.ModuleType("torch._six")
_torch_six.string_classes = str
_torch_six.inf = float("inf")
sys.modules["torch._six"] = _torch_six

import torch
import soundfile as sf
import numpy as np
import torchaudio

# Monkeypatch torchaudio.load to use soundfile internally, avoiding backend errors
def custom_load(filepath, sr=None):
    audio, orig_sr = sf.read(filepath, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio_tensor = torch.from_numpy(audio)
    if sr is not None and sr != orig_sr:
        import soxr
        resampled = soxr.resample(audio, orig_sr, sr, quality="HQ")
        audio_tensor = torch.from_numpy(resampled).float()
    return audio_tensor, sr or orig_sr

torchaudio.load = custom_load

try:
    from df.enhance import enhance, init_df, save_audio
    model, state, _ = init_df()
    wav, _ = custom_load(sys.argv[1], sr=state.sr())
    enhanced = enhance(model, state, wav)
    save_audio(sys.argv[2], enhanced, state.sr())
except Exception as e:
    print(f"Error in deepfilternet subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''

def denoise_audio(source: Path, out: Path, args: argparse.Namespace) -> Path:
    print("\n[4/4] DENOISE")
    if args.skip_denoise:
        shutil.copyfile(source, out)
        print("      DeepFilterNet3 disabled")
        return out

    device = resolve_device(args.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["deepfilternet", "soundfile", "soxr"],
        script_text=DF_SCRIPT,
        script_args=[str(source), str(out)],
        device=device,
    )

    if result.returncode != 0 or not out.exists():
        print("      DeepFilterNet3 unavailable. Keeping upscaled audio.")
        if result.stderr:
            print(result.stderr)
        shutil.copyfile(source, out)
        return out

    print("      DeepFilterNet3 -> final")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BACKEND_HELP_EPILOG = "Backends (setup weight in brackets):\n" + "\n".join(
    f"  {name:<11} [{info['weight']:<6}] {info['desc']}" for name, info in BACKEND_INFO.items()
) + (
    "\n\nSetup weight legend:\n"
    "  light  - plain pip install in an isolated throwaway uv venv, fully automatic\n"
    "  medium - hits a hosted API (fishspeech) or chains onto a base TTS pass (openvoice)\n"
    "  heavy  - clones the official repo into ~/.cache/voice_clone_mini/repos on first "
    "use; may need a one-time manual pretrained-model download (instructions are "
    "printed automatically if weights are missing)\n\n"
    "Upscale backends: flashsr (default, fast one-step diffusion) > audiosr (slower, "
    "original diffusion model) > resample (no generative upscaling).\n\n"
    "Examples:\n"
    "  uv run voice_clone_mini.py --backend f5tts -v ref.wav -t \"Hello there.\" -o out.wav\n"
    "  uv run voice_clone_mini.py --backend xtts --language es -v ref.wav -t \"Hola.\" -o out.wav\n"
    "  uv run voice_clone_mini.py --list-backends\n"
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-backend voice-clone pipeline: sanitize -> TTS backend -> upscale -> DeepFilterNet3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=BACKEND_HELP_EPILOG,
    )
    p.add_argument("--voice", "-v", type=Path, help="Reference audio file (omit if using --record-voice)")
    p.add_argument("--text", "-t", help="Text to speak")
    p.add_argument("--text-file", "-f", type=Path, help="File with text to speak")
    p.add_argument("--out", "-o", type=Path, default=Path("voice_final.wav"), help="Output WAV file")

    p.add_argument("--backend", choices=list(BACKEND_FUNCS.keys()), default="pocket",
                    help="TTS backend to use. See the epilog below (--help) or --list-backends.")
    p.add_argument("--list-backends", action="store_true",
                    help="Print a summary of every backend and its setup weight, then exit.")
    p.add_argument("--openvoice-base", choices=[k for k in BACKEND_FUNCS if k != "openvoice"], default="pocket",
                    help="Base TTS backend used before OpenVoice's tone-color conversion (only used with --backend openvoice)")
    p.add_argument("--ref-text", default="",
                    help="Transcript of the reference audio. Needed by f5tts/cosyvoice/gptsovits/omnivoice; "
                         "auto-transcribed with faster-whisper if omitted.")
    p.add_argument("--language", default="",
                    help="Language code/name for multilingual backends (chatterbox, xtts, qwen3tts, gptsovits). "
                         "Backend-specific format, e.g. 'fr'/'zh' for chatterbox & xtts, 'French' for qwen3tts.")
    p.add_argument("--speaker", default="", help="Named speaker id (only used by backends that support it)")

    # Hands-free mic input
    p.add_argument("--record-voice", action="store_true",
                    help=f"Interactively record the reference voice (Enter to start, Enter to stop, "
                         f"max {RECORD_VOICE_MAX_SECONDS:.0f}s), instead of --voice")
    p.add_argument("--record-command", action="store_true",
                    help=f"Interactively record what to say (Enter to start, Enter to stop, "
                         f"max {RECORD_COMMAND_MAX_SECONDS:.0f}s) and transcribe it with faster-whisper, "
                         f"instead of --text")
    p.add_argument("--whisper-model", default="base.en",
                    help="faster-whisper model size/name for --record-command / auto --ref-text "
                         "(e.g. tiny.en, base.en, small.en, medium.en, large-v3)")
    p.add_argument("--whisper-device", choices=["cpu", "cuda"], default="cpu", help="Device for faster-whisper")
    p.add_argument("--mic-device", type=int, default=None,
                    help="Input device index for recording. If omitted and recording is used, "
                         "you'll be prompted to pick one interactively.")

    p.add_argument("--max-chars", type=int, default=300, help="Max characters per TTS chunk")
    p.add_argument("--gap-ms", type=int, default=180, help="Gap between chunks in ms")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")

    # Sanitization
    p.add_argument("--sanitize-sr", type=int, default=24000)
    p.add_argument("--trim-db", type=float, default=45.0)
    p.add_argument("--sanitize-denoise", action="store_true", help="Apply spectral noise reduction to reference")
    p.add_argument("--prop-decrease", type=float, default=0.5)
    p.add_argument("--raw-ref", action="store_true", help="Use original reference audio directly")
    p.add_argument("--fade-ms", type=float, default=12.0)

    # Upscaling
    p.add_argument("--upscale-backend", choices=["flashsr", "audiosr", "resample"], default="flashsr",
                    help="Upscaling engine. flashsr is the fast default; audiosr is the older/slower "
                         "diffusion model; resample disables generative upscaling entirely.")
    p.add_argument("--skip-upscale", action="store_true", help="Shorthand for --upscale-backend resample")
    p.add_argument("--upscale-device", default=None)
    p.add_argument("--audiosr-model", default="basic", help="Only used with --upscale-backend audiosr")
    p.add_argument("--steps", type=int, default=50, help="Only used with --upscale-backend audiosr")
    p.add_argument("--guidance", type=float, default=3.5, help="Only used with --upscale-backend audiosr")
    p.add_argument("--seed", type=int, default=42, help="Used by --upscale-backend audiosr and --backend chattts")

    # Denoising
    p.add_argument("--skip-denoise", action="store_true", help="Disable DeepFilterNet3")

    # Utility
    p.add_argument("--clean-only", action="store_true", help="Only run sanitization, output reference")
    p.add_argument("--keep-temp", action="store_true", help="Keep intermediate WAVs")
    p.add_argument("--bit-depth", choices=[16, 24], type=int, default=16)

    return p


def print_backend_list() -> None:
    print("Available --backend values:\n")
    for name, info in BACKEND_INFO.items():
        print(f"  {name:<11} [{info['weight']:<6}] {info['desc']}")
    print("\nUpscale backends: flashsr (default) / audiosr / resample. See --help for full details.")


def read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        require_file(args.text_file, "Text file")
        return args.text_file.read_text(encoding="utf-8").strip()
    if args.text:
        return args.text.strip()
    die("Provide --text or --text-file.")


def run(args: argparse.Namespace) -> None:
    if args.skip_upscale:
        args.upscale_backend = "resample"

    work = Path(tempfile.mkdtemp(prefix="voice_clone_mini_"))

    # 0. Mic input (optional): pick a device once, then record the reference
    # voice and/or the spoken command.
    if (args.record_voice or args.record_command) and args.mic_device is None:
        args.mic_device = select_mic_device()

    if args.record_voice:
        recorded_voice = work / "recorded_voice.wav"
        record_to_wav(
            recorded_voice, RECORD_VOICE_MAX_SECONDS,
            samplerate=args.sanitize_sr, device=args.mic_device, label="the reference voice",
        )
        args.voice = recorded_voice
    elif not args.voice:
        die("Provide --voice <file> or --record-voice to supply a reference voice.")
    require_file(args.voice, "Reference audio")

    if args.record_command:
        recorded_command = work / "recorded_command.wav"
        record_to_wav(
            recorded_command, RECORD_COMMAND_MAX_SECONDS,
            samplerate=16000, device=args.mic_device, label="what to say",
        )
        args.text = transcribe_command(recorded_command, args.whisper_model, device=args.whisper_device)
    elif not args.text and not args.text_file:
        die("Provide --text, --text-file, or --record-command to supply what to say.")

    out_dir = args.out.parent if args.out.parent != Path("") else Path(".")
    stem = args.out.stem or "voice_final"
    raw_dir = out_dir / "raw_clone"
    upscaled_dir = out_dir / "upscaled_clone"
    denoised_dir = out_dir / "denoised_clone"

    try:
        # 1. Sanitize
        reference = work / "reference_sanitized.wav"
        Sanitizer(
            sample_rate=args.sanitize_sr,
            trim_db=args.trim_db,
            denoise=args.sanitize_denoise,
            prop_decrease=args.prop_decrease,
            fade_ms=args.fade_ms,
            raw_ref=args.raw_ref,
        ).run(args.voice, reference)

        if args.clean_only:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reference, args.out)
            print(f"\nSanitized reference saved to {args.out.resolve()}")
            return

        # 2. TTS backend
        print(f"\n[2/4] TTS BACKEND: {args.backend}")
        text = read_text(args)
        chunks = split_text(text, args.max_chars)

        raw_outputs = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"      [{idx}/{len(chunks)}] {chunk}")
            chunk_out = work / f"tts_{idx:04d}.wav"
            synthesize_chunk(args.backend, chunk, reference, chunk_out, args)
            raw_outputs.append(chunk_out)

        cloned = work / "cloned.wav"
        concat_audio(raw_outputs, cloned, sr=24000, gap_ms=max(0, args.gap_ms))
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cloned, raw_dir / f"{stem}.wav")

        # 3. Upscale
        upscaled = work / "upscaled.wav"
        upscale_audio(cloned, upscaled, args)
        upscaled_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upscaled, upscaled_dir / f"{stem}.wav")

        # 4. Denoise
        final = work / "final.wav"
        denoise_audio(upscaled, final, args)
        denoised_dir.mkdir(parents=True, exist_ok=True)

        # Output
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.bit_depth == 24:
            audio, sr = load_audio(final)
            save_wav(args.out, audio, sr, "PCM_24")
        else:
            shutil.copyfile(final, args.out)
        shutil.copyfile(args.out, denoised_dir / f"{stem}.wav")

        sf = optional_import("soundfile", "soundfile")
        info = sf.info(str(args.out))
        print("\n" + "=" * 72)
        print("COMPLETE")
        print("=" * 72)
        print(f"Backend:     {args.backend}  (upscale: {args.upscale_backend})")
        print(f"Output:      {args.out.resolve()}")
        print(f"  raw_clone/{stem}.wav      -> pre-upscale, pre-denoise")
        print(f"  upscaled_clone/{stem}.wav -> after upscaling / HQ resample")
        print(f"  denoised_clone/{stem}.wav -> final (same as Output above)")
        print(f"Sample rate: {info.samplerate} Hz")
        print(f"Channels:    {info.channels}")
        print(f"Duration:    {info.duration:.2f} sec")
        print("Pipeline:    SANITIZE -> TTS BACKEND -> UPSCALE -> DENOISE")
        print("=" * 72)

    finally:
        if not args.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"\nIntermediate files kept in {work}")


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.list_backends:
        print_backend_list()
        raise SystemExit(0)
    run(parsed)