#!/usr/bin/env python3
#
# NOTE ON "--all-backends":
# The packages below cover every backend that is a real pip-installable
# Python library (chatterbox, xtts) plus f5tts's CLI and
# the two optional post-processing stages (audiosr, deepfilternet). qwen3
# is deliberately left out: there is no public "qwen-tts" PyPI package as
# of this writing, so that backend only works if you have Qwen3-TTS support
# in a bleeding-edge transformers build (the code already tries that path
# automatically). 
#
# parler-tts is left out because it strictly pins transformers==4.46.1,
# which conflicts with chatterbox-tts.
#
# pocket-tts is left out because it requires numpy>=2, which conflicts
# with chatterbox-tts (requires numpy<2).
#
# The script will automatically skip parler and pocket backends since
# their modules won't be installed.
#
# fish and cosyvoice are NOT pip packages -- upstream ships them as local
# inference servers you run yourself, then point this script at with
# --fish-url / --cosyvoice-url. kokoclone and voicecraft are NOT pip
# packages either -- they're CLI repos you clone yourself and point at with
# --kokoclone-command / --voicecraft-command.
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "chatterbox-tts>=0.1.7",
#   "resemble-perth",
#   "setuptools<70.0.0",
#   "torch",
#   "torchaudio",
#   "librosa>=0.10.0",
#   "soxr>=0.5.0",
#   "noisereduce>=3.0.0",
#   "numpy>=1.26.0",
#   "soundfile>=0.12.0",
#   "transformers>=4.45.0",
#   "coqui-tts",
#   "f5-tts",
#   "datasets>=2.19.0",    # <--- ADD THIS
#   "pyarrow>=15.0.0",     # <--- ADD THIS
#   "deepfilternet",
# ]
# ///
"""
voice_clone_10.py

10-backend local voice/TTS pipeline:

    SANITIZE -> CLONE/TTS -> UPSCALE -> DENOISE

Backends (each adapter uses its real inference interface, not a generic
placeholder):

    1. chatterbox   - Chatterbox TTS  (zero-shot reference-audio cloning)
    2. qwen3        - Qwen3-TTS       (base voice cloning)
    3. pocket       - Pocket TTS      (audio-prompt voice cloning)
    4. kokoclone    - KokoClone       (Kokoro + zero-shot voice cloning, CLI)
    5. xtts         - XTTS v2         (Coqui multilingual voice cloning)
    6. fish         - Fish Speech     (local HTTP API)
    7. f5tts        - F5-TTS          (reference-audio flow-matching TTS, CLI)
    8. cosyvoice    - CosyVoice       (zero-shot local HTTP API)
    9. parler       - Parler-TTS      (prompt/description-controlled TTS)
   10. voicecraft   - VoiceCraft      (local CLI adapter, template-driven)

Important:
- Use only voices/audio you have permission to clone.
- Native adapters are used where the project exposes a usable Python API.
- Server/CLI adapters are used where the project is normally deployed as a
  local service or command-line application.
- Optional heavy stages fail over safely to high-quality resampling/copying.
- Heavy third-party dependencies (numpy, soundfile, soxr, librosa, torch,
  transformers, etc.) are imported lazily inside the functions that need them,
  so `--list-backends` and `--help` work in any Python environment.

Tested:
- Syntax validated with ast.parse().
- CLI smoke-tested: --list-backends, --help, --clean-only end-to-end.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  (type-only)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

BACKENDS: dict[str, str] = {
    "chatterbox": "Chatterbox TTS - zero-shot reference-audio cloning",
    "qwen3":      "Qwen3-TTS - Base voice cloning",
    "pocket":     "Pocket TTS - audio-prompt voice cloning",
    "kokoclone":  "KokoClone - Kokoro + zero-shot voice cloning (CLI)",
    "xtts":       "XTTS v2 - Coqui multilingual voice cloning",
    "fish":       "Fish Speech - local HTTP API",
    "f5tts":      "F5-TTS - reference-audio flow-matching TTS (CLI)",
    "cosyvoice":  "CosyVoice - zero-shot local HTTP API",
    "parler":     "Parler-TTS - prompt/description TTS",
    "voicecraft": "VoiceCraft - local CLI adapter (template-driven)",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def die(message: str, code: int = 1) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def optional_import(module: str, package: str | None = None):
    """Import a module lazily. Raise a helpful RuntimeError if missing."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pkg = package or module
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Install it in this environment "
            f"(e.g. `pip install {pkg}`)."
        ) from exc


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        die(f"{label} not found: {path}")
    if not path.is_file():
        die(f"{label} is not a file: {path}")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        torch = optional_import("torch", "torch")
    except RuntimeError:
        # No torch -> can't run on GPU anyway.
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio(path: Path, target_sr: int | None = None):
    """Load an audio file as mono float32 numpy array. Returns (audio, sr)."""
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
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    audio = np.nan_to_num(audio)
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0

    if maximum > 1e-8:
        audio = audio * (peak / maximum)

    return audio.astype(np.float32)


def save_wav(path: Path, audio, sr: int, subtype: str = "PCM_16") -> None:
    sf = optional_import("soundfile", "soundfile")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), normalize(audio, 0.98), int(sr), subtype=subtype)


def result_to_audio(result: Any):
    """Coerce a backend's audio result into a 1-D float32 numpy array."""
    np = optional_import("numpy", "numpy")

    # Tensor-like (torch) without importing torch at module top.
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    elif hasattr(result, "numpy"):  # tf.Tensor or similar
        result = result.numpy()

    if isinstance(result, (list, tuple)):
        result = result[0] if result else np.zeros(0, dtype=np.float32)

    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()

    return normalize(np.asarray(result, dtype=np.float32).squeeze())


def split_text(text: str, max_chars: int) -> list[str]:
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

        # Split an overlong sentence at whitespace.
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


def concat_audio(files: list[Path], out: Path, sr: int = 24000,
                 gap_ms: int = 180) -> None:
    if not files:
        die("No backend output files were produced.")

    np = optional_import("numpy", "numpy")

    pieces: list = []

    for index, path in enumerate(files):
        audio, source_sr = load_audio(path)
        if source_sr != sr:
            soxr = optional_import("soxr", "soxr")
            audio = soxr.resample(
                audio, source_sr, sr, quality="HQ"
            ).astype(np.float32)

        pieces.append(audio)

        if index < len(files) - 1 and gap_ms:
            pieces.append(np.zeros(
                int(sr * gap_ms / 1000),
                dtype=np.float32,
            ))

    save_wav(out, np.concatenate(pieces), sr)


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
            print("      Raw reference mode (bypassing spectral noise reduction & aggressive trim)")
            shutil.copyfile(source, out)
            print(f"      -> {out}")
            return out

        audio, sr = load_audio(source, self.sample_rate)

        if self.denoise and self.prop_decrease > 0:
            try:
                nr = optional_import("noisereduce", "noisereduce")
                print(f"      Gentle spectral noise reduction (prop_decrease={self.prop_decrease})")
                audio = nr.reduce_noise(
                    y=audio,
                    sr=sr,
                    stationary=False,
                    prop_decrease=self.prop_decrease,
                ).astype("float32")
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
# 2. UPSCALE
# ---------------------------------------------------------------------------

def upscale_audio(source: Path, out: Path, args: argparse.Namespace) -> Path:
    print("\n[3/4] UPSCALE")

    if args.skip_upscale:
        audio, sr = load_audio(source, 48000)
        save_wav(out, audio, sr)
        print("      AudioSR disabled; HQ resample -> 48 kHz")
        return out

    try:
        optional_import("audiosr", "audiosr")
        from audiosr import build_model, super_resolution

        device = args.upscale_device or resolve_device(args.device)
        model = build_model(
            model_name=args.audiosr_model,
            device=device,
        )

        result = super_resolution(
            model,
            str(source),
            seed=args.seed,
            guidance_scale=args.guidance,
            ddim_steps=args.steps,
        )

        audio = result_to_audio(result)
        save_wav(out, audio, 48000)
        print("      AudioSR -> 48 kHz")
        return out

    except Exception as exc:
        print(f"      AudioSR unavailable: {exc}")
        print("      Falling back to HQ resampling -> 48 kHz")

        audio, sr = load_audio(source, 48000)
        save_wav(out, audio, sr)
        return out


# ---------------------------------------------------------------------------
# 3. DENOISE
# ---------------------------------------------------------------------------

def denoise_audio(source: Path, out: Path, args: argparse.Namespace) -> Path:
    print("\n[4/4] DENOISE")

    if args.skip_denoise:
        shutil.copyfile(source, out)
        print("      DeepFilterNet3 disabled")
        return out

    try:
        optional_import("df", "deepfilternet")
        from df.enhance import enhance, init_df, load_audio as df_load, save_audio

        model, state, _ = init_df()
        wav, _ = df_load(str(source), sr=state.sr())
        enhanced = enhance(model, state, wav)
        save_audio(str(out), enhanced, state.sr())

        print("      DeepFilterNet3 -> final")
        return out

    except Exception as exc:
        print(f"      DeepFilterNet3 unavailable: {exc}")
        print("      Keeping upscaled audio")
        shutil.copyfile(source, out)
        return out


# ---------------------------------------------------------------------------
# Backend base
# ---------------------------------------------------------------------------

class Backend:
    name = ""

    def generate(
        self,
        text: str,
        reference: Path,
        out: Path,
        args: argparse.Namespace,
    ) -> Path:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. CHATTERBOX
# ---------------------------------------------------------------------------

class ChatterboxBackend(Backend):
    name = "chatterbox"

    def generate(self, text, reference, out, args):
        mod = optional_import("chatterbox.tts", "chatterbox-tts")
        ChatterboxTTS = mod.ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(
            device=resolve_device(args.device)
        )

        wav = model.generate(
            text,
            audio_prompt_path=str(reference),
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )

        save_wav(out, result_to_audio(wav), model.sr)
        return out


# ---------------------------------------------------------------------------
# 2. QWEN3-TTS
# ---------------------------------------------------------------------------

class Qwen3Backend(Backend):
    name = "qwen3"

    def generate(self, text, reference, out, args):
        """
        Qwen3-TTS exposes a Python inference helper in some distributions
        (package `qwen_tts`) and uses transformers under the hood in others.
        We try the high-level helper first, then fall back to the official
        transformers entrypoint.
        """
        torch = optional_import("torch", "torch")
        device = resolve_device(args.device)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        Qwen3TTSModel = None
        try:
            mod = optional_import("qwen_tts", "qwen-tts")
            Qwen3TTSModel = getattr(mod, "Qwen3TTSModel", None)
            if Qwen3TTSModel is None:
                inf = optional_import("qwen_tts.inference", "qwen-tts")
                Qwen3TTSModel = inf.Qwen3TTSModel
        except RuntimeError:
            # Fall back to the Hugging Face transformers entrypoint.
            transformers = optional_import("transformers", "transformers")
            AutoProcessor = transformers.AutoProcessor
            try:
                Qwen3TTSModel = transformers.Qwen3TTSAudioModel
            except AttributeError as exc:
                raise RuntimeError(
                    "Qwen3-TTS requires either the `qwen-tts` package or a "
                    "recent `transformers` build with Qwen3TTSAudioModel."
                ) from exc

            processor = AutoProcessor.from_pretrained(args.qwen_model)
            model = Qwen3TTSModel.from_pretrained(
                args.qwen_model,
                torch_dtype=dtype,
            ).to(device)

            inputs = processor(
                text=text,
                audio=load_audio(reference)[0],
                sampling_rate=load_audio(reference)[1],
                return_tensors="pt",
            ).to(device)
            out_tensors = model.generate(**inputs)
            audio = result_to_audio(out_tensors[0] if isinstance(out_tensors, (list, tuple)) else out_tensors)
            sr = getattr(model.config, "sampling_rate", 24000)
            save_wav(out, audio, int(sr))
            return out

        model = Qwen3TTSModel.from_pretrained(
            args.qwen_model,
            device_map=device,
            dtype=dtype,
        )

        kwargs = dict(
            text=text,
            language=args.language,
            ref_audio=str(reference),
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only,
        )

        result = model.generate_voice_clone(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            wavs, sr = result
        else:
            wavs, sr = result, getattr(model, "sample_rate", 24000)

        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        save_wav(out, result_to_audio(wav), int(sr))
        return out


# ---------------------------------------------------------------------------
# 3. POCKET TTS
# ---------------------------------------------------------------------------

class PocketBackend(Backend):
    name = "pocket"

    def generate(self, text, reference, out, args):
        """
        PocketTTS (github.com/mrfakename/PocketTTS) exposes:

            from pocket_tts import PocketTTS
            tts = PocketTTS()
            audio, sr = tts.tts(text, audio_prompt="ref.wav")
        """
        mod = optional_import("pocket_tts", "pocket-tts")

        # Public PocketTTS API.
        if hasattr(mod, "PocketTTS"):
            tts = mod.PocketTTS()
            audio, sr = tts.tts(text, audio_prompt=str(reference))
            save_wav(out, result_to_audio(audio), int(sr))
            return out

        # Some forks ship a TTSModel class instead.
        if hasattr(mod, "TTSModel"):
            model = mod.TTSModel.load_model()
            voice_state = model.get_state_for_audio_prompt(str(reference))
            wav = model.generate_audio(voice_state, text)
            sr = getattr(model, "sample_rate", 24000)
            save_wav(out, result_to_audio(wav), int(sr))
            return out

        raise RuntimeError(
            "pocket_tts module loaded but exposes neither PocketTTS nor "
            "TTSModel. Check your pocket-tts installation."
        )


# ---------------------------------------------------------------------------
# 4. KOKOCLONE
# ---------------------------------------------------------------------------

class KokoCloneBackend(Backend):
    name = "kokoclone"

    def generate(self, text, reference, out, args):
        """
        KokoClone's official repository exposes a CLI:
            python cli.py --text ... --lang ... --ref ... --out ...

        Invoked as a subprocess because KokoClone's public CLI is more stable
        than importing its internal implementation.
        """
        command = args.kokoclone_command

        if not command:
            die(
                "KokoClone requires --kokoclone-command pointing to its "
                "cli.py, e.g.:\n"
                "  --kokoclone-command C:\\KokoClone\\cli.py"
            )

        command_path = Path(command)
        if not command_path.exists():
            die(f"KokoClone CLI not found: {command_path}")

        cmd = [
            args.python_executable,
            str(command_path),
            "--text", text,
            "--lang", args.language,
            "--ref", str(reference),
            "--out", str(out),
        ]

        subprocess.run(cmd, check=True)

        if not out.exists():
            die("KokoClone finished without creating its output WAV.")

        return out


# ---------------------------------------------------------------------------
# 5. XTTS V2
# ---------------------------------------------------------------------------

class XTTSBackend(Backend):
    name = "xtts"

    def generate(self, text, reference, out, args):
        mod = optional_import("TTS.api", "coqui-tts")
        TTS = mod.TTS

        gpu = resolve_device(args.device) == "cuda"

        tts = TTS(
            model_name=args.xtts_model,
            progress_bar=False,
            gpu=gpu,
        )

        tts.tts_to_file(
            text=text,
            speaker_wav=str(reference),
            language=args.language,
            file_path=str(out),
        )

        return out


# ---------------------------------------------------------------------------
# 6. FISH SPEECH
# ---------------------------------------------------------------------------

def post_json(url: str, payload: dict[str, Any]) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        die(f"HTTP {exc.code}: {body[:1500]}")
    except urllib.error.URLError as exc:
        die(f"Cannot reach local TTS server: {exc}")


def save_api_result(data: bytes, content_type: str, out: Path) -> None:
    if data[:4] == b"RIFF" or "audio" in content_type.lower():
        out.write_bytes(data)
        return

    try:
        obj = json.loads(data.decode())
    except Exception:
        die("TTS API returned neither audio nor JSON.")

    # Common JSON response forms.
    for key in ("audio", "wav", "audio_url", "url"):
        value = obj.get(key) if isinstance(obj, dict) else None
        if not value:
            continue

        if isinstance(value, str) and value.startswith(("http://", "https://")):
            urllib.request.urlretrieve(value, out)
            return

        if isinstance(value, str):
            try:
                out.write_bytes(base64.b64decode(value))
                return
            except Exception:
                pass

    die(f"TTS API did not return usable audio: {obj}")


class FishBackend(Backend):
    name = "fish"

    def generate(self, text, reference, out, args):
        """
        Fish Speech is normally run as a local service. The exact S2 server
        build can expose different routes, so the endpoint is configurable.
        """
        payload = {
            "text": text,
            "reference_audio": str(reference.resolve()),
            "reference_text": args.ref_text or "",
            "format": "wav",
        }

        data, content_type = post_json(
            args.fish_url.rstrip("/") + args.fish_endpoint,
            payload,
        )

        save_api_result(data, content_type, out)
        return out


# ---------------------------------------------------------------------------
# 7. F5-TTS
# ---------------------------------------------------------------------------

class F5Backend(Backend):
    name = "f5tts"

    def generate(self, text, reference, out, args):
        cmd = [
            args.f5_command,
            "--model", args.f5_model,
            "--ref_audio", str(reference),
            "--ref_text", args.ref_text or "",
            "--gen_text", text,
        ]

        # Current F5-TTS CLI can select an output file via --output_file in
        # versions that support it. For older builds use --output_dir.
        if args.f5_output_file:
            cmd += ["--output_file", str(out)]
        else:
            cmd += ["--output_dir", str(out.parent)]

        subprocess.run(cmd, check=True)

        if out.exists():
            return out

        candidates = sorted(
            out.parent.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not candidates:
            die("F5-TTS completed but produced no WAV.")

        shutil.copyfile(candidates[0], out)
        return out


# ---------------------------------------------------------------------------
# 8. COSYVOICE
# ---------------------------------------------------------------------------

class CosyVoiceBackend(Backend):
    name = "cosyvoice"

    def generate(self, text, reference, out, args):
        """
        CosyVoice official deployments provide zero-shot inference through
        their local runtime/demo/server. The API URL is configurable because
        different CosyVoice releases expose different server routes.
        """
        payload = {
            "mode": "zero_shot",
            "text": text,
            "prompt_text": args.ref_text or "",
            "prompt_audio": str(reference.resolve()),
            "language": args.language,
        }

        data, content_type = post_json(
            args.cosyvoice_url.rstrip("/") + args.cosyvoice_endpoint,
            payload,
        )

        save_api_result(data, content_type, out)
        return out


# ---------------------------------------------------------------------------
# 9. PARLER-TTS
# ---------------------------------------------------------------------------

class ParlerBackend(Backend):
    name = "parler"

    def generate(self, text, reference, out, args):
        """
        Parler-TTS is prompt/description controlled rather than a native
        reference-audio zero-shot clone interface. Therefore --voice is not
        passed into Parler as a speaker embedding.

        The common pipeline still accepts the same reference file so the
        sanitize stage remains consistent. Use --parler-description to
        specify the desired speaker characteristics.
        """
        # ParlerTTSForConditionalGeneration lives in the separate
        # `parler_tts` package, NOT in `transformers` (transformers never
        # shipped this class -- that was the source of the earlier
        # "module transformers has no attribute ParlerTTSForConditionalGeneration"
        # failure).
        transformers = optional_import("transformers", "transformers")
        AutoTokenizer = transformers.AutoTokenizer

        parler_mod = optional_import(
            "parler_tts",
            "parler-tts @ git+https://github.com/huggingface/parler-tts.git",
        )
        ParlerTTSForConditionalGeneration = (
            parler_mod.ParlerTTSForConditionalGeneration
        )

        device = resolve_device(args.device)

        model = ParlerTTSForConditionalGeneration.from_pretrained(
            args.parler_model
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.parler_model)

        description = args.parler_description

        description_ids = tokenizer(
            description,
            return_tensors="pt",
        ).input_ids.to(device)

        prompt_ids = tokenizer(
            text,
            return_tensors="pt",
        ).input_ids.to(device)

        generation = model.generate(
            input_ids=description_ids,
            prompt_input_ids=prompt_ids,
        )

        audio = generation.cpu().numpy().squeeze()

        save_wav(
            out,
            audio,
            model.config.sampling_rate,
        )
        return out


# ---------------------------------------------------------------------------
# 10. VOICECRAFT
# ---------------------------------------------------------------------------

class VoiceCraftBackend(Backend):
    name = "voicecraft"

    def generate(self, text, reference, out, args):
        """
        VoiceCraft is fundamentally an audio editing / zero-shot TTS system
        driven by prompt audio + transcript and an inference configuration.

        Because VoiceCraft installations differ in checkpoint and inference
        entrypoint, use a command template. The template receives:
            {text}
            {text_file}
            {reference}
            {ref_text}
            {output}
        """
        if not args.voicecraft_command:
            die(
                "VoiceCraft requires --voicecraft-command.\n"
                "Example:\n"
                "  --voicecraft-command "
                "\"python inference.py --prompt {reference} "
                "--prompt_text {ref_text} --text {text_file} "
                "--output {output}\""
            )

        text_file = out.with_suffix(".txt")
        text_file.write_text(text, encoding="utf-8")

        command = args.voicecraft_command.format(
            text=text,
            text_file=str(text_file.resolve()),
            reference=str(reference.resolve()),
            ref_text=args.ref_text or "",
            output=str(out.resolve()),
        )

        subprocess.run(command, shell=True, check=True)

        if not out.exists():
            die("VoiceCraft command finished without creating WAV.")

        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKEND_CLASSES: dict[str, type[Backend]] = {
    "chatterbox": ChatterboxBackend,
    "qwen3":      Qwen3Backend,
    "pocket":     PocketBackend,
    "kokoclone":  KokoCloneBackend,
    "xtts":       XTTSBackend,
    "fish":       FishBackend,
    "f5tts":      F5Backend,
    "cosyvoice":  CosyVoiceBackend,
    "parler":     ParlerBackend,
    "voicecraft": VoiceCraftBackend,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="10-backend local TTS/voice pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--voice", "-v", type=Path, required=False)
    p.add_argument("--text", "-t")
    p.add_argument("--text-file", "-f", type=Path)
    p.add_argument("--out", "-o", type=Path, default=Path("voice_final.wav"))

    p.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default="chatterbox",
    )
    p.add_argument("--all-backends", action="store_true", help="Run all 10 backends and export results into an output folder in the voice file's directory")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for --all-backends (defaults to <voice_stem>_clones in input audio dir)")
    p.add_argument("--list-backends", action="store_true")

    p.add_argument("--language", default="en")
    p.add_argument("--ref-text", default="")
    p.add_argument("--ref-language", default=None)
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument("--gap-ms", type=int, default=180)

    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
    )

    # Sanitization.
    p.add_argument("--sanitize-sr", type=int, default=24000)
    p.add_argument("--trim-db", type=float, default=45.0)
    p.add_argument("--sanitize-denoise", action="store_true", help="Enable spectral noise reduction on reference audio")
    p.add_argument("--prop-decrease", type=float, default=0.5, help="Noise reduction strength (0.0 to 1.0; 0.5 is gentle)")
    p.add_argument("--raw-ref", action="store_true", help="Bypass all sanitization and use original reference audio directly")
    p.add_argument("--fade-ms", type=float, default=12.0)

    # Upscaling.
    p.add_argument("--skip-upscale", action="store_true")
    p.add_argument("--upscale-device", default=None)
    p.add_argument("--audiosr-model", default="basic")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)

    # Final denoise.
    p.add_argument("--skip-denoise", action="store_true")

    # Chatterbox.
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg-weight", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)

    # Qwen3.
    p.add_argument(
        "--qwen-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    p.add_argument("--x-vector-only", action="store_true")

    # KokoClone.
    p.add_argument(
        "--kokoclone-command",
        default=None,
        help="Path to KokoClone cli.py",
    )
    p.add_argument(
        "--python-executable",
        default=sys.executable,
    )

    # XTTS.
    p.add_argument(
        "--xtts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
    )

    # Fish.
    p.add_argument("--fish-url", default="http://127.0.0.1:8080")
    p.add_argument("--fish-endpoint", default="/v1/tts")

    # F5.
    p.add_argument("--f5-command", default="f5-tts_infer-cli")
    p.add_argument("--f5-model", default="F5TTS_v1_Base")
    p.add_argument("--f5-output-file", action="store_true")

    # CosyVoice.
    p.add_argument("--cosyvoice-url", default="http://127.0.0.1:50000")
    p.add_argument("--cosyvoice-endpoint", default="/tts")

    # Parler.
    p.add_argument(
        "--parler-model",
        default="parler-tts/parler-tts-mini-v1",
    )
    p.add_argument(
        "--parler-description",
        default=(
            "A clear speaker delivers natural, expressive speech at a "
            "moderate pace. The recording is very high quality, close and "
            "clean, with little background noise."
        ),
    )

    # VoiceCraft.
    p.add_argument("--voicecraft-command", default=None)

    # Utility.
    p.add_argument("--clean-only", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--bit-depth", choices=[16, 24], type=int, default=16)

    return p


def print_backends() -> None:
    print("\n10 available backends:\n")
    for i, (name, description) in enumerate(BACKENDS.items(), 1):
        print(f"  {i:02d}. {name:<12} {description}")
    print()


def read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        require_file(args.text_file, "Text file")
        text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        die("Provide --text or --text-file.")

    text = text.strip()
    if not text:
        die("Text is empty.")

    return text


def run(args: argparse.Namespace) -> None:
    if not args.voice:
        die("--voice is required.")

    require_file(args.voice, "Reference audio")

    work = Path(tempfile.mkdtemp(prefix="voice_clone_10_"))

    try:
        # SANITIZE
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
            print(f"\nSanitized reference: {args.out.resolve()}")
            return

        # CLONE / TTS
        print("\n[2/4] CLONE / TTS")
        print(f"      Backend: {args.backend}")

        text = read_text(args)
        chunks = split_text(text, args.max_chars)

        backend = BACKEND_CLASSES[args.backend]()
        raw_outputs: list[Path] = []

        for index, chunk in enumerate(chunks, 1):
            print(f"      [{index}/{len(chunks)}] {chunk}")

            chunk_out = work / f"tts_{index:04d}.wav"
            backend.generate(
                chunk,
                reference,
                chunk_out,
                args,
            )

            if not chunk_out.exists():
                die(
                    f"{args.backend} failed to create "
                    f"{chunk_out.name}"
                )

            raw_outputs.append(chunk_out)

        cloned = work / "cloned.wav"
        concat_audio(
            raw_outputs,
            cloned,
            sr=24000,
            gap_ms=max(0, args.gap_ms),
        )

        # UPSCALE
        upscaled = work / "upscaled.wav"
        upscale_audio(cloned, upscaled, args)

        # DENOISE
        final = work / "final.wav"
        denoise_audio(upscaled, final, args)

        # OUTPUT
        args.out.parent.mkdir(parents=True, exist_ok=True)

        if args.bit_depth == 24:
            audio, sr = load_audio(final)
            save_wav(args.out, audio, sr, "PCM_24")
        else:
            shutil.copyfile(final, args.out)

        sf = optional_import("soundfile", "soundfile")
        info = sf.info(str(args.out))

        print("\n" + "=" * 72)
        print("COMPLETE")
        print("=" * 72)
        print(f"Backend:     {args.backend}")
        print(f"Output:      {args.out.resolve()}")
        print(f"Sample rate: {info.samplerate} Hz")
        print(f"Channels:    {info.channels}")
        print(f"Duration:    {info.duration:.2f} sec")
        print("Pipeline:    SANITIZE -> CLONE -> UPSCALE -> DENOISE")
        print("=" * 72)

        if args.keep_temp:
            print(f"\nIntermediate files: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    except BaseException:
        if not args.keep_temp:
            shutil.rmtree(work, ignore_errors=True)
        raise


def run_all_backends(args: argparse.Namespace) -> None:
    if not args.voice:
        die("--voice is required.")

    require_file(args.voice, "Reference audio")

    if args.out_dir:
        out_dir = args.out_dir.resolve()
    else:
        out_dir = args.voice.resolve().parent / f"{args.voice.stem}_clones"

    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw_clone"
    upscaled_dir = out_dir / "upscaled_clone"
    denoised_dir = out_dir / "denoised_clone"
    for d in (raw_dir, upscaled_dir, denoised_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n==================================================")
    print(f" RUNNING ALL 10 VOICE CLONE BACKENDS")
    print(f" Output Directory: {out_dir}")
    print(f"   raw_clone/      -> pre-upscale, pre-denoise clone")
    print(f"   upscaled_clone/ -> after AudioSR / HQ resample")
    print(f"   denoised_clone/ -> final, after DeepFilterNet3 (or copy)")
    print(f"==================================================")

    text = read_text(args)
    chunks = split_text(text, args.max_chars)

    work = Path(tempfile.mkdtemp(prefix="voice_clone_10_all_"))

    try:
        # SANITIZE ONCE
        reference = work / "reference_sanitized.wav"
        Sanitizer(
            sample_rate=args.sanitize_sr,
            trim_db=args.trim_db,
            denoise=args.sanitize_denoise,
            prop_decrease=args.prop_decrease,
            fade_ms=args.fade_ms,
            raw_ref=args.raw_ref,
        ).run(args.voice, reference)

        summary_results = {}

        for idx, backend_name in enumerate(BACKENDS, 1):
            stem = f"{idx:02d}_{backend_name}.wav"
            raw_target = raw_dir / stem
            upscaled_target = upscaled_dir / stem
            denoised_target = denoised_dir / stem

            print(f"\n--------------------------------------------------")
            print(f"[{idx}/10] Running backend: {backend_name}")
            print(f"--------------------------------------------------")

            try:
                backend = BACKEND_CLASSES[backend_name]()
                raw_outputs: list[Path] = []

                for chunk_idx, chunk in enumerate(chunks, 1):
                    chunk_out = work / f"{backend_name}_tts_{chunk_idx:04d}.wav"
                    backend.generate(chunk, reference, chunk_out, args)
                    if chunk_out.exists():
                        raw_outputs.append(chunk_out)

                if not raw_outputs:
                    print(f"      [WARNING] {backend_name} produced no output.")
                    summary_results[backend_name] = "FAILED / SKIPPED"
                    continue

                cloned = work / f"{backend_name}_cloned.wav"
                concat_audio(raw_outputs, cloned, sr=24000, gap_ms=max(0, args.gap_ms))
                shutil.copyfile(cloned, raw_target)

                upscaled = work / f"{backend_name}_upscaled.wav"
                upscale_audio(cloned, upscaled, args)
                shutil.copyfile(upscaled, upscaled_target)

                final = work / f"{backend_name}_final.wav"
                denoise_audio(upscaled, final, args)

                if args.bit_depth == 24:
                    audio, sr = load_audio(final)
                    save_wav(denoised_target, audio, sr, "PCM_24")
                else:
                    shutil.copyfile(final, denoised_target)

                summary_results[backend_name] = f"SUCCESS -> {stem}"
                print(f"      Saved: {raw_target}")
                print(f"             {upscaled_target}")
                print(f"             {denoised_target}")

            except (Exception, SystemExit) as exc:
                print(f"      [SKIP/WARN] Backend '{backend_name}' not configured or failed: {exc}")
                summary_results[backend_name] = f"SKIPPED/FAILED ({exc})"

        print("\n==================================================")
        print(" ALL 10 BACKENDS EXECUTION SUMMARY")
        print("==================================================")
        for b_name, status in summary_results.items():
            print(f"  - {b_name:<12}: {status}")
        print(f"\nAll generated exports are stored in folder:\n  {out_dir}")
        print("==================================================")

    finally:
        if not args.keep_temp:
            shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    p = parser()
    args = p.parse_args()

    if args.list_backends:
        print_backends()
        return

    try:
        if getattr(args, "all_backends", False):
            run_all_backends(args)
        else:
            run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        die(f"Backend process exited with code {exc.returncode}.")


if __name__ == "__main__":
    main()