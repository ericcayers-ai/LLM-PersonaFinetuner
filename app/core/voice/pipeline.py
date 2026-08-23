"""Core sanitize/upscale/denoise pipeline stages and shared subprocess-isolation
utilities, ported from voice_clone_mini.py / voice_clone_10.py.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class VoiceCloneError(RuntimeError):
    """Raised for any voice-pipeline failure; message is safe to show to callers."""


# Backend subprocesses (uv resolving + downloading weights) can legitimately take
# a few minutes on first run, but must not hang the FastAPI worker forever.
DEFAULT_SUBPROCESS_TIMEOUT = 900

_HF_AUTH_HINT = (
    "\nThis looks like a Hugging Face authentication/gated-repo error. "
    "Run `huggingface-cli login` (or set the HF_TOKEN environment variable) in the "
    "same environment that runs the app, then retry. If you've already signed in, "
    "make sure you accepted the model's license on huggingface.co."
)


def _looks_like_hf_auth_error(text: str) -> bool:
    lowered = text.lower()
    markers = ("gated repo", "401 client error", "restricted", "you don't have permission", "authentication required")
    return "huggingface" in lowered and any(m in lowered for m in markers)


def optional_import(module: str, package: Optional[str] = None):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        pkg = package or module
        raise VoiceCloneError(f"Missing '{pkg}'. Install it with: pip install {pkg}") from exc


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise VoiceCloneError(f"{label} not found: {path}")
    if not path.is_file():
        raise VoiceCloneError(f"{label} is not a file: {path}")


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
    """Clone a git repo into the persistent cache dir if it isn't there yet."""
    dest = repos_dir() / dirname
    if dest.exists() and any(dest.iterdir()):
        return dest
    cmd = ["git", "clone", "--depth", "1"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([url, str(dest)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise VoiceCloneError(f"git clone failed for {url}:\n{result.stderr}")
    return dest


def run_uv_script(
    *,
    python_version: str,
    packages: list[str],
    script_text: str,
    script_args: list[str],
    device: str,
    cuda_index: str = "https://download.pytorch.org/whl/cu121",
    extra_uv_args: Optional[list[str]] = None,
    workdir: Optional[Path] = None,
    timeout: Optional[int] = DEFAULT_SUBPROCESS_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Write `script_text` to a temp .py and run it via `uv run` with an isolated,
    throwaway venv containing exactly `packages` (+ torch/torchaudio matched to device).
    """
    scratch = Path(tempfile.mkdtemp(prefix="voice_clone_uv_"))
    script_path = scratch / "runner.py"
    script_path.write_text(script_text, encoding="utf-8")

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

    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(workdir) if workdir else None, timeout=timeout,
    )


def run_uv_requirements(
    *,
    python_version: str,
    requirements_file: Path,
    script_text: str,
    script_args: list[str],
    workdir: Path,
    extra_packages: Optional[list[str]] = None,
    timeout: Optional[int] = DEFAULT_SUBPROCESS_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Same as run_uv_script but resolves deps from a cloned repo's requirements.txt."""
    script_path = workdir / "_voice_runner.py"
    script_path.write_text(script_text, encoding="utf-8")
    cmd = ["uv", "run", "--python", python_version, "--with-requirements", str(requirements_file)]
    for pkg in (extra_packages or []):
        cmd.extend(["--with", pkg])
    cmd.extend(["python", str(script_path), *script_args])
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(workdir), timeout=timeout,
    )


def fail(name: str, result: subprocess.CompletedProcess) -> None:
    tail_out = (result.stdout or "")[-4000:]
    tail_err = (result.stderr or "")[-4000:]
    hint = _HF_AUTH_HINT if _looks_like_hf_auth_error(tail_out + tail_err) else ""
    raise VoiceCloneError(f"[{name}] backend subprocess failed (exit {result.returncode}).\n{tail_out}\n{tail_err}{hint}")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path: Path, target_sr: Optional[int] = None):
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


def result_to_audio(result):
    np = optional_import("numpy", "numpy")
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    elif hasattr(result, "numpy"):
        result = result.numpy()
    if isinstance(result, (list, tuple)):
        result = result[0] if result else np.zeros(0, dtype=np.float32)
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    return normalize(np.asarray(result, dtype=np.float32).squeeze())


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
        raise VoiceCloneError("No TTS output files were produced.")
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
        if self.raw_ref:
            shutil.copyfile(source, out)
            return out

        audio, sr = load_audio(source, self.sample_rate)

        if self.denoise and self.prop_decrease > 0:
            try:
                nr = optional_import("noisereduce", "noisereduce")
                audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=self.prop_decrease).astype("float32")
            except Exception:
                pass

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
        return out


# ---------------------------------------------------------------------------
# 2. UPSCALE: FlashSR / AudioSR / plain resample
# ---------------------------------------------------------------------------

FLASHSR_REPO_URL = "https://github.com/jakeoneijk/FlashSR_Inference.git"
FLASHSR_WEIGHTS_REPO = "jakeoneijk/FlashSR_weights"
FLASHSR_CHUNK_SAMPLES = 245760

FLASHSR_SCRIPT = '''
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import soundfile as sf
import numpy as np
import soxr

source, out_path, w_dir, device = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CHUNK = 245760

from FlashSR.FlashSR import FlashSR

flashsr = FlashSR(
    os.path.join(w_dir, "student_ldm.pth"),
    os.path.join(w_dir, "sr_vocoder.pth"),
    os.path.join(w_dir, "vae.pth"),
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
padded = np.zeros(n_chunks * CHUNK, dtype=np.float32)
padded[:n] = audio

out_chunks = []
for i in range(n_chunks):
    seg = padded[i * CHUNK:(i + 1) * CHUNK]
    seg_t = torch.from_numpy(seg).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_out = flashsr.forward(seg_t)
    out_chunks.append(sr_out.detach().cpu().numpy().squeeze().astype(np.float32))

result = np.concatenate(out_chunks)[:n]
maximum = float(np.max(np.abs(result))) if result.size else 0.0
if maximum > 1e-8:
    result = result * (0.95 / maximum)
sf.write(out_path, result.astype(np.float32), 48000, subtype="PCM_16")
'''


def _download_flashsr_weights(target_dir: Path) -> None:
    script = f'''
from huggingface_hub import snapshot_download
snapshot_download(repo_id="{FLASHSR_WEIGHTS_REPO}", repo_type="dataset", local_dir=r"{target_dir}")
'''
    result = run_uv_script(python_version="3.11", packages=["huggingface_hub"], script_text=script, script_args=[], device="cpu")
    if result.returncode != 0:
        hint = _HF_AUTH_HINT if _looks_like_hf_auth_error(result.stderr or "") else ""
        raise VoiceCloneError(
            f"Failed to download FlashSR weights automatically.\n{result.stderr[-2000:]}\n"
            f"Download manually from https://huggingface.co/datasets/{FLASHSR_WEIGHTS_REPO} into {target_dir}, "
            f"or use upscale_backend='resample'.{hint}"
        )


def flashsr_upscale(source: Path, out: Path, device: str) -> Path:
    repo = ensure_repo(FLASHSR_REPO_URL, "FlashSR_Inference")
    w_dir = weights_dir() / "flashsr"
    w_dir.mkdir(parents=True, exist_ok=True)
    needed = ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"]
    if not all((w_dir / f).exists() for f in needed):
        _download_flashsr_weights(w_dir)
    if not all((w_dir / f).exists() for f in needed):
        raise VoiceCloneError("FlashSR weights still missing after download attempt.")

    result = run_uv_script(
        python_version="3.11",
        packages=["torch", "torchaudio", "soundfile", "soxr", "numpy<2"],
        script_text=FLASHSR_SCRIPT,
        script_args=[str(source), str(out), str(w_dir), device],
        device=device,
        workdir=repo,
    )
    if result.returncode != 0 or not out.exists():
        raise VoiceCloneError(result.stderr[-4000:])
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
    result = super_resolution(model, str(sys.argv[1]), seed={seed}, guidance_scale={guidance}, ddim_steps={steps})
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    elif hasattr(result, "numpy"):
        result = result.numpy()
    if isinstance(result, (list, tuple)):
        result = result[0] if result else np.zeros(0, dtype=np.float32)
    sf.write(sys.argv[2], normalize(result), 48000, subtype="PCM_16")
except Exception as e:
    print(f"Error in audiosr subprocess: {{e}}", file=sys.stderr)
    sys.exit(1)
'''


def audiosr_upscale(source: Path, out: Path, device: str, model_name: str = "basic",
                     seed: int = 42, guidance: float = 3.5, steps: int = 50) -> Path:
    script = AUDIOSR_SCRIPT_TEMPLATE.format(model_name=model_name, device=device, seed=seed, guidance=guidance, steps=steps)
    result = run_uv_script(
        python_version="3.10", packages=["audiosr", "matplotlib", "setuptools", "soundfile"],
        script_text=script, script_args=[str(source), str(out)], device=device,
    )
    if result.returncode != 0 or not out.exists():
        raise VoiceCloneError(result.stderr[-4000:])
    return out


def upscale_audio(source: Path, out: Path, backend: str = "flashsr", device: str = "auto",
                   audiosr_model: str = "basic", seed: int = 42, guidance: float = 3.5, steps: int = 50) -> Path:
    resolved = resolve_device(device)

    if backend == "resample":
        audio, sr = load_audio(source, 48000)
        save_wav(out, audio, sr)
        return out

    if backend == "flashsr":
        try:
            return flashsr_upscale(source, out, resolved)
        except Exception:
            audio, sr = load_audio(source, 48000)
            save_wav(out, audio, sr)
            return out

    if backend == "audiosr":
        try:
            return audiosr_upscale(source, out, resolved, audiosr_model, seed, guidance, steps)
        except Exception:
            audio, sr = load_audio(source, 48000)
            save_wav(out, audio, sr)
            return out

    raise VoiceCloneError(f"Unknown upscale backend: {backend}")


# ---------------------------------------------------------------------------
# 3. DENOISE (DeepFilterNet3)
# ---------------------------------------------------------------------------

DEEPFILTER_SCRIPT = '''
import sys
source, out_path = sys.argv[1], sys.argv[2]
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    model, state, _ = init_df()
    wav, _ = load_audio(source, sr=state.sr())
    enhanced = enhance(model, state, wav)
    save_audio(out_path, enhanced, state.sr())
except Exception as e:
    print(f"Error in deepfilternet subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''


def denoise_audio(source: Path, out: Path, skip: bool = False) -> Path:
    if skip:
        shutil.copyfile(source, out)
        return out
    try:
        result = run_uv_script(
            python_version="3.11", packages=["deepfilternet", "soundfile"],
            script_text=DEEPFILTER_SCRIPT, script_args=[str(source), str(out)], device="cpu",
        )
        if result.returncode != 0 or not out.exists():
            raise VoiceCloneError(result.stderr[-2000:])
        return out
    except Exception:
        shutil.copyfile(source, out)
        return out
