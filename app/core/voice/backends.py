"""All TTS/voice-cloning backend adapters, ported from voice_clone_mini.py
(12 backends, isolated `uv run` subprocesses) and voice_clone_10.py (adds
kokoclone, parler, voicecraft as CLI/HTTP adapters; fish as a local-HTTP
variant distinct from fishspeech's hosted API). Nothing from either source
script is omitted -- backends that need a manual model download or an
external local server surface the same instructions the original scripts
printed instead of silently failing.

Every backend function has signature:
    fn(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.voice.pipeline import (
    VoiceCloneError,
    ensure_repo,
    fail,
    resolve_device,
    run_uv_requirements,
    run_uv_script,
)


@dataclass
class BackendOptions:
    device: str = "auto"
    language: Optional[str] = "en"
    seed: int = 42
    # backend-specific knobs, all optional with sane defaults
    openvoice_base: str = "pocket"
    qwen_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    fishspeech_api_key_env: str = "FISH_AUDIO_API_KEY"
    fish_url: str = "http://127.0.0.1:8080"
    fish_endpoint: str = "/v1/tts"
    cosyvoice_api_url: str = "http://127.0.0.1:50000"
    cosyvoice_api_endpoint: str = "/tts"
    kokoclone_command: Optional[str] = None
    voicecraft_command: Optional[str] = None
    parler_model: str = "parler-tts/parler-tts-mini-v1"
    parler_description: str = (
        "A clear speaker delivers natural, expressive speech at a moderate pace. "
        "The recording is very high quality, close and clean, with little background noise."
    )
    seed_chattts: int = 42
    extra: dict = field(default_factory=dict)


BACKEND_INFO: dict[str, dict[str, str]] = {
    "pocket":     {"weight": "light",  "desc": "Pocket-TTS (default/live). Fast, small, good enough for real-time cloning."},
    "f5tts":      {"weight": "light",  "desc": "F5-TTS. Flow-matching DiT; one of the strongest open zero-shot cloners."},
    "chatterbox": {"weight": "light",  "desc": "Chatterbox Multilingual v3 (Resemble AI). 23 languages."},
    "chattts":    {"weight": "light",  "desc": "ChatTTS (2noise). Random-speaker prosody model; does NOT clone the reference voice."},
    "xtts":       {"weight": "light",  "desc": "Coqui XTTS v2. 17 languages, ~6s reference."},
    "openvoice":  {"weight": "medium", "desc": "OpenVoice v2 tone-color conversion, chained after a base TTS pass."},
    "metavoice":  {"weight": "light",  "desc": "MetaVoice-1B. Expressive foundation TTS."},
    "qwen3tts":   {"weight": "light",  "desc": "Qwen3-TTS (Alibaba). 10 languages, low latency."},
    "omnivoice":  {"weight": "light",  "desc": "OmniVoice (k2-fsa). 600+ languages, fast diffusion-LM TTS."},
    "fishspeech": {"weight": "medium", "desc": "Fish Audio hosted API. Needs FISH_AUDIO_API_KEY; not fully local."},
    "cosyvoice":  {"weight": "heavy",  "desc": "CosyVoice2 (FunAudioLLM). Clones repo + downloads pretrained weights (manual step)."},
    "cosyvoice_api": {"weight": "medium", "desc": "CosyVoice via a locally running HTTP API server (voice_clone_10 variant)."},
    "gptsovits":  {"weight": "heavy",  "desc": "GPT-SoVITS (RVC-Boss). Clones repo + runs local HTTP API server (manual model step)."},
    "kokoclone":  {"weight": "heavy",  "desc": "KokoClone (Kokoro + zero-shot cloning). Requires an external CLI you point at with --kokoclone-command."},
    "fish":       {"weight": "medium", "desc": "Fish Speech local HTTP server (distinct from hosted fishspeech). Requires a locally running server."},
    "parler":     {"weight": "medium", "desc": "Parler-TTS. Description-controlled, does NOT clone the reference voice (prompt-only)."},
    "voicecraft": {"weight": "heavy",  "desc": "VoiceCraft. Requires an external inference command template."},
}


# ---- pocket (default live backend) -----------------------------------------

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
        sf.write(sys.argv[3], normalize(audio), int(sr), subtype="PCM_16")
    elif hasattr(p, "TTSModel"):
        model = p.TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(sys.argv[2])
        wav = model.generate_audio(voice_state, sys.argv[1])
        sr = getattr(model, "sample_rate", 24000)
        sf.write(sys.argv[3], normalize(wav), int(sr), subtype="PCM_16")
    else:
        print("Error: pocket_tts module exposes neither PocketTTS nor TTSModel.", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"Error in pocket_tts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''


def backend_pocket(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11", packages=["pocket-tts", "soundfile", "numpy>=2"],
        script_text=POCKET_SCRIPT, script_args=[text, str(reference), str(out)], device=device,
    )
    if result.returncode != 0:
        fail("pocket", result)
    return out


# ---- f5tts -------------------------------------------------------------------

F5TTS_SCRIPT = '''
import sys
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
    from f5_tts.api import F5TTS
    f5 = F5TTS()
    wav, sr, _spect = f5.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=text)
    sf.write(out_path, normalize(wav), int(sr), subtype="PCM_16")
except Exception as e:
    print(f"Error in f5-tts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''


def backend_f5tts(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11", packages=["f5-tts", "soundfile", "numpy<2"],
        script_text=F5TTS_SCRIPT, script_args=[text, str(reference), ref_text, str(out)], device=device,
    )
    if result.returncode != 0:
        fail("f5tts", result)
    return out


# ---- chatterbox (Multilingual v3) --------------------------------------------

CHATTERBOX_SCRIPT = '''
import sys
import torchaudio as ta

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


def backend_chatterbox(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11", packages=["chatterbox-tts", "soundfile"],
        script_text=CHATTERBOX_SCRIPT, script_args=[text, str(reference), opts.language or "", str(out)], device=device,
    )
    if result.returncode != 0:
        fail("chatterbox", result)
    return out


# ---- chattts (does NOT clone the reference; random/seeded speaker) ----------

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


def backend_chattts(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11", packages=["git+https://github.com/2noise/ChatTTS", "soundfile"],
        script_text=CHATTTS_SCRIPT, script_args=[text, str(opts.seed_chattts), str(out)], device=device,
    )
    if result.returncode != 0:
        fail("chattts", result)
    return out


# ---- xtts (Coqui XTTS v2) -----------------------------------------------------

XTTS_SCRIPT = '''
import sys
text, ref_audio, language, model_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
try:
    import torch
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name).to(device)
    tts.tts_to_file(text=text, speaker_wav=ref_audio, language=language or "en", file_path=out_path)
except Exception as e:
    print(f"Error in xtts subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''


def backend_xtts(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11", packages=["coqui-tts", "soundfile"],
        script_text=XTTS_SCRIPT,
        script_args=[text, str(reference), opts.language or "en", opts.xtts_model, str(out)],
        device=device,
    )
    if result.returncode != 0:
        fail("xtts", result)
    return out


# ---- openvoice (tone-color conversion, chained after a base TTS pass) ------

def backend_openvoice(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    base_fn = BACKEND_FUNCS[opts.openvoice_base]
    scratch = out.parent / f"_openvoice_base_{out.stem}.wav"
    base_fn(text, reference, ref_text, scratch, opts)

    device = resolve_device(opts.device)
    cmd = ["uv", "run", "--python", "3.11", "--with", "openvoice-cli", "--with", "soundfile"]
    if device == "cuda":
        cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu121", "--with", "torch", "--with", "torchaudio"])
    else:
        cmd.extend(["--with", "torch", "--with", "torchaudio"])
    cmd.extend(["python", "-m", "openvoice_cli", "single", "-i", str(scratch), "-r", str(reference), "-o", str(out), "-d", device])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not out.exists():
        fail("openvoice", result)
    return out


# ---- metavoice -----------------------------------------------------------------

METAVOICE_SCRIPT = '''
import sys, shutil
text, ref_audio, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from fam.llm.fast_inference import TTS
    tts = TTS()
    wav_path = tts.synthesise(text=text, spk_ref_path=ref_audio)
    shutil.copyfile(wav_path, out_path)
except Exception as e:
    print(f"Error in metavoice subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''


def backend_metavoice(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.10", packages=["metavoice-src", "soundfile"],
        script_text=METAVOICE_SCRIPT, script_args=[text, str(reference), str(out)], device=device,
    )
    if result.returncode != 0:
        fail("metavoice", result)
    return out


# ---- qwen3tts -------------------------------------------------------------------

QWEN3TTS_SCRIPT = '''
import sys
import torch
import soundfile as sf
import numpy as np

text, ref_audio, ref_text, language, model_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

try:
    from qwen_tts import Qwen3TTSModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3TTSModel.from_pretrained(model_name, device_map=device, dtype=dtype)

    wavs = None
    sr = 24000
    errors = []
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


def backend_qwen3tts(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11", packages=["qwen-tts", "soundfile"],
        script_text=QWEN3TTS_SCRIPT,
        script_args=[text, str(reference), ref_text, opts.language or "", opts.qwen_model, str(out)],
        device=device,
    )
    if result.returncode != 0:
        fail("qwen3tts", result)
    return out


# ---- omnivoice --------------------------------------------------------------------

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


def backend_omnivoice(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11", packages=["omnivoice", "soundfile"],
        script_text=OMNIVOICE_SCRIPT, script_args=[text, str(reference), ref_text, str(out)], device=device,
    )
    if result.returncode != 0:
        fail("omnivoice", result)
    return out


# ---- fishspeech (hosted Fish Audio API) -----------------------------------------

FISHSPEECH_SCRIPT = '''
import sys, os
text, ref_audio, ref_text, out_path, api_key_env = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
api_key = os.environ.get(api_key_env)
if not api_key:
    print(f"Error: {api_key_env} is not set. Get one at https://fish.audio.", file=sys.stderr)
    sys.exit(1)
try:
    from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
    session = Session(api_key)
    with open(ref_audio, "rb") as f:
        ref_bytes = f.read()
    references = [ReferenceAudio(audio=ref_bytes, text=ref_text or "")]
    with open(out_path, "wb") as out_f:
        for chunk in session.tts(TTSRequest(text=text, references=references, format="wav")):
            out_f.write(chunk)
except Exception as e:
    print(f"Error in fishspeech (Fish Audio API) subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''


def backend_fishspeech(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    result = run_uv_script(
        python_version="3.11", packages=["fish-audio-sdk", "soundfile"],
        script_text=FISHSPEECH_SCRIPT,
        script_args=[text, str(reference), ref_text, str(out), opts.fishspeech_api_key_env],
        device="cpu",
    )
    if result.returncode != 0:
        fail("fishspeech", result)
    return out


# ---- fish (voice_clone_10's variant: local HTTP server, e.g. Fish Speech S2) --

def post_json(url: str, payload: dict) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise VoiceCloneError(f"HTTP {exc.code}: {body[:1500]}") from exc
    except urllib.error.URLError as exc:
        raise VoiceCloneError(f"Cannot reach local TTS server at {url}: {exc}") from exc


def save_api_result(data: bytes, content_type: str, out: Path) -> None:
    if data[:4] == b"RIFF" or "audio" in content_type.lower():
        out.write_bytes(data)
        return
    try:
        obj = json.loads(data.decode())
    except Exception as exc:
        raise VoiceCloneError("TTS API returned neither audio nor JSON.") from exc
    for key in ("audio", "wav", "audio_url", "url"):
        value = obj.get(key) if isinstance(obj, dict) else None
        if not value:
            continue
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            urllib.request.urlretrieve(value, out)
            return
        if isinstance(value, str):
            import base64
            try:
                out.write_bytes(base64.b64decode(value))
                return
            except Exception:
                pass
    raise VoiceCloneError(f"TTS API did not return usable audio: {obj}")


def backend_fish(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    payload = {
        "text": text,
        "reference_audio": str(reference.resolve()),
        "reference_text": ref_text or "",
        "format": "wav",
    }
    data, content_type = post_json(opts.fish_url.rstrip("/") + opts.fish_endpoint, payload)
    save_api_result(data, content_type, out)
    return out


# ---- cosyvoice (heavy: repo clone + pretrained weights, local Python API) -----

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


def backend_cosyvoice(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    repo = ensure_repo(COSYVOICE_REPO_URL, "CosyVoice", recursive=True)
    model_dir = repo / "pretrained_models" / COSYVOICE_MODEL_DIR_NAME
    requirements = repo / "requirements.txt"
    if not requirements.exists():
        raise VoiceCloneError(
            f"CosyVoice clone at {repo} has no requirements.txt; the clone may be incomplete. "
            f"Delete {repo} and retry."
        )
    if not model_dir.exists():
        raise VoiceCloneError(textwrap.dedent(f"""
            [cosyvoice] Pretrained weights not found at:
                {model_dir}
            One-time manual step (per the official CosyVoice2 instructions):
                pip install modelscope
                modelscope download --model FunAudioLLM/{COSYVOICE_MODEL_DIR_NAME} --local_dir "{model_dir}"
            Re-run once the weights are in place.
        """))

    result = run_uv_requirements(
        python_version="3.10", requirements_file=requirements,
        script_text=COSYVOICE_SCRIPT, script_args=[text, str(reference), ref_text, str(model_dir), str(out)],
        workdir=repo, extra_packages=["soundfile"], timeout=1800,
    )
    if result.returncode != 0:
        fail("cosyvoice", result)
    return out


# ---- cosyvoice (voice_clone_10's variant: local HTTP API server) --------------

def backend_cosyvoice_api(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    payload = {
        "mode": "zero_shot", "text": text, "prompt_text": ref_text or "",
        "prompt_audio": str(reference.resolve()), "language": opts.language,
    }
    data, content_type = post_json(opts.cosyvoice_api_url.rstrip("/") + opts.cosyvoice_api_endpoint, payload)
    save_api_result(data, content_type, out)
    return out


# ---- gptsovits (heavy: repo clone + local HTTP API server) --------------------

GPTSOVITS_REPO_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"
GPTSOVITS_PORT = 9880


def backend_gptsovits(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    repo = ensure_repo(GPTSOVITS_REPO_URL, "GPT-SoVITS")
    requirements = repo / "requirements.txt"
    pretrained = repo / "GPT_SoVITS" / "pretrained_models"
    if not requirements.exists():
        raise VoiceCloneError(f"GPT-SoVITS clone at {repo} has no requirements.txt; delete {repo} and retry.")
    if not pretrained.exists() or not any(pretrained.iterdir()):
        raise VoiceCloneError(textwrap.dedent(f"""
            [gptsovits] Pretrained models not found under:
                {pretrained}
            One-time manual step (per the official GPT-SoVITS README):
                cd "{repo}"
                bash install.sh
            (Windows: use the integrated package linked from the repo README, or run install.sh under WSL.)
            Re-run once pretrained_models is populated.
        """))

    api_script = repo / "api_v2.py"
    if not api_script.exists():
        raise VoiceCloneError(f"Expected {api_script} from the GPT-SoVITS clone but it's missing.")

    proc = subprocess.Popen(
        ["uv", "run", "--python", "3.10", "--with-requirements", str(requirements),
         "python", str(api_script), "-a", "127.0.0.1", "-p", str(GPTSOVITS_PORT)],
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        base_url = f"http://127.0.0.1:{GPTSOVITS_PORT}"
        ready = False
        for _ in range(180):
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
            raise VoiceCloneError(f"gpt-sovits api_v2.py server did not become ready.\n{leftover[-4000:]}")

        payload = {
            "text": text, "text_lang": opts.language or "en",
            "ref_audio_path": str(reference), "prompt_text": ref_text,
            "prompt_lang": opts.language or "en", "media_type": "wav",
        }
        req = urllib.request.Request(
            f"{base_url}/tts", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
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
        raise VoiceCloneError("gpt-sovits API call produced no audio.")
    return out


# ---- kokoclone (external CLI) --------------------------------------------------

def backend_kokoclone(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    command = opts.kokoclone_command
    if not command:
        raise VoiceCloneError(
            "KokoClone requires opts.kokoclone_command pointing to its cli.py, e.g. "
            "C:\\KokoClone\\cli.py"
        )
    command_path = Path(command)
    if not command_path.exists():
        raise VoiceCloneError(f"KokoClone CLI not found: {command_path}")

    import sys as _sys
    cmd = [_sys.executable, str(command_path), "--text", text, "--lang", opts.language or "en",
           "--ref", str(reference), "--out", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VoiceCloneError(f"KokoClone failed: {result.stderr[-2000:]}")
    if not out.exists():
        raise VoiceCloneError("KokoClone finished without creating its output WAV.")
    return out


# ---- parler (description-controlled; does not clone reference audio) ---------

PARLER_SCRIPT = '''
import sys
text, description, model_name, sr_out, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
try:
    import torch
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    import soundfile as sf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    description_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=description_ids, prompt_input_ids=prompt_ids)
    audio = generation.cpu().numpy().squeeze()
    sf.write(out_path, audio, model.config.sampling_rate)
except Exception as e:
    print(f"Error in parler subprocess: {e}", file=sys.stderr)
    sys.exit(1)
'''


def backend_parler(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    device = resolve_device(opts.device)
    result = run_uv_script(
        python_version="3.11",
        packages=["git+https://github.com/huggingface/parler-tts.git", "transformers", "soundfile"],
        script_text=PARLER_SCRIPT,
        script_args=[text, opts.parler_description, opts.parler_model, "", str(out)],
        device=device,
    )
    if result.returncode != 0:
        fail("parler", result)
    return out


# ---- voicecraft (external command template) -----------------------------------

def backend_voicecraft(text: str, reference: Path, ref_text: str, out: Path, opts: BackendOptions) -> Path:
    if not opts.voicecraft_command:
        raise VoiceCloneError(
            "VoiceCraft requires opts.voicecraft_command, e.g.:\n"
            "  python inference.py --prompt {reference} --prompt_text {ref_text} "
            "--text {text_file} --output {output}"
        )
    text_file = out.with_suffix(".txt")
    text_file.write_text(text, encoding="utf-8")
    command = opts.voicecraft_command.format(
        text=text, text_file=str(text_file.resolve()), reference=str(reference.resolve()),
        ref_text=ref_text or "", output=str(out.resolve()),
    )
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise VoiceCloneError(f"VoiceCraft command failed: {result.stderr[-2000:]}")
    if not out.exists():
        raise VoiceCloneError("VoiceCraft command finished without creating WAV.")
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
    "fish": backend_fish,
    "cosyvoice": backend_cosyvoice,
    "cosyvoice_api": backend_cosyvoice_api,
    "gptsovits": backend_gptsovits,
    "kokoclone": backend_kokoclone,
    "parler": backend_parler,
    "voicecraft": backend_voicecraft,
}

BACKENDS_NEEDING_REF_TEXT = {"f5tts", "cosyvoice", "cosyvoice_api", "gptsovits", "omnivoice", "fish", "voicecraft"}
