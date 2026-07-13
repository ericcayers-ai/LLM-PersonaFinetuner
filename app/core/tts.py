"""Watermarked zero-shot voice cloning with Chatterbox."""

from __future__ import annotations

import importlib.util
import io
import threading
from pathlib import Path
from typing import Any, Optional

from app.config import settings


class VoiceCloneEngine:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self._device: Optional[str] = None

    @staticmethod
    def is_installed() -> bool:
        return importlib.util.find_spec("chatterbox") is not None

    @staticmethod
    def _resolve_device() -> str:
        if settings.tts_device != "auto":
            return settings.tts_device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            device = self._resolve_device()
            try:
                if settings.tts_model == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS

                    model = ChatterboxTurboTTS.from_pretrained(device=device)
                elif settings.tts_model == "multilingual-v3":
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                    model = ChatterboxMultilingualTTS.from_pretrained(
                        device=device,
                        t3_model="v3",
                    )
                else:
                    raise RuntimeError(
                        "PF_TTS_MODEL must be 'turbo' or 'multilingual-v3'."
                    )
            except ImportError as exc:
                raise RuntimeError(
                    "chatterbox-tts is not installed. Run the speech dependency setup first."
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load Chatterbox '{settings.tts_model}' on {device}: {exc}"
                ) from exc

            self._model = model
            self._device = device
            return model

    def synthesize(
        self,
        text: str,
        reference_path: str,
        *,
        language: str = "en",
    ) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("Text to synthesize is empty.")
        if len(text) > settings.max_tts_characters:
            raise ValueError(
                f"Text exceeds the {settings.max_tts_characters}-character speech limit."
            )
        if not Path(reference_path).is_file():
            raise ValueError("The voice reference audio is missing. Upload it again.")
        if settings.tts_model == "turbo" and language.lower() != "en":
            raise ValueError(
                "Chatterbox Turbo supports English. Set PF_TTS_MODEL=multilingual-v3 "
                "for multilingual synthesis."
            )

        model = self._ensure_loaded()
        kwargs: dict[str, Any] = {"audio_prompt_path": reference_path}
        if settings.tts_model == "multilingual-v3":
            kwargs["language_id"] = language.lower()

        with self._lock:
            waveform = model.generate(text, **kwargs)

        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required to encode synthesized speech.") from exc

        waveform = waveform.detach().cpu()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        output = io.BytesIO()
        torchaudio.save(output, waveform, model.sr, format="wav")
        return output.getvalue()

    def release(self) -> None:
        with self._lock:
            self._model = None
            self._device = None


voice_engine = VoiceCloneEngine()
