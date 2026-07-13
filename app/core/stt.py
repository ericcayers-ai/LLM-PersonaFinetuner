"""Low-latency local speech recognition powered by faster-whisper."""

from __future__ import annotations

import importlib.util
import io
import threading
from typing import Any, Optional

from app.config import settings


class SpeechToTextEngine:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self._device: Optional[str] = None

    @staticmethod
    def is_installed() -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _resolve_runtime(self) -> tuple[str, str]:
        device = settings.stt_device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = settings.stt_compute_type
        if compute_type == "auto":
            compute_type = "int8_float16" if device == "cuda" else "int8"
        return device, compute_type

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Run the speech dependency setup first."
                ) from exc

            device, compute_type = self._resolve_runtime()
            try:
                self._model = WhisperModel(
                    settings.stt_model,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception as exc:
                if settings.stt_device == "auto" and device == "cuda":
                    try:
                        self._model = WhisperModel(
                            settings.stt_model,
                            device="cpu",
                            compute_type="int8",
                        )
                        device = "cpu"
                    except Exception as cpu_exc:
                        raise RuntimeError(
                            f"Could not load faster-whisper model '{settings.stt_model}' "
                            f"on CUDA ({exc}) or CPU ({cpu_exc})."
                        ) from cpu_exc
                else:
                    raise RuntimeError(
                        f"Could not load faster-whisper model '{settings.stt_model}' "
                        f"on {device}: {exc}"
                    ) from exc
            self._device = device
            return self._model

    def transcribe(
        self,
        content: bytes,
        *,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        if not content:
            raise ValueError("Audio is empty.")

        model = self._ensure_loaded()
        with self._lock:
            segments_iter, info = model.transcribe(
                io.BytesIO(content),
                language=language or None,
                beam_size=1,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 350},
                condition_on_previous_text=False,
            )
            segments = [
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": segment.text.strip(),
                }
                for segment in segments_iter
                if segment.text.strip()
            ]

        text = " ".join(segment["text"] for segment in segments).strip()
        if not text:
            raise ValueError("No speech was detected in the audio.")

        return {
            "text": text,
            "language": getattr(info, "language", None) or language or "unknown",
            "language_probability": round(
                float(getattr(info, "language_probability", 0.0) or 0.0), 4
            ),
            "duration_seconds": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
            "segments": segments,
        }

    def release(self) -> None:
        with self._lock:
            self._model = None
            self._device = None


stt_engine = SpeechToTextEngine()
