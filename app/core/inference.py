"""Inference engine — load base + LoRA adapter, generate chat responses."""

from __future__ import annotations

import threading
from typing import Any, Optional

from app.config import settings
from app.storage import store


class InferenceEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_persona_id: Optional[str] = None
        self._loaded_base_model: Optional[str] = None
        self._use_adapter: bool = True

    def _unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_persona_id = None
        self._loaded_base_model = None

    def _ensure_loaded(
        self,
        persona_id: str,
        use_adapter: bool = True,
    ) -> tuple[Any, Any]:
        persona = store.get_persona(persona_id)
        if not persona:
            raise ValueError(f"Persona {persona_id} not found")

        base_model = persona.get("base_model") or settings.default_base_model
        adapter_path = persona.get("adapter_path")

        if use_adapter and not adapter_path:
            raise ValueError(
                "This persona has not been trained yet. Complete training before chatting."
            )

        with self._lock:
            same_load = (
                self._model is not None
                and self._loaded_persona_id == persona_id
                and self._loaded_base_model == base_model
                and self._use_adapter == use_adapter
            )
            if same_load:
                return self._model, self._tokenizer

            self._unload()

            try:
                import torch
                from unsloth import FastLanguageModel
            except ImportError as e:
                raise RuntimeError(
                    "PyTorch/Unsloth not installed. Install dependencies to run inference."
                ) from e

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "No CUDA device found. Inference requires a local NVIDIA GPU."
                )

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=settings.default_max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            if use_adapter and adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)

            FastLanguageModel.for_inference(model)

            self._model = model
            self._tokenizer = tokenizer
            self._loaded_persona_id = persona_id
            self._loaded_base_model = base_model
            self._use_adapter = use_adapter

            return self._model, self._tokenizer

    def generate(
        self,
        persona_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_adapter: bool = True,
    ) -> str:
        model, tokenizer = self._ensure_loaded(persona_id, use_adapter=use_adapter)

        import torch

        persona = store.get_persona(persona_id)
        system_prompt = persona.get("system_prompt", "") if persona else ""

        chat_messages = []
        has_system = any(m.get("role") == "system" for m in messages)
        if system_prompt and not has_system:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)

        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    def chat(
        self,
        persona_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        compare_base: bool = False,
    ) -> dict[str, str]:
        response = self.generate(
            persona_id,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            use_adapter=True,
        )
        result: dict[str, str] = {"response": response}

        if compare_base:
            # Unload adapter model and load base only
            with self._lock:
                self._unload()
            base_response = self.generate(
                persona_id,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                use_adapter=False,
            )
            result["base_response"] = base_response
            # Restore adapter for next request
            with self._lock:
                self._unload()

        return result


inference_engine = InferenceEngine()
