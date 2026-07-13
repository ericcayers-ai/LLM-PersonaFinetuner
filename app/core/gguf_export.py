"""Export trained adapters for Ollama / GGUF use."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from app.config import settings
from app.storage import store


def export_for_ollama(persona_id: str) -> dict:
    """
    Export a trained persona adapter with Ollama-oriented instructions.
    Attempts GGUF conversion when tooling is available; otherwise saves metadata
    and documents manual conversion steps.
    """
    persona = store.get_persona(persona_id)
    if not persona:
        raise ValueError(f"Persona {persona_id} not found")

    adapter_path = persona.get("adapter_path")
    if not adapter_path or not Path(adapter_path).exists():
        raise ValueError("Persona has no trained adapter. Complete training first.")

    export_dir = settings.outputs_dir / "exports" / persona_id
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy adapter files
    adapter_dest = export_dir / "adapter"
    if adapter_dest.exists():
        shutil.rmtree(adapter_dest)
    shutil.copytree(adapter_path, adapter_dest)

    base_model = persona.get("base_model") or settings.default_base_model
    limitations = [
        "GGUF export merges the LoRA adapter with the base model — this can be large (several GB).",
        "Ollama requires a GGUF file; adapter-only weights cannot be loaded directly.",
        "4-bit quantized base models may need full-precision merge before conversion.",
        "On Windows, install llama.cpp and run conversion manually if auto-export fails.",
    ]

    result = {
        "persona_id": persona_id,
        "export_dir": str(export_dir),
        "adapter_path": str(adapter_dest),
        "base_model": base_model,
        "gguf_path": None,
        "status": "adapter_exported",
        "message": "Adapter copied. GGUF conversion requires additional tooling.",
        "limitations": limitations,
        "manual_steps": _manual_conversion_steps(base_model, str(adapter_dest), str(export_dir)),
    }

    gguf_path = _try_gguf_export(base_model, adapter_path, export_dir)
    if gguf_path:
        result["gguf_path"] = str(gguf_path)
        result["status"] = "gguf_exported"
        result["message"] = f"GGUF exported to {gguf_path.name}"
        result["ollama_command"] = f'ollama create {persona["name"].lower().replace(" ", "-")} -f Modelfile'

    manifest_path = export_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Write Ollama Modelfile template
    modelfile = export_dir / "Modelfile"
    if result.get("gguf_path"):
        modelfile.write_text(
            f'FROM ./{Path(result["gguf_path"]).name}\n'
            f'SYSTEM """{persona.get("system_prompt", "")}"""\n'
            "PARAMETER temperature 0.7\n",
            encoding="utf-8",
        )
    else:
        modelfile.write_text(
            "# Replace MODEL_PATH with your merged GGUF file after conversion\n"
            "FROM ./model.gguf\n"
            f'SYSTEM """{persona.get("system_prompt", "")}"""\n'
            "PARAMETER temperature 0.7\n",
            encoding="utf-8",
        )

    return result


def _manual_conversion_steps(base_model: str, adapter_path: str, export_dir: str) -> list[str]:
    return [
        f"1. Merge LoRA adapter with base model ({base_model}) using PEFT or Unsloth merge.",
        f"2. Convert merged weights to GGUF with llama.cpp: python convert_hf_to_gguf.py <merged_dir>",
        f"3. Place the .gguf file in {export_dir}",
        "4. Update Modelfile FROM path and run: ollama create my-persona -f Modelfile",
    ]


def _try_gguf_export(base_model: str, adapter_path: str, export_dir: Path) -> Path | None:
    """Attempt GGUF export via Unsloth if available."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=settings.default_max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

        gguf_path = export_dir / "persona.gguf"
        if hasattr(model, "save_pretrained_gguf"):
            model.save_pretrained_gguf(str(export_dir), tokenizer, quantization_method="q4_k_m")
            candidates = list(export_dir.glob("*.gguf"))
            return candidates[0] if candidates else None

        # Try unsloth's standalone save if available
        if hasattr(FastLanguageModel, "save_pretrained_gguf"):
            FastLanguageModel.save_pretrained_gguf(model, str(gguf_path), tokenizer)
            return gguf_path if gguf_path.exists() else None

    except Exception:
        pass

    # Check if llama.cpp convert script exists in PATH
    try:
        subprocess.run(["python", "--version"], capture_output=True, check=True)
    except Exception:
        return None

    return None
